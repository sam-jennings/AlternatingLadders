#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

// -----------------------------------------------------------------------------
// Alternating ladder cooperative game Monte Carlo simulator.
//
// The implementation follows the specification provided in the task:
//   * Two players (A/B) draw from independent decks containing one red and one
//     black suit with ranks 1..13.
//   * Two ladders are built concurrently: L1 expects R1,B2,R3,... and L2 expects
//     B1,R2,B3,...  The next needed red and black ranks are tracked separately.
//   * When a rank is completed the duplicate copy of that colour/rank becomes
//     unusable (dead) but it may remain in hands and be discarded later.
//   * Discards normally move to an irretrievable trash, although optional
//     reserve and market mechanics are implemented.
//   * A deterministic “perfect information greedy” policy is followed when
//     choosing plays, moves to reserve and discards.
//
// The simulator supports the command line parameters requested in the task and
// provides both Monte-Carlo statistics and a verbose trace mode for debugging.
// -----------------------------------------------------------------------------

namespace {

// Basic card representation.
struct Card {
    int color;   // 0 = red, 1 = black
    int rank;    // 1..13 (Ace == 1)
    int owner;   // 0 = player A, 1 = player B
};

enum class LocationType { Deck, Hand, Reserve, Market, Trash, Played };

struct CardLocation {
    LocationType type = LocationType::Deck;
    int holder = -1;  // player index for Deck/Hand, otherwise -1
};

struct Settings {
    int64_t trials = 200000;
    int hand_size = 5;
    int reserve_capacity = 0;
    bool ace_grace = false;
    bool market_enabled = false;
    bool skip_forced_discard = false;
    uint64_t seed = 42;
    bool trace = false;
};

// Game state containing all mutable components.
struct GameState {
    const Settings* cfg = nullptr;
    std::vector<Card> cards;              // 52 cards, indexed by card id
    std::vector<CardLocation> locations;  // parallel to cards

    std::array<std::vector<int>, 2> deck;     // decks for player A/B (top == back)
    std::array<std::vector<int>, 2> hand;     // hands for player A/B
    std::vector<int> reserve;                 // shared reserve (face-up parking)
    std::array<std::vector<int>, 28> market;  // [color*14 + rank], ranks 1..13 used
    std::vector<int> trash;                   // irretrievable discards

    std::array<int, 2> next_needed;  // next required rank per colour/ladder
    std::array<int, 2> ladder_direction;  // +1 ascending from Ace, -1 descending from King, 0 undecided
    std::array<bool, 2> ladder_started;   // has the ladder for this colour begun?
    std::array<int, 2> forced_start;      // required initial rank when ladder not yet started (0 = free choice)
    bool ace_started = false;        // has any Ace been played?
    int turns = 0;                   // number of turns elapsed
    std::array<int, 2> hand_limit;   // dynamic per-player hand limits
    int consecutive_skips = 0;       // consecutive skip-forced turns
    int total_skips = 0;             // total skips executed this game
};

bool ladder_completed(const GameState& state, int color) {
    if (!state.ladder_started[color]) {
        return false;
    }
    if (state.ladder_direction[color] > 0) {
        return state.next_needed[color] > 13;
    }
    return state.next_needed[color] < 1;
}

std::vector<int> current_required_ranks(const GameState& state, int color) {
    if (ladder_completed(state, color)) {
        return {};
    }
    if (state.ladder_started[color]) {
        return {state.next_needed[color]};
    }
    if (state.forced_start[color] != 0) {
        return {state.forced_start[color]};
    }
    return {1, 13};
}

std::vector<int> remaining_ranks(const GameState& state, int color) {
    std::vector<int> ranks;
    if (ladder_completed(state, color)) {
        return ranks;
    }
    if (state.ladder_started[color]) {
        if (state.ladder_direction[color] > 0) {
            for (int r = state.next_needed[color]; r <= 13; ++r) {
                ranks.push_back(r);
            }
        } else {
            for (int r = state.next_needed[color]; r >= 1; --r) {
                ranks.push_back(r);
            }
        }
    } else {
        if (state.forced_start[color] == 13) {
            for (int r = 13; r >= 1; --r) {
                ranks.push_back(r);
            }
        } else {
            for (int r = 1; r <= 13; ++r) {
                ranks.push_back(r);
            }
        }
    }
    return ranks;
}

bool is_current_need(const GameState& state, int color, int rank) {
    for (int required : current_required_ranks(state, color)) {
        if (required == rank) {
            return true;
        }
    }
    return false;
}

bool is_future_need(const GameState& state, int color, int rank) {
    if (rank < 1 || rank > 13) {
        return false;
    }
    if (ladder_completed(state, color)) {
        return false;
    }
    if (state.ladder_started[color]) {
        if (state.ladder_direction[color] > 0) {
            return rank >= state.next_needed[color];
        }
        return rank <= state.next_needed[color];
    }
    if (state.forced_start[color] == 13) {
        return rank <= 13;
    }
    return rank >= 1;
}

int progress_metric(const GameState& state, int color) {
    if (!state.ladder_started[color]) {
        if (state.forced_start[color] == 13) {
            return 1;
        }
        return state.next_needed[color];
    }
    if (state.ladder_direction[color] > 0) {
        return state.next_needed[color];
    }
    return 14 - state.next_needed[color];
}

void set_forced_start(GameState& state, int color, int rank) {
    state.forced_start[color] = rank;
    if (!state.ladder_started[color]) {
        state.next_needed[color] = rank;
        state.ladder_direction[color] = (rank == 13) ? -1 : 1;
    }
}

void advance_ladder(GameState& state, int color, int rank_played) {
    if (!state.ladder_started[color]) {
        if (state.ladder_direction[color] == 0) {
            state.ladder_direction[color] = (rank_played == 13) ? -1 : 1;
        } else {
            if (state.ladder_direction[color] == 1) {
                assert(rank_played == 1);
            } else {
                assert(rank_played == 13);
            }
        }
        state.ladder_started[color] = true;
        state.forced_start[color] = 0;
        state.next_needed[color] = rank_played + state.ladder_direction[color];

        int other = 1 - color;
        if (!state.ladder_started[other]) {
            int forced = (state.ladder_direction[color] == 1) ? 13 : 1;
            set_forced_start(state, other, forced);
        }
    } else {
        assert(is_current_need(state, color, rank_played));
        state.next_needed[color] += state.ladder_direction[color];
    }
}

bool all_ladders_completed(const GameState& state) {
    return ladder_completed(state, 0) && ladder_completed(state, 1);
}

// Utility: remove a card id from a vector (order is not preserved but we keep
// deterministic behaviour by erasing via iterator).
void remove_from_vector(std::vector<int>& vec, int card_id) {
    auto it = std::find(vec.begin(), vec.end(), card_id);
    assert(it != vec.end());
    vec.erase(it);
}

int market_index(int color, int rank) {
    return color * 14 + rank;
}

std::string card_to_string(const Card& c) {
    char colour = c.color == 0 ? 'R' : 'B';
    char owner = c.owner == 0 ? 'A' : 'B';
    std::string rank;
    if (c.rank == 1) {
        rank = "A";
    } else if (c.rank == 11) {
        rank = "J";
    } else if (c.rank == 12) {
        rank = "Q";
    } else if (c.rank == 13) {
        rank = "K";
    } else {
        rank = std::to_string(c.rank);
    }
    return std::string(1, colour) + rank + owner;
}

std::string pair_to_string(int color, int rank) {
    char colour = color == 0 ? 'R' : 'B';
    std::string rank_str = rank == 1 ? "A" : std::to_string(rank);
    return std::string(1, colour) + rank_str;
}

// Initialise decks, hands and bookkeeping for a single game.
GameState initialise_game(const Settings& cfg, std::mt19937_64& rng) {
    GameState state;
    state.cfg = &cfg;
    state.cards.reserve(52);
    state.locations.resize(52);

    int id = 0;
    for (int owner = 0; owner < 2; ++owner) {
        for (int color = 0; color < 2; ++color) {
            for (int rank = 1; rank <= 13; ++rank) {
                state.cards.push_back(Card{color, rank, owner});
                state.locations[id] = {LocationType::Deck, owner};
                state.deck[owner].push_back(id);
                ++id;
            }
        }
    }

    for (int p = 0; p < 2; ++p) {
        std::shuffle(state.deck[p].begin(), state.deck[p].end(), rng);
    }

    state.next_needed = {1, 1};
    state.ladder_direction = {0, 0};
    state.ladder_started = {false, false};
    state.forced_start = {0, 0};
    state.ace_started = false;
    state.turns = 0;
    state.hand_limit = {cfg.hand_size, cfg.hand_size};
    state.consecutive_skips = 0;
    state.total_skips = 0;

    // Initial draw up to hand size from each deck.
    for (int p = 0; p < 2; ++p) {
        while (!state.deck[p].empty() &&
               static_cast<int>(state.hand[p].size()) < state.hand_limit[p]) {
            int card_id = state.deck[p].back();
            state.deck[p].pop_back();
            state.hand[p].push_back(card_id);
            state.locations[card_id] = {LocationType::Hand, p};
        }
    }

    return state;
}

// Helper that draws one card for player p if possible.
int draw_one(GameState& state, int player, bool ignore_limit = false) {
    if (state.deck[player].empty()) {
        return -1;
    }
    if (!ignore_limit &&
        static_cast<int>(state.hand[player].size()) >= state.hand_limit[player]) {
        return -1;
    }
    int card_id = state.deck[player].back();
    state.deck[player].pop_back();
    state.hand[player].push_back(card_id);
    state.locations[card_id] = {LocationType::Hand, player};
    return card_id;
}

// Draw until the hand reaches the configured size (used after playing a card
// from hand).
void refill_hand(GameState& state, int player) {
    while (static_cast<int>(state.hand[player].size()) < state.hand_limit[player] &&
           !state.deck[player].empty()) {
        if (draw_one(state, player) < 0) {
            break;
        }
    }
}

// Return true if the other copy of (color,rank) (owned by 1-owner) is unseen –
// i.e. still hidden in the deck (location Deck) rather than visible/trashed.
bool other_copy_unseen(const GameState& state, int color, int rank, int owner) {
    int other_owner = 1 - owner;
    for (size_t i = 0; i < state.cards.size(); ++i) {
        const Card& c = state.cards[i];
        if (c.color == color && c.rank == rank && c.owner == other_owner) {
            const CardLocation& loc = state.locations[i];
            return loc.type == LocationType::Deck;
        }
    }
    assert(false && "other copy not found");
    return false;
}

bool other_copy_in_trash(const GameState& state, int color, int rank, int owner) {
    int other_owner = 1 - owner;
    for (size_t i = 0; i < state.cards.size(); ++i) {
        const Card& c = state.cards[i];
        if (c.color == color && c.rank == rank && c.owner == other_owner) {
            return state.locations[i].type == LocationType::Trash;
        }
    }
    assert(false && "other copy not found");
    return false;
}

// Detect whether a future requirement has become impossible.  Returns the
// failing pair if unwinnable, std::nullopt otherwise.
std::optional<std::pair<int, int>> detect_unwinnable(const GameState& state) {
    for (int color = 0; color < 2; ++color) {
        for (int rank : remaining_ranks(state, color)) {
            bool lost0 = false;
            bool lost1 = false;
            for (size_t i = 0; i < state.cards.size(); ++i) {
                const Card& c = state.cards[i];
                if (c.color != color || c.rank != rank) {
                    continue;
                }
                const CardLocation& loc = state.locations[i];
                if (loc.type == LocationType::Trash) {
                    if (c.owner == 0) {
                        lost0 = true;
                    } else {
                        lost1 = true;
                    }
                }
            }
            if (lost0 && lost1) {
                return std::make_pair(color, rank);
            }
        }
    }
    return std::nullopt;
}

// Check whether the current player has an immediate legal play from hand or
// shared areas (reserve/market).
bool can_play_now(const GameState& state, int player, bool allow_market) {
    for (int color = 0; color < 2; ++color) {
        for (int rank : current_required_ranks(state, color)) {
            if (rank < 1 || rank > 13) {
                continue;
            }
            if (allow_market) {
                const auto& pile = state.market[market_index(color, rank)];
                if (!pile.empty()) {
                    return true;
                }
            }
            for (int card_id : state.reserve) {
                const Card& c = state.cards[card_id];
                if (c.color == color && c.rank == rank) {
                    return true;
                }
            }
            for (int card_id : state.hand[player]) {
                const Card& c = state.cards[card_id];
                if (c.color == color && c.rank == rank) {
                    return true;
                }
            }
        }
    }
    return false;
}

struct PlayCandidate {
    int card_id;
    int color;
    int rank;
    int source_priority;  // 0 = market, 1 = reserve, 2 = hand
    int owner;
};

// Collect the best play and execute it.  Deterministic ordering:
//   1. smallest rank
//   2. market before reserve before hand
//   3. colour tie-breaker (red before black)
//   4. owner A before B (useful when both copies appear in market/reserve)
void play_from_best_source(GameState& state, int player, bool allow_market,
                           std::string& action) {
    std::vector<PlayCandidate> candidates;
    for (int color = 0; color < 2; ++color) {
        for (int rank : current_required_ranks(state, color)) {
            if (rank < 1 || rank > 13) {
                continue;
            }
            if (allow_market) {
                auto idx = market_index(color, rank);
                for (int card_id : state.market[idx]) {
                    const Card& c = state.cards[card_id];
                    candidates.push_back({card_id, color, rank, 0, c.owner});
                }
            }
            for (int card_id : state.reserve) {
                const Card& c = state.cards[card_id];
                if (c.color == color && c.rank == rank) {
                    candidates.push_back({card_id, color, rank, 1, c.owner});
                }
            }
            for (int card_id : state.hand[player]) {
                const Card& c = state.cards[card_id];
                if (c.color == color && c.rank == rank) {
                    candidates.push_back({card_id, color, rank, 2, c.owner});
                }
            }
        }
    }

    assert(!candidates.empty());

    // Determine the minimum rank among candidates.
    int min_rank = candidates.front().rank;
    for (const auto& cand : candidates) {
        min_rank = std::min(min_rank, cand.rank);
    }

    // Filter candidates to the chosen rank.
    std::vector<PlayCandidate> filtered;
    for (const auto& cand : candidates) {
        if (cand.rank == min_rank) {
            filtered.push_back(cand);
        }
    }

    // Apply deterministic ordering.
    std::sort(filtered.begin(), filtered.end(), [](const PlayCandidate& a,
                                                   const PlayCandidate& b) {
        if (a.rank != b.rank) {
            return a.rank < b.rank;
        }
        if (a.source_priority != b.source_priority) {
            return a.source_priority < b.source_priority;
        }
        if (a.color != b.color) {
            return a.color < b.color;  // prefer red over black on ties
        }
        if (a.owner != b.owner) {
            return a.owner < b.owner;
        }
        return a.card_id < b.card_id;
    });

    PlayCandidate best = filtered.front();
    const Card& card = state.cards[best.card_id];

    // Update bookkeeping depending on source.
    if (best.source_priority == 0) {  // market
        auto& pile = state.market[market_index(best.color, best.rank)];
        remove_from_vector(pile, best.card_id);
        action = "play " + card_to_string(card) + " from market";
    } else if (best.source_priority == 1) {  // reserve
        remove_from_vector(state.reserve, best.card_id);
        action = "play " + card_to_string(card) + " from reserve";
    } else {  // hand
        remove_from_vector(state.hand[player], best.card_id);
        action = "play " + card_to_string(card) + " from hand";
        refill_hand(state, player);
    }

    state.locations[best.card_id] = {LocationType::Played, -1};

    // Advance the ladder.
    advance_ladder(state, best.color, best.rank);

    if (card.rank == 1) {
        state.ace_started = true;
    }
}

// Attempt to move a card from the current player's hand into reserve to
// protect future ranks with both copies visible.  Returns true if a move was
// performed (which counts as the discard action for the turn).
bool protect_future_rank(GameState& state, int player, std::string& action) {
    if (state.cfg->reserve_capacity == 0 ||
        static_cast<int>(state.reserve.size()) >= state.cfg->reserve_capacity) {
        return false;
    }

    // Search for the lowest rank still required where both copies are
    // simultaneously visible (in hands/reserve/market) and one of them is in
    // the current player's hand.  We only move from the current player's hand;
    // this respects turn ownership while still following the intent of the
    // policy.
    for (int color = 0; color < 2; ++color) {
        for (int rank : remaining_ranks(state, color)) {
            bool other_visible = false;
            bool other_safe = false;
            // Locate the other copy and check visibility.
            for (size_t i = 0; i < state.cards.size(); ++i) {
                const Card& c = state.cards[i];
                if (c.color == color && c.rank == rank) {
                    if (c.owner == player) {
                        continue;
                    }
                    const CardLocation& loc = state.locations[i];
                    if (loc.type == LocationType::Hand ||
                        loc.type == LocationType::Reserve ||
                        loc.type == LocationType::Market) {
                        other_visible = true;
                    }
                    if (loc.type == LocationType::Reserve ||
                        loc.type == LocationType::Market) {
                        other_safe = true;
                    }
                    break;
                }
            }

            // See whether the current player holds their copy of this rank.
            int my_id = -1;
            for (int card_id : state.hand[player]) {
                const Card& c = state.cards[card_id];
                if (c.color == color && c.rank == rank) {
                    my_id = card_id;
                    break;
                }
            }

            if (my_id >= 0 && other_visible && !other_safe) {
                remove_from_vector(state.hand[player], my_id);
                state.reserve.push_back(my_id);
                state.locations[my_id] = {LocationType::Reserve, -1};
                action = "stash " + card_to_string(state.cards[my_id]) +
                         " into reserve";
                return true;
            }
        }
    }
    return false;
}

// Choose a card to discard from the current player's hand following the policy.
int choose_discard(const GameState& state, int player) {
    const auto& hand = state.hand[player];
    if (hand.empty()) {
        return -1;
    }

    int preferred_color = -1;  // colour to protect (smaller progress metric)
    int metric0 = progress_metric(state, 0);
    int metric1 = progress_metric(state, 1);
    if (metric0 < metric1) {
        preferred_color = 0;
    } else if (metric1 < metric0) {
        preferred_color = 1;
    }

    // Collect candidates that satisfy the safest criterion.
    struct Candidate {
        int card_id;
        int rank;
        int color;
    };

    std::vector<Candidate> safe;
    for (int card_id : hand) {
        const Card& c = state.cards[card_id];
        bool colour_ok = (preferred_color == -1) || (c.color != preferred_color);
        if (!colour_ok) {
            continue;
        }
        if (other_copy_unseen(state, c.color, c.rank, c.owner)) {
            safe.push_back({card_id, c.rank, c.color});
        }
    }

    auto pick_highest = [](const std::vector<Candidate>& v) {
        assert(!v.empty());
        return std::max_element(v.begin(), v.end(),
                                [](const Candidate& a, const Candidate& b) {
                                    if (a.rank != b.rank) {
                                        return a.rank < b.rank;
                                    }
                                    if (a.color != b.color) {
                                        return a.color < b.color;  // prefer black on ties
                                    }
                                    return a.card_id < b.card_id;
                                })
            ->card_id;
    };

    if (!safe.empty()) {
        return pick_highest(safe);
    }

    // Risky path: avoid discarding a currently next-needed rank if possible.
    std::vector<Candidate> risky;
    std::vector<Candidate> critical;
    for (int card_id : hand) {
        const Card& c = state.cards[card_id];
        if (is_current_need(state, c.color, c.rank)) {
            continue;  // keep this rank if we can
        }
        bool is_critical = false;
        if (is_future_need(state, c.color, c.rank)) {
            is_critical = other_copy_in_trash(state, c.color, c.rank, c.owner);
        }
        if (is_critical) {
            critical.push_back({card_id, c.rank, c.color});
        } else {
            risky.push_back({card_id, c.rank, c.color});
        }
    }

    if (!risky.empty()) {
        return pick_highest(risky);
    }
    if (!critical.empty()) {
        return pick_highest(critical);
    }

    // No choice left – discard the highest rank regardless.
    std::vector<Candidate> everything;
    std::vector<Candidate> fallback_critical;
    for (int card_id : hand) {
        const Card& c = state.cards[card_id];
        bool is_critical = false;
        if (is_future_need(state, c.color, c.rank)) {
            is_critical = other_copy_in_trash(state, c.color, c.rank, c.owner);
        }
        if (is_critical) {
            fallback_critical.push_back({card_id, c.rank, c.color});
        } else {
            everything.push_back({card_id, c.rank, c.color});
        }
    }
    if (!everything.empty()) {
        return pick_highest(everything);
    }
    return pick_highest(fallback_critical);
}

// Execute a discard, either to reserve (if the caller already decided to move
// a card there) or to the permanent trash.
void discard_to_trash(GameState& state, int player, int card_id, std::string& action) {
    assert(card_id >= 0);
    remove_from_vector(state.hand[player], card_id);
    if (state.cfg->market_enabled) {
        auto idx = market_index(state.cards[card_id].color, state.cards[card_id].rank);
        state.market[idx].push_back(card_id);
        state.locations[card_id] = {LocationType::Market, -1};
        action = "discard " + card_to_string(state.cards[card_id]) + " to market";
    } else {
        state.trash.push_back(card_id);
        state.locations[card_id] = {LocationType::Trash, -1};
        action = "discard " + card_to_string(state.cards[card_id]) + " to trash";
    }
    if (static_cast<int>(state.hand[player].size()) < state.hand_limit[player]) {
        int drawn = draw_one(state, player);
        if (drawn >= 0) {
            action += " (drew " + card_to_string(state.cards[drawn]) + ")";
        }
    }
}

// Prepare a human-readable description of a vector of card ids, sorted for
// reproducibility.
std::string describe_cards(const GameState& state, const std::vector<int>& ids) {
    std::vector<std::string> names;
    names.reserve(ids.size());
    for (int id : ids) {
        names.push_back(card_to_string(state.cards[id]));
    }
    std::sort(names.begin(), names.end());
    std::ostringstream oss;
    for (size_t i = 0; i < names.size(); ++i) {
        if (i) {
            oss << ' ';
        }
        oss << names[i];
    }
    if (names.empty()) {
        return "-";
    }
    return oss.str();
}

// Render the market for tracing.
std::string describe_market(const GameState& state) {
    if (!state.cfg->market_enabled) {
        return "(disabled)";
    }
    std::vector<std::string> parts;
    for (int color = 0; color < 2; ++color) {
        for (int rank = 1; rank <= 13; ++rank) {
            const auto& pile = state.market[market_index(color, rank)];
            if (pile.empty()) {
                continue;
            }
            std::ostringstream oss;
            oss << pair_to_string(color, rank) << ':'
                << describe_cards(state, pile);
            parts.push_back(oss.str());
        }
    }
    if (parts.empty()) {
        return "-";
    }
    std::sort(parts.begin(), parts.end());
    std::ostringstream oss;
    for (size_t i = 0; i < parts.size(); ++i) {
        if (i) {
            oss << " | ";
        }
        oss << parts[i];
    }
    return oss.str();
}

struct TrialResult {
    bool completed = false;
    bool unwinnable = false;
    int turns = 0;
    int skips = 0;
    std::optional<std::pair<int, int>> failure_reason;
};

TrialResult play_single_game(const Settings& cfg, std::mt19937_64& rng, bool trace_output) {
    GameState state = initialise_game(cfg, rng);
    TrialResult result;
    int current_player = 0;  // player 0 == A goes first

    auto trace_state = [&](const std::string& preface, const std::string& action) {
        if (!trace_output) {
            return;
        }
        auto describe_next = [&](int color) {
            auto required = current_required_ranks(state, color);
            if (required.empty()) {
                return std::string("-");
            }
            std::ostringstream oss;
            for (size_t i = 0; i < required.size(); ++i) {
                if (i) {
                    oss << '/';
                }
                int rank = required[i];
                if (rank == 1) {
                    oss << 'A';
                } else if (rank == 11) {
                    oss << 'J';
                } else if (rank == 12) {
                    oss << 'Q';
                } else if (rank == 13) {
                    oss << 'K';
                } else {
                    oss << rank;
                }
            }
            return oss.str();
        };
        std::cout << preface << "\n";
        std::cout << "  Next needed: R" << describe_next(0)
                  << " B" << describe_next(1) << "\n";
        std::cout << "  Hand A: " << describe_cards(state, state.hand[0]) << "\n";
        std::cout << "  Hand B: " << describe_cards(state, state.hand[1]) << "\n";
        std::cout << "  Reserve: " << describe_cards(state, state.reserve) << "\n";
        if (cfg.market_enabled) {
            std::cout << "  Market:  " << describe_market(state) << "\n";
        }
        std::cout << "  Trash:   " << describe_cards(state, state.trash) << "\n";
        std::cout << "  Action:  " << action << "\n\n";
    };

    while (true) {
        auto fail = detect_unwinnable(state);
        if (fail.has_value()) {
            result.unwinnable = true;
            result.failure_reason = fail;
            break;
        }
        if (all_ladders_completed(state)) {
            result.completed = true;
            break;
        }

        ++state.turns;

        std::string action;
        std::string preface = "Turn " + std::to_string(state.turns) +
                              " (Player " + (current_player == 0 ? "A" : "B") + ")";

        bool market_allowed = state.cfg->market_enabled;
        if (state.cfg->skip_forced_discard && state.cfg->market_enabled &&
            state.consecutive_skips == 1) {
            market_allowed = false;
        }

        bool action_was_skip = false;
        bool action_performed = false;

        if (can_play_now(state, current_player, market_allowed)) {
            play_from_best_source(state, current_player, market_allowed, action);
            action_performed = true;
        } else {
            bool ace_grace_used = false;
            if (cfg.ace_grace && !state.ace_started) {
                if (!state.deck[current_player].empty() &&
                    static_cast<int>(state.hand[current_player].size()) <
                        state.hand_limit[current_player]) {
                    int drawn = draw_one(state, current_player);
                    if (drawn >= 0) {
                        action = "pass (ace grace, drew " +
                                 card_to_string(state.cards[drawn]) + ")";
                        ace_grace_used = true;
                        action_performed = true;
                    }
                }
            }

            if (!ace_grace_used) {
                bool moved_to_reserve =
                    protect_future_rank(state, current_player, action);
                if (moved_to_reserve) {
                    action_performed = true;
                } else {
                    int discard_id = choose_discard(state, current_player);
                    if (discard_id >= 0) {
                        bool skip_due_to_forced = false;
                        if (cfg.skip_forced_discard) {
                            const Card& candidate = state.cards[discard_id];
                            skip_due_to_forced = other_copy_in_trash(
                                state, candidate.color, candidate.rank,
                                candidate.owner);
                        }
                        if (skip_due_to_forced) {
                            const Card& candidate = state.cards[discard_id];
                            action = "skip (protect " +
                                     card_to_string(candidate) + ")";
                            if (cfg.market_enabled) {
                                int previous_limit =
                                    state.hand_limit[current_player];
                                if (previous_limit > 1) {
                                    state.hand_limit[current_player] =
                                        previous_limit - 1;
                                    action += " (hand limit now " +
                                              std::to_string(
                                                  state.hand_limit
                                                      [current_player]) +
                                              ")";
                                } else {
                                    action += " (hand limit already minimal)";
                                }
                            } else {
                                int drawn = draw_one(state, current_player, true);
                                if (drawn >= 0) {
                                    action += " (drew " +
                                              card_to_string(state.cards[drawn]) +
                                              ")";
                                } else if (state.deck[current_player].empty()) {
                                    action += " (deck empty)";
                                }
                            }
                            action_performed = true;
                            action_was_skip = true;
                        } else {
                            discard_to_trash(state, current_player, discard_id,
                                             action);
                            action_performed = true;
                        }
                    } else {
                        action = "pass (hand empty)";
                        action_performed = true;
                    }
                }
            }
        }

        if (action_was_skip) {
            state.consecutive_skips++;
            state.total_skips++;
        } else if (action_performed) {
            state.consecutive_skips = 0;
        }

        trace_state(preface, action);

        auto fail_after = detect_unwinnable(state);
        if (fail_after.has_value()) {
            result.unwinnable = true;
            result.failure_reason = fail_after;
            break;
        }
        if (all_ladders_completed(state)) {
            result.completed = true;
            break;
        }

        current_player = 1 - current_player;
    }

    result.turns = state.turns;
    result.skips = state.total_skips;
    return result;
}

// Simple command line parser.
Settings parse_args(int argc, char** argv) {
    Settings cfg;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto require_value = [&](const std::string& name) {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << name << "\n";
                std::exit(1);
            }
        };

        if (arg == "--trials") {
            require_value(arg);
            cfg.trials = std::stoll(argv[++i]);
        } else if (arg == "--hand") {
            require_value(arg);
            cfg.hand_size = std::stoi(argv[++i]);
        } else if (arg == "--reserve") {
            require_value(arg);
            cfg.reserve_capacity = std::stoi(argv[++i]);
        } else if (arg == "--ace-grace") {
            require_value(arg);
            cfg.ace_grace = std::stoi(argv[++i]) != 0;
        } else if (arg == "--market") {
            require_value(arg);
            cfg.market_enabled = std::stoi(argv[++i]) != 0;
        } else if (arg == "--skip") {
            require_value(arg);
            cfg.skip_forced_discard = std::stoi(argv[++i]) != 0;
        } else if (arg == "--seed") {
            require_value(arg);
            cfg.seed = static_cast<uint64_t>(std::stoull(argv[++i]));
        } else if (arg == "--trace") {
            require_value(arg);
            cfg.trace = std::stoi(argv[++i]) != 0;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: ./ladder_sim [options]\n"
                         "  --trials N       number of Monte Carlo trials (default 200000)\n"
                         "  --hand H         hand size (default 5)\n"
                         "  --reserve R      reserve capacity (default 0)\n"
                         "  --ace-grace 0/1  enable Ace grace (default 0)\n"
                         "  --market 0/1     enable reclaimable market (default 0)\n"
                         "  --skip 0/1       skip instead of forced discard (default 0)\n"
                         "  --seed S         RNG seed (default 42)\n"
                         "  --trace 0/1      verbose single game trace (default 0)\n";
            std::exit(0);
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            std::exit(1);
        }
    }
    return cfg;
}

}  // namespace

int main(int argc, char** argv) {
    Settings cfg = parse_args(argc, argv);
    std::mt19937_64 rng(cfg.seed);

    if (cfg.trace) {
        TrialResult res = play_single_game(cfg, rng, true);
        std::cout << "Trace complete. turns=" << res.turns
                  << " skips=" << res.skips;
        if (res.turns > 0) {
            std::ostringstream ratio_stream;
            ratio_stream << std::fixed << std::setprecision(4)
                         << static_cast<double>(res.skips) / res.turns;
            std::cout << " skip_ratio=" << ratio_stream.str();
        }
        if (res.completed) {
            std::cout << " result=success\n";
        } else if (res.unwinnable && res.failure_reason.has_value()) {
            std::cout << " result=unwinnable (" << pair_to_string(res.failure_reason->first,
                                                                      res.failure_reason->second)
                      << ")\n";
        } else {
            std::cout << " result=unknown\n";
        }
        return 0;
    }

    std::vector<int> success_turns;
    success_turns.reserve(static_cast<size_t>(cfg.trials));
    int64_t successes = 0;
    long double sum_turns = 0.0L;
    long double sum_sq_turns = 0.0L;
    long double sum_skips = 0.0L;
    long double sum_skip_ratios = 0.0L;
    std::map<std::string, int64_t> failure_counts;

    for (int64_t t = 0; t < cfg.trials; ++t) {
        TrialResult res = play_single_game(cfg, rng, false);
        sum_turns += res.turns;
        sum_sq_turns += static_cast<long double>(res.turns) * res.turns;
        sum_skips += res.skips;
        if (res.turns > 0) {
            sum_skip_ratios += static_cast<long double>(res.skips) / res.turns;
        }
        if (res.completed) {
            ++successes;
            success_turns.push_back(res.turns);
        } else if (res.unwinnable) {
            if (res.failure_reason.has_value()) {
                failure_counts[pair_to_string(res.failure_reason->first,
                                              res.failure_reason->second)]++;
            } else {
                failure_counts["unknown"]++;
            }
        }
    }

    long double mean = sum_turns / cfg.trials;
    long double variance = (sum_sq_turns / cfg.trials) - mean * mean;
    if (variance < 0) {
        variance = 0;  // guard against floating point noise
    }
    long double sd = std::sqrt(variance);

    std::sort(success_turns.begin(), success_turns.end());
    auto quantile = [&](double q) -> int {
        if (success_turns.empty()) {
            return 0;
        }
        double pos = q * (success_turns.size() - 1);
        size_t idx = static_cast<size_t>(std::round(pos));
        if (idx >= success_turns.size()) {
            idx = success_turns.size() - 1;
        }
        return success_turns[idx];
    };

    std::cout << "trials=" << cfg.trials << " hand=" << cfg.hand_size
              << " reserve=" << cfg.reserve_capacity
              << " ace_grace=" << (cfg.ace_grace ? 1 : 0)
              << " market=" << (cfg.market_enabled ? 1 : 0)
              << " skip=" << (cfg.skip_forced_discard ? 1 : 0)
              << " seed=" << cfg.seed << "\n";

    long double mean_skips = sum_skips / cfg.trials;
    long double mean_skip_ratio = sum_skip_ratios / cfg.trials;
    long double overall_skip_fraction =
        sum_turns > 0 ? sum_skips / sum_turns : 0.0L;

    std::ostringstream skip_stats;
    skip_stats << std::fixed;
    skip_stats << " mean_skips=" << std::setprecision(2)
               << static_cast<double>(mean_skips);
    skip_stats << " mean_skip_ratio=" << std::setprecision(4)
               << static_cast<double>(mean_skip_ratio);
    skip_stats << " overall_skip_fraction="
               << static_cast<double>(overall_skip_fraction);

    long double completable = static_cast<long double>(successes) / cfg.trials;
    long double winnable_rate = completable;  // identical because we stop on failure
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "completable=" << completable
              << " winnable=" << winnable_rate
              << " mean_turns=" << std::setprecision(2) << static_cast<double>(mean)
              << " sd=" << static_cast<double>(sd) << std::setprecision(4)
              << " p50=" << quantile(0.5)
              << " p90=" << quantile(0.9)
              << skip_stats.str() << "\n";

    long double mean_skips = sum_skips / cfg.trials;
    long double mean_skip_ratio = sum_skip_ratios / cfg.trials;
    long double overall_skip_fraction =
        sum_turns > 0 ? sum_skips / sum_turns : 0.0L;

    std::cout << "mean_skips=" << std::setprecision(2)
              << static_cast<double>(mean_skips)
              << " mean_skip_ratio=" << std::setprecision(4)
              << static_cast<double>(mean_skip_ratio)
              << " overall_skip_fraction=" << static_cast<double>(overall_skip_fraction)
              << std::setprecision(4) << "\n";

    if (!failure_counts.empty()) {
        std::cout << "failures_by_need:";
        for (const auto& [key, value] : failure_counts) {
            std::cout << ' ' << key << ':' << value;
        }
        std::cout << "\n";
    }

    return 0;
}

