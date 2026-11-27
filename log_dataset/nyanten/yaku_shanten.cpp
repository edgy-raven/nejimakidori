#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <numeric>
#include <string_view>
#include <vector>

namespace {

enum Kind { Tanyao, Flush, Outside, Honors };
// Four unary bits per rank encode required copies. Missing draws become a
// population count; fixed-meld copy limits become a bit-mask intersection.
using Patterns = std::array<
    std::array<std::array<std::array<std::vector<std::uint64_t>, 2>, 2>, 5>, 4>;

Patterns const &patterns()
{
  static Patterns const table = [] {
    Patterns result;
    for (unsigned kind = 0; kind < 4; ++kind) {
      std::vector<unsigned> ranks;
      std::vector<std::array<unsigned, 3>> groups;
      for (unsigned rank = 0; rank < (kind == Honors ? 7u : 9u); ++rank) {
        if (kind == Tanyao && (rank == 0 || rank == 8)) continue;
        if (kind == Outside && rank != 0 && rank != 8) continue;
        ranks.push_back(rank);
        groups.push_back({rank, rank, rank});
      }
      if (kind != Honors) {
        for (unsigned rank = 0; rank < 7; ++rank) {
          if (kind == Tanyao && (rank == 0 || rank == 6)) continue;
          if (kind == Outside && rank != 0 && rank != 6) continue;
          groups.push_back({rank, rank + 1, rank + 2});
        }
      }
      for (unsigned melds = 0; melds <= 4; ++melds) {
        for (unsigned pair = 0; pair < 2; ++pair) {
          std::array<std::uint8_t, 10> hand{};
          std::vector<std::array<std::uint8_t, 10>> rows;
          std::function<void(unsigned, unsigned)> visit =
              [&](unsigned first, unsigned remaining) {
            if (!remaining) {
              if (!pair) rows.push_back(hand);
              else for (unsigned rank : ranks) {
                if (hand[rank] > 2) continue;
                hand[rank] += 2;
                rows.push_back(hand);
                hand[rank] -= 2;
              }
              return;
            }
            for (unsigned group = first; group < groups.size(); ++group) {
              for (unsigned rank : groups[group]) ++hand[rank];
              auto const previous = hand[9];
              hand[9] |= groups[group][0] != groups[group][1];
              if (*std::max_element(hand.begin(), hand.begin() + 9) <= 4) {
                visit(group, remaining - 1);
              }
              hand[9] = previous;
              for (unsigned rank : groups[group]) --hand[rank];
            }
          };
          visit(0, melds);
          std::sort(rows.begin(), rows.end());
          rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
          for (auto const &shape : rows) {
            std::uint64_t copies = 0;
            for (unsigned rank = 0; rank < 9; ++rank) {
              copies |= ((std::uint64_t{1} << shape[rank]) - 1) << (4 * rank);
            }
            result[kind][melds][pair][shape[9]].push_back(copies);
          }
        }
      }
    }
    return result;
  }();
  return table;
}

// Candidate hands and yaku families repeatedly reuse unchanged suit groups.
// Cache exact per-group decomposition costs, independent of meld budget/flags.
using Costs = std::array<std::array<std::array<int, 2>, 2>, 5>;
Costs const &group_costs(
    std::uint8_t const *counts, unsigned const *fixed, unsigned kind,
    unsigned needed)
{
  struct Entry {
    std::array<std::uint8_t, 19> key{};
    Costs costs{};
    unsigned prepared = 0;
    bool valid = false;
  };
  thread_local std::array<Entry, 4096> cache;
  std::array<std::uint8_t, 19> key{};
  unsigned const ranks = kind == Honors ? 7 : 9;
  std::copy_n(counts, ranks, key.begin());
  std::copy_n(fixed, ranks, key.begin() + 9);
  key[18] = kind;
  auto &entry = cache[std::hash<std::string_view>{}(std::string_view(
      reinterpret_cast<char const *>(key.data()), key.size())) % cache.size()];
  if (!entry.valid || entry.key != key) {
    entry.key = key;
    entry.valid = true;
    entry.prepared = 0;
  }
  if (entry.prepared <= needed) {
    std::uint64_t held = 0, forbidden = 0;
    int usable = 0;
    for (unsigned rank = 0; rank < ranks; ++rank) {
      held |= ((std::uint64_t{1} << counts[rank]) - 1) << (4 * rank);
      if ((kind != Tanyao || (rank > 0 && rank < 8)) &&
          (kind != Outside || rank < 3 || rank > 5))
        usable += counts[rank];
      forbidden |= ((std::uint64_t{15} << (4 - fixed[rank])) & 15)
                   << (4 * rank);
    }
    for (unsigned added = entry.prepared; added <= needed; ++added) {
      for (unsigned pair = 0; pair < 2; ++pair) {
        auto &costs = entry.costs[added][pair];
        costs = {15, 15};
        // No completion can retain more than the usable tiles already held.
        // Scan each sequence class only until it attains that lower bound.
        int const lower = std::max(0, int(3 * added + 2 * pair) - usable);
        for (unsigned sequence = 0; sequence < 2; ++sequence) {
          for (auto const copies : patterns()[kind][added][pair][sequence]) {
            if (copies & forbidden) continue;
            costs[sequence] = std::min(
                costs[sequence], __builtin_popcountll(copies & ~held));
            if (costs[sequence] == lower) break;
          }
        }
      }
    }
  }
  entry.prepared = std::max(entry.prepared, needed + 1);
  return entry.costs;
}

int distance(
    std::uint8_t const *counts, std::array<unsigned, 34> const &fixed,
    unsigned melds, unsigned flags, unsigned family, unsigned suit)
{
  unsigned const needed = 4 - melds;
  int best = 15;
  if (family == Flush) {
    // Only one suit and honors participate. Combine complementary budgets
    // directly, rather than running the general flags DP for each suit.
    auto const suited = group_costs(counts + suit * 9,
        fixed.data() + suit * 9, Flush, needed);
    auto const &honors = group_costs(counts + 27, fixed.data() + 27, Honors, needed);
    for (unsigned added = 0; added <= needed; ++added) {
      for (unsigned pair = 0; pair < 2; ++pair) {
        if (!(flags & 2) && added + pair == 0) continue;
        best = std::min(best,
            std::min(suited[added][pair][0], suited[added][pair][1]) +
            honors[needed - added][1 - pair][0]);
      }
    }
  } else {
    // Tanyao/outside need only meld/pair/sequence state. Suit/honor flags
    // carry no additional information once fixed-meld legality is checked.
    std::array<int, 20> states;
    states.fill(15);
    states[flags & 1] = 0;
    for (unsigned group = 0; group < (family == Outside ? 4u : 3u); ++group) {
      auto const &costs = group_costs(counts + group * 9,
          fixed.data() + group * 9, group == 3 ? Honors : family, needed);
      std::array<int, 20> next;
      next.fill(15);
      for (unsigned used = 0; used <= needed; ++used) {
        for (unsigned head = 0; head < 2; ++head) {
          for (unsigned sequence = 0; sequence < 2; ++sequence) {
            int const cost = states[(used * 2 + head) * 2 + sequence];
            if (cost == 15) continue;
            for (unsigned added = 0; used + added <= needed; ++added) {
              for (unsigned pair = 0; head + pair < 2; ++pair) {
                for (unsigned extra_sequence = 0; extra_sequence < 2;
                     ++extra_sequence) {
                  auto &target = next[((used + added) * 2 + head + pair) * 2
                                      + (sequence | extra_sequence)];
                  target = std::min(target,
                      cost + costs[added][pair][extra_sequence]);
                }
              }
            }
          }
        }
      }
      states = next;
    }
    best = std::min(states[(needed * 2 + 1) * 2 + 1],
                    states[(needed * 2 + 1) * 2]);
  }
  if (!melds) {
    std::vector<int> suited, honors;
    for (unsigned tile = 0; tile < 34; ++tile) {
      int const cost = std::max(0, 2 - int(counts[tile]));
      if (family == Tanyao) {
        if (tile < 27 && tile % 9 != 0 && tile % 9 != 8) suited.push_back(cost);
      } else if (family == Outside) {
        if (tile >= 27 || tile % 9 == 0 || tile % 9 == 8) suited.push_back(cost);
      } else if (tile >= 27) honors.push_back(cost);
      else if (tile / 9 == suit) suited.push_back(cost);
    }
    std::sort(suited.begin(), suited.end());
    std::sort(honors.begin(), honors.end());
    if (family != Flush) {
      best = std::min(best, std::accumulate(suited.begin(), suited.begin() + 7, 0));
    } else for (unsigned pairs = 0; pairs < 7; ++pairs) {
      best = std::min(best,
          std::accumulate(honors.begin(), honors.begin() + pairs, 0) +
          std::accumulate(suited.begin(), suited.begin() + 7 - pairs, 0));
    }
  }
  return best;
}

}  // namespace

// Valid concealed counts and fixed chii/pon/kan groups use base tile IDs.
// Melds have four slots; a three-tile meld ends with sentinel 34.
// Packed replacement counts: tanyao, flush, outside. 15 = impossible.
extern "C" std::uint32_t nyanten_yaku_shanten(
    std::uint8_t const *counts, std::uint8_t const *melds, unsigned meld_count)
{
  struct Entry {
    std::array<std::uint8_t, 51> key{};
    std::uint32_t result = 0;
    bool valid = false;
  };
  thread_local std::array<Entry, 32768> cache;
  std::array<std::uint8_t, 51> key{};
  std::copy_n(counts, 34, key.begin());
  std::copy_n(melds, meld_count * 4, key.begin() + 34);
  key[50] = meld_count;
  auto &entry = cache[std::hash<std::string_view>{}(std::string_view(
      reinterpret_cast<char const *>(key.data()), key.size())) % cache.size()];
  if (entry.valid && entry.key == key) return entry.result;

  std::array<unsigned, 34> fixed{};
  unsigned flags = 0;
  bool outside = true;
  for (unsigned meld = 0; meld < meld_count; ++meld) {
    bool terminal = false;
    flags |= melds[4 * meld] != melds[4 * meld + 1];
    for (unsigned copy = 0; copy < 4; ++copy) {
      unsigned const tile = melds[4 * meld + copy];
      if (tile == 34) continue;
      ++fixed[tile];
      flags |= tile < 27 ? 2 : 0;
      terminal |= tile >= 27 || tile % 9 == 0 || tile % 9 == 8;
    }
    outside &= terminal;
  }
  std::uint32_t result = 0;
  for (unsigned family = 0; family < 3; ++family) {
    int best = 15;
    for (unsigned suit = 0; suit < (family == Flush ? 3u : 1u); ++suit) {
      bool allowed = family != Outside || outside;
      for (unsigned tile = 0; tile < 34; ++tile) {
        if (!fixed[tile]) continue;
        if (family == Tanyao) {
          allowed &= tile < 27 && tile % 9 != 0 && tile % 9 != 8;
        } else if (family == Flush) {
          allowed &= tile >= 27 || tile / 9 == suit;
        }
      }
      if (allowed) best = std::min(
          best, distance(counts, fixed, meld_count, flags, family, suit));
    }
    result |= std::uint32_t(best) << (8 * family);
  }
  entry.key = key;
  entry.result = result;
  entry.valid = true;
  return result;
}
