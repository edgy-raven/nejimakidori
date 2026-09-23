#include <nyanten/standard/replacement_number.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <numeric>
#include <string>
#include <unordered_set>

namespace {

using Hand = std::array<std::uint8_t, 34>;

std::uint8_t replacement_number(Hand const &hand)
{
  return Nyanten::Standard_::calculateReplacementNumber(
      hand.cbegin(), std::accumulate(hand.cbegin(), hand.cend(), 0u));
}

std::array<std::uint8_t, 2> special_replacement_numbers(Hand const &hand)
{
  unsigned pairs = 0, distinct = 0, orphans = 0;
  bool orphan_pair = false;
  for (unsigned tile = 0; tile < 34; ++tile) {
    pairs += hand[tile] >= 2;
    distinct += hand[tile] > 0;
    if (tile >= 27 || tile % 9 == 0 || tile % 9 == 8) {
      orphans += hand[tile] > 0;
      orphan_pair |= hand[tile] >= 2;
    }
  }
  return {
      static_cast<std::uint8_t>(
          7 - pairs + std::max(0, 7 - static_cast<int>(distinct))),
      static_cast<std::uint8_t>(14 - orphans - orphan_pair)};
}

std::uint8_t all_replacement_number(Hand const &hand, std::uint8_t best)
{
  if (best <= 1 || std::accumulate(hand.begin(), hand.end(), 0u) < 12) {
    return best;
  }
  auto const special = special_replacement_numbers(hand);
  return std::min({best, special[0], special[1]});
}

// There are at most 91 removable unordered pairs in a 14-tile hand.
struct StructuralNode {
  std::array<std::uint64_t, 2> improving{};
  std::array<std::uint8_t, 2> base{};
  bool valid = false;
  bool expanded = false;
};

struct Neighborhood {
  Hand hand{};
  std::uint64_t touched = 0;
  std::array<std::array<std::uint16_t, 34>, 34> offsets{};
  std::array<StructuralNode, 34 + 91 * 34> nodes{};
};

template <typename Space>
auto &neighbor(
    Space &space, unsigned first, unsigned second, unsigned draw)
{
  if (draw == first) return space.nodes[second];
  if (draw == second) return space.nodes[first];
  return space.nodes[space.offsets[first][second] + draw];
}

void inherit(StructuralNode &target, StructuralNode const &source)
{
  if (source.expanded || (!target.valid && source.valid)) target = source;
}

Neighborhood &neighborhood(Hand const &hand)
{
  thread_local std::array<Neighborhood, 16> spaces;
  thread_local std::uint64_t clock = 0;
  ++clock;
  auto *oldest = &spaces[0];
  for (auto &space : spaces) {
    if (space.touched && space.hand == hand) {
      space.touched = clock;
      return space;
    }
    if (space.touched < oldest->touched) oldest = &space;
  }
  *oldest = {};
  oldest->hand = hand;
  oldest->touched = clock;
  unsigned offset = 34;
  for (unsigned first = 0; first < 34; ++first) {
    if (!hand[first]) continue;
    for (unsigned second = first; second < 34; ++second) {
      if (hand[second] <= unsigned(first == second)) continue;
      oldest->offsets[first][second] = offset;
      oldest->offsets[second][first] = offset;
      offset += 34;
    }
  }
  // Rebase evaluated states from a nearby previous hand. If new = old-A+B,
  // new-C is old-A-C+B; new-C-D+X overlaps when C/D=B or X=A.
  for (auto const &previous : spaces) {
    if (&previous == oldest || !previous.touched) continue;
    int removed = -1;
    int added = -1;
    bool adjacent = true;
    for (unsigned tile = 0; tile < 34; ++tile) {
      int const delta = int(hand[tile]) - int(previous.hand[tile]);
      if (delta == -1 && removed == -1) removed = tile;
      else if (delta == 1 && added == -1) added = tile;
      else if (delta) { adjacent = false; break; }
    }
    if (!adjacent || removed == -1 || added == -1) continue;
    for (unsigned cut = 0; cut < 34; ++cut) {
      if (hand[cut]) inherit(
          oldest->nodes[cut], neighbor(previous, removed, cut, added));
    }
    unsigned pair = 34;
    for (unsigned first = 0; first < 34; ++first) {
      if (!hand[first]) continue;
      for (unsigned second = first; second < 34; ++second) {
        if (hand[second] <= unsigned(first == second)) continue;
        for (unsigned draw = 0; draw < 34; ++draw) {
          if (draw == unsigned(removed)) {
            inherit(oldest->nodes[pair + draw],
                    neighbor(previous, first, second, added));
          } else if (first == unsigned(added)) {
            inherit(oldest->nodes[pair + draw],
                    neighbor(previous, removed, second, draw));
          } else if (second == unsigned(added)) {
            inherit(oldest->nodes[pair + draw],
                    neighbor(previous, removed, first, draw));
          }
        }
        pair += 34;
      }
    }
  }
  return *oldest;
}

void prepare(Hand const &hand, StructuralNode &node)
{
  if (node.valid) return;
  node.base[0] = replacement_number(hand);
  node.base[1] = all_replacement_number(hand, node.base[0]);
  node.valid = true;
}

std::uint64_t ukeire(
    Hand &hand, Hand const &unavailable, StructuralNode &node,
    std::uint16_t &count, unsigned mode)
{
  if (!node.expanded) {
    // A trial draw changes one group. Keep the other groups' keys and the
    // unchanged prefix combinations instead of hashing the full hand again.
    namespace N = Nyanten::Standard_;
    auto const first = N::shupai_keys[
        N::hashShupai(hand.begin(), hand.begin() + 9)];
    auto const second = N::shupai_keys[
        N::hashShupai(hand.begin() + 9, hand.begin() + 18)];
    auto const third = N::shupai_keys[
        N::hashShupai(hand.begin() + 18, hand.begin() + 27)];
    auto const honors = N::zipai_keys[
        N::hashZipai(hand.begin() + 27, hand.end())];
    auto const pair = N::keys1[first][second];
    auto const suits = N::keys2[pair][third];
    unsigned const melds =
        (std::accumulate(hand.begin(), hand.end(), 0u) + 1) / 3;
    if (!node.valid) {
      node.base[0] = N::keys3[suits][honors][melds];
      node.base[1] = all_replacement_number(hand, node.base[0]);
      node.valid = true;
    }
    // A draw changes distinct/pair counts only at the 0->1/1->2 boundary.
    // Compute special-form statistics once instead of rescanning every draw.
    unsigned pairs = 0, distinct = 0, orphans = 0;
    bool orphan_pair = false;
    if (melds == 4) {
      for (unsigned tile = 0; tile < 34; ++tile) {
        pairs += hand[tile] >= 2;
        distinct += hand[tile] > 0;
        if (tile >= 27 || tile % 9 == 0 || tile % 9 == 8) {
          orphans += hand[tile] > 0;
          orphan_pair |= hand[tile] >= 2;
        }
      }
    }
    for (unsigned tile = 0; tile < 34; ++tile) {
      if (hand[tile] == 4) continue;
      ++hand[tile];
      std::uint8_t normal;
      if (tile < 9) {
        auto const changed = N::shupai_keys[
            N::hashShupai(hand.begin(), hand.begin() + 9)];
        normal = N::keys3[
            N::keys2[N::keys1[changed][second]][third]][honors][melds];
      } else if (tile < 18) {
        auto const changed = N::shupai_keys[
            N::hashShupai(hand.begin() + 9, hand.begin() + 18)];
        normal = N::keys3[
            N::keys2[N::keys1[first][changed]][third]][honors][melds];
      } else if (tile < 27) {
        auto const changed = N::shupai_keys[
            N::hashShupai(hand.begin() + 18, hand.begin() + 27)];
        normal = N::keys3[N::keys2[pair][changed]][honors][melds];
      } else {
        auto const changed = N::zipai_keys[
            N::hashZipai(hand.begin() + 27, hand.end())];
        normal = N::keys3[suits][changed][melds];
      }
      auto all = normal;
      if (melds == 4 && all > 1) {
        all = std::min(all, static_cast<std::uint8_t>(
            7 - pairs - (hand[tile] == 2) +
            std::max(0, 7 - int(distinct) - (hand[tile] == 1))));
        bool const terminal = tile >= 27 || tile % 9 == 0 || tile % 9 == 8;
        all = std::min(all, static_cast<std::uint8_t>(
            14 - orphans - (terminal && hand[tile] == 1) -
            (orphan_pair || (terminal && hand[tile] == 2))));
      }
      if (normal < node.base[0]) node.improving[0] |= std::uint64_t{1} << tile;
      if (all < node.base[1]) node.improving[1] |= std::uint64_t{1} << tile;
      --hand[tile];
    }
    node.expanded = true;
  }
  std::uint64_t mask = node.improving[mode];
  count = 0;
  for (auto remaining = mask; remaining; remaining &= remaining - 1) {
    unsigned const tile = __builtin_ctzll(remaining);
    if (unavailable[tile] == 4) mask &= ~(std::uint64_t{1} << tile);
    count += 4 - unavailable[tile];
  }
  return mask;
}

void analyze(
    std::uint8_t const *counts,
    std::uint8_t const *unavailable,
    std::uint64_t const candidate_mask,
    bool const calculate_upgrades,
    std::int8_t *shanten,
    std::uint64_t *ukeire_masks,
    std::uint16_t *ukeire_counts,
    std::uint64_t *upgrade_masks,
    std::uint8_t *upgrade_type_counts,
    std::uint16_t *upgrade_tile_counts,
    std::uint16_t *upgrade_weighted_gains,
    float *completion_probability,
    std::uint8_t *completion_valid,
    bool calculate_completion,
    unsigned mode)
{
  Hand hand;
  Hand unavailable_hand;
  std::copy_n(counts, 34, hand.begin());
  std::copy_n(unavailable, 34, unavailable_hand.begin());
  auto &space = neighborhood(hand);
  for (unsigned cut = 0; cut < 34; ++cut) {
    if (!(candidate_mask & (std::uint64_t{1} << cut))) continue;
    --hand[cut];
    auto &node = space.nodes[cut];
    ukeire_masks[cut] = ukeire(
        hand, unavailable_hand, node, ukeire_counts[cut], mode);
    shanten[cut] = static_cast<std::int8_t>(node.base[mode]) - 1;
    ++hand[cut];
  }
  unsigned const unseen = calculate_completion ? 136 -
      std::accumulate(unavailable_hand.begin(), unavailable_hand.end(), 0u) : 0;
  bool completion_search = false;
  if (calculate_completion) {
    for (unsigned cut = 0; cut < 34; ++cut) {
      if (!(candidate_mask & (std::uint64_t{1} << cut))) continue;
      if (shanten[cut] > 1) continue;
      completion_valid[cut] = 1;
      if (shanten[cut] == 0 && unseen) {
        completion_probability[cut] = float(ukeire_counts[cut]) / unseen;
      }
      completion_search |= shanten[cut] == 1 && unseen > 1;
    }
  }
  if (!calculate_upgrades && !completion_search) return;

  std::array<std::array<std::uint16_t, 34>, 34> best_completion{};
  std::array<std::array<std::uint16_t, 34>, 34> best_ukeire{};
  for (unsigned cut = 0; cut < 34; ++cut) {
    if (candidate_mask & (std::uint64_t{1} << cut)) {
      best_ukeire[cut].fill(ukeire_counts[cut]);
    }
  }
  // Removing A then B yields the same hand as removing B then A. Evaluate
  // each unordered pair once, retaining each initial discard's own baseline.
  for (unsigned first = 0; first < 34; ++first) {
    if (!hand[first]) continue;
    --hand[first];
    for (unsigned second = first; second < 34; ++second) {
      if (!hand[second]) continue;
      bool const first_candidate = candidate_mask & (std::uint64_t{1} << first);
      bool const second_candidate = candidate_mask & (std::uint64_t{1} << second);
      if (!first_candidate && !second_candidate) continue;
      --hand[second];
      for (unsigned draw = 0; draw < 34; ++draw) {
        if (unavailable_hand[draw] == 4) continue;
        auto const draw_bit = std::uint64_t{1} << draw;
        // Cutting the drawn type cannot increase acceptance. Improving draws
        // belong to acceptance, rather than the upgrade search.
        bool const improve_first = calculate_upgrades && first_candidate &&
            draw != second && !(ukeire_masks[first] & draw_bit);
        bool const improve_second = calculate_upgrades && second_candidate &&
            draw != first && !(ukeire_masks[second] & draw_bit);
        bool const complete_first = completion_search && first_candidate &&
            shanten[first] == 1 && (ukeire_masks[first] & draw_bit);
        bool const complete_second = completion_search && second_candidate &&
            shanten[second] == 1 && (ukeire_masks[second] & draw_bit);
        if (!improve_first && !improve_second &&
            !complete_first && !complete_second) continue;
        ++hand[draw];
        auto &node = neighbor(space, first, second, draw);
        prepare(hand, node);
        std::uint8_t const base = node.base[mode];
        bool const same_first = improve_first && base == shanten[first] + 1;
        bool const same_second = improve_second && base == shanten[second] + 1;
        if (same_first || same_second ||
            ((complete_first || complete_second) && base == 1)) {
          ++unavailable_hand[draw];
          std::uint16_t next_ukeire;
          ukeire(hand, unavailable_hand, node, next_ukeire, mode);
          --unavailable_hand[draw];
          if (complete_first && base == 1) {
            best_completion[first][draw] = std::max(
                best_completion[first][draw], next_ukeire);
          }
          if (complete_second && base == 1) {
            best_completion[second][draw] = std::max(
                best_completion[second][draw], next_ukeire);
          }
          if (same_first) {
            best_ukeire[first][draw] = std::max(
                best_ukeire[first][draw], next_ukeire);
          }
          if (same_second) {
            best_ukeire[second][draw] = std::max(
                best_ukeire[second][draw], next_ukeire);
          }
        }
        --hand[draw];
      }
      ++hand[second];
    }
    ++hand[first];
  }
  for (unsigned cut = 0; cut < 34; ++cut) {
    if (!(candidate_mask & (std::uint64_t{1} << cut))) continue;
    // Winning on the next two draws: chance-weight the first draw, then
    // choose its best legal continuation. Alternatives are not summed.
    if (completion_search && shanten[cut] == 1) {
      unsigned routes = 0;
      for (unsigned draw = 0; draw < 34; ++draw) {
        routes += (4 - unavailable_hand[draw]) * best_completion[cut][draw];
      }
      completion_probability[cut] = float(routes) / (unseen * (unseen - 1));
    }
    for (unsigned draw = 0; draw < 34; ++draw) {
      if (best_ukeire[cut][draw] <= ukeire_counts[cut]) continue;
      unsigned const remaining = 4 - unavailable_hand[draw];
      unsigned const gain = best_ukeire[cut][draw] - ukeire_counts[cut];
      upgrade_masks[cut] |= std::uint64_t{1} << draw;
      ++upgrade_type_counts[cut];
      upgrade_tile_counts[cut] += remaining;
      upgrade_weighted_gains[cut] += remaining * gain;
    }
  }
}

struct TenpaiPaths {
  Neighborhood &space;
  Hand const &visible;
  unsigned horizon;
  unsigned remaining_nodes;
  unsigned initial_cut = 0;
  bool exhausted = false;
  std::uint64_t total = 0;
  std::array<std::uint64_t, 34> retained_totals{};
  std::unordered_set<std::string> visited;

  void collect(
      Hand &hand, Hand &retained, Hand &unavailable, unsigned draws,
      StructuralNode &node)
  {
    prepare(hand, node);
    if (node.base[1] - 1 > draws) return;
    // A state identifies both the resulting shape and the original copies
    // it retains. Draw/discard order cannot create another copy of a route.
    std::string key(reinterpret_cast<char const *>(hand.data()), 34);
    key.append(reinterpret_cast<char const *>(retained.data()), 34);
    if (!visited.insert(key).second) return;
    if (node.base[1] <= 1) {
      // Count unordered physical draw combinations, not draw probabilities
      // or permutations. Different retained-copy counts identify different
      // acquisition multisets for the same tenpai shape.
      std::uint64_t weight = 1;
      for (unsigned tile = 0; tile < 34; ++tile) {
        unsigned combinations = 1;
        unsigned const added = hand[tile] - retained[tile];
        for (unsigned copy = 1; copy <= added; ++copy) {
          combinations = combinations * (5 - visible[tile] - copy) / copy;
        }
        weight *= combinations;
      }
      total += weight;
      for (unsigned tile = 0; tile < 34; ++tile) {
        retained_totals[tile] += weight * retained[tile];
      }
      return;
    }
    if (!remaining_nodes) { exhausted = true; return; }
    --remaining_nodes;
    std::uint16_t count;
    auto const improving = ukeire(hand, unavailable, node, count, 1);
    for (auto remaining = improving; remaining; remaining &= remaining - 1) {
      unsigned const draw = __builtin_ctzll(remaining);
      ++hand[draw];
      ++unavailable[draw];
      for (unsigned cut = 0; cut < 34; ++cut) {
        // On a shortest improving route, every new draw must survive to
        // tenpai. Discarding a new draw would waste one of the required draws.
        if (!retained[cut]) continue;
        --hand[cut];
        --retained[cut];
        // The first draw/cut expansion is also used by ukeire upgrades and
        // winning-shape completion. Deeper nodes are local to this search;
        // do not evict the shared neighborhood while its nodes are in use.
        StructuralNode next_node;
        collect(
            hand, retained, unavailable, draws - 1,
            draws == horizon
                ? neighbor(space, initial_cut, cut, draw) : next_node);
        ++retained[cut];
        ++hand[cut];
        if (exhausted) break;
      }
      --unavailable[draw];
      --hand[draw];
      if (exhausted) return;
    }
  }
};

}  // namespace

extern "C" bool nyanten_tenpai_neededness(
    std::uint8_t const *counts, std::uint8_t const *unavailable,
    std::uint64_t candidate_mask, unsigned max_draws, unsigned nodes,
    float *output)
{
  Hand hand;
  Hand unseen_counts;
  std::copy_n(counts, 34, hand.begin());
  std::copy_n(unavailable, 34, unseen_counts.begin());
  auto &space = neighborhood(hand);
  unsigned horizon = 14;
  for (unsigned cut = 0; cut < 34; ++cut) {
    if (!(candidate_mask & (std::uint64_t{1} << cut))) continue;
    --hand[cut];
    prepare(hand, space.nodes[cut]);
    horizon = std::min(horizon, unsigned(space.nodes[cut].base[1] - 1));
    ++hand[cut];
  }
  if (horizon > max_draws) return false;
  Hand const original = hand;
  Hand const visible = unseen_counts;
  TenpaiPaths search{space, visible, horizon, nodes};
  for (unsigned cut = 0; cut < 34; ++cut) {
    if (!(candidate_mask & (std::uint64_t{1} << cut))) continue;
    --hand[cut];
    search.initial_cut = cut;
    Hand retained = hand;
    search.collect(hand, retained, unseen_counts, horizon, space.nodes[cut]);
    ++hand[cut];
    if (search.exhausted) return false;
  }
  if (!search.total) return false;
  for (unsigned tile = 0; tile < 34; ++tile) {
    if (original[tile]) {
      output[tile] = double(search.retained_totals[tile]) /
          (search.total * original[tile]);
    }
  }
  return true;
}

extern "C" std::uint8_t nyanten_shanten(std::uint8_t const *counts)
{
  Hand hand;
  std::copy_n(counts, 34, hand.begin());
  return all_replacement_number(hand, replacement_number(hand));
}

extern "C" std::uint32_t nyanten_families(std::uint8_t const *counts)
{
  Hand hand;
  std::copy_n(counts, 34, hand.begin());
  auto const special = special_replacement_numbers(hand);
  return replacement_number(hand) | (std::uint32_t{special[0]} << 8) |
      (std::uint32_t{special[1]} << 16);
}

extern "C" std::uint64_t nyanten_draw(std::uint8_t const *counts, unsigned mode)
{
  Hand hand;
  std::copy_n(counts, 34, hand.begin());
  StructuralNode node;
  std::uint16_t count;
  // Draw analysis retains structural waits even when public tiles exhaust
  // them. The Python consumer weights this mask with current availability.
  auto const mask = ukeire(hand, hand, node, count, mode);
  return mask | (std::uint64_t{node.base[mode]} << 34);
}

struct Analysis {
  std::int8_t shanten[34];
  std::uint64_t ukeire_mask[34];
  std::uint16_t ukeire_count[34];
  std::uint64_t upgrade_mask[34];
  std::uint8_t upgrade_type_count[34];
  std::uint16_t upgrade_tile_count[34];
  std::uint16_t upgrade_weighted_gain[34];
  float completion_probability[34];
  std::uint8_t completion_valid[34];
};

extern "C" void nyanten_analyze_modes(
    std::uint8_t const *counts, std::uint8_t const *unavailable,
    std::uint64_t candidate_mask, bool calculate_upgrades,
    bool calculate_completion, std::uint8_t const *modes, std::size_t mode_count, Analysis *output)
{
  bool const shared = std::accumulate(counts, counts + 34, 0u) < 14;
  for (std::size_t i = 0; i < mode_count; ++i) {
    // Special forms require a closed hand; reuse a preceding normal result.
    if (i && modes[i] == 1 && modes[i - 1] == 0 && shared) {
      output[i] = output[i - 1];
      continue;
    }
    analyze(counts, unavailable, candidate_mask, calculate_upgrades,
            output[i].shanten, output[i].ukeire_mask,
            output[i].ukeire_count, output[i].upgrade_mask,
            output[i].upgrade_type_count, output[i].upgrade_tile_count,
            output[i].upgrade_weighted_gain, output[i].completion_probability,
            output[i].completion_valid,
            calculate_completion, modes[i]);
  }
}
