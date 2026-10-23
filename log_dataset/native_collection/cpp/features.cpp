#include "features.hpp"
#include "analysis.hpp"
#include <numeric>
#include <tuple>
namespace feature {
using namespace fields;
void pack_base(Row row, State const &s, feature::Analysis &a) {
  int actor = s.actor;
  uint8_t hand_red[3]{}, public_red[3]{};
  for (int i = 0; i < s.hand.size; ++i) {
    int t = s.hand.tiles[i];
    ++a.counts[t / 4];
    if (red(t))
      ++hand_red[t / 36];
  }
  row.put(candidate_hand_counts, 0, a.counts, 34);
  row.put(candidate_hand_red, 0, hand_red, 3);
  bool closed = true;
  for (int i = 0; i < s.meld_count[actor]; ++i)
    closed &= !s.melds[actor][i].opened;
  row.set(closed_hand, 0, closed);
  auto packed = nyanten_families(a.counts);
  int families[3];
  Field family_fields[] = {normal_self_shanten, chiitoitsu_self_shanten,
                           kokushi_self_shanten};
  for (int i = 0; i < 3; ++i) {
    families[i] = int((packed >> (8 * i)) & 255) - 1;
    row.set(family_fields[i], 0,
            i > 0 && s.meld_count[actor] ? 5 : std::min(families[i] + 1, 4));
  }
  uint8_t flags[4][19]{};
  for (int seat = 0; seat < 4; ++seat) {
    int available[34];
    for (int t = 0; t < 34; ++t)
      available[t] = 4 - s.exposed[t];
    for (int i = 0; i < s.meld_count[seat]; ++i)
      for (int j = 0; j < s.melds[seat][i].size; ++j)
        ++available[s.melds[seat][i].tiles[j] / 4];
    if (s.trigger_seat >= 0 && s.trigger_seat != seat)
      ++available[base(s.trigger_tile)];
    yaku(s.melds[seat], s.meld_count[seat], available, s.wind, (seat + 4 - s.oya) % 4,
         flags[seat]);
  }
  row.put(self_yaku_possibility, 0, flags[actor], 19);
  if (s.draw >= 0) {
    row.set(current_draw_valid, 0, 1);
    row.set(current_draw_tile_id, 0, base(s.draw));
    row.set(current_draw_is_red, 0, s.draw >= 34);
  }
  row.set(decision_phase, 0, s.phase);
  if (s.phase == 4)
    row.set(win_kind, 0, s.tsumo ? 2 : 1);
  row.set(rinshan_state, 0, s.rinshan == actor);
  for (int i = 0; i < s.dora_size; ++i) {
    int t = base(s.dora[i]);
    int next = t < 27   ? t / 9 * 9 + (t + 1) % 9
               : t < 31 ? 27 + (t - 26) % 4
                        : 31 + (t - 30) % 3;
    ++a.dora_counts[next];
    row.set(dora_tile_id, i, t);
    row.set(dora_valid, i, 1);
  }
  row.put(dora_multiplicity, 0, a.dora_counts, 34);
  std::copy_n(a.counts, 34, a.unavailable);
  for (int t = 0; t < 37; ++t) {
    a.unavailable[base(t)] += s.public_counts[t];
    if (t >= 34)
      public_red[t - 34] += s.public_counts[t];
  }
  row.put(known_unavailable_counts, 0, a.unavailable, 34);
  row.put(public_visible_red_counts, 0, public_red, 3);
  row.set(prevailing_wind, 0, s.wind);
  row.set(kyoku_number, 0, s.kyoku - 1);
  row.set(dealer_relative_seat, 0, (s.oya + 4 - actor) % 4);
  row.set(honba, 0, s.honba);
  row.set(riichi_sticks, 0, s.sticks);
  row.set(live_wall_count, 0, s.wall);
  int kans = 0, first_kan = -1;
  bool multiple = false, no_melds = true;
  for (int seat = 0; seat < 4; ++seat) {
    no_melds &= s.meld_count[seat] == 0;
    for (int i = 0; i < s.meld_count[seat]; ++i)
      if (s.melds[seat][i].kind >= 2) {
        ++kans;
        if (first_kan < 0)
          first_kan = seat;
        multiple |= seat != first_kan;
      }
  }
  row.set(total_kan_count, 0, kans);
  for (int decl = 0; decl < 2; ++decl) {
    bool all = true;
    for (int seat = 0; seat < 4; ++seat)
      all &= s.declared[seat] || (seat == actor && decl == 1);
    bool abort = (kans == 4 && multiple) || all;
    for (int t = 0; t < 37; ++t)
      row.set(discard_abort, decl * 37 + t, abort);
  }
  bool first = no_melds && s.river_count[actor] == 0;
  for (int seat = 0; seat < 4; ++seat)
    if (seat != actor)
      first &= s.river_count[seat] == 1;
  if (first) {
    int t = s.rivers[(actor + 1) % 4][0].tile;
    bool same = t >= 27 && t < 31;
    for (int seat = 0; seat < 4; ++seat)
      if (seat != actor)
        same &= s.rivers[seat][0].tile == t;
    if (same) {
      row.set(discard_abort, t, 1);
      row.set(discard_abort, 37 + t, 1);
    }
  }
  int number = s.wind * 4 + s.kyoku - 1, last = s.east_only ? 3 : 7;
  a.wind = s.wind;
  a.kyoku = s.kyoku - 1;
  a.oya = s.oya;
  a.remaining = std::max(0, last - number);
  row.set(nominal_all_last, 0, number == last);
  row.set(scheduled_hands_remaining, 0, a.remaining);
  int maximum = *std::max_element(s.scores, s.scores + 4);
  row.set(dealer_currently_leads, 0, s.scores[s.oya] == maximum);
  row.set(any_player_at_or_above_30000, 0, maximum >= 30000);
  int first_dealer = (s.oya + 5 - s.kyoku) % 4;
  std::iota(a.order, a.order + 4, 0);
  std::sort(a.order, a.order + 4, [&](int x, int y) {
    return std::make_pair(-s.scores[x], (x + 4 - first_dealer) % 4) <
           std::make_pair(-s.scores[y], (y + 4 - first_dealer) % 4);
  });
  for (int i = 0; i < 3; ++i)
    a.unavailable_red[i] = hand_red[i] + public_red[i];
  a.distance = std::min(
      3, s.meld_count[actor] ? families[0] : *std::min_element(families, families + 3));
  row.set(score_extrema, 0, *std::min_element(s.scores, s.scores + 4) / 100000.);
  row.set(score_extrema, 1, maximum / 100000.);
  uint64_t riichi_safe[4]{};
  std::vector<std::pair<int, River const *>> discards;
  for (int seat = 0; seat < 4; ++seat)
    for (size_t i = 0; i < s.river_count[seat]; ++i)
      discards.emplace_back(seat, &s.rivers[seat][i]);
  std::sort(discards.begin(), discards.end(), [](auto const &x, auto const &y) {
    return x.second->event < y.second->event;
  });
  uint8_t pushes[16]{};
  for (int target = 0; target < 4; ++target)
    if (s.accepted[target]) {
      int declared = s.accepted_event[target];
      uint64_t safe = 0;
      for (size_t i = 0; i < s.river_count[target]; ++i)
        if (s.rivers[target][i].event < declared)
          safe |= 1ull << base(s.rivers[target][i].tile);
      for (auto const &[seat, ptr] : discards) {
        auto const &cut = *ptr;
        if (cut.event <= declared)
          continue;
        int tile = base(cut.tile);
        if (seat != target && !cut.tsumogiri && !(safe & (1ull << tile)))
          ++pushes[((seat + 4 - actor) % 4) * 4 + (target + 4 - actor) % 4];
        if (cut.resolution)
          safe |= 1ull << tile;
      }
      riichi_safe[target] = safe;
    }
  row.put(riichi_push_count, 0, pushes, 16);
  for (int relative = 0; relative < 4; ++relative) {
    int seat = (actor + relative) % 4;
    public_values(s.melds[seat], s.meld_count[seat], a.dora_counts, a.public_fu[relative],
                  a.public_bonus[relative]);
    row.set(public_fu, relative, a.public_fu[relative]);
    row.set(public_bonus, relative, a.public_bonus[relative]);
    auto river = s.rivers[seat];
    int size = s.river_count[seat];
    for (int i = size - 1; i >= 0; --i)
      if (!river[i].tsumogiri) {
        row.set(last_tedashi_tile, relative * 37 + river[i].tile, 1);
        break;
      }
    row.set(seat_wind, relative, (seat + 4 - s.oya) % 4);
    row.set(seat_context, relative * 2, s.scores[seat] / 100000.);
    row.set(seat_context, relative * 2 + 1,
            (std::find(a.order, a.order + 4, seat) - a.order + 1) / 4.);
    row.set(riichi_state, relative, s.declared[seat] + s.accepted[seat]);
    row.set(ippatsu_alive, relative, s.ippatsu[seat]);
    row.set(opponent_discard_count, relative, size);
    int called = 0;
    for (int i = 0; i < s.meld_count[seat]; ++i)
      if (s.melds[seat][i].opened)
        called += s.melds[seat][i].size;
    if (relative) {
      row.set(opponent_called_tile_count, relative, called);
      constexpr int indices[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 14, 16, 17};
      if (called)
        for (int i = 0; i < 13; ++i)
          row.set(open_opponent_yaku_possibility, relative * 13 + i,
                  flags[seat][indices[i]]);
    }
    int last_change = -1;
    std::vector<int> call_indices;
    for (int i = 0; i < s.meld_count[seat]; ++i) {
      auto const &m = s.melds[seat][i];
      int slot = relative * 4 + i;
      call_indices.push_back(m.event);
      if (m.upgrade >= 0)
        call_indices.push_back(m.upgrade);
      last_change = std::max({last_change, m.event, m.upgrade});
      row.set(meld_valid, slot, 1);
      row.set(meld_type, slot, m.kind + 1);
      for (int j = 0; j < m.event_size; ++j) {
        row.set(meld_tile_ids, slot * 4 + j, base(m.event_tiles[j]));
        row.set(meld_tile_red, slot * 4 + j, m.event_tiles[j] >= 34);
        row.set(meld_tile_valid, slot * 4 + j, 1);
      }
      row.set(meld_source, slot, m.opened ? (m.target + 4 - actor) % 4 + 1 : 0);
      row.set(meld_call_age, slot, std::min(1., (s.events - m.event) / 100.));
      if (m.upgrade >= 0)
        row.set(meld_upgrade_age, slot, std::min(1., (s.events - m.upgrade) / 100.));
    }
    for (int i = 0; i < size; ++i)
      if (!river[i].tsumogiri)
        last_change = std::max(last_change, river[i].event);
    uint8_t passed[34]{}, earliest[34]{};
    for (int other = 0; other < 4; ++other)
      if (other != seat)
        for (size_t i = 0; i < s.river_count[other]; ++i) {
          auto const &cut = s.rivers[other][i];
          if (cut.resolution && cut.event > last_change)
            passed[base(cut.tile)] = 1;
        }
    row.put(passed_unchanged_to_seat, relative * 34, passed, 34);
    for (int t = 0; t < 34; ++t)
      a.genbutsu[relative][t] = (riichi_safe[seat] >> t) & 1;
    int boundary = s.boundary[seat] < 0 ? size : s.boundary[seat];
    int prior = -1, turn = -1, prior_event = -1;
    Field river_fields[15][2] = {
        {river_pre_tile_id, river_post_tile_id},
        {river_pre_is_red, river_post_is_red},
        {river_pre_tsumogiri, river_post_tsumogiri},
        {river_pre_riichi, river_post_riichi},
        {river_pre_valid, river_post_valid},
        {river_pre_origin, river_post_origin},
        {river_pre_call_resolution, river_post_call_resolution},
        {river_pre_caller, river_post_caller},
        {river_pre_event_age, river_post_event_age},
        {river_pre_own_gap, river_post_own_gap},
        {river_pre_live_wall, river_post_live_wall},
        {river_pre_current_dora_multiplicity, river_post_current_dora_multiplicity},
        {river_pre_dora_multiplicity_at_discard, river_post_dora_multiplicity_at_discard},
        {river_pre_tedashi_relation, river_post_tedashi_relation},
        {river_pre_tedashi_gap, river_post_tedashi_gap}};
    for (int i = 0; i < size; ++i) {
      auto const &cut = river[i];
      int t = base(cut.tile);
      if (prior >= 0 && std::any_of(call_indices.begin(), call_indices.end(), [&](int e) {
            return prior_event < e && e < cut.event;
          }))
        prior = -1;
      a.genbutsu[relative][t] = 1;
      if (i < 6 && !cut.riichi && t < 27)
        for (int o : outside[t % 9])
          if (o >= 0 && !earliest[t / 9 * 9 + o])
            earliest[t / 9 * 9 + o] = i + 1;
      if (!relative)
        continue;
      int phase = i >= boundary, end = phase ? size : boundary;
      if (32 + i < end)
        continue;
      int slot = (relative - 1) * 32 + 32 + i - end;
      int origin = seat == s.oya && i == 0 ? 1
                   : cut.origin == 2       ? 4
                   : cut.origin == 3       ? 5
                   : cut.tsumogiri         ? 2
                                           : 3;
      double values[] = {double(t),
                         double(cut.tile >= 34),
                         double(cut.tsumogiri),
                         double(cut.riichi),
                         1.,
                         double(origin),
                         double(cut.resolution + 1),
                         cut.resolution == 0 ? 1.
                         : cut.caller < 0    ? 0.
                                             : double((cut.caller + 4 - actor) % 4 + 2),
                         std::min(1., (s.events - cut.event) / 100.),
                         std::min(1., (s.river_count[actor] - cut.turns[actor]) / 24.),
                         cut.wall / 70.,
                         double(a.dora_counts[t]),
                         double(cut.dora),
                         0.,
                         0.};
      if (!cut.tsumogiri) {
        if (prior >= 0) {
          int gap = std::abs(t - prior);
          values[13] = t == prior                                 ? 1
                       : t < 27 && t / 9 == prior / 9 && gap <= 2 ? gap + 1
                                                                  : 0;
          values[14] = std::min(i - turn, 31);
        }
        prior = t;
        turn = i;
        prior_event = cut.event;
      }
      for (int k = 0; k < 15; ++k)
        row.set(river_fields[k][phase], slot, values[k]);
    }
    row.put(genbutsu_to_seat, relative * 34, a.genbutsu[relative], 34);
    row.put(sotogawa_earliest_turn_to_seat, relative * 34, earliest, 34);
  }
  uint8_t blockers[170]{};
  for (int t = 0; t < 27; ++t)
    for (int o : outside[t % 9])
      if (o >= 0)
        ++blockers[(t / 9 * 9 + o) * 5 + 4 - a.unavailable[t]];
  row.put(blocker_counts, 0, blockers, 170);
  if (s.trigger_seat >= 0) {
    row.set(trigger_valid, 0, 1);
    row.set(trigger_tile_id, 0, base(s.trigger_tile));
    row.set(trigger_is_red, 0, s.trigger_tile >= 34);
    row.set(trigger_source, 0, (s.trigger_seat + 4 - actor) % 4);
  }
}
} // namespace feature
extern "C" void feature_pack_base(uint8_t *row, feature::State const *state,
                                  feature::Analysis *analysis) {
  feature::pack_base({row}, *state, *analysis);
}
extern "C" void feature_row_set(uint8_t *row, size_t offset, int dtype, size_t index,
                                double value) {
  feature::Row{row}.set({offset, dtype}, index, value);
}
// Check the Rust/C++ transport ABI without relying on particular game states.
extern "C" void feature_layout(size_t *out) {
  using namespace feature;
  size_t layout[] = {sizeof(Meld),
                     sizeof(River),
                     sizeof(Hand),
                     sizeof(State),
                     sizeof(feature::Analysis),
                     sizeof(Branch),
                     sizeof(Legal),
                     sizeof(Focus),
                     sizeof(Ranges),
                     sizeof(RangeContext),
                     sizeof(Bank),
                     offsetof(State, rivers),
                     offsetof(State, exposed),
                     offsetof(feature::Analysis, genbutsu),
                     offsetof(Focus, yakuhai),
                     offsetof(RangeContext, unavailable_red)};
  std::copy_n(layout, 16, out);
}
