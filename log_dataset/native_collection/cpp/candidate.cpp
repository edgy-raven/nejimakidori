#include "analysis.hpp"
#include "features.hpp"
#include <tuple>
namespace feature {
using namespace fields;
void Ranges::merge(Ranges const &other) {
  eligible |= other.eligible;
  for (int s = 0; s < 4; ++s)
    if (other.valid[s]) {
      values[s * 2] =
          valid[s] ? std::min(values[s * 2], other.values[s * 2]) : other.values[s * 2];
      values[s * 2 + 1] = valid[s] ? std::max(values[s * 2 + 1], other.values[s * 2 + 1])
                                   : other.values[s * 2 + 1];
      valid[s] = 1;
    }
  for (int i = 0; i < 2; ++i)
    restricted[i] |= other.restricted[i];
}
void write(Row row, Ranges const &v, int action) {
  row.put(action_value_range, action * 8, v.values, 8);
  row.put(action_value_valid, action * 4, v.valid, 4);
  row.put(action_open_yaku_restricted, action * 2, v.restricted, 2);
  row.set(action_value_eligible, action, v.eligible);
}
std::array<uint8_t, 34> counts(Hand const &h) {
  std::array<uint8_t, 34> out{};
  for (int i = 0; i < h.size; ++i)
    ++out[h.tiles[i] / 4];
  return out;
}
Hand cut_hand(Hand h, int cut) {
  int pos = 0;
  while (code(h.tiles[pos]) != cut)
    ++pos;
  std::move(h.tiles + pos + 1, h.tiles + h.size, h.tiles + pos);
  --h.size;
  return h;
}
int Bank::add(Row row, Hand const &h, int fu, int bonus) {
  std::array<uint8_t, 44> key{};
  for (int i = 0; i < h.size; ++i) {
    int t = h.tiles[i];
    ++key[t / 4];
    if (red(t))
      ++key[34 + t / 36];
  }
  uint8_t fixed[16];
  std::fill_n(fixed, 16, 34);
  for (int i = 0; i < h.meld_count; ++i) {
    auto const &m = h.melds[i];
    int lowest = 34;
    for (int j = 0; j < m.size; ++j) {
      int t = m.tiles[j];
      lowest = std::min(lowest, t / 4);
      fixed[i * 4 + j] = t / 4;
      if (red(t))
        ++key[41 + t / 36];
    }
    key[37 + i] = lowest * 5 + m.kind + 1;
  }
  std::sort(key.begin() + 37, key.begin() + 41);
  for (int i = 0; i < size; ++i)
    if (keys[i] == key)
      return i;
  int index = size++;
  keys[index] = key;
  uint8_t encoded[204]{};
  for (int i = 37; i < 41; ++i)
    if (key[i])
      ++encoded[(key[i] - 1) / 5 * 6 + (key[i] - 1) % 5];
  for (int suit = 0; suit < 3; ++suit)
    encoded[(suit * 9 + 4) * 6 + 5] = key[41 + suit];
  row.put(candidate_hand_counts, index * 34, key.data(), 34);
  row.put(candidate_hand_red, index * 3, key.data() + 34, 3);
  row.put(candidate_melds, index * 204, encoded, 204);
  auto y = nyanten_yaku_shanten(key.data(), fixed, h.meld_count);
  for (int i = 0; i < 3; ++i)
    row.set(candidate_yaku_shanten, index * 3 + i, int((y >> (8 * i)) & 255) - 1);
  row.set(candidate_hand_valid, index, 1);
  row.set(candidate_public_fu, index, fu);
  row.set(candidate_public_bonus, index, bonus);
  return index;
}
std::vector<int> cuts(uint8_t const *legal) {
  std::vector<int> out;
  for (int t = 0; t < 37; ++t)
    if (legal[t])
      out.push_back(t);
  return out;
}
void sort_cuts(std::vector<int> &cuts) {
  std::sort(cuts.begin(), cuts.end(), [](int x, int y) {
    return std::make_pair(base(x), x < 34) < std::make_pair(base(y), y < 34);
  });
}
std::array<double, 3> efficiency(::Analysis const &a, std::vector<int> const &cuts) {
  int t = base(*std::min_element(cuts.begin(), cuts.end(), [&](int x, int y) {
    x = base(x);
    y = base(y);
    return std::make_tuple(a.shanten[x], -int(a.ukeire_count[x]),
                           -__builtin_popcountll(a.ukeire_mask[x])) <
           std::make_tuple(a.shanten[y], -int(a.ukeire_count[y]),
                           -__builtin_popcountll(a.ukeire_mask[y]));
  }));
  return {a.shanten[t] / 8., __builtin_popcountll(a.ukeire_mask[t]) / 34.,
          a.ukeire_count[t] / 80.};
}
std::array<double, 3> efficiency(uint64_t v, uint8_t const *unavailable) {
  uint64_t mask = v & ((1ull << 34) - 1);
  int total = 0;
  for (int t = 0; t < 34; ++t)
    if (mask & (1ull << t))
      total += 4 - unavailable[t];
  return {double(v >> 34) / 8. - 1. / 8., __builtin_popcountll(mask) / 34., total / 80.};
}
void flags(Hand const &h, int const *exposed, int round, int seat, uint8_t *out,
           int cut = -1) {
  int available[34];
  for (int t = 0; t < 34; ++t)
    available[t] = 4 - exposed[t] - (t == cut);
  for (int i = 0; i < h.meld_count; ++i)
    for (int j = 0; j < h.melds[i].size; ++j)
      ++available[h.melds[i].tiles[j] / 4];
  yaku(h.melds, h.meld_count, available, round, seat, out);
}
void pack_candidates(Row row, State const &s, feature::Analysis const &a,
                     Branch const *branches, size_t branch_count, Legal &legal,
                     Focus &focus, RangeContext const &availability, void *context,
                     Score score) {
  row.set(riichi_legal_mask, 0, 1);
  row.set(response_legal_mask, 0, 1);
  row.set(kan_legal_mask, 0, 1);
  row.put(discard_action_legal, 0, legal.discards, 37);
  row.put(riichi_discard_legal_mask, 0, legal.riichi, 37);
  for (int t = 0; t < 37; ++t)
    if (legal.riichi[t])
      row.set(riichi_legal_mask, 1, 1);
  bool removed = false;
  for (int i = 0; i < s.hand.size; ++i) {
    int t = code(s.hand.tiles[i]);
    if (!removed && s.draw == t) {
      removed = true;
      continue;
    }
    row.set(discard_tedashi_legal, t, legal.discards[t] && !s.accepted[s.actor]);
  }
  Bank bank;
  bank.add(row, s.hand, a.public_fu[0], a.public_bonus[0]);
  auto choices = cuts(legal.discards);
  std::array<double, 3> self;
  if (!choices.empty()) {
    uint64_t mask = 0;
    for (int t : choices)
      mask |= 1ull << base(t);
    uint8_t modes[] = {0, 1};
    ::Analysis analyses[2]{};
    uint64_t structural[34]{};
    nyanten_analyze_features(a.counts, a.unavailable, mask, modes, 2, analyses,
                             structural);
    if (focus.threat && focus.phase == 0) {
      focus.threat = 0;
      for (int t : choices) {
        for (int seat = 1; seat < 4; ++seat)
          focus.threat |= a.genbutsu[seat][base(t)] != a.genbutsu[seat][base(choices[0])];
        focus.threat |=
            analyses[1].shanten[base(t)] != analyses[1].shanten[base(choices[0])] ||
            analyses[1].ukeire_count[base(t)] !=
                analyses[1].ukeire_count[base(choices[0])];
      }
    }
    Field shanten[] = {discard_action_normal_shanten, discard_action_all_shanten};
    Field ukeire[] = {discard_action_normal_ukeire_count,
                      discard_action_all_ukeire_count};
    for (int i = 0; i < 2; ++i)
      for (int cut : choices) {
        row.set(shanten[i], cut, analyses[i].shanten[base(cut)] + 1);
        row.set(ukeire[i], cut, analyses[i].ukeire_count[base(cut)]);
      }
    uint8_t y[19];
    flags(s.hand, s.exposed, s.wind, (s.actor + 4 - s.oya) % 4, y);
    int retained[4]{};
    for (int seat = 0; seat < 4; ++seat)
      for (int t = 0; t < 34; ++t)
        retained[seat] += a.genbutsu[seat][t] * a.counts[t];
    Ranges decline{};
    self = efficiency(analyses[1], choices);
    sort_cuts(choices);
    for (int cut : choices) {
      int t = base(cut);
      auto after = cut_hand(s.hand, cut);
      row.set(discard_candidate_hand, cut,
              bank.add(row, after, a.public_fu[0], a.public_bonus[0]));
      uint8_t cut_y[19];
      if (t < 27 && t % 9 >= 1 && t % 9 <= 7)
        std::copy_n(y, 19, cut_y);
      else
        flags(s.hand, s.exposed, s.wind, (s.actor + 4 - s.oya) % 4, cut_y, t);
      row.put(discard_action_yaku_possibility, cut * 19, cut_y, 19);
      auto v = ranges(after, availability, t, context, score, structural[t]);
      write(row, v, cut);
      write(row, v, 37 + cut);
      if (legal.riichi[cut])
        write(row, v, 74 + cut);
      decline.merge(v);
      for (int seat = 0; seat < 4; ++seat)
        row.set(discard_retained_genbutsu, cut * 4 + seat,
                retained[seat] - a.genbutsu[seat][t]);
    }
    write(row, decline, 260);
  } else {
    auto draw = nyanten_draw(a.counts, 1);
    self = efficiency(draw, a.unavailable);
    auto v = ranges(s.hand, availability, -1, context, score, draw);
    write(row, v, 111);
    write(row, v, 260);
  }
  row.put(self_efficiency, 0, self.data(), 3);
  for (size_t b = 0; b < branch_count; ++b) {
    auto const &branch = branches[b];
    auto const &h = branch.hand;
    int fu, bonus;
    public_values(h.melds, h.meld_count, a.dora_counts, fu, bonus);
    if (branch.response)
      row.set(response_legal_mask, branch.response, 1);
    uint8_t y[19];
    flags(h, branch.exposed, s.wind, (s.actor + 4 - s.oya) % 4, y);
    if (branch.response != 1 && branch.response != 2) {
      int t = branch.tile;
      row.set(kan_legal_mask, t + 1, 1);
      row.set(kan_candidate_hand, t, bank.add(row, h, fu, bonus));
      auto draw = nyanten_draw(counts(h).data(), 1);
      auto v = ranges(h, availability, -1, context, score, draw);
      write(row, v, 261 + t);
      auto eff = efficiency(draw, a.unavailable);
      row.put(kan_efficiency, t * 3, eff.data(), 3);
      row.put(kan_yaku_possibility, t * 19, y, 19);
      continue;
    }
    bool chi = branch.response == 1;
    int option = branch.option;
    if (chi)
      row.set(chi_pattern_legal_mask, option, 1);
    auto call_cuts = cuts(branch.cuts);
    uint64_t mask = 0;
    for (int t : call_cuts)
      mask |= 1ull << base(t);
    uint8_t mode = 1;
    ::Analysis analysis{};
    uint64_t structural[34]{};
    nyanten_analyze_features(counts(h).data(), a.unavailable, mask, &mode, 1, &analysis,
                             structural);
    bool viable =
        focus.yakuhai > 0 ||
        (!chi && (branch.tile >= 31 || branch.tile == 27 + (s.actor + 4 - s.oya) % 4 ||
                  branch.tile == 27 + s.wind));
    int indices[] = {0, 6, 16};
    for (int i = 0; i < 3; ++i)
      viable |= focus.routes[i] && y[indices[i]];
    if (!call_cuts.empty()) {
      auto eff = efficiency(analysis, call_cuts);
      row.put(chi ? chi_call_efficiency : pon_call_efficiency, option * 3, eff.data(), 3);
    }
    sort_cuts(call_cuts);
    for (int cut : call_cuts) {
      auto after = cut_hand(h, cut);
      int position = option * 37 + cut;
      row.set(chi ? chi_candidate_hand : pon_candidate_hand, position,
              bank.add(row, after, fu, bonus));
      legal.calls[chi ? option : 3][cut] = 1;
      row.set(chi ? chi_discard_legal : pon_discard_legal, position, 1);
      row.set(chi ? chi_discard_shanten : pon_discard_shanten, position,
              analysis.shanten[base(cut)] + 1);
      row.set(chi ? chi_discard_ukeire_count : pon_discard_ukeire_count, position,
              analysis.ukeire_count[base(cut)]);
      auto v =
          ranges(after, availability, base(cut), context, score, structural[base(cut)]);
      write(row, v, (chi ? 112 : 223) + position);
      focus.fast |= (analysis.shanten[base(cut)] < a.distance ||
                     (analysis.shanten[base(cut)] == a.distance &&
                      analysis.ukeire_count[base(cut)] > self[1])) &&
                    (viable || v.values[5] > 0);
    }
    row.put(chi ? chi_call_yaku_possibility : pon_call_yaku_possibility, option * 19, y,
            19);
  }
}
} // namespace feature
extern "C" void feature_pack_candidates(uint8_t *row, feature::State const *s,
                                        feature::Analysis const *a,
                                        feature::Branch const *branches, size_t size,
                                        feature::Legal *legal, feature::Focus *focus,
                                        feature::RangeContext const *availability,
                                        void *context, feature::Score score) {
  feature::pack_candidates({row}, *s, *a, branches, size, *legal, *focus, *availability,
                           context, score);
}

extern "C" int feature_bank_hand(uint8_t *row, feature::Bank *bank,
                                 feature::Hand const *hand, int fu, int bonus) {
  return bank->add({row}, *hand, fu, bonus);
}
