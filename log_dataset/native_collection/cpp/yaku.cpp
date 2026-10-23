#include "features.hpp"
namespace feature {
void public_values(Meld const *melds, int size, int const *dora, int &fu, int &bonus) {
  fu = bonus = 0;
  constexpr int values[] = {0, 2, 8, 16, 8};
  for (int i = 0; i < size; ++i) {
    auto const &m = melds[i];
    int t = m.tiles[0] / 4;
    fu += values[m.kind] * (1 + (t >= 27 || t % 9 == 0 || t % 9 == 8));
    for (int j = 0; j < m.size; ++j)
      bonus += dora[m.tiles[j] / 4] + red(m.tiles[j]);
  }
}
void yaku(Meld const *melds, int size, int const *available, int round, int seat,
          uint8_t *out) {
  bool fixed[34]{};
  bool simple = true, suited = true, honors = true, terminals = true,
       terminal_honor = true, outside_all = true, closed = true;
  unsigned suits = 0;
  int sequences = 0, free = 4 - size;
  auto terminal = [](int t) { return t < 27 && (t % 9 == 0 || t % 9 == 8); };
  for (int i = 0; i < size; ++i) {
    auto const &m = melds[i];
    sequences += m.kind == 0;
    if (m.kind != 0)
      fixed[m.tiles[0] / 4] = true;
    bool outside = false;
    closed &= !m.opened;
    for (int j = 0; j < m.size; ++j) {
      int t = m.tiles[j] / 4;
      if (t < 27)
        suits |= 1u << (t / 9);
      simple &= t < 27 && !terminal(t);
      suited &= t < 27;
      honors &= t >= 27;
      terminals &= terminal(t);
      terminal_honor &= t >= 27 || terminal(t);
      outside |= t >= 27 || terminal(t);
    }
    outside_all &= outside;
  }
  auto triplet = [&](int t) { return fixed[t] || (free > 0 && available[t] >= 3); };
  std::fill_n(out, 19, 0);
  out[0] = simple;
  int dragons[] = {31, 32, 33, 27 + round, 27 + seat};
  for (int i = 0; i < 5; ++i)
    out[1 + i] = triplet(dragons[i]);
  out[6] = __builtin_popcount(suits) <= 1;
  out[7] = out[6] && suited;
  out[8] = sequences == 0;
  out[9] = size == 0;
  for (int pair = 31; pair < 34; ++pair) {
    bool valid = !fixed[pair] && available[pair] >= 2;
    int needed = 0;
    for (int t = 31; t < 34; ++t)
      if (t != pair) {
        valid &= triplet(t);
        needed += !fixed[t];
      }
    out[10] |= valid && needed <= free;
  }
  int needed = 0;
  bool valid = true;
  for (int t = 31; t < 34; ++t) {
    valid &= triplet(t);
    needed += !fixed[t];
  }
  out[11] = valid && needed <= free;
  out[12] = honors;
  out[13] = terminals;
  out[14] = terminal_honor;
  out[15] = size == 0;
  for (int t = 0; t < 34; ++t)
    if (t >= 27 || terminal(t))
      out[15] &= available[t] > 0;
  out[16] = outside_all && (sequences > 0 || free > 0);
  out[17] = out[16] && suited;
  out[18] = closed && free >= 2;
}
} // namespace feature
extern "C" void feature_yaku(feature::Meld const *melds, int size, int const *available,
                             int round, int seat, uint8_t *out) {
  feature::yaku(melds, size, available, round, seat, out);
}
extern "C" void feature_public_values(feature::Meld const *melds, int size,
                                      int const *dora, int *out) {
  feature::public_values(melds, size, dora, out[0], out[1]);
}
