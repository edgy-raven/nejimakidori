#include "analysis.hpp"
#include "features.hpp"
namespace feature {
Ranges ranges(Hand const &hand, RangeContext const &ctx, int cut, void *context,
              Score score, uint64_t draw) {
  uint8_t counts[34]{};
  for (int i = 0; i < hand.size; ++i)
    ++counts[hand.tiles[i] / 4];
  int shanten = int(draw >> 34) - 1;
  uint64_t improving = draw & ((1ull << 34) - 1);
  Ranges result{};
  if (shanten < 0 || shanten > 1)
    return result;
  result.eligible = 1;
  if (shanten == 0) {
    score(context, &hand, &ctx, cut, improving, &result);
    return result;
  }
  auto next = ctx;
  next.missed_agari = 0;
  if (cut >= 0)
    next.own_discards |= 1ull << cut;
  for (int base = 0; base < 34; ++base)
    if ((improving & (1ull << base)) && ctx.unavailable[base] < 4) {
      ++counts[base];
      ++next.unavailable[base];
      uint64_t waits[34]{};
      nyanten_tenpai_discards(counts, waits);
      bool five = base == 4 || base == 13 || base == 22;
      int reds = five ? 1 - ctx.unavailable_red[base / 9] : 0;
      int copies[] = {4 - ctx.unavailable[base] - reds, reds};
      for (int is_red = 0; is_red < 2; ++is_red)
        if (copies[is_red] > 0) {
          int tile = base * 4 + (five && !is_red);
          while (std::find(hand.tiles, hand.tiles + hand.size, tile) !=
                 hand.tiles + hand.size)
            ++tile;
          auto after = hand;
          after.tiles[after.size++] = tile;
          if (is_red)
            ++next.unavailable_red[base / 9];
          bool seen[37]{};
          for (int pos = 0; pos < after.size; ++pos) {
            int discarded = after.tiles[pos], code = feature::code(discarded);
            if (seen[code] || !waits[discarded / 4])
              continue;
            seen[code] = true;
            auto ready = after;
            std::move(ready.tiles + pos + 1, ready.tiles + ready.size, ready.tiles + pos);
            --ready.size;
            Ranges value{};
            score(context, &ready, &next, discarded / 4, waits[discarded / 4], &value);
            result.merge(value);
          }
          if (is_red)
            --next.unavailable_red[base / 9];
        }
      --counts[base];
      --next.unavailable[base];
    }
  return result;
}
} // namespace feature
extern "C" void feature_ranges(feature::Hand const *hand,
                               feature::RangeContext const *ctx, int cut, void *opaque,
                               feature::Score score, feature::Ranges *out) {
  uint8_t counts[34]{};
  for (int i = 0; i < hand->size; ++i)
    ++counts[hand->tiles[i] / 4];
  *out = feature::ranges(*hand, *ctx, cut, opaque, score, nyanten_draw(counts, 1));
}
