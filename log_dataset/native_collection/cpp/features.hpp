#pragma once
#include "feature_fields.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <vector>

namespace feature {
struct Meld {
  int kind, opened, size, tiles[4], called;
  int event_size, event_tiles[4], target, event, upgrade;
};
struct River {
  int tile, event, wall, turns[4], origin, riichi, resolution, caller, dora, tsumogiri;
};
struct Hand {
  int size, tiles[14], meld_count;
  Meld melds[4];
};
struct State {
  Hand hand;
  Meld melds[4][4];
  int meld_count[4];
  River const *rivers[4];
  size_t river_count[4];
  int boundary[4], accepted_event[4], declared[4], accepted[4], ippatsu[4];
  int draw, actor, phase, east_only, tsumo, rinshan, events, wall;
  int oya, kyoku, wind, honba, sticks, scores[4];
  int dora_size, dora[5], public_counts[37], exposed[34];
  int trigger_seat, trigger_tile;
};
struct Analysis {
  uint8_t counts[34], unavailable[34];
  int unavailable_red[3];
  uint8_t genbutsu[4][34];
  int dora_counts[34], public_fu[4], public_bonus[4];
  int distance, wind, kyoku, oya, remaining, order[4];
};
struct Branch {
  Hand hand;
  int response, option, tile;
  uint8_t cuts[37];
  int exposed[34];
};
struct Legal {
  uint8_t discards[37], riichi[37], calls[4][37];
};
struct Focus {
  uint8_t routes[3], fast, threat, phase;
  int yakuhai;
};
struct Ranges {
  double values[8];
  uint8_t valid[4], restricted[2], eligible;
  void merge(Ranges const &other);
};
struct RangeContext {
  uint64_t own_discards;
  int missed_agari;
  uint8_t unavailable[34];
  int unavailable_red[3];
};
using Score = void (*)(void *, Hand const *, RangeContext const *, int, uint64_t,
                       Ranges *);
Ranges ranges(Hand const &, RangeContext const &, int, void *, Score, uint64_t);

// Packed rows retain their existing wire layout. Field offsets are generated
// from the same typed declarations that expose INPUTS to Python.
struct Row {
  uint8_t *bytes;
  void set(fields::Field field, size_t index, double value) {
    switch (field.dtype) {
    case 0:
      bytes[field.offset + index] = static_cast<int8_t>(value);
      break;
    case 1: {
      auto v = static_cast<int16_t>(value);
      std::memcpy(bytes + field.offset + index * 2, &v, 2);
      break;
    }
    case 2: {
      auto v = static_cast<_Float16>(value);
      std::memcpy(bytes + field.offset + index * 2, &v, 2);
      break;
    }
    case 3: {
      auto v = static_cast<float>(value);
      std::memcpy(bytes + field.offset + index * 4, &v, 4);
      break;
    }
    }
  }
  template <class T>
  void put(fields::Field field, size_t offset, T const *values, size_t size) {
    if (field.dtype == 0 && sizeof(T) == 1) {
      std::memcpy(bytes + field.offset + offset, values, size);
    } else {
      for (size_t i = 0; i < size; ++i)
        set(field, offset + i, values[i]);
    }
  }
};
struct Bank {
  std::array<std::array<uint8_t, 44>, 96> keys{};
  int size = 0;
  int add(Row, Hand const &, int, int);
};
inline int base(int code) { return code < 34 ? code : 4 + 9 * (code - 34); }
inline bool red(int tile) { return tile == 16 || tile == 52 || tile == 88; }
inline int code(int tile) { return red(tile) ? 34 + tile / 36 : tile / 4; }
inline constexpr int outside[9][2] = {{-1, -1}, {0, -1}, {0, 1},  {1, 2},  {2, 6},
                                      {6, 7},   {7, 8},  {8, -1}, {-1, -1}};
void yaku(Meld const *melds, int size, int const *available, int round, int seat,
          uint8_t *out);
void public_values(Meld const *melds, int size, int const *dora, int &fu, int &bonus);
void pack_base(Row row, State const &state, Analysis &analysis);
void pack_candidates(Row row, State const &state, Analysis const &analysis,
                     Branch const *branches, size_t branch_count, Legal &legal,
                     Focus &focus, RangeContext const &availability, void *context,
                     Score score);
} // namespace feature
