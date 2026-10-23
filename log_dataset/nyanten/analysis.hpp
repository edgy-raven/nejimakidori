#pragma once
#include <cstddef>
#include <cstdint>
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

extern "C" {
std::uint8_t nyanten_shanten(std::uint8_t const *);
std::uint32_t nyanten_families(std::uint8_t const *);
std::uint32_t nyanten_yaku_shanten(std::uint8_t const *, std::uint8_t const *, unsigned);
std::uint64_t nyanten_draw(std::uint8_t const *, unsigned);
void nyanten_tenpai_discards(std::uint8_t const *, std::uint64_t *);
void nyanten_analyze_features(std::uint8_t const *, std::uint8_t const *, std::uint64_t,
                              std::uint8_t const *, std::size_t, Analysis *,
                              std::uint64_t *);
void nyanten_analyze_modes(std::uint8_t const *, std::uint8_t const *, std::uint64_t,
                           bool, bool, std::uint8_t const *, std::size_t, Analysis *);
}
