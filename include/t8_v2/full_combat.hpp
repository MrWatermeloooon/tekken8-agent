#pragma once

#include "t8_v2/sim.hpp"

#include <cstdint>
#include <optional>
#include <string>

namespace t8::v2 {

enum class Posture : std::uint8_t {
    Standing,
    Crouching,
    Grounded,
    Airborne,
    Knockdown,
    Wakeup,
};

enum FullCombatFlag : std::uint32_t {
    FullHoming = 1U << 0U,
    FullPowerCrush = 1U << 1U,
    FullHighCrush = 1U << 2U,
    FullLowCrush = 1U << 3U,
    FullParry = 1U << 4U,
    FullReversal = 1U << 5U,
    FullThrow = 1U << 6U,
    FullLauncher = 1U << 7U,
    FullTornado = 1U << 8U,
    FullWallSplat = 1U << 9U,
    FullWallBreak = 1U << 10U,
    FullFloorBreak = 1U << 11U,
    FullBalconyBreak = 1U << 12U,
    FullHeatEngager = 1U << 13U,
    FullHeatDash = 1U << 14U,
    FullHeatSmash = 1U << 15U,
    FullRageArt = 1U << 16U,
};

struct FullCombatMove {
    std::string stable_id;
    HitLevel hit_level = HitLevel::Mid;
    int startup = 1;
    int active = 1;
    int recovery = 1;
    double damage = 0.0;
    double range = 0.0;
    double pushback = 0.0;
    double tracking_left = 0.0;
    double tracking_right = 0.0;
    int hit_advantage = 0;
    int block_advantage = 0;
    std::uint32_t flags = 0;
    Posture required_posture = Posture::Standing;
    int required_stance = -1;
    double required_resource = 0.0;
    double resource_delta = 0.0;
    double health_cost = 0.0;
    double health_restore = 0.0;
    bool requires_heat = false;
    bool requires_rage = false;
    bool validated = false;
};

struct FullFighterState {
    double health = 180.0;
    double recoverable_health = 0.0;
    double x = 0.0;
    double axis = 0.0;
    Posture posture = Posture::Standing;
    HitLevel guard = HitLevel::None;
    int stance = -1;
    double resource = 0.0;
    double heat = 0.0;
    bool rage = false;
    int combo_count = 0;
    double combo_scale = 1.0;
    bool tornado_used = false;
    bool wall_splat = false;
    bool throw_held = false;
};

struct FullStageState {
    double half_width = 3.6;
    bool wall_broken = false;
    bool floor_broken = false;
    bool balcony_broken = false;
};

struct FullCombatState {
    FullFighterState p1{};
    FullFighterState p2{};
    FullStageState stage{};
};

struct FullAttackOutcome {
    bool legal = false;
    bool landed = false;
    bool blocked = false;
    bool parried = false;
    bool armored = false;
    bool crushed = false;
    bool wall_splat = false;
    double damage = 0.0;
};

struct FullExchangeOutcome {
    FullAttackOutcome p1{};
    FullAttackOutcome p2{};
};

class FullCombatOracle {
public:
    [[nodiscard]] static bool legal(const FullFighterState& fighter, const FullCombatMove& move);
    [[nodiscard]] static FullExchangeOutcome resolve(
        FullCombatState& state,
        const std::optional<FullCombatMove>& p1_move,
        const std::optional<FullCombatMove>& p2_move);
};

}  // namespace t8::v2
