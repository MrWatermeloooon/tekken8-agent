#include "t8_v2/full_combat.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

t8::v2::FullCombatMove move(double damage = 20.0) {
    t8::v2::FullCombatMove value{};
    value.stable_id = "test:1";
    value.damage = damage;
    value.range = 1.0;
    value.tracking_left = value.tracking_right = 0.25;
    value.pushback = 0.2;
    value.validated = true;
    return value;
}

void symmetric_trade() {
    t8::v2::FullCombatState state{};
    state.p1.x = 0.0;
    state.p2.x = 0.8;
    const auto outcome = t8::v2::FullCombatOracle::resolve(state, move(), move());
    require(outcome.p1.landed && outcome.p2.landed, "simultaneous trade must resolve from a snapshot");
    require(state.p1.health == 160.0 && state.p2.health == 160.0,
            "simultaneous trade must damage both fighters equally");
}

void armor_crush_parry_and_tracking() {
    t8::v2::FullCombatState state{};
    state.p2.x = 0.8;
    auto high = move();
    high.hit_level = t8::v2::HitLevel::High;
    auto crush = move(10.0);
    crush.flags = t8::v2::FullHighCrush;
    auto outcome = t8::v2::FullCombatOracle::resolve(state, high, crush);
    require(outcome.p1.crushed && outcome.p2.landed, "high crush must evade a simultaneous high");

    state = {};
    state.p2.x = 0.8;
    auto parry = move(0.0);
    parry.flags = t8::v2::FullParry;
    outcome = t8::v2::FullCombatOracle::resolve(state, high, parry);
    require(outcome.p1.parried, "parry must stop a parryable strike");

    state = {};
    state.p2.x = 0.8;
    state.p2.axis = 0.8;
    outcome = t8::v2::FullCombatOracle::resolve(state, high, std::nullopt);
    require(!outcome.p1.landed, "linear move must miss beyond its tracking window");
    high.flags = t8::v2::FullHoming;
    outcome = t8::v2::FullCombatOracle::resolve(state, high, std::nullopt);
    require(outcome.p1.landed, "homing move must ignore lateral tracking window");
}

void resources_combo_and_stage_mechanics() {
    t8::v2::FullCombatState state{};
    state.p1.x = 3.0;
    state.p2.x = 3.5;
    state.p1.heat = 1.0;
    state.p1.rage = true;
    state.p1.resource = 2.0;
    auto launcher = move(30.0);
    launcher.requires_heat = true;
    launcher.required_resource = 2.0;
    launcher.resource_delta = -2.0;
    launcher.flags = t8::v2::FullLauncher | t8::v2::FullTornado |
        t8::v2::FullWallSplat | t8::v2::FullWallBreak | t8::v2::FullFloorBreak;
    const auto outcome = t8::v2::FullCombatOracle::resolve(state, launcher, std::nullopt);
    require(outcome.p1.landed && state.p2.posture == t8::v2::Posture::Airborne,
            "launcher must start an airborne combo");
    require(state.p2.combo_count == 1 && state.p2.tornado_used,
            "launcher/tornado state must be recorded");
    require(state.stage.wall_broken && state.stage.floor_broken && outcome.p1.wall_splat,
            "validated stage-break flags must update stage state");
    require(state.p1.heat == 0.0 && state.p1.resource == 0.0,
            "Heat and character resources must be consumed");
}

}  // namespace

int main() {
    try {
        symmetric_trade();
        armor_crush_parry_and_tracking();
        resources_combo_and_stage_mechanics();
        std::cout << "full combat oracle tests passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
