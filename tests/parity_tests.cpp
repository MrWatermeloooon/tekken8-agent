#include "t8_v2/sim.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string_view>

namespace {

using t8::v2::Action;
using t8::v2::FighterRuntime;
using t8::v2::HitLevel;
using t8::v2::Simulator;
using t8::v2::State;

int failures = 0;

void check(bool condition, std::string_view message) {
    if (!condition) {
        ++failures;
        std::cerr << "FAIL: " << message << '\n';
    }
}

void near(double actual, double expected, std::string_view message, double tolerance = 1e-6) {
    if (std::abs(actual - expected) > tolerance) {
        ++failures;
        std::cerr << "FAIL: " << message << " expected=" << expected << " actual=" << actual << '\n';
    }
}

FighterRuntime fighter(double health, double x) {
    FighterRuntime value{};
    value.health = health;
    value.x = x;
    return value;
}

void test_contract_constants() {
    check(t8::v2::kActionCount == 24, "V1 trainable action count is frozen at 24");
    check(t8::v2::kObservationSize == 19, "V1 privileged observation size is frozen at 19");
    check(t8::v2::kVisualObservationSize == 13, "V1 visual observation size is frozen at 13");
    check(t8::v2::kActionNames.front() == "neutral", "action index 0 is neutral");
    check(t8::v2::kActionNames[17] == "throw_break_1p2", "throw-break spelling and index are frozen");
    check(t8::v2::kActionNames.back() == "throw", "action index 23 is throw");
    check(t8::v2::kMoves[0].total_frames() == 25, "jab total frames match V1");
    check(t8::v2::kMoves[4].launches, "hopkick launch flag matches V1");
}

void test_reset_and_observation() {
    Simulator sim;
    const auto& state = sim.reset(123);
    near(state.p1.health, 180.0, "reset p1 health");
    near(state.p2.health, 180.0, "reset p2 health");
    near(state.p1.x, -0.85, "reset p1 x");
    near(state.p2.x, 0.85, "reset p2 x");
    check(state.frame == 0, "reset frame");
    check(!state.round_over && state.winner == 0, "reset round state");

    const auto obs = sim.observation(1);
    near(obs[0], 1.0, "reset own health observation");
    near(obs[1], 1.0, "reset opponent health observation");
    near(obs[2], 0.2361111044883728, "reset signed distance observation");
    near(obs[3], 0.2361111044883728, "reset absolute distance observation");
    near(obs[4], 0.6180555820465088, "reset forward wall observation");
    near(obs[5], 0.3819444477558136, "reset back wall observation");
    near(obs[18], 1.0, "observation bias");

    const auto p2_obs = sim.observation(2);
    near(p2_obs[2], obs[2], "P2 signed distance is ego-relative");
    near(p2_obs[4], obs[4], "P2 forward wall is ego-relative");

    const auto visual = sim.visual_observation(1);
    near(visual[0], 1.0, "reset visual own health");
    near(visual[1], 1.0, "reset visual opponent health");
    near(visual[2], -0.85, "reset visual own x");
    near(visual[3], 0.85, "reset visual opponent x");
    near(visual[4], 1.70, "reset visual distance");
    near(visual[5], 0.0, "reset visual own velocity");
    near(visual[12], 0.0, "reset visual opponent attack likelihood");

    const State previous = sim.state();
    static_cast<void>(sim.step(Action::WalkForward, Action::WalkForward));
    const auto moving_visual = sim.visual_observation(1, &previous);
    near(moving_visual[5], 0.10, "visual own decision velocity");
    near(moving_visual[6], -0.10, "visual opponent decision velocity");
    near(moving_visual[7], 0.10, "visual own motion");
    near(moving_visual[8], 0.10, "visual opponent motion");
}

void test_neutral_step_fixture() {
    Simulator sim;
    const auto result = sim.step(Action::Neutral, Action::Neutral);
    check(result.state.frame == 4, "neutral step advances four frames");
    check(result.state.stall_frames == 4, "neutral step increments stall frames");
    check(result.state.no_action_frames == 4, "neutral step increments no-action frames");
    check(result.state.p1.guard == HitLevel::Mid && result.state.p2.guard == HitLevel::Mid,
          "neutral auto-blocks highs and mids");
    near(result.reward_p1, -0.02, "neutral P1 idle reward");
    near(result.reward_p2, -0.02, "neutral P2 idle reward");
    check(!result.terminated && !result.truncated, "neutral step remains live");
}

void test_busy_action_mask() {
    Simulator sim;
    static_cast<void>(sim.step(Action::Jab, Action::Neutral));
    const auto mask = sim.legal_action_mask(1);
    check(mask[t8::v2::action_index(Action::Neutral)], "neutral stays legal while busy");
    for (std::size_t index = 1; index < mask.size(); ++index) {
        check(!mask[index], "non-neutral action is masked while busy");
    }
}

void test_simultaneous_jab_trade_fixture() {
    Simulator sim;
    State state{};
    state.p1 = fighter(180.0, 0.0);
    state.p2 = fighter(180.0, 0.82);
    sim.set_state(state);

    t8::v2::StepResult result{};
    for (int decision = 0; decision < 20; ++decision) {
        const Action action = decision == 0 ? Action::Jab : Action::Neutral;
        result = sim.step(action, action);
    }
    near(result.state.p1.health, 173.0, "trade P1 health");
    near(result.state.p2.health, 173.0, "trade P2 health");
    near(result.state.p1.x, -0.08, "trade P1 pushback position");
    near(result.state.p2.x, 0.90, "trade P2 pushback position");
    check(result.state.frame == 80, "trade fixture frame");
    check(result.state.no_action_frames == 52, "trade fixture no-action frames");
}

void test_throw_break_fixture() {
    Simulator sim;
    State state{};
    state.p1 = fighter(180.0, 0.0);
    state.p2 = fighter(180.0, 0.48);
    state.p1.move = 5;
    state.p1.move_frame = 11;
    sim.set_state(state);

    const auto result = sim.step(Action::Neutral, Action::ThrowBreak1);
    near(result.state.p2.health, 180.0, "throw break prevents damage");
    near(result.state.p2.x, 0.612, "throw break applies blocked pushback");
    check(result.state.p1.move_frame == 15, "throw move advances four frames");
    check(result.state.p1.has_hit, "broken throw consumes the attack");
    check(result.state.p2.throw_break_active == 5, "throw-break timer decrements per frame");
    check(result.info.p2_blocks == 1, "throw break counts as a block in V1 metrics");
    check(result.info.p2_throw_breaks == 1, "throw break metric");
    near(result.reward_p1, -0.30, "throw-broken attacker reward");
    near(result.reward_p2, 0.47, "successful throw-break reward");
}

void test_timeout_fixture() {
    Simulator sim;
    State state{};
    state.p1 = fighter(180.0, 0.0);
    state.p2 = fighter(170.0, 0.82);
    state.frame = sim.config().max_frames - 1;
    sim.set_state(state);

    const auto result = sim.step(Action::Neutral, Action::Neutral);
    check(result.terminated && !result.truncated, "V1 timeout is terminal, not truncated");
    check(result.state.frame == 3600 && result.state.winner == 1, "timeout winner and frame");
    check(result.info.timed_out, "timeout metric");
    near(result.reward_p1, -116.71666666666665, "timeout P1 reward");
    near(result.reward_p2, -126.71666666666667, "timeout P2 reward");
}

}  // namespace

int main() {
    test_contract_constants();
    test_reset_and_observation();
    test_neutral_step_fixture();
    test_busy_action_mask();
    test_simultaneous_jab_trade_fixture();
    test_throw_break_fixture();
    test_timeout_fixture();

    if (failures != 0) {
        std::cerr << failures << " parity assertion(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "All Tekken-lite V1 -> V2 parity fixtures passed\n";
    return EXIT_SUCCESS;
}
