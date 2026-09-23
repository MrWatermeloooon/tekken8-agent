#include "t8_v2/full_combat_engine.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace t8::v2;

int failures = 0;

void check(bool condition, std::string_view message) {
    if (!condition) {
        ++failures;
        if (failures < 40) std::cerr << "FAIL: " << message << '\n';
    }
}

FullHitSpec hit(FullHitLevel level, int first, int active, double damage, double reach = 1.0) {
    FullHitSpec value{};
    value.level = level;
    value.first_active_frame = first;
    value.active_frames = active;
    value.damage = damage;
    value.reach = reach;
    value.tracking_left = value.tracking_right = 0.1;
    return value;
}

// i10 high, 2 active, recovery 13: 24 frames total, +1 on block, +8 on hit.
FullMoveSpec jab() {
    FullMoveSpec move{};
    move.stable_id = "test:jab";
    move.hits = {hit(FullHitLevel::High, 10, 2, 5.0)};
    move.recovery = 13;
    move.block_advantage = 1;
    move.hit_advantage = 8;
    move.pushback_hit = move.pushback_block = 0.0;
    return move;
}

FullMoveSpec strike(FullHitLevel level, double damage = 12.0) {
    auto move = jab();
    move.stable_id = "test:strike";
    move.hits = {hit(level, 12, 3, damage)};
    move.recovery = 18;
    move.block_advantage = -4;
    move.hit_advantage = 3;
    return move;
}

FullMoveSpec launcher() {
    auto move = strike(FullHitLevel::Mid, 15.0);
    move.stable_id = "test:launcher";
    move.hit_effect = FullHitEffect::Launch;
    move.block_advantage = -14;
    return move;
}

constexpr FullInput kIdle{};
FullInput use(std::size_t index) { return FullInput{index, FullUniversal::Idle}; }
FullInput act(FullUniversal action) { return FullInput{std::nullopt, action}; }

struct Recorded {
    std::vector<FullContactEvent> contacts;
    std::vector<FullTransition> transitions;
};

Recorded run(FullCombatEngine& engine, int frames, FullInput p1, FullInput p2,
             FullInput p1_after = kIdle, FullInput p2_after = kIdle) {
    Recorded recorded;
    for (int frame = 0; frame < frames; ++frame) {
        const auto events = engine.step(frame == 0 ? p1 : p1_after, frame == 0 ? p2 : p2_after);
        recorded.contacts.insert(recorded.contacts.end(), events.contacts.begin(), events.contacts.end());
        recorded.transitions.insert(recorded.transitions.end(), events.transitions.begin(), events.transitions.end());
    }
    return recorded;
}

// Steps idle inputs from the current frame; returns (P1, P2) first actionable frame offsets.
std::pair<int, int> recovery_order(FullCombatEngine& engine, FullInput p2_hold = kIdle) {
    int p1 = -1;
    int p2 = -1;
    for (int frame = 0; frame < 200 && (p1 < 0 || p2 < 0); ++frame) {
        if (p1 < 0 && engine.actionable(0)) p1 = frame;
        if (p2 < 0 && engine.actionable(1)) p2 = frame;
        engine.step(kIdle, p2_hold);
    }
    return {p1, p2};
}

FullCombatEngine engine_with(std::vector<FullMoveSpec> p1, std::vector<FullMoveSpec> p2 = {jab()},
                             FullEngineConfig config = {}) {
    FullCombatEngine engine({std::move(p1), std::move(p2)}, config);
    engine.reset(-0.4, 0.4);  // close range: 0.8 apart, inside the 1.0 test reach
    return engine;
}

void test_frame_advantage_is_exact() {
    // Block: P2 holds standing guard.
    auto engine = engine_with({jab()});
    auto recorded = run(engine, 10, use(0), kIdle);
    check(recorded.contacts.size() == 1 && recorded.contacts[0].contact == FullContact::Blocked,
          "jab connects on its 10th frame and is blocked by standing guard");
    auto [p1, p2] = recovery_order(engine);
    check(p2 - p1 == 1, "blocked jab is exactly +1");

    // Hit: P2 walks forward (no guard).
    engine = engine_with({jab()});
    recorded = run(engine, 10, use(0), act(FullUniversal::WalkForward), kIdle, act(FullUniversal::WalkForward));
    check(recorded.contacts.size() == 1 && recorded.contacts[0].contact == FullContact::Hit, "walking into a jab is a hit");
    std::tie(p1, p2) = recovery_order(engine);
    check(p2 - p1 == 8, "jab on hit is exactly +8");

    // Connecting on the second active frame leaves one frame less recovery: +9.
    engine = engine_with({jab()});
    engine.reset(0.0, 1.26);
    recorded = run(engine, 11, use(0), act(FullUniversal::WalkForward), kIdle, act(FullUniversal::WalkForward));
    check(recorded.contacts.size() == 1 && engine.state().frame == 11, "late connection on active frame 11");
    std::tie(p1, p2) = recovery_order(engine);
    check(p2 - p1 == 9, "a later active frame gives one more frame of advantage");

    // Minus on block: P2 punishes afterwards.
    engine = engine_with({strike(FullHitLevel::Mid)});
    run(engine, 12, use(0), kIdle);
    std::tie(p1, p2) = recovery_order(engine);
    check(p1 - p2 == 4, "a -4 mid leaves the attacker 4 frames behind");
}

void test_active_window_and_reach() {
    auto engine = engine_with({jab()});
    engine.reset(0.0, 1.5);
    const auto recorded = run(engine, 30, use(0), kIdle);
    check(recorded.contacts.empty(), "a hit out of reach whiffs");
    check(engine.state().fighters[1].health == 180.0, "whiff deals no damage");

    // A second hit in a string resolves separately.
    auto string = jab();
    string.stable_id = "test:1,2";
    string.hits.push_back(hit(FullHitLevel::Mid, 20, 2, 8.0));
    engine = engine_with({string});
    const auto both = run(engine, 20, use(0), act(FullUniversal::WalkForward), kIdle, act(FullUniversal::WalkForward));
    check(both.contacts.size() == 2 && both.contacts[1].hit_index == 1, "both string hits resolve in order");
    check(std::abs(engine.state().fighters[1].health - (180.0 - 13.0)) < 1e-9, "string damage sums");
}

void test_hit_levels_and_guards() {
    // High whiffs over a crouching guard.
    auto engine = engine_with({jab()});
    auto recorded = run(engine, 30, use(0), act(FullUniversal::Crouch), kIdle, act(FullUniversal::Crouch));
    check(recorded.contacts.empty(), "high whiffs against crouching");
    const struct {
        FullHitLevel level;
        FullUniversal guard;
        FullContact expected;
        const char* message;
    } cases[] = {
        {FullHitLevel::Low, FullUniversal::Idle, FullContact::Hit, "low hits standing guard"},
        {FullHitLevel::Low, FullUniversal::Crouch, FullContact::Blocked, "low is blocked crouching"},
        {FullHitLevel::Mid, FullUniversal::Crouch, FullContact::Hit, "mid hits crouching guard"},
        {FullHitLevel::Mid, FullUniversal::Idle, FullContact::Blocked, "mid is blocked standing"},
        {FullHitLevel::SpecialMid, FullUniversal::Crouch, FullContact::Blocked, "special mid is blocked crouching"},
        {FullHitLevel::Throw, FullUniversal::Idle, FullContact::Hit, "throws are unblockable"},
    };
    for (const auto& item : cases) {
        engine = engine_with({strike(item.level)});
        recorded = run(engine, 12, use(0), act(item.guard), kIdle, act(item.guard));
        check(recorded.contacts.size() == 1 && recorded.contacts[0].contact == item.expected, item.message);
    }
    engine = engine_with({strike(FullHitLevel::Throw)});
    recorded = run(engine, 12, use(0), act(FullUniversal::Crouch), kIdle, act(FullUniversal::Crouch));
    check(recorded.contacts.empty(), "throws whiff against crouching");
    engine = engine_with({strike(FullHitLevel::Throw)});
    recorded = run(engine, 12, use(0), act(FullUniversal::ThrowBreak));
    check(recorded.contacts.size() == 1 && recorded.contacts[0].contact == FullContact::ThrowBroken,
          "an armed throw break breaks the throw");
    check(engine.state().fighters[1].health == 180.0, "a broken throw deals no damage");
}

void test_hitstun_does_not_guard() {
    // Two jabs: the second lands during the first's hitstun and is not blocked.
    auto fast = jab();
    fast.hit_advantage = 30;
    auto engine = engine_with({fast});
    run(engine, 10, use(0), act(FullUniversal::WalkForward), kIdle, act(FullUniversal::WalkForward));
    while (!engine.actionable(0)) engine.step(kIdle, kIdle);
    const auto recorded = run(engine, 10, use(0), kIdle);
    check(recorded.contacts.size() == 1 && recorded.contacts[0].contact == FullContact::Hit,
          "hitstun does not grant a guard");
}

void test_tracking_sidestep_and_homing() {
    auto linear = strike(FullHitLevel::Mid);
    auto homing = linear;
    homing.hits[0].homing = true;
    for (const auto* move : {&linear, &homing}) {
        auto engine = engine_with({*move});
        const auto recorded = run(engine, 12, use(0), act(FullUniversal::SidestepRight));
        const bool connected = !recorded.contacts.empty();
        check(move->hits[0].homing ? connected : !connected,
              move->hits[0].homing ? "homing hits a sidestep" : "a linear mid misses a sidestep");
    }
}

void test_collision_and_walls() {
    auto engine = engine_with({jab()});
    engine.reset(-0.5, 0.5);
    run(engine, 80, act(FullUniversal::WalkForward), act(FullUniversal::WalkForward),
        act(FullUniversal::WalkForward), act(FullUniversal::WalkForward));
    const auto& state = engine.state();
    check(state.fighters[1].x - state.fighters[0].x >= 2.0 * engine.config().body_radius - 1e-9,
          "bodies never overlap when walking into each other");

    // Pushback into a wall moves the attacker back instead.
    auto push = strike(FullHitLevel::Mid);
    push.pushback_block = 0.3;
    engine = engine_with({push});
    engine.reset(2.8, 3.6);
    run(engine, 12, use(0), kIdle);
    check(std::abs(engine.state().fighters[1].x - 3.6) < 1e-9, "the defender stays at the wall");
    check(std::abs(engine.state().fighters[0].x - 2.5) < 1e-9, "blocked pushback at the wall moves the attacker back");
}

void test_travel_closes_distance() {
    auto lunge = strike(FullHitLevel::Mid);
    lunge.travel = 0.55;
    auto engine = engine_with({lunge});
    engine.reset(0.0, 1.5);
    const auto recorded = run(engine, 12, use(0), kIdle);
    check(recorded.contacts.size() == 1, "forward travel during startup brings the hit into reach");
}

void test_juggle_scaling_landing_and_wakeup() {
    auto follow = jab();
    follow.hits[0].level = FullHitLevel::Mid;
    follow.hits[0].damage = 10.0;
    auto low = strike(FullHitLevel::Low, 6.0);
    auto engine = engine_with({launcher(), follow, low});
    run(engine, 12, use(0), act(FullUniversal::WalkForward), kIdle, act(FullUniversal::WalkForward));
    const auto& p2 = engine.state().fighters[1];
    check(p2.posture == FullPosture::Airborne && p2.combo_hits == 1, "the launcher sends the defender airborne");
    while (!engine.actionable(0)) engine.step(kIdle, kIdle);
    const double before = p2.health;
    run(engine, 10, use(1), kIdle);
    check(std::abs((before - p2.health) - 9.0) < 1e-9, "a juggle hit is scaled by the combo (10 x 0.9)");
    check(p2.combo_hits == 2, "juggle hits count toward the combo");
    int guard = 0;
    while (p2.posture == FullPosture::Airborne && guard++ < 300) engine.step(kIdle, kIdle);
    check(p2.posture == FullPosture::Grounded, "the juggle ends grounded");
    while (!engine.actionable(0)) engine.step(kIdle, kIdle);
    auto recorded = run(engine, 10, use(1), kIdle);
    check(recorded.contacts.empty(), "a mid whiffs on a grounded opponent");
    while (!engine.actionable(0)) engine.step(kIdle, kIdle);
    recorded = run(engine, 12, use(2), kIdle);
    check(recorded.contacts.size() == 1 && recorded.contacts[0].contact == FullContact::Hit, "a low hits a grounded opponent");
    check(!engine.legal(1, act(FullUniversal::WakeupStand)), "wake-up needs the minimum knockdown time");
    for (int frame = 0; frame < engine.config().knockdown_frames; ++frame) engine.step(kIdle, kIdle);
    check(engine.legal(1, act(FullUniversal::WakeupStand)), "wake-up becomes legal after the knockdown time");
    engine.step(kIdle, act(FullUniversal::WakeupStand));
    check(p2.posture == FullPosture::Wakeup, "wake-up starts");
    for (int frame = 0; frame < engine.config().wakeup_stand_frames; ++frame) engine.step(kIdle, kIdle);
    check(p2.posture == FullPosture::Standing && p2.combo_hits == 0, "standing up ends the combo");
}

void test_tornado_extends_once_and_tech_roll() {
    auto tornado = jab();
    tornado.hits[0].level = FullHitLevel::Mid;
    tornado.hits[0].tornado = true;
    auto engine = engine_with({launcher(), tornado});
    run(engine, 12, use(0), act(FullUniversal::WalkForward), kIdle, act(FullUniversal::WalkForward));
    while (!engine.actionable(0)) engine.step(kIdle, kIdle);
    auto& p2 = engine.mutable_state().fighters[1];
    run(engine, 10, use(1), kIdle);
    check(p2.tornado_used, "tornado is applied");
    const int after_first = p2.posture_frames;
    while (!engine.actionable(0)) engine.step(kIdle, kIdle);
    run(engine, 10, use(1), kIdle);
    check(p2.posture_frames < after_first + engine.config().tornado_extra_frames,
          "a second tornado in the same combo does not extend again");

    auto knockdown = strike(FullHitLevel::Mid);
    knockdown.hit_effect = FullHitEffect::Knockdown;
    engine = engine_with({knockdown});
    run(engine, 12, use(0), act(FullUniversal::WalkForward), kIdle, act(FullUniversal::WalkForward));
    check(engine.state().fighters[1].posture == FullPosture::Grounded, "knockdown effect grounds the defender");
    check(engine.legal(1, act(FullUniversal::TechRoll)), "a knockdown can be teched right away");
    for (int frame = 0; frame < engine.config().tech_window_frames; ++frame) engine.step(kIdle, kIdle);
    check(!engine.legal(1, act(FullUniversal::TechRoll)), "the tech window closes");
}

void test_wall_splat_and_breaks() {
    auto carry = jab();
    carry.hits[0].level = FullHitLevel::Mid;
    carry.pushback_hit = 1.0;
    auto breaker = carry;
    breaker.hits[0].wall_break = true;
    auto floor = carry;
    floor.pushback_hit = 0.0;
    floor.hits[0].floor_break = true;
    auto balcony = carry;
    balcony.hits[0].balcony_break = true;

    const auto launched_near_wall = [&](FullMoveSpec follow_up) {
        auto engine = engine_with({launcher(), std::move(follow_up)});
        engine.reset(2.4, 3.2);
        run(engine, 12, use(0), act(FullUniversal::WalkForward), kIdle, act(FullUniversal::WalkForward));
        while (!engine.actionable(0)) engine.step(kIdle, kIdle);
        return engine;
    };

    auto engine = launched_near_wall(carry);
    run(engine, 10, use(1), kIdle);
    check(engine.state().fighters[1].posture == FullPosture::WallSplat, "a juggle carried into the wall splats");
    check(engine.state().fighters[1].wall_splat_used, "the splat is recorded for the combo");

    engine = launched_near_wall(breaker);
    auto recorded = run(engine, 10, use(1), kIdle);
    check(recorded.transitions.size() == 1 && recorded.transitions[0] == FullTransition::WallBreak, "wall break transition");
    check(engine.state().stage.right_wall > 3.6 && !engine.state().stage.right_wall_breakable, "the broken wall opens a new area");
    check(engine.state().fighters[1].posture == FullPosture::Airborne, "a wall break leaves the defender airborne");

    engine = launched_near_wall(floor);
    recorded = run(engine, 10, use(1), kIdle);
    check(recorded.transitions.size() == 1 && recorded.transitions[0] == FullTransition::FloorBreak, "floor break transition");
    check(engine.state().stage.level == 1 && !engine.state().stage.floor_breakable, "floor break drops to the next level once");

    engine = launched_near_wall(balcony);
    engine.mutable_state().stage.right_balcony = true;
    recorded = run(engine, 10, use(1), kIdle);
    check(recorded.transitions.size() == 1 && recorded.transitions[0] == FullTransition::BalconyBreak, "balcony break transition");
    check(engine.state().stage.level == 1 && engine.state().fighters[1].x > 3.6, "balcony break moves the fight past the old wall");
}

void test_crush_armor_parry_invincibility_and_counter_hits() {
    auto hopkick = strike(FullHitLevel::Mid, 14.0);
    hopkick.low_crush = {5, 20};
    auto power = strike(FullHitLevel::Mid, 20.0);
    power.power_crush = {5, 14};
    auto parry = strike(FullHitLevel::Mid, 0.0);
    parry.hits[0].reach = 0.01;  // pure parry
    parry.parry = {3, 20};
    auto invincible = strike(FullHitLevel::Mid, 0.0);
    invincible.hits[0].reach = 0.01;
    invincible.invincible = {1, 20};
    auto slow = strike(FullHitLevel::Mid);
    slow.hits[0].first_active_frame = 25;
    const auto defended = [&](FullMoveSpec attack, FullMoveSpec defense) {
        auto engine = engine_with({std::move(attack)}, {std::move(defense)});
        return std::make_pair(run(engine, 13, use(0), use(0)), engine);
    };
    auto [low_vs_crush, _a] = defended(strike(FullHitLevel::Low), hopkick);
    check(!low_vs_crush.contacts.empty() && low_vs_crush.contacts[0].contact == FullContact::Crushed, "low crush evades a low");
    auto [mid_vs_power, engine_power] = defended(strike(FullHitLevel::Mid), power);
    check(!mid_vs_power.contacts.empty() && mid_vs_power.contacts[0].contact == FullContact::Armored, "power crush absorbs a mid");
    check(engine_power.state().fighters[1].health < 180.0 && engine_power.state().fighters[1].move.has_value(),
          "armor takes the damage and keeps attacking");
    auto [mid_vs_parry, engine_parry] = defended(strike(FullHitLevel::Mid), parry);
    check(!mid_vs_parry.contacts.empty() && mid_vs_parry.contacts[0].contact == FullContact::Parried, "parry stops a mid");
    check(engine_parry.state().fighters[0].stun > 0, "a parried attacker is stunned");
    auto [mid_vs_invincible, _b] = defended(strike(FullHitLevel::Mid), invincible);
    check(!mid_vs_invincible.contacts.empty() && mid_vs_invincible.contacts[0].contact == FullContact::Evaded, "invincibility evades");
    auto counter = strike(FullHitLevel::Mid, 10.0);
    counter.counter_hit_advantage = 20;
    auto [counter_hit, engine_counter] = defended(counter, slow);
    check(!counter_hit.contacts.empty() && counter_hit.contacts[0].contact == FullContact::CounterHit,
          "hitting a move in startup is a counter-hit");
    check(std::abs(engine_counter.state().fighters[1].health - (180.0 - 12.0)) < 1e-9, "counter-hit damage bonus");
}

void test_determinism_and_validation() {
    const auto play = [] {
        auto engine = engine_with({jab(), launcher(), strike(FullHitLevel::Low)});
        const FullInput script[] = {use(1), kIdle, act(FullUniversal::WalkForward), use(0), use(2)};
        for (int frame = 0; frame < 240; ++frame) {
            engine.step(script[frame % 5], frame % 7 == 0 ? act(FullUniversal::SidestepLeft) : act(FullUniversal::WalkForward));
        }
        return engine.state();
    };
    const auto first = play();
    const auto second = play();
    check(first.frame == second.frame && first.fighters[0].x == second.fighters[0].x &&
          first.fighters[1].health == second.fighters[1].health && first.fighters[1].axis == second.fighters[1].axis,
          "identical inputs give identical fights");
    auto unordered = jab();
    unordered.hits.push_back(hit(FullHitLevel::Mid, 5, 1, 1.0));
    bool rejected = false;
    try {
        engine_with({unordered});
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    check(rejected, "unordered hits are rejected");
}

}  // namespace

int main() {
    try {
        test_frame_advantage_is_exact();
        test_active_window_and_reach();
        test_hit_levels_and_guards();
        test_hitstun_does_not_guard();
        test_tracking_sidestep_and_homing();
        test_collision_and_walls();
        test_travel_closes_distance();
        test_juggle_scaling_landing_and_wakeup();
        test_tornado_extends_once_and_tech_roll();
        test_wall_splat_and_breaks();
        test_crush_armor_parry_invincibility_and_counter_hits();
        test_determinism_and_validation();
    } catch (const std::exception& error) {
        std::cerr << "engine test error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
    if (failures != 0) {
        std::cerr << failures << " full combat engine assertion(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "full combat engine tests passed\n";
    return EXIT_SUCCESS;
}
