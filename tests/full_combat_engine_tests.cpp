#include "t8_v2/full_combat_engine.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
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
    move.hit_advantage = 30;  // "+30a": airborne until the attacker has recovered for 30 frames
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
        {FullHitLevel::Throw, FullUniversal::Idle, FullContact::ThrowHeld, "throws are unblockable"},
    };
    for (const auto& item : cases) {
        engine = engine_with({strike(item.level)});
        recorded = run(engine, 12, use(0), act(item.guard), kIdle, act(item.guard));
        check(recorded.contacts.size() == 1 && recorded.contacts[0].contact == item.expected, item.message);
    }
    engine = engine_with({strike(FullHitLevel::Throw)});
    recorded = run(engine, 12, use(0), act(FullUniversal::Crouch), kIdle, act(FullUniversal::Crouch));
    check(recorded.contacts.empty(), "throws whiff against crouching");
}

FullMoveSpec throw_move(FullThrowBreak rule) {
    auto move = strike(FullHitLevel::Throw);
    move.stable_id = "test:throw";
    move.hits[0].throw_break = rule;
    return move;
}

// Steps idle until a contact of `kind` appears; returns it, or nullopt.
std::optional<FullContactEvent> until_contact(FullCombatEngine& engine, FullContact kind, int limit = 60) {
    for (int frame = 0; frame < limit; ++frame) {
        for (const auto& contact : engine.step(kIdle, kIdle).contacts) {
            if (contact.contact == kind) return contact;
        }
    }
    return std::nullopt;
}

void test_throw_break_window() {
    // The right button inside the window breaks the throw; both recover together.
    auto engine = engine_with({throw_move(FullThrowBreak::OneOrTwo)});
    check(!engine.legal(1, act(FullUniversal::ThrowBreak1)), "a break input needs a held throw");
    auto recorded = run(engine, 12, use(0), kIdle);
    check(recorded.contacts.size() == 1 && recorded.contacts[0].contact == FullContact::ThrowHeld, "the throw connects");
    check(!engine.actionable(1), "a held fighter cannot act");
    auto events = engine.step(kIdle, act(FullUniversal::ThrowBreak2));
    check(events.contacts.size() == 1 && events.contacts[0].contact == FullContact::ThrowBroken, "2 breaks a 1-or-2 throw");
    check(engine.state().fighters[1].health == 180.0, "a broken throw deals no damage");
    auto [p1, p2] = recovery_order(engine);
    check(p1 == p2 && p1 > 0, "a throw break is neutral");

    // One attempt only: a wrong button spends it, and the throw lands with exact advantage.
    engine = engine_with({throw_move(FullThrowBreak::OneOrTwo)});
    run(engine, 12, use(0), kIdle);
    engine.step(kIdle, act(FullUniversal::ThrowBreak));
    check(!engine.legal(1, act(FullUniversal::ThrowBreak1)), "the break attempt is spent");
    const auto landed = until_contact(engine, FullContact::Hit);
    check(landed.has_value() && std::abs(engine.state().fighters[1].health - 168.0) < 1e-9,
          "an unbroken throw lands when the window closes");
    std::tie(p1, p2) = recovery_order(engine);
    check(p2 - p1 == 3, "a throw's hit advantage is exact after its break window");

    // A throw whose recovery is shorter than the window still resolves exactly.
    auto quick = throw_move(FullThrowBreak::One);
    quick.recovery = 2;
    engine = engine_with({quick});
    run(engine, 12, use(0), kIdle);
    check(until_contact(engine, FullContact::Hit).has_value(), "a short throw completes after its window");
    std::tie(p1, p2) = recovery_order(engine);
    check(p2 - p1 == 3, "a short throw keeps its hit advantage");

    // Unbreakable throws land at once; side switches swap positions.
    auto unbreakable = throw_move(FullThrowBreak::None);
    unbreakable.side_switch_on_hit = true;
    engine = engine_with({unbreakable});
    recorded = run(engine, 12, use(0), kIdle);
    check(recorded.contacts.size() == 1 && recorded.contacts[0].contact == FullContact::Hit, "unbreakable throws land at once");
    check(engine.state().fighters[0].x > engine.state().fighters[1].x, "a side switch swaps sides");
}

FullMoveSpec engager() {
    auto move = strike(FullHitLevel::Mid, 20.0);
    move.stable_id = "test:engager";
    move.heat_engager = true;
    move.heat_dash_block_advantage = 5;
    move.heat_dash_hit_advantage = 43;
    move.heat_dash_hit_effect = FullHitEffect::Knockdown;
    return move;
}

void test_bug_pass_regressions() {
    // Pushback after a side switch drives the defender away from the attacker's new side.
    auto switcher = strike(FullHitLevel::Mid);
    switcher.side_switch_on_hit = true;
    switcher.pushback_hit = 0.3;
    auto engine = engine_with({switcher});
    // A crouching defender stays put and is hit by the mid.
    run(engine, 12, use(0), act(FullUniversal::Crouch), kIdle, act(FullUniversal::Crouch));
    const double gap = engine.state().fighters[0].x - engine.state().fighters[1].x;
    check(std::abs(gap - (0.8 + 0.3)) < 1e-9, "pushback after a side switch widens the gap");

    // A strike landing on the same frame a throw connects frees the held fighter.
    auto grab = throw_move(FullThrowBreak::OneTwo);
    auto poke = strike(FullHitLevel::Mid);
    engine = engine_with({grab}, {poke});
    run(engine, 12, use(0), use(0));
    const auto& thrower = engine.state().fighters[0];
    const auto& target = engine.state().fighters[1];
    check(thrower.pending_throw_hit < 0 && target.throw_break_frames == 0, "a strike on the same frame beats the throw");
    int frames = 0;
    while (!engine.actionable(0) && frames++ < 200) engine.step(kIdle, kIdle);
    check(engine.actionable(0), "the interrupted thrower recovers");

    // A Heat version with published Heat Dash data can Heat Dash while in Heat.
    auto heat_version = strike(FullHitLevel::Mid, 20.0);
    heat_version.requires_heat = true;
    heat_version.heat_dash_block_advantage = 5;
    engine = engine_with({heat_version});
    engine.mutable_state().fighters[0].heat = true;
    engine.mutable_state().fighters[0].heat_frames = 600;
    engine.mutable_state().fighters[0].heat_available = false;
    run(engine, 12, use(0), kIdle);
    check(engine.legal(0, act(FullUniversal::HeatDash)), "Heat versions with Heat Dash data can Heat Dash");

    // Heat lasts exactly its duration whether it starts at a move's start or on contact.
    FullEngineConfig short_heat{};
    short_heat.heat_duration_frames = 30;
    auto burst = jab();
    burst.engages_heat = true;
    engine = engine_with({burst}, {jab()}, short_heat);
    int active = 0;
    for (int frame = 0; frame < 60; ++frame) {
        engine.step(frame == 0 ? use(0) : kIdle, kIdle);
        active += engine.state().fighters[0].heat ? 1 : 0;
    }
    check(active == 30, "Heat Burst Heat lasts exactly its duration");
    engine = engine_with({engager()}, {jab()}, short_heat);
    active = 0;
    for (int frame = 0; frame < 60; ++frame) {
        engine.step(frame == 0 ? use(0) : kIdle, kIdle);
        active += engine.state().fighters[0].heat ? 1 : 0;
    }
    check(active == 30, "engager Heat lasts exactly its duration");

    // Heat Dash on hit replaces a launch with the published knockdown.
    auto launch_engager = engager();
    launch_engager.hit_effect = FullHitEffect::Launch;
    engine = engine_with({launch_engager});
    run(engine, 12, use(0), act(FullUniversal::WalkForward), kIdle, act(FullUniversal::WalkForward));
    check(engine.state().fighters[1].posture == FullPosture::Airborne, "the engager launches");
    engine.step(act(FullUniversal::HeatDash), kIdle);
    check(engine.state().fighters[1].posture == FullPosture::Standing && engine.state().fighters[1].crumpled,
          "Heat Dash on hit replaces the launch with its own outcome");
}

void test_stances_and_resources() {
    FullCharacterRules rules{};
    FullStanceRule open{};
    open.name = "OPEN";                 // stance 0: can guard, times out
    open.max_frames = 30;
    FullStanceRule shut{};
    shut.name = "SHUT";                 // stance 1: cannot guard, pulses, auto-parries lows into an outcome
    shut.can_guard = false;
    shut.auto_parry = FullLevelLow;
    shut.parry_outcomes[2] = "test:counter";
    shut.pulse_interval_frames = 10;
    shut.pulse_recoverable = 2.0;
    shut.pulse_resource = 5.0;
    rules.stances = {open, shut};
    rules.resource_max = 100.0;
    rules.install_threshold = 50.0;
    rules.resource_persists = true;

    FullMoveSpec enter_open{};
    enter_open.stable_id = "test:enter_open";
    enter_open.recovery = 8;
    enter_open.result_stance = 0;
    auto enter_shut = enter_open;
    enter_shut.stable_id = "test:enter_shut";
    enter_shut.result_stance = 1;
    auto counter = strike(FullHitLevel::Mid, 20.0);
    counter.stable_id = "test:counter";
    counter.reactive = true;
    counter.hit_effect = FullHitEffect::Knockdown;
    counter.resource_gain_start = 10.0;
    auto poke = strike(FullHitLevel::Mid, 10.0);
    poke.stable_id = "test:poke";
    poke.resource_gain_hit = 30.0;
    poke.resource_gain_block = 5.0;
    poke.result_stance_on_block = 0;
    poke.install_damage_bonus = 6.0;
    poke.install_reach = 3.0;
    auto crouch_end = strike(FullHitLevel::Mid);
    crouch_end.stable_id = "test:crouch_end";
    crouch_end.result_crouching = true;
    auto heat_cost = jab();
    heat_cost.stable_id = "test:heat_cost";
    heat_cost.heat_cost_frames = 450;
    auto low = strike(FullHitLevel::Low, 8.0);
    const std::vector<FullMoveSpec> moves = {enter_open, enter_shut, counter, poke, crouch_end, heat_cost, low};
    const auto fight = [&] {
        FullCombatEngine engine(std::array<std::vector<FullMoveSpec>, 2>{moves, moves}, FullEngineConfig{},
                                std::array<FullCharacterRules, 2>{rules, rules});
        engine.reset(-0.4, 0.4);
        return engine;
    };
    const auto settle = [](FullCombatEngine& engine, FullInput p2 = kIdle) {
        for (int frame = 0; frame < 200 && !engine.actionable(0); ++frame) engine.step(kIdle, p2);
    };

    // Zero-hit stance entry; the stance times out; reactive moves are never input.
    auto engine = fight();
    check(!engine.legal(0, use(2)), "reactive outcomes cannot be input");
    engine.step(use(0), kIdle);
    settle(engine);
    check(engine.state().fighters[0].stance == 0, "a zero-hit move enters its stance");
    for (int frame = 0; frame < 30; ++frame) engine.step(kIdle, kIdle);
    check(engine.state().fighters[0].stance < 0, "a stance with a time limit returns to neutral");

    // A stance that cannot guard is hit; pulses restore recoverable health and resource.
    engine = fight();
    engine.mutable_state().fighters[0].health = 150.0;
    engine.mutable_state().fighters[0].recoverable_health = 10.0;
    engine.step(use(1), kIdle);
    settle(engine);
    for (int frame = 0; frame < 10; ++frame) engine.step(kIdle, kIdle);
    check(std::abs(engine.state().fighters[0].recoverable_health - 8.0) < 1e-9 &&
              std::abs(engine.state().fighters[0].resource - 5.0) < 1e-9,
          "a stance pulse restores recoverable health and resource");
    auto recorded = run(engine, 12, kIdle, use(3));
    check(recorded.contacts.size() == 1 && recorded.contacts[0].contact == FullContact::Hit &&
              engine.state().fighters[0].stance < 0,
          "a stance that cannot guard is hit, and the hit ends the stance");

    // Automatic stance parry plays its outcome.
    engine = fight();
    engine.step(use(1), kIdle);
    settle(engine);
    recorded = run(engine, 12, kIdle, use(6));
    check(recorded.contacts.size() == 1 && recorded.contacts[0].contact == FullContact::Parried,
          "the stance parries a low automatically");
    check(engine.state().fighters[1].posture == FullPosture::Grounded &&
              std::abs(engine.state().fighters[1].health - 160.0) < 1e-9 && engine.state().fighters[0].resource >= 10.0,
          "the parry outcome deals its damage, knocks down, and gains resource");

    // Resource gains, install bonuses, on-block stance transitions, and next_round.
    engine = fight();
    run(engine, 12, use(3), act(FullUniversal::WalkForward), kIdle, act(FullUniversal::WalkForward));
    check(std::abs(engine.state().fighters[0].resource - 30.0) < 1e-9, "a hit gains its published resource once");
    engine = fight();
    run(engine, 12, use(3), kIdle);
    settle(engine);
    check(std::abs(engine.state().fighters[0].resource - 5.0) < 1e-9 && engine.state().fighters[0].stance == 0,
          "a blocked move gains its block resource and takes its on-block stance");
    engine = fight();
    engine.reset(-1.5, 1.5);  // out of normal reach
    engine.mutable_state().fighters[0].resource = 60.0;
    engine.mutable_state().fighters[0].installed = true;
    recorded = run(engine, 12, use(3), act(FullUniversal::WalkForward), kIdle, act(FullUniversal::WalkForward));
    check(recorded.contacts.size() == 1 && std::abs(recorded.contacts[0].damage - 16.0) < 1e-9,
          "the install extends reach and adds its damage bonus");
    engine.next_round();
    check(engine.state().fighters[0].resource >= 60.0 && engine.state().fighters[0].installed,
          "a persistent resource and its install carry into the next round");

    // Crouched endings and Heat-timer costs.
    engine = fight();
    engine.step(use(4), kIdle);
    settle(engine);
    check(engine.state().fighters[0].posture == FullPosture::Crouching, "a move can end crouched");
    engine = fight();
    check(!engine.legal(0, use(5)), "a move that spends Heat time needs Heat");
    engine.mutable_state().fighters[0].heat = true;
    engine.mutable_state().fighters[0].heat_frames = 600;
    engine.step(use(5), kIdle);
    check(engine.state().fighters[0].heat_frames == 600 - 1 - 450, "a move spends its Heat time");

    // Zero-hit moves need a recovery.
    FullMoveSpec nothing{};
    nothing.stable_id = "test:nothing";
    bool rejected = false;
    try {
        engine_with({nothing});
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    check(rejected, "a move with neither hits nor recovery is rejected");
}

void test_attack_throws_situations_and_special_states() {
    // Attack throws: on hit, on counter-hit only, and front/standing conditions.
    auto grab_on_hit = strike(FullHitLevel::Mid, 10.0);
    grab_on_hit.attack_throw = FullMoveSpec::AttackThrow::OnHit;
    grab_on_hit.attack_throw_damage = 30.0;
    auto grab_on_counter = grab_on_hit;
    grab_on_counter.attack_throw = FullMoveSpec::AttackThrow::OnCounterHit;
    auto grab_standing = grab_on_hit;
    grab_standing.attack_throw_standing_only = true;
    grab_standing.attack_throw_airborne = false;
    auto engine = engine_with({grab_on_hit, grab_on_counter, grab_standing});
    run(engine, 12, use(0), act(FullUniversal::WalkForward), kIdle, act(FullUniversal::WalkForward));
    check(std::abs(engine.state().fighters[1].health - 140.0) < 1e-9, "an attack throw adds its damage on hit");
    engine = engine_with({grab_on_hit, grab_on_counter, grab_standing});
    run(engine, 12, use(1), act(FullUniversal::WalkForward), kIdle, act(FullUniversal::WalkForward));
    check(std::abs(engine.state().fighters[1].health - 170.0) < 1e-9, "a counter-hit attack throw needs a counter-hit");
    engine = engine_with({grab_on_hit, grab_on_counter, grab_standing}, {strike(FullHitLevel::Mid, 5.0)});
    run(engine, 13, use(1), use(0));  // P2 is mid-attack: counter-hit
    check(engine.state().fighters[1].health < 180.0 - 12.0 - 29.0, "a counter-hit triggers the attack throw");
    engine = engine_with({grab_on_hit, grab_on_counter, grab_standing});
    run(engine, 12, use(2), act(FullUniversal::Crouch), kIdle, act(FullUniversal::Crouch));
    check(std::abs(engine.state().fighters[1].health - 170.0) < 1e-9, "a standing-only attack throw skips a crouching opponent");

    // Back-turned: only BT moves, no guard, hits use the back-turned values, movement turns around.
    auto turn = strike(FullHitLevel::Mid);
    turn.result_back_turned = true;
    auto from_back = jab();
    from_back.requires_back_turned = true;
    auto versus_back = strike(FullHitLevel::Mid, 10.0);
    versus_back.back_turned_hit_advantage = 10;
    versus_back.back_turned_hit_effect = FullHitEffect::Launch;
    auto back_throw = throw_move(FullThrowBreak::None);
    back_throw.requires_opponent_back_turned = true;
    engine = engine_with({turn, from_back, versus_back, back_throw}, {turn, from_back, versus_back, back_throw});
    for (int frame = 0; frame < 40; ++frame) engine.step(frame == 0 ? use(0) : kIdle, kIdle);
    check(engine.state().fighters[0].back_turned, "a move can end back-turned");
    check(engine.legal(0, use(1)) && !engine.legal(0, use(2)), "back-turned fighters use BT moves only");
    check(engine.legal(1, use(3)) && !engine.legal(0, use(3)), "a back throw needs a back-turned opponent");
    auto recorded = run(engine, 12, kIdle, use(2));
    check(recorded.contacts.size() == 1 && recorded.contacts[0].contact == FullContact::Hit &&
              engine.state().fighters[0].posture == FullPosture::Airborne,
          "a back-turned fighter cannot guard and takes the back-turned hit result");
    engine = engine_with({turn, from_back});
    for (int frame = 0; frame < 40; ++frame) engine.step(frame == 0 ? use(0) : kIdle, kIdle);
    engine.step(act(FullUniversal::WalkBack), kIdle);
    check(!engine.state().fighters[0].back_turned, "moving turns a back-turned fighter around");

    // Back to the wall, and exposed sides for side throws.
    auto wall_jump = strike(FullHitLevel::Mid);
    wall_jump.requires_back_to_wall = true;
    auto left_throw = throw_move(FullThrowBreak::One);
    left_throw.requires_opponent_left_side = true;
    auto right_throw = left_throw;
    right_throw.requires_opponent_left_side = false;
    right_throw.requires_opponent_right_side = true;
    engine = engine_with({wall_jump, left_throw, right_throw});
    check(!engine.legal(0, use(0)), "a back-to-wall move needs the wall behind");
    engine.reset(engine.state().stage.left_wall + 0.1, engine.state().stage.left_wall + 0.9);
    check(engine.legal(0, use(0)), "a back-to-wall move is legal with the wall behind");
    engine.reset(-0.4, 0.4);
    check(!engine.legal(0, use(1)) && !engine.legal(0, use(2)), "side throws need an exposed side");
    // P2 faces -x toward P1, so its left is at +axis: P1 moving to +axis stands on P2's left.
    engine.mutable_state().fighters[0].axis = 0.6;
    check(engine.legal(0, use(1)) && !engine.legal(0, use(2)), "P2's left side is exposed");
    engine.mutable_state().fighters[0].axis = -0.6;
    check(engine.legal(0, use(2)) && !engine.legal(0, use(1)), "P2's right side is exposed");

    // Ki Charge: no guard, hits taken are counter-hits, the next attack is stronger.
    FullMoveSpec ki{};
    ki.stable_id = "test:ki";
    ki.recovery = 55;
    ki.ki_charge = true;
    engine = engine_with({ki, jab()}, {jab()});
    engine.step(use(0), kIdle);
    check(engine.state().fighters[0].ki_charge_frames > 0, "Ki Charge starts");
    for (int frame = 0; frame < 60; ++frame) engine.step(kIdle, kIdle);
    recorded = run(engine, 10, kIdle, use(0));
    check(recorded.contacts.size() == 1 && recorded.contacts[0].contact == FullContact::CounterHit,
          "a Ki-charged fighter cannot guard and takes counter-hits");
    engine = engine_with({ki, jab()}, {jab()});
    engine.step(use(0), kIdle);
    for (int frame = 0; frame < 60; ++frame) engine.step(kIdle, kIdle);
    recorded = run(engine, 10, use(1), act(FullUniversal::WalkForward), kIdle, act(FullUniversal::WalkForward));
    check(recorded.contacts.size() == 1 &&
              std::abs(recorded.contacts[0].damage - 5.0 * engine.config().ki_charge_damage_scale) < 1e-9,
          "the attack after Ki Charge is boosted");

    // Partial recoverable damage, K.O. protection, and Rage Art damage growing with lost health.
    auto soft = jab();
    soft.recoverable_damage = 2.0;
    soft.cannot_ko = true;
    engine = engine_with({soft});
    engine.mutable_state().fighters[1].health = 4.0;
    run(engine, 10, use(0), act(FullUniversal::WalkForward), kIdle, act(FullUniversal::WalkForward));
    check(engine.state().fighters[1].health == 1.0 && engine.state().winner < 0, "a move that cannot K.O. leaves 1 health");
    engine = engine_with({soft});
    run(engine, 10, use(0), act(FullUniversal::WalkForward), kIdle, act(FullUniversal::WalkForward));
    check(std::abs(engine.state().fighters[1].recoverable_health - 2.0) < 1e-9, "part of the damage is recoverable");
    auto rage_art = strike(FullHitLevel::Mid, 55.0);
    rage_art.consumes_rage = true;
    rage_art.rage_art_max_damage = 82.0;
    const auto rage_damage = [&](double health) {
        auto fight = engine_with({rage_art});
        fight.mutable_state().fighters[0].rage = true;
        fight.mutable_state().fighters[0].health = health;
        run(fight, 12, use(0), act(FullUniversal::WalkForward), kIdle, act(FullUniversal::WalkForward));
        return 180.0 - fight.state().fighters[1].health;
    };
    // The Rage Art spends Rage as it starts, so the Rage damage bonus does not apply to it.
    check(std::abs(rage_damage(180.0) - 55.0) < 1e-9 && std::abs(rage_damage(90.0) - 68.5) < 1e-9,
          "Rage Art damage grows toward its maximum as health drops");
}

void test_heat_parry_outcome() {
    // f+1+2 style: parries only in Heat; the outcome plays with its own recovery, stance, and Heat cost.
    FullCharacterRules rules{};
    FullStanceRule mia{};
    mia.name = "MIA";
    rules.stances = {mia};
    auto parry = strike(FullHitLevel::High, 5.0);
    parry.stable_id = "test:parry";
    parry.hits[0].reach = 0.01;
    parry.heat_parry = {1, 20};
    parry.heat_parry_levels = FullLevelHigh | FullLevelMid;
    parry.heat_parry_outcome = "test:outcome";
    auto outcome = strike(FullHitLevel::High, 0.0);
    outcome.stable_id = "test:outcome";
    outcome.reactive = true;
    outcome.recovery = 30;
    outcome.hit_advantage = 40;
    outcome.result_stance = 0;
    outcome.heat_cost_frames = 240;
    const std::vector<FullMoveSpec> moves = {parry, outcome, strike(FullHitLevel::Mid)};
    const auto fight = [&](bool heat) {
        FullCombatEngine engine(std::array<std::vector<FullMoveSpec>, 2>{moves, moves}, FullEngineConfig{},
                                std::array<FullCharacterRules, 2>{rules, rules});
        engine.reset(-0.4, 0.4);
        engine.mutable_state().fighters[0].heat = heat;
        engine.mutable_state().fighters[0].heat_frames = heat ? 600 : 0;
        return engine;
    };
    auto engine = fight(false);
    auto recorded = run(engine, 13, use(0), use(2));
    check(!recorded.contacts.empty() && recorded.contacts[0].contact == FullContact::CounterHit,
          "the Heat parry does nothing outside Heat");
    engine = fight(true);
    recorded = run(engine, 13, use(0), use(2));
    check(!recorded.contacts.empty() && recorded.contacts[0].contact == FullContact::Parried, "the Heat parry catches a mid");
    const auto& p1 = engine.state().fighters[0];
    check(p1.move.has_value() && *p1.move == 1 && p1.heat_frames <= 600 - 240, "the outcome plays and spends Heat time");
    auto [a, b] = recovery_order(engine);
    check(b - a == 40, "the outcome's advantage is exact");
    check(engine.state().fighters[0].stance == 0, "the outcome ends in its stance");
}

void test_airtime_follows_published_advantage() {
    // "+30a": the opponent stays airborne until the attacker has recovered for 30 frames.
    auto engine = engine_with({launcher()});
    run(engine, 12, use(0), act(FullUniversal::WalkForward), kIdle, act(FullUniversal::WalkForward));
    int attacker_free = -1;
    int landed = -1;
    for (int frame = 0; frame < 200 && landed < 0; ++frame) {
        if (attacker_free < 0 && engine.actionable(0)) attacker_free = frame;
        if (engine.state().fighters[1].posture != FullPosture::Airborne) landed = frame;
        engine.step(kIdle, kIdle);
    }
    check(landed - attacker_free == 30, "a launch keeps the opponent airborne for its published advantage");

    // A juggle hit keeps the opponent up until the attacker recovers plus the juggle window.
    auto follow = jab();
    follow.hits[0].level = FullHitLevel::Mid;
    engine = engine_with({launcher(), follow});
    run(engine, 12, use(0), act(FullUniversal::WalkForward), kIdle, act(FullUniversal::WalkForward));
    while (!engine.actionable(0)) engine.step(kIdle, kIdle);
    run(engine, 10, use(1), kIdle);
    attacker_free = landed = -1;
    for (int frame = 0; frame < 200 && landed < 0; ++frame) {
        if (attacker_free < 0 && engine.actionable(0)) attacker_free = frame;
        if (engine.state().fighters[1].posture != FullPosture::Airborne) landed = frame;
        engine.step(kIdle, kIdle);
    }
    // Second hit of the combo: the window has shrunk once.
    check(landed - attacker_free == engine.config().juggle_window_frames - engine.config().juggle_window_decay,
          "a juggle hit keeps the opponent up exactly the juggle window after the attacker recovers");

    // Gravity: jabs cannot juggle forever.
    auto quick = jab();
    quick.hits[0].level = FullHitLevel::Mid;
    engine = engine_with({launcher(), quick});
    run(engine, 12, use(0), act(FullUniversal::WalkForward), kIdle, act(FullUniversal::WalkForward));
    int juggle_hits = 0;
    for (int frame = 0; frame < 3000 && engine.state().fighters[1].posture == FullPosture::Airborne; ++frame) {
        const auto events = engine.step(engine.actionable(0) ? use(1) : kIdle, kIdle);
        juggle_hits += static_cast<int>(events.contacts.size());
    }
    check(juggle_hits > 3 && juggle_hits < 30 && engine.state().fighters[1].posture != FullPosture::Airborne,
          "repeated fast juggle hits eventually drop the opponent");

    // A multi-hit Heat Engager allows Heat Dash only after its last hit.
    auto two_hit = engager();
    two_hit.hits.push_back(hit(FullHitLevel::Mid, 20, 2, 10.0));
    engine = engine_with({two_hit});
    bool dash_before_last = false;
    for (int frame = 0; frame < 20; ++frame) {
        engine.step(frame == 0 ? use(0) : kIdle, frame == 0 ? act(FullUniversal::WalkForward) : kIdle);
        if (frame < 19) dash_before_last |= engine.legal(0, act(FullUniversal::HeatDash));
    }
    check(!dash_before_last && engine.legal(0, act(FullUniversal::HeatDash)),
          "Heat Dash waits for the engager's last hit");
}

void test_heat() {
    FullEngineConfig short_heat{};
    short_heat.heat_duration_frames = 30;
    auto burst = strike(FullHitLevel::Mid, 12.0);
    burst.stable_id = "test:heat_burst";
    burst.engages_heat = true;
    burst.power_crush = {1, 16};
    burst.recoverable_only = true;
    auto enhanced = jab();
    enhanced.requires_heat = true;
    auto smash = strike(FullHitLevel::Mid, 30.0);
    smash.consumes_heat = true;
    smash.requires_heat = true;
    auto engine = engine_with({burst, enhanced, smash}, {jab()}, short_heat);
    check(!engine.legal(0, use(1)) && !engine.legal(0, use(2)), "Heat moves need Heat");
    auto events = engine.step(use(0), kIdle);
    const auto& p1 = engine.state().fighters[0];
    check(p1.heat && !p1.heat_available && events.heat_activated[0], "Heat Burst activates Heat");
    while (!engine.actionable(0)) engine.step(kIdle, kIdle);
    check(std::abs(engine.state().fighters[1].health - 180.0) < 1e-9 &&
              std::abs(engine.state().fighters[1].recoverable_health) < 1e-9,
          "a blocked Heat Burst deals no damage without chip");
    for (int frame = 0; frame < 40 && p1.heat; ++frame) engine.step(kIdle, kIdle);
    check(!p1.heat && !engine.legal(0, use(0)), "Heat ends with its timer and is used once per round");

    engine = engine_with({burst, enhanced, smash});
    engine.step(use(0), kIdle);
    while (!engine.actionable(0)) engine.step(kIdle, kIdle);
    check(engine.legal(0, use(1)), "Heat-enhanced moves are legal in Heat");
    engine.step(use(2), kIdle);
    check(!engine.state().fighters[0].heat, "Heat Smash spends the remaining Heat");

    // Heat Engager on block: activates Heat, then Heat Dash is exactly +5.
    engine = engine_with({engager()});
    auto recorded = run(engine, 12, use(0), kIdle);
    check(recorded.contacts.size() == 1 && recorded.contacts[0].contact == FullContact::Blocked &&
              engine.state().fighters[0].heat,
          "a blocked Heat Engager activates Heat");
    check(engine.legal(0, act(FullUniversal::HeatDash)), "Heat Dash is available after an engager connects");
    engine.step(act(FullUniversal::HeatDash), kIdle);
    check(!engine.state().fighters[0].heat, "Heat Dash spends the remaining Heat");
    auto [a, b] = recovery_order(engine);
    check(b - a == 5, "Heat Dash on block is exactly +5");

    // On hit, the Heat Dash follow-up knocks down.
    engine = engine_with({engager()});
    run(engine, 12, use(0), act(FullUniversal::WalkForward), kIdle, act(FullUniversal::WalkForward));
    engine.step(act(FullUniversal::HeatDash), kIdle);
    check(engine.state().fighters[1].crumpled, "Heat Dash on hit follows the published knockdown");

    // Without Heat left, an engager neither activates Heat nor allows Heat Dash.
    engine = engine_with({engager()});
    engine.mutable_state().fighters[0].heat_available = false;
    run(engine, 12, use(0), kIdle);
    check(!engine.state().fighters[0].heat && !engine.legal(0, act(FullUniversal::HeatDash)),
          "Heat Dash needs Heat");
}

void test_recoverable_health_rage_and_ko() {
    // Chip on block is recoverable and cannot K.O.; Heat chip replaces it in Heat.
    auto chipper = strike(FullHitLevel::Mid, 20.0);
    chipper.hits[0].chip_damage = 6.0;
    chipper.hits[0].heat_chip_damage = 9.0;
    auto engine = engine_with({chipper});
    run(engine, 12, use(0), kIdle);
    const auto& p2 = engine.state().fighters[1];
    check(std::abs(p2.health - 174.0) < 1e-9 && std::abs(p2.recoverable_health - 6.0) < 1e-9,
          "chip damage on block becomes recoverable health");
    engine = engine_with({chipper});
    engine.mutable_state().fighters[0].heat = true;
    engine.mutable_state().fighters[0].heat_frames = 600;
    run(engine, 12, use(0), kIdle);
    check(std::abs(engine.state().fighters[1].health - 171.0) < 1e-9, "chip in Heat uses the Heat amount");
    engine = engine_with({chipper});
    engine.mutable_state().fighters[1].health = 1.0;
    run(engine, 12, use(0), kIdle);
    check(engine.state().fighters[1].health == 1.0 && engine.state().winner < 0, "chip damage cannot K.O.");

    // Landing attacks regains recoverable health; restores add to it.
    auto restorer = jab();
    restorer.restore_recoverable_hit = 4.0;
    restorer.restore_health_hit = 2.0;
    engine = engine_with({restorer});
    auto& own = engine.mutable_state().fighters[0];
    own.health = 150.0;
    own.recoverable_health = 20.0;
    run(engine, 10, use(0), act(FullUniversal::WalkForward), kIdle, act(FullUniversal::WalkForward));
    const double regained = engine.config().recoverable_regain_hit * 5.0 + 4.0;
    check(std::abs(engine.state().fighters[0].health - (150.0 + regained + 2.0)) < 1e-9 &&
              std::abs(engine.state().fighters[0].recoverable_health - (20.0 - regained)) < 1e-9,
          "hits regain recoverable health and restores add to it");

    // Recoverable-only damage cannot K.O.; some moves remove recoverable health.
    auto soft = jab();
    soft.recoverable_only = true;
    auto remover = jab();
    remover.removes_recoverable = true;
    engine = engine_with({soft, remover});
    engine.mutable_state().fighters[1].health = 3.0;
    run(engine, 10, use(0), act(FullUniversal::WalkForward), kIdle, act(FullUniversal::WalkForward));
    check(engine.state().fighters[1].health == 1.0 && std::abs(engine.state().fighters[1].recoverable_health - 2.0) < 1e-9,
          "recoverable-only damage leaves the opponent at 1");
    while (!engine.actionable(0)) engine.step(kIdle, kIdle);
    engine.mutable_state().fighters[1].health = 100.0;
    run(engine, 10, use(1), act(FullUniversal::WalkForward), kIdle, act(FullUniversal::WalkForward));
    check(engine.state().fighters[1].recoverable_health == 0.0, "a move can remove recoverable health");

    // Self-damage, partly recoverable, and skipped in Heat when published that way.
    auto costly = strike(FullHitLevel::Mid);
    costly.self_damage = 12.0;
    costly.self_recoverable = 8.0;
    engine = engine_with({costly});
    engine.step(use(0), kIdle);
    check(std::abs(engine.state().fighters[0].health - 168.0) < 1e-9 &&
              std::abs(engine.state().fighters[0].recoverable_health - 8.0) < 1e-9,
          "self-damage is applied when the move starts");
    costly.self_damage_without_heat_only = true;
    engine = engine_with({costly});
    engine.mutable_state().fighters[0].heat = true;
    engine.mutable_state().fighters[0].heat_frames = 600;
    engine.step(use(0), kIdle);
    check(engine.state().fighters[0].health == 180.0, "self-damage without Heat is skipped in Heat");

    // Rage: activates at low health, boosts damage, and is spent by a Rage Art.
    auto rage_art = strike(FullHitLevel::Mid, 55.0);
    rage_art.consumes_rage = true;
    engine = engine_with({jab(), rage_art}, {jab()});
    engine.mutable_state().fighters[1].health = 50.0;
    FullFrameEvents last{};
    for (int frame = 0; frame < 10; ++frame) {
        last = engine.step(frame == 0 ? use(0) : kIdle, act(FullUniversal::WalkForward));
        if (last.rage_activated[1]) break;
    }
    check(engine.state().fighters[1].rage && last.rage_activated[1], "Rage activates at low health");
    check(!engine.legal(0, use(1)), "a Rage Art needs Rage");
    while (!engine.actionable(0) || !engine.actionable(1)) engine.step(kIdle, kIdle);
    const double before = engine.state().fighters[0].health;
    run(engine, 10, act(FullUniversal::WalkForward), use(0), act(FullUniversal::WalkForward), kIdle);
    check(std::abs((before - engine.state().fighters[0].health) - 5.0 * engine.config().rage_damage_scale) < 1e-9,
          "Rage boosts damage");
    engine.mutable_state().fighters[0].rage = true;
    while (!engine.actionable(0)) engine.step(kIdle, kIdle);
    engine.step(use(1), kIdle);
    check(!engine.state().fighters[0].rage, "a Rage Art spends Rage");

    // K.O. ends the fight.
    engine = engine_with({jab()});
    engine.mutable_state().fighters[1].health = 3.0;
    run(engine, 10, use(0), act(FullUniversal::WalkForward), kIdle, act(FullUniversal::WalkForward));
    check(engine.state().winner == 0, "a lethal hit wins the round");
    const int frame = engine.state().frame;
    check(engine.step(use(0), kIdle).contacts.empty() && engine.state().frame == frame, "nothing changes after a K.O.");
}

void test_reversal_armor_spike_float_and_trades() {
    auto reversal = strike(FullHitLevel::Mid, 0.0);
    reversal.hits[0].reach = 0.01;
    reversal.reversal = {1, 20};
    reversal.reversal_damage = 20.0;
    auto parry = reversal;
    parry.reversal = {};
    parry.parry = {1, 20};
    auto armor = strike(FullHitLevel::Mid, 0.0);
    armor.hits[0].reach = 0.01;
    armor.armor = {1, 20};
    auto power = armor;
    power.armor = {};
    power.power_crush = {1, 20};
    const auto contact_of = [&](FullMoveSpec attack, FullMoveSpec defense) {
        auto engine = engine_with({std::move(attack)}, {std::move(defense)});
        const auto recorded = run(engine, 13, use(0), use(0));
        return std::make_pair(recorded.contacts.empty() ? FullContact::Evaded : recorded.contacts[0].contact, engine);
    };
    auto [reversed, reversed_engine] = contact_of(strike(FullHitLevel::Mid), reversal);
    check(reversed == FullContact::Reversed, "a reversal catches a mid");
    const auto& attacker = reversed_engine.state().fighters[0];
    check(attacker.health == 160.0 && attacker.posture == FullPosture::Grounded && !attacker.techable,
          "the reversed attacker takes the reversal damage and goes down");
    auto unreversable = strike(FullHitLevel::Mid);
    unreversable.hits[0].reversal_break = true;
    check(contact_of(unreversable, reversal).first == FullContact::CounterHit, "reversal break moves cannot be reversed");
    auto unparryable = strike(FullHitLevel::Mid);
    unparryable.hits[0].unparryable = true;
    check(contact_of(unparryable, parry).first == FullContact::CounterHit, "unparryable moves go through parries");
    check(contact_of(strike(FullHitLevel::Low), armor).first == FullContact::Armored, "armor absorbs lows");
    check(contact_of(strike(FullHitLevel::Low), power).first == FullContact::CounterHit, "power crush does not absorb lows");

    // A spike slams a juggled opponent down; a launch's float height is data.
    auto floaty = launcher();
    floaty.launch_air_frames = 80;
    auto spike = jab();
    spike.hits[0].level = FullHitLevel::Mid;
    spike.hits[0].spike = true;
    auto engine = engine_with({floaty, spike});
    run(engine, 12, use(0), act(FullUniversal::WalkForward), kIdle, act(FullUniversal::WalkForward));
    check(engine.state().fighters[1].posture_frames == 80, "a launch uses its own float time");
    while (!engine.actionable(0)) engine.step(kIdle, kIdle);
    run(engine, 10, use(1), kIdle);
    check(engine.state().fighters[1].posture == FullPosture::Grounded && !engine.state().fighters[1].techable,
          "a spike grounds a juggled opponent untechably");

    // Wall scaling: hits after a wall splat deal less.
    const auto post_splat_damage = [&](double wall_scale) {
        FullEngineConfig config{};
        config.wall_combo_scale = wall_scale;
        auto carry = jab();
        carry.hits[0].level = FullHitLevel::Mid;
        carry.hits[0].damage = 10.0;
        carry.pushback_hit = 1.0;
        auto follow = carry;  // reaches the splatted opponent after the carry's wall pushback
        follow.hits[0].reach = 3.0;
        follow.pushback_hit = 0.0;
        auto engine = engine_with({launcher(), carry, follow}, {jab()}, config);
        engine.reset(2.4, 3.2);
        run(engine, 12, use(0), act(FullUniversal::WalkForward), kIdle, act(FullUniversal::WalkForward));
        while (!engine.actionable(0)) engine.step(kIdle, kIdle);
        run(engine, 10, use(1), kIdle);
        while (!engine.actionable(0)) engine.step(kIdle, kIdle);
        const double before = engine.state().fighters[1].health;
        run(engine, 10, use(2), kIdle);
        return before - engine.state().fighters[1].health;
    };
    const double unscaled = post_splat_damage(1.0);
    check(unscaled > 0.0 && std::abs(post_splat_damage(0.5) - unscaled * 0.5) < 1e-9, "wall scaling applies after a splat");

    // A same-frame trade applies both hits.
    engine = engine_with({jab()}, {jab()});
    run(engine, 10, use(0), use(0));
    check(engine.state().fighters[0].health < 180.0 && engine.state().fighters[1].health < 180.0, "trades hit both fighters");
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

void test_sidesteps_follow_each_fighters_facing() {
    // A right-tracking-only mid catches a step to the attacker's right; a left-only one does not.
    auto right_only = strike(FullHitLevel::Mid);
    right_only.hits[0].tracking_left = 0.0;
    right_only.hits[0].tracking_right = 1.0;
    auto left_only = right_only;
    left_only.hits[0].tracking_left = 1.0;
    left_only.hits[0].tracking_right = 0.0;
    // P2 faces -x: its left is P1's right.
    for (const auto& [move, expect_hit, message] : {
             std::tuple{right_only, true, "P2 stepping to its left moves to P1's right"},
             std::tuple{left_only, false, "a left-tracking move misses a step to P1's right"}}) {
        auto engine = engine_with({move});
        const auto recorded = run(engine, 12, use(0), act(FullUniversal::SidestepLeft));
        check(recorded.contacts.empty() != expect_hit, message);
    }
    auto engine = engine_with({jab()});
    engine.step(act(FullUniversal::SidestepLeft), act(FullUniversal::SidestepLeft));
    check(engine.state().fighters[0].axis < 0.0 && engine.state().fighters[1].axis > 0.0,
          "facing each other, both fighters' left steps go opposite ways on the axis");
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
        test_sidesteps_follow_each_fighters_facing();
        test_collision_and_walls();
        test_travel_closes_distance();
        test_juggle_scaling_landing_and_wakeup();
        test_tornado_extends_once_and_tech_roll();
        test_wall_splat_and_breaks();
        test_crush_armor_parry_invincibility_and_counter_hits();
        test_determinism_and_validation();
        test_throw_break_window();
        test_heat();
        test_bug_pass_regressions();
        test_stances_and_resources();
        test_attack_throws_situations_and_special_states();
        test_heat_parry_outcome();
        test_airtime_follows_published_advantage();
        test_recoverable_health_rage_and_ko();
        test_reversal_armor_spike_float_and_trades();
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
