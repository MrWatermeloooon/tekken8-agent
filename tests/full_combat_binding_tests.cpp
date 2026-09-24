#include "t8_v2/full_combat_binding.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

using namespace t8::v2;

int failures = 0;

void check(bool condition, std::string_view message) {
    if (!condition) {
        ++failures;
        if (failures < 40) std::cerr << "FAIL: " << message << '\n';
    }
}

bool has_blocker(const FullBoundMove& move, std::string_view prefix) {
    return std::any_of(move.blockers.begin(), move.blockers.end(),
                       [&](const std::string& blocker) { return blocker.rfind(prefix, 0) == 0; });
}

const std::vector<std::string> kJunStances = {"GEN", "IZU", "MIA"};

bool has_hits(const FullMoveSpec& spec) { return !spec.hits.empty() && !spec.reactive; }

FullCombatEngine engine_for(const FullMoveSpec& move) {
    FullCombatEngine engine(std::array<std::vector<FullMoveSpec>, 2>{
        std::vector<FullMoveSpec>{move}, std::vector<FullMoveSpec>{move}});
    engine.reset(-0.4, 0.4);
    return engine;
}

// Frames from now until each fighter can act again (P1, P2), stepping idle.
std::pair<int, int> recovery_order(FullCombatEngine& engine, FullInput hold) {
    int p1 = -1;
    int p2 = -1;
    for (int frame = 0; frame < 400 && (p1 < 0 || p2 < 0); ++frame) {
        if (p1 < 0 && engine.actionable(0)) p1 = frame;
        if (p2 < 0 && engine.actionable(1)) p2 = frame;
        engine.step({}, hold);
    }
    return {p1, p2};
}

// Performs the move against a dummy holding `hold` until its last hit resolves.
// Returns the contacts, or an empty list if the move could not be started.
std::vector<FullContactEvent> perform(FullCombatEngine& engine, const FullMoveSpec& spec, FullInput hold) {
    if (spec.requires_back_to_wall) {
        const double wall = engine.state().stage.left_wall;
        engine.reset(wall, wall + 0.8);
    }
    auto& p1 = engine.mutable_state().fighters[0];
    p1.stance = spec.required_stance;
    p1.rage = spec.requires_rage || spec.consumes_rage;
    p1.heat = spec.requires_heat || spec.heat_cost_frames > 0;
    p1.heat_frames = p1.heat ? engine.config().heat_duration_frames : 0;
    if (spec.required_posture == FullPosture::Crouching) {
        engine.step({std::nullopt, FullUniversal::Crouch}, hold);
    }
    if (spec.requires_sidestep) engine.step({std::nullopt, FullUniversal::SidestepLeft}, hold);
    std::vector<FullContactEvent> contacts;
    const FullInput attack{std::size_t{0}, FullUniversal::Idle};
    if (!engine.legal(0, attack)) return contacts;
    auto events = engine.step(attack, hold);
    contacts.insert(contacts.end(), events.contacts.begin(), events.contacts.end());
    const int last_active = spec.hits.back().first_active_frame + spec.hits.back().active_frames;
    for (int frame = 1; frame < last_active && contacts.size() < spec.hits.size(); ++frame) {
        events = engine.step({}, hold);
        contacts.insert(contacts.end(), events.contacts.begin(), events.contacts.end());
    }
    return contacts;
}

void test_strict_binding_blocks_everything_honestly(const std::vector<FullBindingRow>& rows) {
    FullBindingOptions strict{};
    strict.stance_names = kJunStances;
    const auto bound = bind_full_combat_moves(rows, strict);
    const auto base = std::count_if(rows.begin(), rows.end(), [](const auto& row) { return row.variant_of.empty(); });
    const auto reactive = std::count_if(rows.begin(), rows.end(), [](const auto& row) { return row.reactive; });
    check(base == 149, "Jun has 149 catalog moves");
    check(rows.size() - static_cast<std::size_t>(base) == 15, "Jun has 15 stance-branch variants");
    check(reactive == 4, "Jun has four parry outcomes");
    const auto bound_count = std::count_if(bound.begin(), bound.end(), [](const auto& move) { return move.bound(); });
    check(bound_count == 0, "no Jun move is bound before Practice validation and measurement");
    std::size_t unvalidated = 0;
    for (std::size_t index = 0; index < bound.size(); ++index) {
        const auto& move = bound[index];
        unvalidated += has_blocker(move, "Practice validation");
        // Every move with hitboxes needs its geometry measured; stance entries and outcomes have none.
        if (!rows[index].reactive && !rows[index].hit_levels.empty()) {
            check(has_blocker(move, "missing measurement"), move.stable_id + ": awaits measurement");
        }
        check(!move.provisional, "strict binding never marks a move provisional");
    }
    check(unvalidated == rows.size(), "every move awaits Practice validation");
    std::cout << "strict: 0/" << rows.size() << " bound (" << base << " moves, " << rows.size() - base
              << " variants, " << reactive << " parry outcomes)\n";
}

void test_provisional_binding_reproduces_catalog_frame_data(const std::vector<FullBindingRow>& rows) {
    FullBindingOptions provisional{};
    provisional.stance_names = kJunStances;
    provisional.require_practice_validation = false;
    provisional.allow_variable_frame_data = true;
    // Frame data is under test here, not spacing: with unmeasured per-hit
    // pushback, a fixed stand-in would push long strings out of the stand-in
    // reach, so pushback is zeroed.
    FullProvisionalGeometry geometry{};
    geometry.pushback = 0.0;
    provisional.provisional = geometry;
    const auto bound = bind_full_combat_moves(rows, provisional);

    std::map<std::string, int> remaining;
    std::vector<FullMoveSpec> moveset;
    for (const auto& move : bound) {
        if (move.bound()) {
            check(move.provisional || !has_hits(*move.spec), "moves with hits bound without measurements are provisional");
            moveset.push_back(*move.spec);
        }
        for (const auto& blocker : move.blockers) {
            remaining[blocker.substr(0, blocker.find(':'))] += 1;
        }
    }
    std::cout << "provisional: " << moveset.size() << "/" << rows.size() << " bound; remaining blockers:\n";
    for (const auto& [blocker, count] : remaining) std::cout << "  " << count << " x " << blocker << '\n';
    check(moveset.size() >= 140, "most Jun moves bind once geometry is supplied");

    bool constructed = true;
    try {
        FullCombatEngine all(std::array<std::vector<FullMoveSpec>, 2>{
            moveset, std::vector<FullMoveSpec>{moveset.front()}});
    } catch (const std::exception& error) {
        constructed = false;
        std::cerr << "engine rejected bound moves: " << error.what() << '\n';
    }
    check(constructed, "the engine accepts every bound Jun move");

    std::size_t block_checked = 0;
    std::size_t hit_checked = 0;
    std::size_t block_skipped = 0;
    for (const auto& move : moveset) {
        if (!has_hits(move)) continue;
        const auto& hits = move.hits;
        const bool any_throw = std::any_of(hits.begin(), hits.end(), [](const auto& hit) { return hit.level == FullHitLevel::Throw; });
        const bool all_low = std::all_of(hits.begin(), hits.end(), [](const auto& hit) { return hit.level == FullHitLevel::Low; });
        const bool no_low = std::none_of(hits.begin(), hits.end(), [](const auto& hit) { return hit.level == FullHitLevel::Low; });
        const bool no_crouch_hittable = std::none_of(hits.begin(), hits.end(), [](const auto& hit) {
            return hit.level == FullHitLevel::Mid || hit.level == FullHitLevel::High;
        });
        const int stun = move.total_frames() - hits.back().first_active_frame + move.block_advantage;
        // Block: one guard must block every hit, and the advantage must not clamp at zero stun.
        if (!any_throw && (no_low || (all_low && no_crouch_hittable)) && stun > 0) {
            auto engine = engine_for(move);
            const FullInput guard{std::nullopt, no_low ? FullUniversal::Idle : FullUniversal::Crouch};
            const auto contacts = perform(engine, move, guard);
            const bool all_blocked = contacts.size() == hits.size() &&
                std::all_of(contacts.begin(), contacts.end(), [](const auto& c) { return c.contact == FullContact::Blocked; });
            check(all_blocked, move.stable_id + ": every hit is blocked");
            if (all_blocked) {
                const auto [attacker, defender] = recovery_order(engine, guard);
                check(defender - attacker == move.block_advantage,
                      move.stable_id + ": engine block advantage " + std::to_string(defender - attacker) +
                      " equals catalog " + std::to_string(move.block_advantage));
                ++block_checked;
            }
        } else {
            ++block_skipped;
        }
        // Hit: single-hit plain-stun moves against a dummy walking in (no guard).
        if (hits.size() == 1 && move.hit_effect == FullHitEffect::Stun && !any_throw &&
            move.total_frames() - hits.back().first_active_frame + move.hit_advantage > 0) {
            auto engine = engine_for(move);
            const FullInput walk{std::nullopt, FullUniversal::WalkForward};
            const auto contacts = perform(engine, move, walk);
            if (contacts.size() == 1 && contacts[0].contact == FullContact::Hit) {
                const auto [attacker, defender] = recovery_order(engine, walk);
                check(defender - attacker == move.hit_advantage,
                      move.stable_id + ": engine hit advantage equals catalog");
                ++hit_checked;
            } else {
                check(false, move.stable_id + ": single hit connects on a walking dummy");
            }
        }
    }
    std::cout << "frame data reproduced: " << block_checked << " moves on block, " << hit_checked
              << " on hit (" << block_skipped << " not block-testable: throws, mixed levels, or clamped)\n";
    check(block_checked >= 50 && hit_checked >= 20, "a substantial share of Jun's frame data is verified");
}

struct JunFight {
    std::vector<FullMoveSpec> moves;
    FullCharacterRuleSet rules;
    std::size_t index(const std::string& id) const {
        for (std::size_t i = 0; i < moves.size(); ++i) {
            if (moves[i].stable_id == id) return i;
        }
        throw std::runtime_error("move not bound: " + id);
    }
    FullCombatEngine engine() const {
        FullCombatEngine result(std::array<std::vector<FullMoveSpec>, 2>{moves, moves}, FullEngineConfig{},
                                std::array<FullCharacterRules, 2>{rules.rules, rules.rules});
        result.reset(-0.4, 0.4);
        return result;
    }
};

void idle_until_actionable(FullCombatEngine& engine, int player, FullInput other = {}) {
    for (int frame = 0; frame < 400 && !engine.actionable(player); ++frame) {
        engine.step(player == 0 ? FullInput{} : other, player == 0 ? other : FullInput{});
    }
}

void test_jun_state_machines(const std::vector<FullBindingRow>& rows, const std::filesystem::path& directory) {
    JunFight jun;
    jun.rules = load_full_combat_character_rules(directory / "full_combat_stances.csv",
                                                 directory / "full_combat_resources.csv", "jun");
    check(jun.rules.stance_names == kJunStances, "Jun's stances are GEN, IZU, MIA");
    check(jun.rules.rules.resource_max == 100.0 && jun.rules.rules.install_threshold == 100.0 &&
              jun.rules.rules.resource_persists,
          "Kazama Essence: 100 max, Divine Aura at 100, kept across rounds");
    const auto& gen = jun.rules.rules.stances[0];
    check(!gen.can_guard && gen.auto_parry == (FullLevelLow | FullLevelThrow) &&
              gen.parry_outcomes[2] == "jun:142" && gen.parry_outcomes[3] == "jun:143",
          "GEN cannot guard and parries lows and throws into GEN.P");
    check(jun.rules.rules.stances[2].pulse_interval_frames == 120, "MIA pulses every two seconds");

    FullBindingOptions options{};
    options.stance_names = jun.rules.stance_names;
    options.require_practice_validation = false;
    options.allow_variable_frame_data = true;
    FullProvisionalGeometry geometry{};
    geometry.pushback = 0.0;
    geometry.range = 1.5;
    options.provisional = geometry;
    for (const auto& move : bind_full_combat_moves(rows, options)) {
        if (move.bound()) jun.moves.push_back(*move.spec);
    }
    const FullInput walk{std::nullopt, FullUniversal::WalkForward};
    const auto use = [&](const std::string& id) { return FullInput{jun.index(id), FullUniversal::Idle}; };

    // 1,1 ends in Izumo; IZU moves become legal.
    auto engine = jun.engine();
    engine.step(use("jun:7"), walk);
    idle_until_actionable(engine, 0, walk);
    check(engine.state().fighters[0].stance == 1, "1,1 ends in IZU");
    check(engine.legal(0, use("jun:123")) && !engine.legal(0, use("jun:118")), "IZU moves are legal in IZU, GEN moves are not");
    check(engine.state().fighters[0].resource > 0.0, "a landed move gains Kazama Essence");

    // Miare: stance entry with no hits; healing pulses of recoverable health and Essence.
    engine = jun.engine();
    auto& p1 = engine.mutable_state().fighters[0];
    p1.health = 150.0;
    p1.recoverable_health = 10.0;
    engine.step(use("jun:77"), {});
    idle_until_actionable(engine, 0);
    check(p1.stance == 2, "b+1+2 enters MIA");
    const double before = p1.recoverable_health;
    for (int frame = 0; frame < 120; ++frame) engine.step({}, {});
    check(std::abs(p1.recoverable_health - (before - 3.0)) < 1e-9 && std::abs(p1.resource - 1.0) < 1e-9,
          "a MIA pulse restores 3 recoverable health and 1 Essence");
    check(engine.legal(0, use("jun:132")), "MIA moves are legal in MIA");

    // Genjitsu: no guard, automatic parry of lows into GEN.P (Low).
    engine = jun.engine();
    engine.mutable_state().fighters[0].health = 150.0;
    engine.step(use("jun:39"), {});
    idle_until_actionable(engine, 0);
    check(engine.state().fighters[0].stance == 0, "f+3+4 enters GEN");
    // P2 attacks with a plain standing single-hit low.
    std::size_t low = jun.moves.size();
    for (std::size_t i = 0; i < jun.moves.size() && low == jun.moves.size(); ++i) {
        const auto& spec = jun.moves[i];
        if (has_hits(spec) && spec.required_stance < 0 && !spec.requires_heat && !spec.requires_sidestep &&
            spec.required_posture == FullPosture::Standing && spec.hits.size() == 1 &&
            level_class(spec.hits[0].level) == FullLevelLow) {
            low = i;
        }
    }
    check(low < jun.moves.size(), "Jun has a plain standing low");
    std::vector<FullContactEvent> contacts;
    for (int frame = 0; frame < 60 && contacts.empty() && low < jun.moves.size(); ++frame) {
        contacts = engine.step({}, frame == 0 ? FullInput{low, FullUniversal::Idle} : FullInput{}).contacts;
    }
    check(contacts.size() == 1 && contacts[0].contact == FullContact::Parried, "GEN parries a low");
    const auto& p2 = engine.state().fighters[1];
    check(p2.posture == FullPosture::Grounded && p2.health < 180.0, "GEN.P (Low) knocks the attacker down");
    check(engine.state().fighters[0].resource >= 20.0 && engine.state().fighters[0].health > 150.0,
          "GEN.P restores health and gains 20 Essence");

    // Divine Aura at 100 Essence adds d+1+2's published damage bonus; Essence carries into the next round.
    const auto damage_of = [&](bool installed) {
        auto fight = jun.engine();
        fight.mutable_state().fighters[0].resource = installed ? 100.0 : 0.0;
        fight.mutable_state().fighters[0].installed = installed;
        for (int frame = 0; frame < 60; ++frame) fight.step(frame == 0 ? use("jun:53") : FullInput{}, walk);
        return 180.0 - fight.state().fighters[1].health;
    };
    check(std::abs(damage_of(true) - damage_of(false) - 4.0) < 1e-9, "Divine Aura adds 4 damage to d+1+2");
    engine = jun.engine();
    engine.mutable_state().fighters[0].resource = 60.0;
    engine.next_round();
    check(engine.state().fighters[0].resource == 60.0, "Kazama Essence carries into the next round");

    // f+1+2 enters MIA on hit only; SS.2 needs a sidestep.
    engine = jun.engine();
    engine.step(use("jun:37"), walk);
    idle_until_actionable(engine, 0, walk);
    check(engine.state().fighters[0].stance == 2, "f+1+2 enters MIA on hit");
    engine = jun.engine();
    engine.step(use("jun:37"), {});
    idle_until_actionable(engine, 0);
    check(engine.state().fighters[0].stance < 0, "f+1+2 does not enter MIA on block");
    engine = jun.engine();
    check(!engine.legal(0, use("jun:111")), "SS.2 needs a sidestep");
    engine.step({std::nullopt, FullUniversal::SidestepRight}, {});
    check(engine.legal(0, use("jun:111")), "SS.2 is legal out of a sidestep");

    // Stance branches: b+2,1 held forward ends in GEN with its own frames.
    check(jun.moves[jun.index("jun:67~GEN")].result_stance == 0 && jun.moves[jun.index("jun:67~GEN")].recovery == 18,
          "b+2,1~F ends in GEN after r18");
    // Skipped data now bound: damage parts, attack throws, situations, Ki Charge, the Heat parry.
    const auto& falling_rain = jun.moves[jun.index("jun:141")];
    check(falling_rain.hits.size() == 1 && falling_rain.hits[0].damage == 45.0, "Falling Rain deals its three parts as one throw");
    check(jun.moves[jun.index("jun:1")].hits[0].damage == 12.0, "Heat Burst's [12;12] is one value, not two parts");
    const auto& spirit = jun.moves[jun.index("jun:119")];
    check(spirit.hits.size() == 1 && spirit.attack_throw == FullMoveSpec::AttackThrow::OnHit &&
              spirit.attack_throw_damage == 32.0 && spirit.attack_throw_front_only && spirit.back_turned_hit_advantage == 10,
          "GEN.2 is a mid that becomes a 32-damage throw on a front hit");
    const auto& violet = jun.moves[jun.index("jun:64")];
    check(violet.attack_throw == FullMoveSpec::AttackThrow::OnCounterHit && violet.attack_throw_damage == 22.0,
          "b+1 becomes an attack throw on counter-hit");
    const auto& rage_art = jun.moves[jun.index("jun:5")];
    check(rage_art.hits.size() == 1 && rage_art.hits[0].damage == 55.0 && rage_art.rage_art_max_damage == 82.0 &&
              rage_art.consumes_rage,
          "the Rage Art binds with its damage range");
    check(jun.moves[jun.index("jun:117")].requires_back_to_wall, "the Wall Jump needs the wall behind");
    check(jun.moves[jun.index("jun:88")].ki_charge && jun.moves[jun.index("jun:88")].hits.empty(), "Ki Charge binds");
    const auto& inner_peace = jun.moves[jun.index("jun:37")];
    check(inner_peace.heat_parry_outcome == "jun:3" && inner_peace.heat_parry.contains(5) &&
              inner_peace.recoverable_damage == 5.0 && inner_peace.cannot_ko,
          "f+1+2 parries in Heat into H.f+1+2,P and deals 5 recoverable damage");
    engine = jun.engine();
    engine.mutable_state().fighters[0].heat = true;
    engine.mutable_state().fighters[0].heat_frames = 600;
    engine.mutable_state().fighters[0].heat_available = false;
    std::size_t mid = jun.moves.size();
    for (std::size_t i = 0; i < jun.moves.size() && mid == jun.moves.size(); ++i) {
        const auto& spec = jun.moves[i];
        if (has_hits(spec) && spec.required_stance < 0 && !spec.requires_heat && !spec.requires_sidestep &&
            spec.required_posture == FullPosture::Standing && spec.hits.size() == 1 && spec.attack_throw ==
                FullMoveSpec::AttackThrow::None && level_class(spec.hits[0].level) == FullLevelMid &&
            spec.hits[0].first_active_frame >= 12 && !spec.hits[0].unparryable) {
            mid = i;
        }
    }
    check(mid < jun.moves.size(), "Jun has a plain mid of at least 12 frames");
    // P2's mid starts first and arrives on f+1+2's 8th frame, inside its Heat parry (5~12).
    bool parried = false;
    const int arrival = mid < jun.moves.size() ? jun.moves[mid].hits[0].first_active_frame : 0;
    for (int frame = 0; frame < 40 && !parried && mid < jun.moves.size(); ++frame) {
        const auto events = engine.step(frame == arrival - 8 ? use("jun:37") : FullInput{},
                                        frame == 0 ? FullInput{mid, FullUniversal::Idle} : FullInput{});
        for (const auto& contact : events.contacts) parried |= contact.contact == FullContact::Parried;
    }
    check(parried, "f+1+2 in Heat parries a mid");
    idle_until_actionable(engine, 0);
    check(engine.state().fighters[0].stance == 2 && engine.state().fighters[0].heat_frames <= 600 - 240,
          "H.f+1+2,P spends 240 frames of Heat and ends in MIA");
    std::cout << "Jun state machines: " << jun.moves.size() << " moves bound with rules\n";
}

void test_windows_parse() {
    const auto window = parse_frame_window("5~13|34~36");
    check(window.spans.size() == 2 && window.contains(13) && !window.contains(20) && window.contains(35),
          "multi-span windows parse");
    check(parse_frame_window("").empty(), "an empty window parses");
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: full_combat_binding_tests <full_combat_bindings.csv>\n";
        return EXIT_FAILURE;
    }
    try {
        const auto rows = load_full_combat_bindings(argv[1], "jun");
        test_windows_parse();
        test_strict_binding_blocks_everything_honestly(rows);
        test_provisional_binding_reproduces_catalog_frame_data(rows);
        test_jun_state_machines(rows, std::filesystem::path(argv[1]).parent_path());
        // Every character's rows load and bind (with blockers) without errors.
        const auto roster = load_full_combat_bindings(argv[1], "");
        FullBindingOptions any{};
        any.provisional = FullProvisionalGeometry{};
        any.require_practice_validation = false;
        any.allow_variable_frame_data = true;
        const auto bound = bind_full_combat_moves(roster, any);
        const auto count = std::count_if(bound.begin(), bound.end(), [](const auto& move) { return move.bound(); });
        check(roster.size() > 6000 && count > 0, "the whole roster binds without errors");
        std::vector<FullMoveSpec> every;
        for (const auto& move : bound) {
            if (move.bound()) every.push_back(*move.spec);
        }
        try {
            FullCombatEngine all(std::array<std::vector<FullMoveSpec>, 2>{every, every});
        } catch (const std::exception& error) {
            check(false, std::string("the engine accepts every bound roster move: ") + error.what());
        }
        std::cout << "roster: " << count << "/" << roster.size() << " rows bind with stand-in geometry\n";
    } catch (const std::exception& error) {
        std::cerr << "binding test error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
    if (failures != 0) {
        std::cerr << failures << " binding assertion(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "full combat binding tests passed\n";
    return EXIT_SUCCESS;
}
