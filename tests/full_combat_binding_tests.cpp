#include "t8_v2/full_combat_binding.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
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
    auto& p1 = engine.mutable_state().fighters[0];
    p1.stance = spec.required_stance;
    p1.heat = spec.requires_heat;
    if (spec.required_posture == FullPosture::Crouching) {
        engine.step({std::nullopt, FullUniversal::Crouch}, hold);
    }
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
    check(rows.size() == 149, "Jun has 149 binding rows");
    const auto bound_count = std::count_if(bound.begin(), bound.end(), [](const auto& move) { return move.bound(); });
    check(bound_count == 0, "no Jun move is bound before Practice validation and measurement");
    std::size_t reactive = 0;
    std::size_t unvalidated = 0;
    std::size_t unmeasured = 0;
    for (const auto& move : bound) {
        reactive += has_blocker(move, "reactive");
        unvalidated += has_blocker(move, "Practice validation");
        unmeasured += has_blocker(move, "missing measurement");
        check(!move.provisional, "strict binding never marks a move provisional");
    }
    check(reactive == 4, "the four parry outcomes are reactive");
    check(unvalidated == 149, "every move awaits Practice validation");
    check(unmeasured == 149, "every move awaits measurement");
    std::cout << "strict: 0/" << rows.size() << " bound (reactive " << reactive << ")\n";
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
            check(move.provisional, "moves bound without measurements are marked provisional");
            moveset.push_back(*move.spec);
        }
        for (const auto& blocker : move.blockers) {
            remaining[blocker.substr(0, blocker.find(':'))] += 1;
        }
    }
    std::cout << "provisional: " << moveset.size() << "/" << rows.size() << " bound; remaining blockers:\n";
    for (const auto& [blocker, count] : remaining) std::cout << "  " << count << " x " << blocker << '\n';
    check(moveset.size() >= 100, "most Jun moves bind once geometry is supplied");

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
