#pragma once

// Randomized full-combat play shared by the trace, refactor, and CPU/GPU
// parity tests: real catalog movesets (stand-in geometry), random legal
// inputs from a fixed seed, and a complete text dump of the engine state.

#include "t8_v2/full_combat_binding.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

namespace t8::v2::testing {

struct FullFixture {
    std::array<std::vector<FullMoveSpec>, 2> moves;
    std::array<FullCharacterRules, 2> rules;
};

inline std::vector<FullMoveSpec> bind_character(const std::filesystem::path& generated, const std::string& character,
                                                FullCharacterRules& rules) {
    FullBindingOptions options{};
    options.require_practice_validation = false;
    options.allow_variable_frame_data = true;
    options.provisional = FullProvisionalGeometry{};
    try {
        auto set = load_full_combat_character_rules(generated / "full_combat_stances.csv",
                                                    generated / "full_combat_resources.csv", character);
        rules = set.rules;
        options.stance_names = set.stance_names;
    } catch (const std::exception&) {
        rules = {};
    }
    // As the roster loader binds a stand-in-geometry character (src/full_move_roster.cpp).
    const auto rows = load_full_combat_bindings(generated / "full_combat_bindings.csv", character);
    add_provisional_stances(rows, rules, options.stance_names);
    std::vector<FullMoveSpec> moves;
    for (const auto& move : bind_full_combat_moves(rows, options)) {
        if (move.bound()) moves.push_back(*move.spec);
    }
    return moves;
}

inline FullFixture make_fixture(const std::filesystem::path& generated, const std::string& first,
                                const std::string& second) {
    FullFixture fixture;
    fixture.moves[0] = bind_character(generated, first, fixture.rules[0]);
    fixture.moves[1] = bind_character(generated, second, fixture.rules[1]);
    return fixture;
}

struct SplitMix {
    std::uint64_t state;
    std::uint64_t next() {
        std::uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30U)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27U)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31U);
    }
    std::size_t below(std::size_t bound) { return static_cast<std::size_t>(next() % bound); }
};

// A random legal input: usually a move when one is legal, otherwise a
// universal action (movement, guard, breaks, wake-ups, Heat Dash).
inline FullInput random_input(const FullCombatEngine& engine, int player, std::size_t move_count, SplitMix& rng) {
    std::vector<FullInput> legal;
    const auto& own = engine.state().fighters[static_cast<std::size_t>(player)];
    const auto& other = engine.state().fighters[static_cast<std::size_t>(1 - player)];
    const bool close = std::abs(own.x - other.x) < 1.6;
    // Approach when far, so fights produce contacts; guard sometimes when close.
    const std::size_t roll = rng.below(100);
    if (!close && roll < 55) {
        const FullInput approach{std::nullopt, roll < 30 ? FullUniversal::DashForward : FullUniversal::WalkForward};
        if (engine.legal(player, approach)) return approach;
    }
    if (close && roll < 15) {
        const FullInput guard{std::nullopt, roll < 8 ? FullUniversal::Idle : FullUniversal::Crouch};
        if (engine.legal(player, guard)) return guard;
    }
    if (rng.below(100) < (close ? 70U : 25U)) {
        for (std::size_t index = 0; index < move_count; ++index) {
            const FullInput input{index, FullUniversal::Idle};
            if (engine.legal(player, input)) legal.push_back(input);
        }
    }
    if (legal.empty()) {
        for (int value = 0; value <= static_cast<int>(FullUniversal::TechRoll); ++value) {
            const FullInput input{std::nullopt, static_cast<FullUniversal>(value)};
            if (engine.legal(player, input)) legal.push_back(input);
        }
    }
    if (legal.empty()) return {};
    return legal[rng.below(legal.size())];
}

inline std::string dump(const FullEngineState& state) {
    std::string out;
    char buffer[96];
    const auto add = [&](const char* format, auto value) {
        std::snprintf(buffer, sizeof buffer, format, value);
        out += buffer;
    };
    add("f%d ", state.frame);
    add("w%d ", state.winner);
    const auto& s = state.stage;
    add("s%.17g,", s.left_wall);
    add("%.17g,", s.right_wall);
    add("%d", static_cast<int>(s.left_wall_breakable) | (static_cast<int>(s.right_wall_breakable) << 1) |
                  (static_cast<int>(s.left_balcony) << 2) | (static_cast<int>(s.right_balcony) << 3) |
                  (static_cast<int>(s.floor_breakable) << 4));
    add(",%d", s.level);
    for (const auto& f : state.fighters) {
        add(" |h%.17g", f.health);
        add(" r%.17g", f.recoverable_health);
        add(" x%.17g", f.x);
        add(" a%.17g", f.axis);
        add(" p%d", static_cast<int>(f.posture));
        add(" st%d", f.stance);
        add(" H%d", static_cast<int>(f.heat) | (static_cast<int>(f.heat_available) << 1) |
                        (static_cast<int>(f.rage) << 2) | (static_cast<int>(f.rage_available) << 3));
        add(" hf%d", f.heat_frames);
        add(" hd%d", f.heat_dash_from ? static_cast<int>(*f.heat_dash_from) : -1);
        add(" tb%d", f.throw_break_frames);
        add(",%d", static_cast<int>(f.throw_break_attempted));
        add(",%d", f.pending_throw_hit);
        add(",%d", f.pending_throw_elapsed);
        add(" m%d", f.move ? static_cast<int>(*f.move) : -1);
        add(" u%d", static_cast<int>(f.universal));
        add(" af%d", f.action_frame);
        add(" hl%u", f.hits_landed);
        add(" sf%d", f.stance_frames);
        add(" res%.17g", f.resource);
        add(" fl%d", static_cast<int>(f.installed) | (static_cast<int>(f.move_hit) << 1) |
                         (static_cast<int>(f.move_blocked) << 2) | (static_cast<int>(f.resource_gained) << 3) |
                         (static_cast<int>(f.back_turned) << 4) | (static_cast<int>(f.move_in_heat) << 5) |
                         (static_cast<int>(f.ki_charge_boost) << 6) | (static_cast<int>(f.move_boosted) << 7) |
                         (static_cast<int>(f.in_blockstun) << 8) | (static_cast<int>(f.tornado_used) << 9) |
                         (static_cast<int>(f.wall_splat_used) << 10) | (static_cast<int>(f.wall_scaled) << 11) |
                         (static_cast<int>(f.techable) << 12) | (static_cast<int>(f.crumpled) << 13));
        add(" kc%d", f.ki_charge_frames);
        add(" stun%d", f.stun);
        add(" pf%d", f.posture_frames);
        add(" ge%d", f.grounded_elapsed);
        add(" ch%d", f.combo_hits);
    }
    return out;
}

inline std::string dump(const FullFrameEvents& events) {
    std::string out;
    char buffer[96];
    for (const auto& contact : events.contacts) {
        std::snprintf(buffer, sizeof buffer, " c%d:%zu:%d:%.17g", contact.attacker, contact.hit_index,
                      static_cast<int>(contact.contact), contact.damage);
        out += buffer;
    }
    for (const auto transition : events.transitions) {
        std::snprintf(buffer, sizeof buffer, " t%d", static_cast<int>(transition));
        out += buffer;
    }
    std::snprintf(buffer, sizeof buffer, " ha%d%d ra%d%d", static_cast<int>(events.heat_activated[0]),
                  static_cast<int>(events.heat_activated[1]), static_cast<int>(events.rage_activated[0]),
                  static_cast<int>(events.rage_activated[1]));
    return out + buffer;
}

// One recorded random fight: the inputs chosen on every frame (so another
// engine can replay them) and the state after each frame.
struct FullRecordedFight {
    double start = 0.0;                     // P1 x; P2 starts 0.9 to its right
    std::vector<std::uint8_t> new_round;    // next_round(start, start + 0.9) before this frame
    std::vector<std::array<FullInput, 2>> inputs;
    std::vector<std::string> states;
    std::vector<std::string> events;
};

inline FullRecordedFight record_random_fight(const FullFixture& fixture, std::uint64_t seed, int frames) {
    FullCombatEngine engine(fixture.moves, FullEngineConfig{}, fixture.rules);
    SplitMix rng{seed};
    // Odd seeds start near a wall, so splats, wall breaks, and pushback against walls occur.
    const double start = seed % 2 == 1 ? 2.4 : -0.6 + 0.1 * static_cast<double>(seed % 5);
    engine.reset(start, start + 0.9);
    FullRecordedFight fight;
    fight.start = start;
    for (int frame = 0; frame < frames; ++frame) {
        const bool round_over = engine.state().winner >= 0;
        if (round_over) engine.next_round(start, start + 0.9);
        fight.new_round.push_back(round_over ? 1U : 0U);
        const std::array<FullInput, 2> chosen = {random_input(engine, 0, fixture.moves[0].size(), rng),
                                                 random_input(engine, 1, fixture.moves[1].size(), rng)};
        const auto events = engine.step(chosen[0], chosen[1]);
        fight.inputs.push_back(chosen);
        fight.states.push_back(dump(engine.state()));
        fight.events.push_back(dump(events));
    }
    return fight;
}

}  // namespace t8::v2::testing
