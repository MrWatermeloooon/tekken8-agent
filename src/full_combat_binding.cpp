#include "t8_v2/full_combat_binding.hpp"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <fstream>
#include <set>
#include <stdexcept>
#include <unordered_map>

namespace t8::v2 {
namespace {

std::vector<std::string> split_csv_line(const std::string& line) {
    std::vector<std::string> fields;
    std::string field;
    bool quoted = false;
    for (std::size_t index = 0; index < line.size(); ++index) {
        const char character = line[index];
        if (quoted) {
            if (character == '"' && index + 1 < line.size() && line[index + 1] == '"') {
                field.push_back('"');
                ++index;
            } else if (character == '"') {
                quoted = false;
            } else {
                field.push_back(character);
            }
        } else if (character == '"') {
            quoted = true;
        } else if (character == ',') {
            fields.push_back(std::move(field));
            field.clear();
        } else {
            field.push_back(character);
        }
    }
    fields.push_back(std::move(field));
    return fields;
}

std::vector<std::string> split(const std::string& text, char separator) {
    std::vector<std::string> parts;
    if (text.empty()) return parts;
    std::size_t start = 0;
    while (true) {
        const auto end = text.find(separator, start);
        parts.push_back(text.substr(start, end - start));
        if (end == std::string::npos) break;
        start = end + 1;
    }
    return parts;
}

std::optional<int> to_int(const std::string& text) {
    if (text.empty()) return std::nullopt;
    int value = 0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size()) {
        throw std::runtime_error("invalid integer in full combat bindings: " + text);
    }
    return value;
}

std::optional<double> to_double(const std::string& text) {
    if (text.empty()) return std::nullopt;
    std::size_t used = 0;
    const double value = std::stod(text, &used);
    if (used != text.size()) throw std::runtime_error("invalid number in full combat bindings: " + text);
    return value;
}

std::optional<FullHitLevel> hit_level(const std::string& text) {
    if (text == "high") return FullHitLevel::High;
    if (text == "mid") return FullHitLevel::Mid;
    if (text == "low") return FullHitLevel::Low;
    if (text == "special_mid") return FullHitLevel::SpecialMid;
    if (text == "special_low") return FullHitLevel::SpecialLow;
    if (text == "throw") return FullHitLevel::Throw;
    return std::nullopt;
}

FullHitEffect hit_effect(const std::string& text) {
    if (text == "launch") return FullHitEffect::Launch;
    if (text == "knockdown") return FullHitEffect::Knockdown;
    return FullHitEffect::Stun;
}

}  // namespace

FrameWindow parse_frame_window(const std::string& text) {
    FrameWindow window{};
    for (const auto& span : split(text, '|')) {
        const auto parts = split(span, '~');
        if (parts.size() != 2) throw std::runtime_error("invalid frame window: " + text);
        window.spans.push_back({*to_int(parts[0]), *to_int(parts[1])});
    }
    return window;
}

std::vector<FullBindingRow> load_full_combat_bindings(const std::filesystem::path& path, const std::string& character) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("could not open full combat bindings: " + path.string());
    std::string line;
    if (!std::getline(input, line)) throw std::runtime_error("empty full combat bindings: " + path.string());
    const auto header = split_csv_line(line);
    std::unordered_map<std::string, std::size_t> column;
    for (std::size_t index = 0; index < header.size(); ++index) column[header[index]] = index;
    const auto at = [&](const std::vector<std::string>& fields, const char* name) -> const std::string& {
        const auto found = column.find(name);
        if (found == column.end()) throw std::runtime_error(std::string("full combat bindings missing column: ") + name);
        return fields.at(found->second);
    };
    const auto flag = [&](const std::vector<std::string>& fields, const char* name) { return at(fields, name) == "1"; };
    std::vector<FullBindingRow> rows;
    while (std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        const auto fields = split_csv_line(line);
        if (fields.size() != header.size()) throw std::runtime_error("full combat bindings row has the wrong field count");
        if (!character.empty() && at(fields, "character") != character) continue;
        FullBindingRow row{};
        row.stable_id = at(fields, "stable_id");
        row.character = at(fields, "character");
        row.command = at(fields, "command");
        row.parser_status = at(fields, "parser_status");
        row.source_consistency = at(fields, "source_consistency");
        row.reactive = flag(fields, "reactive");
        row.hit_levels = split(at(fields, "hit_levels"), '|');
        for (const auto& value : split(at(fields, "damages"), '|')) row.damages.push_back(*to_double(value));
        row.startup = to_int(at(fields, "startup"));
        row.recovery = to_int(at(fields, "recovery"));
        row.block_advantage = to_int(at(fields, "block_advantage"));
        row.hit_advantage = to_int(at(fields, "hit_advantage"));
        row.hit_effect = at(fields, "hit_effect");
        row.counter_hit_advantage = to_int(at(fields, "counter_hit_advantage"));
        row.counter_hit_effect = at(fields, "counter_hit_effect");
        row.frame_data_variable = flag(fields, "frame_data_variable");
        row.homing = flag(fields, "homing");
        row.tornado = flag(fields, "tornado");
        row.wall_break = flag(fields, "wall_break");
        row.floor_break = flag(fields, "floor_break");
        row.balcony_break = flag(fields, "balcony_break");
        row.heat_engager = flag(fields, "heat_engager");
        row.requires_heat = flag(fields, "requires_heat");
        row.requires_rage = flag(fields, "requires_rage");
        row.rage_art = flag(fields, "rage_art");
        row.posture = at(fields, "posture");
        row.stances = split(at(fields, "stances"), '|');
        row.situations = split(at(fields, "situations"), '|');
        row.automatic_transition = at(fields, "automatic_transition");
        row.low_crush = at(fields, "low_crush");
        row.high_crush = at(fields, "high_crush");
        row.power_crush = at(fields, "power_crush");
        row.parry = at(fields, "parry");
        row.airborne = at(fields, "airborne");
        row.invincible = at(fields, "invincible");
        row.heat_burst = flag(fields, "heat_burst");
        row.heat_smash = flag(fields, "heat_smash");
        row.heat_dash_block = to_int(at(fields, "heat_dash_block"));
        row.heat_dash_hit = to_int(at(fields, "heat_dash_hit"));
        row.heat_dash_effect = at(fields, "heat_dash_effect");
        row.chip_block = at(fields, "chip_block");
        row.chip_block_heat = at(fields, "chip_block_heat");
        row.recoverable_only = flag(fields, "recoverable_only");
        row.removes_recoverable = flag(fields, "removes_recoverable");
        row.armor_damage_recoverable = flag(fields, "armor_damage_recoverable");
        row.self_damage = at(fields, "self_damage");
        row.self_recoverable = at(fields, "self_recoverable");
        row.self_damage_without_heat = flag(fields, "self_damage_without_heat");
        row.restore_health_hit = at(fields, "restore_health_hit");
        row.restore_recoverable_hit = at(fields, "restore_recoverable_hit");
        row.restore_recoverable_block = at(fields, "restore_recoverable_block");
        row.throw_break = at(fields, "throw_break");
        row.side_switch_on_hit = flag(fields, "side_switch_on_hit");
        row.side_switch_on_break = flag(fields, "side_switch_on_break");
        row.spike = flag(fields, "spike");
        row.unparryable = flag(fields, "unparryable");
        row.reversal_break = flag(fields, "reversal_break");
        row.variant_of = at(fields, "variant_of");
        row.result_crouching = flag(fields, "result_crouching");
        row.result_stance = at(fields, "result_stance");
        row.result_stance_on_hit = at(fields, "result_stance_on_hit");
        row.result_stance_on_block = at(fields, "result_stance_on_block");
        row.requires_sidestep = flag(fields, "requires_sidestep");
        row.requires_running = flag(fields, "requires_running");
        row.parry_levels = at(fields, "parry_levels");
        row.parry_outcomes = split(at(fields, "parry_outcomes"), '|');
        row.heat_cost_frames = to_int(at(fields, "heat_cost_frames")).value_or(0);
        row.resource_gain_start = to_double(at(fields, "resource_gain_start")).value_or(0.0);
        row.resource_gain_hit = to_double(at(fields, "resource_gain_hit")).value_or(0.0);
        row.resource_gain_airborne_hit = to_double(at(fields, "resource_gain_airborne_hit")).value_or(0.0);
        row.resource_gain_block = to_double(at(fields, "resource_gain_block")).value_or(0.0);
        row.resource_gain_heat_activation = to_double(at(fields, "resource_gain_heat_activation")).value_or(0.0);
        row.install_damage_bonus = to_double(at(fields, "install_damage_bonus")).value_or(0.0);
        row.install_chip = to_double(at(fields, "install_chip"));
        row.install_range = to_double(at(fields, "install_range"));
        row.attack_throw = at(fields, "attack_throw");
        row.attack_throw_front_only = flag(fields, "attack_throw_front_only");
        row.attack_throw_standing_only = flag(fields, "attack_throw_standing_only");
        row.attack_throw_airborne = flag(fields, "attack_throw_airborne");
        row.attack_throw_damage = to_double(at(fields, "attack_throw_damage"));
        row.back_turned_hit_advantage = to_int(at(fields, "back_turned_hit_advantage"));
        row.back_turned_hit_effect = at(fields, "back_turned_hit_effect");
        row.result_back_turned = flag(fields, "result_back_turned");
        row.heat_parry = at(fields, "heat_parry");
        row.heat_parry_levels = at(fields, "heat_parry_levels");
        row.heat_parry_outcome = at(fields, "heat_parry_outcome");
        row.recoverable_damage = to_double(at(fields, "recoverable_damage")).value_or(0.0);
        row.cannot_ko = flag(fields, "cannot_ko");
        row.rage_art_max_damage = to_double(at(fields, "rage_art_max_damage"));
        row.ki_charge = flag(fields, "ki_charge");
        row.practice_status = at(fields, "practice_status");
        for (const auto& value : split(at(fields, "measured_active_frames"), '|')) {
            row.measured_active_frames.push_back(*to_int(value));
        }
        row.measured_range = to_double(at(fields, "measured_range"));
        row.measured_tracking_left = to_double(at(fields, "measured_tracking_left"));
        row.measured_tracking_right = to_double(at(fields, "measured_tracking_right"));
        row.measured_pushback = to_double(at(fields, "measured_pushback"));
        row.measured_travel = to_double(at(fields, "measured_travel"));
        for (const auto& value : split(at(fields, "measured_hit_frames"), '|')) {
            row.measured_hit_frames.push_back(*to_int(value));
        }
        rows.push_back(std::move(row));
    }
    return rows;
}

// Stand-in values for the gaps in a move's published data (stand-in geometry runs only). Every
// filled move is provisional: the Practice validator (scripts/validate_moves.py) confirms or
// corrects it in the game. The figures are medians of the moves that do publish them
// (2026-10-11 roster): hit advantage is 15 frames above block advantage (5,820 moves); startup is
// 12 (high), 16 (mid), 18 (low), 12 (throw); block -9, hit +7 when neither is known.
namespace {

std::optional<std::string> provisional_level(const std::string& text) {
    // TekkenDocs abbreviations: throws by where they grab, special/odd mids and lows.
    if (text == "th" || text == "thh" || text == "thm" || text == "thg" || text == "tg" || text == "ta" ||
        text == "tc" || text == "tw" || text == "ts") return std::string("throw");
    if (text == "mm" || text == "rm" || text == "ubm" || text == "m") return std::string("mid");
    if (text == "sp" || text == "sm") return std::string("special_mid");
    if (text == "sll" || text == "sl") return std::string("special_low");
    if (text == "h" || text == "hh") return std::string("high");
    if (text == "l" || text == "ll") return std::string("low");
    return std::nullopt;
}

}  // namespace

std::string automatic_transition_target(const std::string& text) {
    // "Transition to RFF", "Transition to HAM on hit automatically": the stance is the word after "to".
    const auto at = text.find("to ");
    if (at == std::string::npos) return {};
    std::string name;
    for (std::size_t index = at + 3; index < text.size(); ++index) {
        const char c = text[index];
        if (std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '.') name += c;
        else if (!name.empty()) break;
    }
    return name;
}

void add_provisional_stances(const std::vector<FullBindingRow>& rows, FullCharacterRules& rules,
                             std::vector<std::string>& stance_names) {
    const auto add = [&](const std::string& name) {
        if (name.empty() || name == "BT") return;  // BT is the fighter's own back-turned state
        if (std::find(stance_names.begin(), stance_names.end(), name) != stance_names.end()) return;
        stance_names.push_back(name);
        FullStanceRule stance{};
        stance.name = name;
        stance.max_frames = 60;
        rules.stances.push_back(stance);
    };
    for (const auto& row : rows) {
        for (const auto& stance : row.stances) add(stance);
        add(row.result_stance);
        add(row.result_stance_on_hit);
        add(row.result_stance_on_block);
        add(automatic_transition_target(row.automatic_transition));
    }
}

bool fill_provisional_gaps(FullBindingRow& row) {
    bool filled = false;
    if (!row.reactive) {
        // Unknown level abbreviations.
        for (auto& level : row.hit_levels) {
            if (hit_level(level)) continue;
            const bool stance_marker = level == "sp" && row.damages.empty();
            if (stance_marker) continue;  // "sp" without damage marks a move with no hits (handled below)
            if (const auto mapped = provisional_level(level)) {
                level = *mapped;
                filled = true;
            }
        }
        // More damage values than hits are usually alternatives (a charge level each: Jack-8's 1+4*
        // lists 35|45|55|199), not parts to add up: keep the first (base) value per hit. Missing
        // ones repeat the last known. A stand-in hit never exceeds 100 damage (a charged maximum
        // like 199 or 200 is not what the move usually does).
        // One hit with several parts (Falling Rain's 15|15|15 throw) is the binder's own case and
        // adds up; but a charge move (a "*" in the command, Jack-8's 1+4*: 35|45|55|199) lists one
        // value per charge level, and parts adding to over 100 are alternatives too: keep the base.
        if (row.hit_levels.size() == 1 && row.damages.size() > 1) {
            double total = 0.0;
            for (const double damage : row.damages) total += damage;
            if (row.command.find('*') != std::string::npos || total > 100.0) {
                row.damages.resize(1);
                filled = true;
            }
        }
        if (row.hit_levels.size() > 1 && !row.damages.empty() && row.damages.size() != row.hit_levels.size()) {
            auto& damages = row.damages;
            if (damages.size() > row.hit_levels.size()) damages.resize(row.hit_levels.size());
            while (damages.size() < row.hit_levels.size()) damages.push_back(damages.back());
            for (auto& damage : damages) damage = std::min(damage, 100.0);
            filled = true;
        }
        const bool has_hits = !row.hit_levels.empty() &&
            !std::all_of(row.hit_levels.begin(), row.hit_levels.end(), [](const std::string& l) { return l == "sp"; });
        if (has_hits) {
            if (!row.startup) {
                const std::string& first = row.hit_levels.front();
                row.startup = first == "high" ? 12 : first == "low" ? 18 : first == "throw" ? 12 :
                              first == "special_low" ? 11 : first == "special_mid" ? 20 : 16;
                filled = true;
            }
            if (!row.block_advantage && !row.hit_advantage) {
                row.block_advantage = -9;
                row.hit_advantage = 7;
                filled = true;
            } else if (!row.block_advantage) {
                row.block_advantage = *row.hit_advantage - 15;
                filled = true;
            } else if (!row.hit_advantage) {
                row.hit_advantage = *row.block_advantage + 15;
                filled = true;
            }
        } else if (row.damages.empty()) {
            // A move without hits (a stance entry, Ki Charge, crouch dash, a parry stance): it binds
            // as a stance entry, whose recovery is estimated with the rest.
            row.hit_levels.clear();
            filled = true;
        }
        // "Transition to RFF": the move ends in that stance (on hit, if the note says so).
        if (!row.automatic_transition.empty()) {
            const std::string target = automatic_transition_target(row.automatic_transition);
            if (!target.empty()) {
                const bool on_hit = row.automatic_transition.find("on hit") != std::string::npos;
                if (target == "BT") {
                    // The fighter's own back-turned state, not a stance. The engine turns a move's
                    // user around whatever the outcome, so a turn only on hit is left out.
                    if (!on_hit) row.result_back_turned = true;
                } else {
                    std::string& result = on_hit ? row.result_stance_on_hit : row.result_stance;
                    if (result.empty()) result = target;
                }
                row.automatic_transition.clear();
                filled = true;
            }
        }
    }
    return filled;
}

std::vector<FullBoundMove> bind_full_combat_moves(
    const std::vector<FullBindingRow>& rows,
    const FullBindingOptions& options) {
    std::vector<FullBoundMove> result;
    result.reserve(rows.size());
    std::set<std::string> commands;  // "character command", to spot a stance chain's plain version
    for (const auto& row : rows) commands.insert(row.character + ' ' + row.command);
    for (const auto& source : rows) {
        FullBindingRow row = source;
        FullBoundMove bound{};
        bound.stable_id = row.stable_id;
        bound.command = row.command;
        auto& blockers = bound.blockers;
        const auto& geometry = options.provisional;
        if (geometry && fill_provisional_gaps(row)) bound.provisional = true;
        const auto measured = [&](const auto& value, const char* name, auto fallback) {
            using Value = std::decay_t<decltype(fallback)>;
            if (value) return static_cast<Value>(*value);
            if (geometry) {
                bound.provisional = true;
                return fallback;
            }
            blockers.push_back(std::string("missing measurement: ") + name);
            return fallback;
        };

        // Parry outcomes bind as reactive moves the engine plays; they need no startup or recovery.
        const bool reactive = row.reactive;
        // Stance entries ("b+1+2", r10 MIA) have no hits: only recovery and the state they lead to.
        // Moves without hits: stance entries ("b+1+2", r10 MIA), Ki Charge, and evasive movement
        // ("UB,b", hit level "sp" = special, r49 with crush and floating windows).
        const bool no_hit_levels = std::all_of(row.hit_levels.begin(), row.hit_levels.end(),
                                               [](const std::string& level) { return level == "sp"; });
        const bool stance_entry = !reactive && no_hit_levels && row.damages.empty() &&
            (row.recovery.has_value() || geometry.has_value());
        if (row.requires_running) blockers.push_back("running state not modeled");
        if (row.parser_status != "parsed") blockers.push_back("command notation unresolved");
        if (row.source_consistency != "valid") blockers.push_back("source frame data inconsistent");
        if (options.require_practice_validation && row.practice_status != "pass") {
            blockers.push_back("Practice validation " + (row.practice_status.empty() ? "pending" : row.practice_status));
        }
        if (row.frame_data_variable && !options.allow_variable_frame_data) {
            blockers.push_back("frame data is a range (needs Practice resolution)");
        }
        bool back_to_wall = false;
        bool opponent_back_turned = false;
        bool opponent_left = false;
        bool opponent_right = false;
        for (const auto& situation : row.situations) {
            if (situation == "BACK_TO_WALL") back_to_wall = true;
            else if (situation == "OPPONENT_BACK_TURNED") opponent_back_turned = true;
            else if (situation == "OPPONENT_LEFT_SIDE") opponent_left = true;
            else if (situation == "OPPONENT_RIGHT_SIDE") opponent_right = true;
            else blockers.push_back("situational requirement not modeled: " + situation);
        }
        if (!row.automatic_transition.empty()) {
            blockers.push_back("automatic transition not modeled: " + row.automatic_transition);
        }
        // Amounts the notes state without a number are measurements still to take.
        const auto amounts = [&](const std::string& text, const char* name) {
            std::vector<double> values;
            if (text == "?") {
                if (geometry) bound.provisional = true;
                else blockers.push_back(std::string("missing measurement: ") + name);
                return values;
            }
            for (const auto& value : split(text, '|')) values.push_back(*to_double(value));
            return values;
        };
        const auto amount = [&](const std::string& text, const char* name) {
            const auto values = amounts(text, name);
            return values.empty() ? 0.0 : values.front();
        };
        const auto chip = amounts(row.chip_block, "chip_block");
        const auto heat_chip = amounts(row.chip_block_heat, "chip_block_heat");
        const double self_damage = amount(row.self_damage, "self_damage");
        const double self_recoverable = amount(row.self_recoverable, "self_recoverable");
        const double restore_health = amount(row.restore_health_hit, "restore_health_hit");
        const double restore_hit = amount(row.restore_recoverable_hit, "restore_recoverable_hit");
        const double restore_block = amount(row.restore_recoverable_block, "restore_recoverable_block");
        std::optional<FullThrowBreak> throw_break;
        if (row.throw_break == "1") throw_break = FullThrowBreak::One;
        else if (row.throw_break == "2") throw_break = FullThrowBreak::Two;
        else if (row.throw_break == "1+2") throw_break = FullThrowBreak::OneTwo;
        else if (row.throw_break == "1|2") throw_break = FullThrowBreak::OneOrTwo;
        else if (row.throw_break == "none") throw_break = FullThrowBreak::None;

        std::vector<FullHitLevel> levels;
        for (const auto& text : row.hit_levels) {
            if (stance_entry && text == "sp") continue;
            const auto level = hit_level(text);
            if (level) levels.push_back(*level);
            else blockers.push_back("unsupported hit level: " + text);
        }
        std::vector<double> damages = row.damages;
        double attack_throw_damage = row.attack_throw_damage.value_or(0.0);
        if (!row.attack_throw.empty()) {
            // The throw part follows the strike automatically: it adds damage, not a separate hit.
            while (!levels.empty() && levels.back() == FullHitLevel::Throw) {
                if (damages.size() == levels.size()) {
                    if (!row.attack_throw_damage) attack_throw_damage += damages.back();
                    damages.pop_back();
                }
                levels.pop_back();
            }
        }
        if (!reactive && levels.size() == 1 && damages.size() > 1) {
            // One hit dealing several damage parts ("Falling Rain": 15,15,15).
            double total = 0.0;
            for (const double part : damages) total += part;
            damages = {total};
        }
        if (reactive) {
            // A parry outcome deals its damage parts together when the parry lands; its hit list is
            // fixed here, before anything is sized from it.
            levels.assign(damages.size(), levels.empty() ? FullHitLevel::Mid : levels.front());
        }
        const bool all_throws = !levels.empty() &&
            std::all_of(levels.begin(), levels.end(), [](auto level) { return level == FullHitLevel::Throw; });
        const bool any_throw = std::any_of(levels.begin(), levels.end(),
                                           [](auto level) { return level == FullHitLevel::Throw; });
        if (any_throw && !throw_break && !reactive) {  // parry outcomes cannot be broken
            if (row.throw_break.empty() && geometry) {
                bound.provisional = true;
                throw_break = FullThrowBreak::OneTwo;
            } else {
                blockers.push_back(row.throw_break.empty() ? std::string("throw break input not stated")
                                   : row.throw_break == "?" ? std::string("throw break input varies with the thrower's input")
                                                            : "unsupported throw break input: " + row.throw_break);
            }
        }
        // One chip value for a multi-hit move does not say which hit deals it.
        const auto per_hit = [&](const std::vector<double>& values, const char* name) {
            std::vector<double> result(levels.size(), 0.0);
            if (values.empty() || levels.empty()) return result;
            if (values.size() == levels.size()) return values;
            if (values.size() == 1 && geometry) {
                bound.provisional = true;
                result.back() = values.front();
            } else if (geometry) {
                // A count that does not match the hits: on a one-hit move the values are its
                // versions (charged, in Heat; the first is the base move's, as for damage); on a
                // string the values belong to its last hits, where chip comes from.
                bound.provisional = true;
                if (levels.size() == 1) {
                    result.front() = values.front();
                } else {
                    const std::size_t count = std::min(values.size(), levels.size());
                    for (std::size_t k = 0; k < count; ++k) {
                        result[levels.size() - count + k] = values[values.size() - count + k];
                    }
                }
            } else {
                blockers.push_back(std::string("missing measurement: ") + name + " per hit");
            }
            return result;
        };
        const auto chip_per_hit = per_hit(chip, "chip_block");
        const auto heat_chip_per_hit = per_hit(heat_chip, "chip_block_heat");
        if (levels.empty() && !stance_entry) blockers.push_back("no hit levels");
        if (levels.size() != damages.size() && !(reactive && !damages.empty())) {
            blockers.push_back("damage and hit-level counts differ");
        }
        if (!row.startup && !stance_entry && !reactive) blockers.push_back("missing source startup");
        // Missing recovery (35% of the roster's moves; 170 of Lee's) blocks a move. With stand-in
        // geometry it gets the median recovery of the moves with a similar block advantage (from the
        // 4,029 moves that publish one); on block the engine sets the defender's stun from the
        // attacker's remaining frames plus the block advantage, so that stays exact and recovery only
        // decides how long a whiff leaves the move open. Marked provisional like the geometry.
        std::optional<int> recovery = row.recovery;
        if (!recovery && !reactive && geometry) {
            const int advantage = row.block_advantage.value_or(-8);
            recovery = advantage <= -15 ? 36 : advantage <= -10 ? 32 : advantage <= -5 ? 29 : advantage <= 0 ? 25 : 30;
            bound.provisional = true;
        }
        if (!recovery && !reactive) blockers.push_back("missing source recovery");
        if (!row.block_advantage && !all_throws && !stance_entry && !reactive) {
            blockers.push_back("missing source block advantage");
        }
        if (!row.hit_advantage && !stance_entry) blockers.push_back("missing source hit advantage");
        const auto stance_index = [&](const std::string& name) {
            if (name.empty()) return -1;
            const auto found = std::find(options.stance_names.begin(), options.stance_names.end(), name);
            if (found == options.stance_names.end()) {
                blockers.push_back("unknown stance: " + name);
                return -1;
            }
            return static_cast<int>(found - options.stance_names.begin());
        };
        const int result_stance = stance_index(row.result_stance);
        const int result_on_hit = stance_index(row.result_stance_on_hit);
        const int result_on_block = stance_index(row.result_stance_on_block);
        std::uint8_t parry_levels = 0;
        if (row.parry_levels == "?") {
            if (geometry) {
                bound.provisional = true;
                parry_levels = FullLevelHigh | FullLevelMid;
            } else {
                blockers.push_back("missing measurement: parry_levels");
            }
        } else {
            for (const auto& level : split(row.parry_levels, '|')) {
                if (level == "high") parry_levels |= FullLevelHigh;
                else if (level == "mid") parry_levels |= FullLevelMid;
                else if (level == "low") parry_levels |= FullLevelLow;
                else if (level == "throw") parry_levels |= FullLevelThrow;
            }
        }

        std::vector<int> first_frames;
        if (reactive) {
            for (std::size_t index = 0; index < levels.size(); ++index) first_frames.push_back(static_cast<int>(index) + 1);
        } else if (levels.size() == 1 && row.startup) {
            first_frames = {*row.startup};
        } else if (row.measured_hit_frames.size() == levels.size() && !levels.empty()) {
            first_frames = row.measured_hit_frames;
        } else if (geometry && row.startup) {
            bound.provisional = true;
            for (std::size_t index = 0; index < levels.size(); ++index) {
                first_frames.push_back(*row.startup + static_cast<int>(index) * geometry->string_gap);
            }
        } else if (levels.size() > 1) {
            blockers.push_back("missing measurement: hit_frames (string timing)");
        }
        std::vector<int> active;
        if (reactive) {
            active.assign(levels.size(), 1);
        } else if (row.measured_active_frames.size() == levels.size()) {
            active = row.measured_active_frames;
        } else if (row.measured_active_frames.size() == 1) {
            active.assign(levels.size(), row.measured_active_frames.front());
        } else if (geometry) {
            bound.provisional = true;
            active.assign(levels.size(), geometry->active_frames);
        } else {
            blockers.push_back("missing measurement: active_frames");
        }
        const FullProvisionalGeometry fallback = geometry.value_or(FullProvisionalGeometry{});
        // Stance entries and parry outcomes have no hitboxes to measure.
        const bool needs_geometry = !stance_entry && !reactive;
        const auto geometry_value = [&](const auto& value, const char* name, double stand_in) {
            return needs_geometry ? measured(value, name, stand_in) : stand_in;
        };
        const double reach = geometry_value(row.measured_range, "range", fallback.range);
        const double left = geometry_value(row.measured_tracking_left, "tracking_left", fallback.tracking);
        const double right = geometry_value(row.measured_tracking_right, "tracking_right", fallback.tracking);
        const double pushback = geometry_value(row.measured_pushback, "pushback", fallback.pushback);
        const double travel = geometry_value(row.measured_travel, "travel", fallback.travel);

        int required_stance = -1;
        bool requires_back_turned = false;
        std::vector<std::string> stances;
        for (const auto& stance : row.stances) {
            if (stance == "BT") requires_back_turned = true;  // the user's own back-turned state
            else stances.push_back(stance);
        }
        // A stance chain ("NSS.KIN.f+1+2", "MD1.HSP.b+4", "LTG.KNK.4,1+2"): the move is done from the
        // stance written right before its input; the earlier ones are a mode or the way in.
        // Conditions written like stances (counter hit, a grounded or airborne opponent) are not
        // modeled; a chain move that is a mode's version of a move the stance already has is left
        // to that move (the engine has one stance at a time); any other chain binds from its
        // last stance, provisionally.
        if (stances.size() > 1) {
            static const std::vector<std::string> conditions = {"CH", "OTG", "(AIRBORNE)", "AIRBORNE"};
            std::string proximate;
            std::size_t prefix_end = 0;
            for (std::size_t start = 0;;) {  // the prefix tokens in command order
                const auto dot = row.command.find('.', start);
                if (dot == std::string::npos) break;
                const std::string token = row.command.substr(start, dot - start);
                if (std::find(stances.begin(), stances.end(), token) == stances.end()) break;
                proximate = token;
                prefix_end = dot + 1;
                start = dot + 1;
            }
            const auto condition = std::find_first_of(stances.begin(), stances.end(), conditions.begin(), conditions.end());
            if (condition != stances.end()) {
                blockers.push_back("situational requirement not modeled: " + *condition);
            } else if (proximate.empty() || !geometry) {
                blockers.push_back("more than one stance requirement");
            } else if (commands.count(row.character + ' ' + proximate + '.' + row.command.substr(prefix_end))) {
                blockers.push_back("mode version of " + proximate + "." + row.command.substr(prefix_end) +
                                   " (one stance at a time)");
            } else {
                bound.provisional = true;
                stances = {proximate};
            }
        }
        for (const auto& stance : stances) {
            const auto found = std::find(options.stance_names.begin(), options.stance_names.end(), stance);
            if (found == options.stance_names.end()) {
                blockers.push_back("unknown stance: " + stance);
            } else {
                required_stance = static_cast<int>(found - options.stance_names.begin());
            }
        }

        const bool geometry_complete = stance_entry || (first_frames.size() == levels.size() &&
            active.size() == levels.size() && levels.size() == damages.size() && !levels.empty());
        if (!geometry_complete || (!recovery && !reactive) || (!row.hit_advantage && !stance_entry)) {
            result.push_back(std::move(bound));
            continue;
        }
        FullMoveSpec spec{};
        spec.stable_id = row.stable_id;
        for (std::size_t index = 0; index < levels.size(); ++index) {
            FullHitSpec hit{};
            hit.level = levels[index];
            hit.first_active_frame = first_frames[index];
            hit.active_frames = active[index];
            hit.damage = damages[index];
            hit.reach = reach;
            hit.tracking_left = left;
            hit.tracking_right = right;
            hit.homing = row.homing;
            const bool last = index + 1 == levels.size();
            hit.chip_damage = chip_per_hit[index];
            if (!heat_chip.empty()) hit.heat_chip_damage = heat_chip_per_hit[index];
            if (throw_break) hit.throw_break = *throw_break;
            hit.unparryable = row.unparryable;
            hit.reversal_break = row.reversal_break;
            hit.spike = last && row.spike;
            hit.tornado = last && row.tornado;
            hit.wall_break = last && row.wall_break;
            hit.floor_break = last && row.floor_break;
            hit.balcony_break = last && row.balcony_break;
            spec.hits.push_back(hit);
        }
        // A parry outcome without published recovery uses the engine's default (recovery 0).
        spec.recovery = reactive ? row.recovery.value_or(0) : recovery.value_or(1);
        spec.block_advantage = row.block_advantage.value_or(0);
        spec.hit_advantage = row.hit_advantage.value_or(0);
        spec.reactive = reactive;
        spec.requires_sidestep = row.requires_sidestep;
        spec.result_crouching = row.result_crouching;
        spec.result_stance_on_hit = result_on_hit;
        spec.result_stance_on_block = result_on_block;
        spec.parry_levels = parry_levels == 0 ? static_cast<std::uint8_t>(FullLevelHigh | FullLevelMid) : parry_levels;
        for (std::size_t index = 0; index < row.parry_outcomes.size() && index < spec.parry_outcomes.size(); ++index) {
            spec.parry_outcomes[index] = row.parry_outcomes[index];
        }
        spec.heat_cost_frames = row.heat_cost_frames;
        spec.resource_gain_start = row.resource_gain_start;
        spec.resource_gain_hit = row.resource_gain_hit;
        spec.resource_gain_airborne_hit = row.resource_gain_airborne_hit;
        spec.resource_gain_block = row.resource_gain_block;
        spec.resource_gain_heat_activation = row.resource_gain_heat_activation;
        spec.install_damage_bonus = row.install_damage_bonus;
        spec.install_chip_damage = row.install_chip;
        spec.install_reach = row.install_range;
        spec.requires_back_to_wall = back_to_wall;
        spec.requires_back_turned = requires_back_turned;
        spec.requires_opponent_back_turned = opponent_back_turned;
        spec.requires_opponent_left_side = opponent_left;
        spec.requires_opponent_right_side = opponent_right;
        spec.result_back_turned = row.result_back_turned;
        spec.back_turned_hit_advantage = row.back_turned_hit_advantage;
        spec.back_turned_hit_effect = hit_effect(row.back_turned_hit_effect);
        if (row.attack_throw == "hit") spec.attack_throw = FullMoveSpec::AttackThrow::OnHit;
        else if (row.attack_throw == "counter_hit") spec.attack_throw = FullMoveSpec::AttackThrow::OnCounterHit;
        spec.attack_throw_front_only = row.attack_throw_front_only;
        spec.attack_throw_standing_only = row.attack_throw_standing_only;
        spec.attack_throw_airborne = row.attack_throw_airborne;
        spec.attack_throw_damage = attack_throw_damage;
        if (!row.heat_parry.empty()) {
            spec.heat_parry = parse_frame_window(row.heat_parry);
            spec.heat_parry_outcome = row.heat_parry_outcome;
            spec.heat_parry_levels = 0;
            for (const auto& level : split(row.heat_parry_levels, '|')) {
                if (level == "high") spec.heat_parry_levels |= FullLevelHigh;
                else if (level == "mid") spec.heat_parry_levels |= FullLevelMid;
                else if (level == "low") spec.heat_parry_levels |= FullLevelLow;
                else if (level == "throw") spec.heat_parry_levels |= FullLevelThrow;
            }
            if (spec.heat_parry_levels == 0) {
                if (geometry) {
                    bound.provisional = true;
                    spec.heat_parry_levels = FullLevelHigh | FullLevelMid;
                } else {
                    blockers.push_back("missing measurement: heat_parry_levels");
                }
            }
        }
        spec.recoverable_damage = row.recoverable_damage;
        spec.cannot_ko = row.cannot_ko;
        spec.rage_art_max_damage = row.rage_art_max_damage;
        spec.ki_charge = row.ki_charge;
        spec.hit_effect = hit_effect(row.hit_effect);
        if (row.counter_hit_advantage) {
            spec.counter_hit_advantage = row.counter_hit_advantage;
            spec.counter_hit_effect = hit_effect(row.counter_hit_effect);
        }
        spec.travel = stance_entry ? 0.0 : travel;
        spec.pushback_hit = spec.pushback_block = pushback;
        spec.low_crush = parse_frame_window(row.low_crush);
        spec.high_crush = parse_frame_window(row.high_crush);
        spec.power_crush = parse_frame_window(row.power_crush);
        spec.parry = parse_frame_window(row.parry);
        spec.airborne = parse_frame_window(row.airborne);
        spec.invincible = parse_frame_window(row.invincible);
        // "while standing" moves are performed rising from a crouch.
        spec.required_posture = row.posture == "crouching" || row.posture == "while_standing"
            ? FullPosture::Crouching : FullPosture::Standing;
        spec.required_stance = required_stance;
        spec.result_stance = result_stance;
        spec.requires_heat = row.requires_heat;
        spec.requires_rage = row.requires_rage;
        spec.engages_heat = row.heat_burst;
        spec.heat_engager = row.heat_engager;
        spec.consumes_heat = row.heat_smash;
        spec.consumes_rage = row.rage_art;
        spec.heat_dash_block_advantage = row.heat_dash_block;
        spec.heat_dash_hit_advantage = row.heat_dash_hit;
        spec.heat_dash_hit_effect = hit_effect(row.heat_dash_effect);
        spec.recoverable_only = row.recoverable_only;
        spec.removes_recoverable = row.removes_recoverable;
        spec.armor_damage_recoverable = row.armor_damage_recoverable;
        spec.self_damage = self_damage;
        spec.self_recoverable = std::min(self_recoverable, self_damage);
        spec.self_damage_without_heat_only = row.self_damage_without_heat;
        spec.restore_health_hit = restore_health;
        spec.restore_recoverable_hit = restore_hit;
        spec.restore_recoverable_block = restore_block;
        spec.side_switch_on_hit = row.side_switch_on_hit;
        spec.side_switch_on_break = row.side_switch_on_break;
        // Hits must be strictly ordered for the engine; report instead of throwing.
        for (std::size_t index = 1; index < spec.hits.size(); ++index) {
            if (spec.hits[index].first_active_frame <= spec.hits[index - 1].first_active_frame) {
                blockers.push_back("hit frames are not increasing");
                break;
            }
        }
        bound.spec = std::move(spec);
        result.push_back(std::move(bound));
    }
    return result;
}

namespace {

std::vector<std::unordered_map<std::string, std::string>> read_table(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("could not open " + path.string());
    std::string line;
    if (!std::getline(input, line)) throw std::runtime_error("empty table: " + path.string());
    if (!line.empty() && line.back() == '\r') line.pop_back();
    const auto header = split_csv_line(line);
    std::vector<std::unordered_map<std::string, std::string>> rows;
    while (std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        const auto fields = split_csv_line(line);
        if (fields.size() != header.size()) throw std::runtime_error("wrong field count in " + path.string());
        std::unordered_map<std::string, std::string> row;
        for (std::size_t index = 0; index < header.size(); ++index) row[header[index]] = fields[index];
        rows.push_back(std::move(row));
    }
    return rows;
}

}  // namespace

FullCharacterRuleSet load_full_combat_character_rules(const std::filesystem::path& stances_csv,
                                                      const std::filesystem::path& resources_csv,
                                                      const std::string& character) {
    FullCharacterRuleSet result{};
    for (const auto& row : read_table(stances_csv)) {
        if (row.at("character") != character) continue;
        FullStanceRule stance{};
        stance.name = row.at("stance");
        stance.can_guard = row.at("can_guard") == "1";
        stance.max_frames = to_int(row.at("max_frames"));
        for (const auto& level : split(row.at("auto_parry"), '|')) {
            if (level == "high") stance.auto_parry |= FullLevelHigh;
            else if (level == "mid") stance.auto_parry |= FullLevelMid;
            else if (level == "low") stance.auto_parry |= FullLevelLow;
            else if (level == "throw") stance.auto_parry |= FullLevelThrow;
        }
        stance.parry_outcomes = {row.at("parry_outcome_high"), row.at("parry_outcome_mid"),
                                 row.at("parry_outcome_low"), row.at("parry_outcome_throw")};
        stance.pulse_interval_frames = to_int(row.at("pulse_interval_frames")).value_or(0);
        stance.pulse_recoverable = to_double(row.at("pulse_recoverable")).value_or(0.0);
        stance.pulse_resource = to_double(row.at("pulse_resource")).value_or(0.0);
        result.stance_names.push_back(stance.name);
        result.rules.stances.push_back(std::move(stance));
    }
    for (const auto& row : read_table(resources_csv)) {
        if (row.at("character") != character) continue;
        result.rules.resource_max = to_double(row.at("max")).value_or(0.0);
        result.rules.install_threshold = to_double(row.at("install_threshold"));
        result.rules.resource_persists = row.at("persists_across_rounds") == "1";
        result.rules.install_consumes = row.at("consumed_on_install") == "1";
    }
    return result;
}

}  // namespace t8::v2
