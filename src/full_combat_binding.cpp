#include "t8_v2/full_combat_binding.hpp"

#include <algorithm>
#include <charconv>
#include <fstream>
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

std::vector<FullBoundMove> bind_full_combat_moves(
    const std::vector<FullBindingRow>& rows,
    const FullBindingOptions& options) {
    std::vector<FullBoundMove> result;
    result.reserve(rows.size());
    for (const auto& row : rows) {
        FullBoundMove bound{};
        bound.stable_id = row.stable_id;
        bound.command = row.command;
        auto& blockers = bound.blockers;
        const auto& geometry = options.provisional;
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

        if (row.reactive) blockers.push_back("reactive outcome of a parry (not an input)");
        if (row.parser_status != "parsed") blockers.push_back("command notation unresolved");
        if (row.source_consistency != "valid") blockers.push_back("source frame data inconsistent");
        if (options.require_practice_validation && row.practice_status != "pass") {
            blockers.push_back("Practice validation " + (row.practice_status.empty() ? "pending" : row.practice_status));
        }
        if (row.frame_data_variable && !options.allow_variable_frame_data) {
            blockers.push_back("frame data is a range (needs Practice resolution)");
        }
        for (const auto& situation : row.situations) {
            blockers.push_back("situational requirement not modeled: " + situation);
        }
        if (!row.automatic_transition.empty()) {
            blockers.push_back("automatic transition not modeled: " + row.automatic_transition);
        }
        if (row.rage_art || row.requires_rage) blockers.push_back("Rage state not modeled");

        std::vector<FullHitLevel> levels;
        for (const auto& text : row.hit_levels) {
            const auto level = hit_level(text);
            if (level) levels.push_back(*level);
            else blockers.push_back("unsupported hit level: " + text);
        }
        const bool all_throws = !levels.empty() &&
            std::all_of(levels.begin(), levels.end(), [](auto level) { return level == FullHitLevel::Throw; });
        if (levels.empty()) blockers.push_back("no hit levels");
        if (levels.size() != row.damages.size()) blockers.push_back("damage and hit-level counts differ");
        if (!row.startup) blockers.push_back("missing source startup");
        if (!row.recovery) blockers.push_back("missing source recovery");
        if (!row.block_advantage && !all_throws) blockers.push_back("missing source block advantage");
        if (!row.hit_advantage) blockers.push_back("missing source hit advantage");

        std::vector<int> first_frames;
        if (levels.size() == 1 && row.startup) {
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
        if (row.measured_active_frames.size() == levels.size()) {
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
        const double reach = measured(row.measured_range, "range", fallback.range);
        const double left = measured(row.measured_tracking_left, "tracking_left", fallback.tracking);
        const double right = measured(row.measured_tracking_right, "tracking_right", fallback.tracking);
        const double pushback = measured(row.measured_pushback, "pushback", fallback.pushback);
        const double travel = measured(row.measured_travel, "travel", fallback.travel);

        int required_stance = -1;
        for (const auto& stance : row.stances) {
            const auto found = std::find(options.stance_names.begin(), options.stance_names.end(), stance);
            if (found == options.stance_names.end()) {
                blockers.push_back("unknown stance: " + stance);
            } else {
                required_stance = static_cast<int>(found - options.stance_names.begin());
            }
        }
        if (row.stances.size() > 1) blockers.push_back("more than one stance requirement");

        const bool geometry_complete = first_frames.size() == levels.size() && active.size() == levels.size() &&
            levels.size() == row.damages.size() && !levels.empty();
        if (!geometry_complete || !row.recovery || !row.hit_advantage) {
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
            hit.damage = row.damages[index];
            hit.reach = reach;
            hit.tracking_left = left;
            hit.tracking_right = right;
            hit.homing = row.homing;
            const bool last = index + 1 == levels.size();
            hit.tornado = last && row.tornado;
            hit.wall_break = last && row.wall_break;
            hit.floor_break = last && row.floor_break;
            hit.balcony_break = last && row.balcony_break;
            spec.hits.push_back(hit);
        }
        spec.recovery = *row.recovery;
        spec.block_advantage = row.block_advantage.value_or(0);
        spec.hit_advantage = *row.hit_advantage;
        spec.hit_effect = hit_effect(row.hit_effect);
        if (row.counter_hit_advantage) {
            spec.counter_hit_advantage = row.counter_hit_advantage;
            spec.counter_hit_effect = hit_effect(row.counter_hit_effect);
        }
        spec.travel = travel;
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
        spec.requires_heat = row.requires_heat;
        spec.requires_rage = row.requires_rage;
        spec.engages_heat = row.heat_engager;
        spec.consumes_rage = row.rage_art;
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

}  // namespace t8::v2
