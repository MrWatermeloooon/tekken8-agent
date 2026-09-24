#include "t8_v2/full_combat_routes.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace t8::v2 {
namespace {

std::vector<std::string> split_fields(const std::string& line) {
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

std::string describe(const FullFighter& fighter) {
    static const char* postures[] = {"standing", "crouching", "airborne", "grounded", "wakeup", "wall_splat"};
    std::ostringstream text;
    text << postures[static_cast<int>(fighter.posture)] << " x=" << fighter.x << " stun=" << fighter.stun
         << " air=" << fighter.posture_frames << " combo=" << fighter.combo_hits;
    return text.str();
}

std::optional<std::size_t> find_move(const std::vector<FullMoveSpec>& moveset, const std::string& stable_id) {
    for (std::size_t index = 0; index < moveset.size(); ++index) {
        if (moveset[index].stable_id == stable_id) return index;
    }
    return std::nullopt;
}

}  // namespace

std::vector<FullRoute> load_full_combat_routes(const std::filesystem::path& path, const std::string& character) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("could not open full combat routes: " + path.string());
    std::string line;
    if (!std::getline(input, line)) throw std::runtime_error("empty full combat routes: " + path.string());
    if (!line.empty() && line.back() == '\r') line.pop_back();
    const auto header = split_fields(line);
    std::vector<FullRoute> routes;
    while (std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        const auto fields = split_fields(line);
        if (fields.size() != header.size()) throw std::runtime_error("full combat routes row has the wrong field count");
        std::unordered_map<std::string, std::string> row;
        for (std::size_t index = 0; index < header.size(); ++index) row[header[index]] = fields[index];
        if (!character.empty() && row.at("character") != character) continue;
        if (routes.empty() || routes.back().name != row.at("route") || routes.back().character != row.at("character")) {
            FullRoute route{};
            route.character = row.at("character");
            route.name = row.at("route");
            route.category = row.at("category");
            route.source = row.at("source");
            route.distance = std::stod(row.at("distance"));
            route.near_wall = row.at("near_wall") == "1";
            route.opponent = row.at("opponent");
            route.opponent_move = row.at("opponent_move");
            route.heat = row.at("heat") == "1";
            route.resource = std::stod(row.at("resource"));
            if (!row.at("expected_hits").empty()) route.expected_hits = std::stoi(row.at("expected_hits"));
            route.counter_hit_starter = row.at("counter_hit_starter") == "1";
            route.heat_activated = row.at("heat_activated") == "1";
            routes.push_back(std::move(route));
        }
        routes.back().steps.push_back({row.at("input"), std::stoi(row.at("delay"))});
    }
    return routes;
}

FullRouteResult run_full_combat_route(const FullRoute& route, const std::vector<FullMoveSpec>& moveset,
                                      const FullCharacterRules& rules, const FullEngineConfig& config,
                                      int max_wait_frames) {
    FullRouteResult result{};
    const auto fail = [&](std::string message) { result.failures.push_back(route.name + ": " + std::move(message)); };

    // Resolve every input before fighting.
    std::vector<FullInput> inputs;
    std::vector<std::size_t> expected_step_hits;
    for (const auto& step : route.steps) {
        if (step.input == "@HeatDash") {
            inputs.push_back({std::nullopt, FullUniversal::HeatDash});
            expected_step_hits.push_back(0);
        } else if (const auto index = find_move(moveset, step.input)) {
            inputs.push_back({*index, FullUniversal::Idle});
            expected_step_hits.push_back(moveset[*index].hits.size());
        } else {
            fail("move " + step.input + " is not bound");
            return result;
        }
    }
    std::optional<std::size_t> opponent_move;
    if (route.opponent == "attack") {
        opponent_move = find_move(moveset, route.opponent_move);
        if (!opponent_move) {
            fail("opponent move " + route.opponent_move + " is not bound");
            return result;
        }
    }

    FullCombatEngine engine(std::array<std::vector<FullMoveSpec>, 2>{moveset, moveset}, config,
                            std::array<FullCharacterRules, 2>{rules, rules});
    if (route.near_wall) {
        const double wall = engine.state().stage.right_wall;
        engine.reset(wall - route.distance, wall);
    } else {
        engine.reset(-route.distance / 2.0, route.distance / 2.0);
    }
    auto& p1 = engine.mutable_state().fighters[0];
    if (route.heat) {
        p1.heat = true;
        p1.heat_available = false;
        p1.heat_frames = config.heat_duration_frames;
    }
    p1.resource = route.resource;
    p1.installed = rules.install_threshold && route.resource >= *rules.install_threshold;

    std::size_t next = 0;
    int waited = 0;
    int delay_left = -1;
    int current = -1;
    std::vector<std::size_t> step_hits(route.steps.size(), 0);
    bool combo_started = false;
    bool opponent_was_free = false;
    bool heat_seen = false;
    const int limit = 60 * 30;
    for (int frame = 0; frame < limit; ++frame) {
        FullInput p1_input{};
        if (next < inputs.size()) {
            if (engine.legal(0, inputs[next])) {
                if (delay_left < 0) delay_left = route.steps[next].delay;
                if (delay_left-- == 0) {
                    p1_input = inputs[next];
                    result.issue_frames.push_back(frame);
                    std::ostringstream line;
                    line << "f" << frame << " input step " << next << " " << route.steps[next].input
                         << " | opponent " << describe(engine.state().fighters[1]);
                    result.trace.push_back(line.str());
                    current = static_cast<int>(next);
                    ++next;
                    waited = 0;
                    delay_left = -1;
                }
            } else if (++waited > max_wait_frames) {
                fail("step " + std::to_string(next) + " (" + route.steps[next].input + ") never became legal");
                break;
            }
        }
        FullInput p2_input{};
        if (route.opponent == "walk" && !combo_started) p2_input = {std::nullopt, FullUniversal::WalkForward};
        if (route.opponent == "crouch" && !combo_started) p2_input = {std::nullopt, FullUniversal::Crouch};
        if (route.opponent == "attack" && frame == 0) p2_input = {*opponent_move, FullUniversal::Idle};

        const auto events = engine.step(p1_input, p2_input);
        heat_seen |= events.heat_activated[0];
        for (const auto& contact : events.contacts) {
            if (contact.attacker != 0) {
                fail("the opponent's attack connected first");
                continue;
            }
            const std::string where = "step " + std::to_string(current) + " (" + route.steps[static_cast<std::size_t>(current)].input + ")";
            if (contact.contact != FullContact::Hit && contact.contact != FullContact::CounterHit) {
                fail(where + " was " + std::string(to_string(contact.contact)));
                continue;
            }
            if (!combo_started && route.counter_hit_starter && contact.contact != FullContact::CounterHit) {
                fail("the starter is not a counter-hit");
            }
            if (combo_started && opponent_was_free) {
                fail(where + " landed after the opponent could act (not a true combo)");
            }
            {
                std::ostringstream line;
                line << "f" << frame << " " << to_string(contact.contact) << " step " << current << " hit "
                     << contact.hit_index << " damage " << contact.damage << " | opponent "
                     << describe(engine.state().fighters[1]);
                result.trace.push_back(line.str());
            }
            combo_started = true;
            ++result.hits;
            result.damage += contact.damage;
            ++step_hits[static_cast<std::size_t>(current)];
        }
        if (!result.failures.empty()) break;
        if (combo_started && engine.actionable(1)) opponent_was_free = true;
        result.frames = frame + 1;
        if (next == inputs.size() && engine.actionable(0)) break;
        if (engine.state().winner >= 0) break;
    }
    for (std::size_t index = 0; index < route.steps.size() && result.failures.empty(); ++index) {
        if (index >= result.issue_frames.size()) {
            fail("step " + std::to_string(index) + " was never input");
        } else if (step_hits[index] != expected_step_hits[index]) {
            fail("step " + std::to_string(index) + " (" + route.steps[index].input + ") landed " +
                 std::to_string(step_hits[index]) + " of " + std::to_string(expected_step_hits[index]) + " hits");
        }
    }
    if (result.failures.empty() && route.expected_hits && result.hits != *route.expected_hits) {
        fail("landed " + std::to_string(result.hits) + " hits, expected " + std::to_string(*route.expected_hits));
    }
    if (result.failures.empty() && route.heat_activated && !heat_seen) fail("Heat was not activated");
    result.end_posture = engine.state().fighters[1].posture;
    return result;
}

}  // namespace t8::v2
