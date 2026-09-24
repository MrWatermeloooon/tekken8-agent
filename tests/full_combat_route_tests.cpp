#include "t8_v2/full_combat_binding.hpp"
#include "t8_v2/full_combat_routes.hpp"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <set>
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

// Jun's moves bound with stand-in geometry: frame data from the catalog, reach 1.5, no pushback.
std::vector<FullMoveSpec> jun_moves(const std::filesystem::path& directory, const FullCharacterRuleSet& rules) {
    FullBindingOptions options{};
    options.stance_names = rules.stance_names;
    options.require_practice_validation = false;
    options.allow_variable_frame_data = true;
    FullProvisionalGeometry geometry{};
    geometry.range = 1.5;
    geometry.pushback = 0.0;
    options.provisional = geometry;
    std::vector<FullMoveSpec> moves;
    for (const auto& move : bind_full_combat_moves(load_full_combat_bindings(directory / "full_combat_bindings.csv", "jun"), options)) {
        if (move.bound()) moves.push_back(*move.spec);
    }
    return moves;
}

void test_jun_routes(const std::filesystem::path& directory) {
    const auto rules = load_full_combat_character_rules(directory / "full_combat_stances.csv",
                                                        directory / "full_combat_resources.csv", "jun");
    const auto moves = jun_moves(directory, rules);
    const auto routes = load_full_combat_routes(directory / "full_combat_routes.csv", "jun");
    std::set<std::string> categories;
    for (const auto& route : routes) categories.insert(route.category);
    check(categories == std::set<std::string>{"counter_hit", "heat", "midscreen", "wall"},
          "Jun has a route for every gate category");
    for (const auto& route : routes) {
        const auto result = run_full_combat_route(route, moves, rules.rules);
        std::cout << (result.passed() ? "PASS " : "FAIL ") << route.category << " " << route.name << " ("
                  << route.source << "): " << result.hits << " hits, " << result.damage << " damage, "
                  << result.frames << " frames\n";
        for (const auto& failure : result.failures) std::cout << "    " << failure << '\n';
        if (!result.passed() || std::getenv("T8_ROUTE_TRACE")) {
            for (const auto& line : result.trace) std::cout << "      " << line << '\n';
        }
        check(result.passed(), route.name + " executes as a legal true combo");
    }
}

void test_runner_catches_broken_routes(const std::filesystem::path& directory) {
    const auto rules = load_full_combat_character_rules(directory / "full_combat_stances.csv",
                                                        directory / "full_combat_resources.csv", "jun");
    const auto moves = jun_moves(directory, rules);
    auto route = load_full_combat_routes(directory / "full_combat_routes.csv", "jun").front();

    auto illegal = route;  // an IZU move with no way into IZU
    illegal.steps = {{"jun:129", 0}};
    auto result = run_full_combat_route(illegal, moves, rules.rules);
    check(!result.passed() && result.failures.front().find("never became legal") != std::string::npos,
          "a step that is never legal fails the route");

    auto blocked = route;  // the opponent guards the starter
    blocked.opponent = "guard";
    result = run_full_combat_route(blocked, moves, rules.rules);
    check(!result.passed() && result.failures.front().find("blocked") != std::string::npos,
          "a blocked starter fails the route");

    auto gap = route;  // a slow follow-up after a plain hit is not a true combo
    gap.steps = {{"jun:20", 0}, {"jun:53", 0}};  // 4 (+7 on hit) into d+1+2 (i26)
    gap.expected_hits.reset();
    result = run_full_combat_route(gap, moves, rules.rules);
    check(!result.passed(), "a follow-up slower than the advantage is caught");

    auto wrong_count = route;
    wrong_count.expected_hits = 99;
    result = run_full_combat_route(wrong_count, moves, rules.rules);
    check(!result.passed(), "a wrong hit count fails the route");
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: full_combat_route_tests <data/generated directory>\n";
        return EXIT_FAILURE;
    }
    try {
        test_jun_routes(argv[1]);
        test_runner_catches_broken_routes(argv[1]);
    } catch (const std::exception& error) {
        std::cerr << "route test error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
    if (failures != 0) {
        std::cerr << failures << " route assertion(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "full combat route tests passed\n";
    return EXIT_SUCCESS;
}
