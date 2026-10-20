// Prints randomized full-combat traces (state and events after every frame).
// Used to check that a refactor of the engine leaves its behavior unchanged:
//   t8_v2_full_combat_trace data/generated > before.txt   (old engine)
//   t8_v2_full_combat_trace data/generated > after.txt    (new engine)
// and the files must be identical.

#include "full_combat_random_play.hpp"

#include <cstdlib>
#include <iostream>

int main(int argc, char** argv) {
    using namespace t8::v2::testing;
    if (argc < 2) {
        std::cerr << "usage: full_combat_trace <data/generated> [frames]\n";
        return EXIT_FAILURE;
    }
    const int frames = argc > 2 ? std::atoi(argv[2]) : 3000;
    const std::pair<const char*, const char*> matchups[] = {
        {"jun", "jun"}, {"jun", "king"}, {"yoshimitsu", "lidia"}, {"kazuya", "alisa"},
        {"armor-king", "jack-8"}, {"leroy", "zafina"},
    };
    try {
        for (const auto& [first, second] : matchups) {
            const auto fixture = make_fixture(argv[1], first, second);
            for (std::uint64_t seed = 1; seed <= 6; ++seed) {
                const auto fight = record_random_fight(fixture, seed * 7919, frames);
                for (std::size_t frame = 0; frame < fight.states.size(); ++frame) {
                    std::cout << first << '-' << second << ' ' << seed << ' ' << fight.states[frame]
                              << " ||" << fight.events[frame] << '\n';
                }
            }
        }
    } catch (const std::exception& error) {
        std::cerr << "trace error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
