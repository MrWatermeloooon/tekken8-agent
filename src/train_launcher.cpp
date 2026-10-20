// T8Train: one-click training launcher.
//
// Double-click T8Train.exe (copied to the repository root by the build) and it
// trains. On the first run it writes training.ini next to itself with the
// default settings, so nothing has to be typed; edit that file to change the
// fighter, run length, or mode. Each start:
//
//  * finds the newest built trainer (full-move or compatibility);
//  * resumes the run directory's latest checkpoint if there is one, otherwise
//    starts a new run (a run directory with results but no resumable
//    checkpoint is never overwritten; a numbered sibling is used instead);
//  * keeps the computer awake until the trainer exits;
//  * shows the trainer's output and also writes it to <run_dir>/launcher.log;
//  * starts the live dashboard (<run_dir>/dashboard.html) when the Python
//    environment exists;
//  * waits for Enter at the end, so a double-clicked window stays readable.
//
// While training, press V to open the visualizer (scripts/visualizer.py): the
// latest champion (the best policy found so far, not the newest) fights itself
// across random matchups, picking up each new champion per match. N opens the
// neural network visualizer (scripts/network_visualizer.py) on the same champion,
// following each new one. C ranks the champion's characters (tools/character_ranking.py)
// in a new window. Q saves a checkpoint after the current update and quits.
//
// Ctrl+C stops training; the last checkpoint is kept and the next start
// resumes from it.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include <thread>

#ifdef _WIN32
#define NOMINMAX
#include <conio.h>
#include <windows.h>
#else
#include <csignal>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
extern char** environ;
#endif

namespace fs = std::filesystem;

namespace {

constexpr const char* kDefaultSettings = R"(# T8Train settings. Edit, save, and run T8Train again.
# A run directory that already has checkpoints is resumed with the settings it
# was started with; change run_dir to start a different run.

# full-move: every documented move of the fighter (t8_v2_full_move_train).
# compatibility: the six-move compatibility trainer (t8_v2_train).
mode = full-move

# One policy learns every listed character: "all", one fighter such as jun, or a
# comma-separated list such as jun,king,reina. Its opponents get better with it,
# because in self-play they are the same policy playing those characters.
learner_character = all
# "roster" = every fighter with data, or a comma-separated list such as king,reina,paul
opponents = roster
run_dir = runs/train_all

# Required until a fighter passes its gate (measured and Practice-validated
# moves): unmeasured moves use stand-in geometry. Such runs are development
# runs and their checkpoints are refused for controller output.
allow_provisional = true

# Run size. 1000000 updates (2 trillion steps) is "train until stopped": Ctrl+C
# any time and run T8Train again to continue.
updates = 1000000
envs = 32768
horizon = 64
seed = 2027
reward = shaped

# PPO, chosen by tools/hyperparameter_sweep.py (2026-10-08, runs/sweep*): equal-step
# variants ranked in a mixed all-character arena. 16384-sample minibatches (128
# per update) and a 512-wide network learned most per step; TF32 tensor-core
# multiplies are about 12% faster and learned the same as FP32.
minibatch = 16384
epochs = 4
hidden = 512
learning_rate = 5e-4
final_learning_rate = 1e-4
# The learning rate and entropy bonus anneal over this many updates (about
# 12.6 billion steps), then stay at their final values. Blank = the run length.
anneal_updates = 6000
# fp32 | tf32 | bf16 (storage and optimizer stay FP32 in every mode)
matmul = tf32

# Full-move only. Settings below won the 2026-10-09/10 tests (OVERNIGHT_RECAP.md).
# Entropy floor: while the policy's per-decision entropy (nats) is below this,
# the exploration bonus is raised so it keeps trying different moves. 0 = off.
# 1.0: forked from the 24B run for 3B steps (2026-10-10), 1.0 beat 1.5 70% head-to-head
# and kept ~9 effective moves per character; 0.5 beat 1.5 79% but collapsed to ~5 moves (a
# one-trick bot) and punished less. 1.5 held the exploration bonus at ~8x its schedule.
entropy_target = 1.0
self_play_share = 0.3
self_play_start = 200
# The scripted roster's share of updates starts at 1 - self_play_share (70%) and
# shrinks toward this floor while the bot wins at least 95% of those rounds; it
# grows back under 90%, so the bot never forgets them. 0 = a fixed share.
scripted_floor = 0.1
# Tiered scripted bots: 331 bots in 7 tiers (dummies, beginner, intermediate,
# advanced, expert, master, pro styles) instead of one random opponent. Training
# climbs the ladder from tier 0 as the bot beats each tier (80% to move up).
# Off: in tests it trained worse, and every tier is beaten ~100% anyway.
scripted_tiers = false
# Opponent pool: the share of newest-policy opponents anneals from 0.8 to 0.1, so
# it meets more varied past versions of itself as training goes on.
pool_anneal = true
# Movement and guard inputs are held this many frames per decision (8 = steadier
# walking and blocking; it was the biggest single win in the tests).
hold_frames = 8
# The opponent's move is hidden from the policy until it is this many frames old,
# like human reaction time, so it cannot react to a move on its first frame.
perception_delay = 12
# The live bot (2026-10-11): train on exactly what the screen reader sees, with its noise and the
# controller's latency, so the policy transfers to the real game. In the round 3 tests a
# noise-trained bot beat a clean-trained one 86% under screen-like noise (and lost 29% in perfect
# conditions, which the real game never gives). Only screen-observation checkpoints may drive
# the controller (src/t8_agent/live/full_move.py).
screen_observation = true
live_noise = true
# Noise ranges, drawn per environment and round: position error (m), detection error rate,
# input latency (frames). The pose tracker's calibrated distance error is 0.28 m RMS.
noise_position = 0.25
noise_flip = 0.10
latency_min = 2
latency_max = 6
scenario_share = 0.25
# Scenario starts (punish, throw break, wakeup, juggle, ...) in self-play rounds too; without it
# only scripted-bot updates get them (6.6% of rounds in the 24B run; punishing stayed at 11%).
# Off until its test (runs/r3_scenarios) confirms it.
self_play_scenarios = false
# Anti-stalling: a draw (0-0 or equal-health timeout) scores -draw_penalty for the
# learner, a loss for both, and each broken throw costs the thrower
# throw_break_penalty (a jab's damage is about 0.03). Without them self-play
# drifts into timeouts and throw spam.
draw_penalty = 0.5
throw_break_penalty = 0.03
# Dense time pressure, as in the pro-level Blade & Soul agents (Oh et al. 2019): a
# cost per second of round time, so waiting is never free. A full 60 s round costs
# 0.6 of a win at 0.01.
time_penalty = 0.01
# Champion matches (sampled play) per check; more rounds = less noise.
champion_rounds = 2048

checkpoint_interval = 100
eval_interval = 25
eval_episodes = 512

# Live dashboard.html in the run directory (needs the .venv Python environment).
dashboard = true

# Further trainer options, passed through unchanged, for example: --gamma 0.995
extra =
)";

std::string trim(std::string value) {
    const auto start = value.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return {};
    const auto end = value.find_last_not_of(" \t\r\n");
    return value.substr(start, end - start + 1);
}

std::map<std::string, std::string> parse_settings(const std::string& text) {
    std::map<std::string, std::string> settings;
    std::istringstream input(text);
    std::string line;
    while (std::getline(input, line)) {
        const auto comment = line.find('#');
        if (comment != std::string::npos) line = line.substr(0, comment);
        const auto equals = line.find('=');
        if (equals == std::string::npos) continue;
        settings[trim(line.substr(0, equals))] = trim(line.substr(equals + 1));
    }
    return settings;
}

fs::path executable_path() {
#ifdef _WIN32
    std::wstring buffer(MAX_PATH, L'\0');
    const DWORD length = GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
    buffer.resize(length);
    return fs::path(buffer);
#else
    return fs::canonical("/proc/self/exe");
#endif
}

std::optional<fs::path> find_repository(fs::path start) {
    for (auto directory = start; ; directory = directory.parent_path()) {
        if (fs::exists(directory / "data" / "generated" / "full_move_catalog.csv")) return directory;
        if (directory == directory.root_path() || directory.empty()) return std::nullopt;
    }
}

std::optional<fs::path> find_trainer(const fs::path& repository, const fs::path& launcher_directory,
                                     const std::string& name) {
#ifdef _WIN32
    const std::string file = name + ".exe";
#else
    const std::string file = name;
#endif
    std::vector<fs::path> candidates = {launcher_directory / file};
    for (const char* directory : {"build-gpu/Release", "build/Release", "build-gpu", "build"}) {
        candidates.push_back(repository / directory / file);
    }
    std::optional<fs::path> newest;
    for (const auto& candidate : candidates) {
        std::error_code error;
        if (!fs::is_regular_file(candidate, error)) continue;
        if (!newest || fs::last_write_time(candidate) > fs::last_write_time(*newest)) newest = candidate;
    }
    return newest;
}

// The newest update_N checkpoint that has its trainer state next to it.
std::optional<std::pair<fs::path, std::uint64_t>> latest_checkpoint(const fs::path& run, const std::string& state_ext) {
    std::optional<std::pair<fs::path, std::uint64_t>> best;
    std::error_code error;
    if (!fs::is_directory(run / "checkpoints", error)) return best;
    for (const auto& entry : fs::directory_iterator(run / "checkpoints")) {
        const auto name = entry.path().filename().string();
        if (entry.path().extension() != ".t8ppo" || name.rfind("update_", 0) != 0) continue;
        auto state = entry.path();
        state.replace_extension(state_ext);
        if (!fs::exists(state)) continue;
        try {
            const auto update = std::stoull(name.substr(7, name.size() - 7 - 6));
            if (!best || update > best->second) best = std::make_pair(entry.path(), update);
        } catch (const std::exception&) {
        }
    }
    return best;
}

// The newest update_N.t8ppo in a directory that has its .t8fmstate.
std::optional<fs::path> newest_with_state(const fs::path& directory) {
    std::optional<fs::path> best;
    std::uint64_t best_update = 0;
    std::error_code error;
    if (!fs::is_directory(directory, error)) return best;
    for (const auto& entry : fs::directory_iterator(directory)) {
        const auto name = entry.path().filename().string();
        if (entry.path().extension() != ".t8ppo" || name.rfind("update_", 0) != 0) continue;
        auto state = entry.path();
        state.replace_extension(".t8fmstate");
        if (!fs::exists(state)) continue;
        try {
            const auto update = std::stoull(name.substr(7, name.size() - 7 - 6));
            if (!best || update > best_update) {
                best = entry.path();
                best_update = update;
            }
        } catch (const std::exception&) {
        }
    }
    return best;
}

std::vector<std::string> split_words(const std::string& text) {
    std::vector<std::string> words;
    std::istringstream input(text);
    std::string word;
    while (input >> word) words.push_back(word);
    return words;
}

std::string quote(const std::string& argument) {
    if (!argument.empty() && argument.find_first_of(" \t\"") == std::string::npos) return argument;
    std::string out = "\"";
    for (const char character : argument) {
        if (character == '"') out += '\\';
        out += character;
    }
    return out + "\"";
}

#ifdef _WIN32
std::atomic<bool> interrupted{false};

BOOL WINAPI on_control(DWORD) {
    interrupted = true;
    return TRUE;  // the trainer receives the same event and stops; the launcher reports and waits
}

std::wstring widen(const std::string& text) {
    if (text.empty()) return {};
    const int size = MultiByteToWideChar(CP_UTF8, 0, text.data(), static_cast<int>(text.size()), nullptr, 0);
    std::wstring wide(static_cast<std::size_t>(size), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, text.data(), static_cast<int>(text.size()), wide.data(), size);
    return wide;
}

// Runs the trainer, copying its output to the console and the log; returns its exit code.
int run_and_log(const std::vector<std::string>& command, const fs::path& directory, const fs::path& log_path) {
    SECURITY_ATTRIBUTES attributes{sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};
    HANDLE read_end = nullptr, write_end = nullptr;
    if (!CreatePipe(&read_end, &write_end, &attributes, 0)) throw std::runtime_error("cannot create a pipe");
    SetHandleInformation(read_end, HANDLE_FLAG_INHERIT, 0);
    STARTUPINFOW startup{};
    startup.cb = sizeof startup;
    startup.dwFlags = STARTF_USESTDHANDLES;
    startup.hStdOutput = write_end;
    startup.hStdError = write_end;
    startup.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    PROCESS_INFORMATION process{};
    std::string line;
    for (const auto& argument : command) line += (line.empty() ? "" : " ") + quote(argument);
    std::wstring wide = widen(line);
    if (!CreateProcessW(nullptr, wide.data(), nullptr, nullptr, TRUE, 0, nullptr, directory.c_str(), &startup,
                        &process)) {
        CloseHandle(read_end);
        CloseHandle(write_end);
        throw std::runtime_error("cannot start " + command.front());
    }
    CloseHandle(write_end);
    std::ofstream log(log_path, std::ios::app | std::ios::binary);
    char buffer[4096];
    DWORD count = 0;
    while (ReadFile(read_end, buffer, sizeof buffer, &count, nullptr) && count > 0) {
        std::cout.write(buffer, count);
        std::cout.flush();
        log.write(buffer, count);
        log.flush();
    }
    CloseHandle(read_end);
    WaitForSingleObject(process.hProcess, INFINITE);
    DWORD code = 1;
    GetExitCodeProcess(process.hProcess, &code);
    CloseHandle(process.hProcess);
    CloseHandle(process.hThread);
    return static_cast<int>(code);
}

std::optional<PROCESS_INFORMATION> start_background(const std::vector<std::string>& command, const fs::path& directory,
                                                    bool own_console = false) {
    std::string line;
    for (const auto& argument : command) line += (line.empty() ? "" : " ") + quote(argument);
    std::wstring wide = widen(line);
    STARTUPINFOW startup{};
    startup.cb = sizeof startup;
    PROCESS_INFORMATION process{};
    if (!CreateProcessW(nullptr, wide.data(), nullptr, nullptr, FALSE,
                        own_console ? CREATE_NEW_CONSOLE : CREATE_NO_WINDOW, nullptr, directory.c_str(),
                        &startup, &process)) {
        return std::nullopt;
    }
    return process;
}
#else
int run_and_log(const std::vector<std::string>& command, const fs::path& directory, const fs::path& log_path) {
    int pipe_ends[2];
    if (pipe(pipe_ends) != 0) throw std::runtime_error("cannot create a pipe");
    const pid_t child = fork();
    if (child == 0) {
        dup2(pipe_ends[1], STDOUT_FILENO);
        dup2(pipe_ends[1], STDERR_FILENO);
        close(pipe_ends[0]);
        if (chdir(directory.c_str()) != 0) _exit(127);
        std::vector<char*> arguments;
        for (const auto& argument : command) arguments.push_back(const_cast<char*>(argument.c_str()));
        arguments.push_back(nullptr);
        execv(arguments.front(), arguments.data());
        _exit(127);
    }
    close(pipe_ends[1]);
    std::ofstream log(log_path, std::ios::app | std::ios::binary);
    char buffer[4096];
    ssize_t count = 0;
    while ((count = read(pipe_ends[0], buffer, sizeof buffer)) > 0) {
        std::cout.write(buffer, count);
        std::cout.flush();
        log.write(buffer, count);
        log.flush();
    }
    close(pipe_ends[0]);
    int status = 0;
    waitpid(child, &status, 0);
    return WIFEXITED(status) ? WEXITSTATUS(status) : 1;
}
#endif

void keep_awake(bool on) {
#ifdef _WIN32
    SetThreadExecutionState(on ? (ES_CONTINUOUS | ES_SYSTEM_REQUIRED) : ES_CONTINUOUS);
#else
    static_cast<void>(on);  // on Linux, run under systemd-inhibit if the machine might sleep
#endif
}

void wait_for_enter() {
    std::cout << "\nPress Enter to close." << std::endl;
    std::string line;
    std::getline(std::cin, line);
}

}  // namespace

int main(int argc, char** argv) {
    const bool no_pause = argc > 1 && std::string(argv[1]) == "--no-pause";
    bool quit_now = false;  // Q ended the run: close without waiting for Enter
    int exit_code = 0;
    try {
        const fs::path launcher = executable_path();
        const auto repository = find_repository(launcher.parent_path());
        if (!repository) {
            throw std::runtime_error("T8Train must be inside the tekken 8 repository (data/generated not found)");
        }
        const fs::path settings_path = *repository / "training.ini";
        if (!fs::exists(settings_path)) {
            std::ofstream(settings_path) << kDefaultSettings;
            std::cout << "Wrote default settings to " << settings_path.string() << " (edit it to change the run).\n";
        }
        std::stringstream text;
        text << std::ifstream(settings_path).rdbuf();
        auto settings = parse_settings(text.str());
        const auto get = [&](const std::string& key, const std::string& fallback) {
            const auto found = settings.find(key);
            return found == settings.end() || found->second.empty() ? fallback : found->second;
        };
        const std::string mode = get("mode", "full-move");
        if (mode != "full-move" && mode != "compatibility") {
            throw std::runtime_error("training.ini: mode must be full-move or compatibility, not " + mode);
        }
        const bool full_move = mode == "full-move";
        const auto trainer = find_trainer(*repository, launcher.parent_path(),
                                          full_move ? "t8_v2_full_move_train" : "t8_v2_train");
        if (!trainer) {
            throw std::runtime_error(std::string("the ") + (full_move ? "t8_v2_full_move_train" : "t8_v2_train") +
                                     " trainer is not built; build with: cmake --build build-gpu --config Release");
        }
        const std::string state_ext = full_move ? ".t8fmstate" : ".t8state";

        // Run directory: resume it, start it, or (results without a checkpoint) use a fresh sibling.
        std::string run_dir = get("run_dir", "runs/train_all");
        auto resume = latest_checkpoint(*repository / run_dir, state_ext);
        if (!resume && fs::exists(*repository / run_dir / "metrics.jsonl")) {
            for (int suffix = 2; ; ++suffix) {
                const std::string candidate = run_dir + "_" + std::to_string(suffix);
                if (!fs::exists(*repository / candidate / "metrics.jsonl")) {
                    std::cout << run_dir << " has results but no resumable checkpoint; starting " << candidate
                              << " instead (nothing is overwritten).\n";
                    run_dir = candidate;
                    resume = latest_checkpoint(*repository / run_dir, state_ext);
                    break;
                }
            }
        }
        fs::create_directories(*repository / run_dir);

        std::vector<std::string> command = {trainer->string()};
        const auto add = [&](const std::string& flag, const std::string& key, const std::string& fallback) {
            command.push_back(flag);
            command.push_back(get(key, fallback));
        };
        const std::string updates = get("updates", "3000");
        add("--envs", "envs", "32768");
        add("--horizon", "horizon", "64");
        add("--updates", "updates", "3000");
        // The learning-rate and entropy schedules end here (blank: the run length),
        // so a very long run still settles instead of decaying over its whole length.
        const std::string anneal = get("anneal_updates", "");
        command.insert(command.end(), {"--anneal-updates", anneal.empty() ? updates : anneal});
        add("--minibatch", "minibatch", "16384");
        add("--seed", "seed", "2027");
        add("--reward", "reward", "shaped");
        add("--checkpoint-interval", "checkpoint_interval", "100");
        add("--eval-interval", "eval_interval", "25");
        add("--eval-episodes", "eval_episodes", "512");
        add("--learner-character", "learner_character", "all");
        if (full_move) {
            add("--opponents", "opponents", "roster");
            add("--self-play-share", "self_play_share", "0.3");
            add("--self-play-start", "self_play_start", "200");
            add("--scripted-floor", "scripted_floor", "0.1");
            if (get("scripted_tiers", "false") == "true") command.push_back("--scripted-tiers");
            if (get("pool_anneal", "true") == "true") command.push_back("--pool-anneal");
            if (get("self_play_scenarios", "false") == "true") command.push_back("--self-play-scenarios");
            add("--hold-frames", "hold_frames", "8");
            add("--perception-delay", "perception_delay", "12");
            if (get("screen_observation", "true") == "true") command.push_back("--screen-observation");
            if (get("live_noise", "true") == "true") {
                command.push_back("--live-noise");
                add("--noise-position", "noise_position", "0.25");
                add("--noise-flip", "noise_flip", "0.10");
                add("--latency-min", "latency_min", "2");
                add("--latency-max", "latency_max", "6");
            }
            add("--draw-penalty", "draw_penalty", "0.5");
            add("--throw-break-penalty", "throw_break_penalty", "0.03");
            add("--time-penalty", "time_penalty", "0.01");
            add("--champion-rounds", "champion_rounds", "2048");
            add("--scenario-share", "scenario_share", "0.25");
            add("--entropy-target", "entropy_target", "1.0");
            add("--hidden", "hidden", "512");
            add("--epochs", "epochs", "4");
            add("--learning-rate", "learning_rate", "5e-4");
            add("--final-learning-rate", "final_learning_rate", "1e-4");
            add("--matmul", "matmul", "tf32");
            if (get("allow_provisional", "true") == "true") command.push_back("--allow-provisional");
        } else {
            command.insert(command.end(), {"--opponents", "roster", "--curriculum-stage", "auto",
                                           "--observation-mode", "visual", "--full-move-catalog",
                                           (*repository / "data" / "generated" / "full_move_catalog.csv").string()});
        }
        for (const auto& word : split_words(get("extra", ""))) command.push_back(word);
        command.insert(command.end(), {"--run-dir", run_dir});
        if (resume) command.insert(command.end(), {"--resume", resume->first.string()});

        if (resume && resume->second >= std::stoull(updates)) {
            std::cout << run_dir << " already finished all " << updates << " updates. To train more, set a new run_dir "
                      << "(and a larger updates) in training.ini; the learning-rate schedule of a finished run "
                      << "cannot be stretched on resume.\n";
            if (!no_pause) wait_for_enter();
            return 0;
        }
        const bool provisional = full_move && get("allow_provisional", "true") == "true";
        std::atomic<bool> quit_requested{false};
        std::cout << "========================================\n"
                  << " T8Train - Tekken 8 " << (full_move ? "full-move" : "compatibility") << " training\n"
                  << "========================================\n"
                  << "Learners: " << get("learner_character", "all") << "   Opponents: " << get("opponents", "roster") << '\n'
                  << "Run: " << run_dir
                  << (resume ? "   (resuming from update " + std::to_string(resume->second) + ")" : "   (new run)") << '\n'
                  << "Updates: " << updates << "   Environments: " << get("envs", "32768") << '\n'
                  << "Trainer: " << trainer->filename().string() << '\n';
        if (provisional) std::cout << "Geometry: stand-in for unmeasured moves (development run, not game-accurate)\n";

#ifdef _WIN32
        SetConsoleCtrlHandler(on_control, TRUE);
        std::optional<PROCESS_INFORMATION> dashboard;
        const fs::path python = *repository / ".venv" / "Scripts" / "python.exe";
        if (get("dashboard", "true") == "true" && fs::exists(python)) {
            dashboard = start_background({python.string(), (*repository / "scripts" / "training.py").string(), "report",
                                          "--run-dir", run_dir, "--watch", "60"}, *repository);
            if (dashboard) std::cout << "Dashboard: " << (*repository / run_dir / "dashboard.html").string() << '\n';
        }
        // V: the fight visualizer, N: the neural network visualizer, both on the latest champion.
        std::atomic<bool> training{true};
        const fs::path stop_request = *repository / run_dir / "stop_request";
        const fs::path champion_directory = *repository / run_dir / "champion";
        const auto open_visualizer = [&](const char* key, const char* script, const char* what,
                                         bool own_console = false) {
            if (!fs::exists(python)) {
                std::cout << "\n[" << key << "] the visualizer needs the .venv Python environment\n" << std::flush;
                return;
            }
            if (!newest_with_state(champion_directory)) {
                std::cout << "\n[" << key << "] no champion yet: the first one is crowned at the first champion match\n"
                          << std::flush;
                return;
            }
            std::vector<std::string> viewer = {python.string(), (*repository / script).string(), "--run",
                                               (*repository / run_dir).string()};
            if (own_console) viewer.push_back("--pause");  // a console tool: keep its window open to read
            if (auto process = start_background(viewer, *repository, own_console)) {
                CloseHandle(process->hProcess);
                CloseHandle(process->hThread);
                std::cout << "\n[" << key << "] " << what << (own_console ? "\n" : " (Esc closes it)\n") << std::flush;
            }
        };
        std::thread keys([&]() {
            while (training.load()) {
                if (_kbhit()) {
                    const int key = _getch();
                    if (key == 'v' || key == 'V') {
                        open_visualizer("V", "scripts/visualizer.py", "visualizer opening: the latest champion vs itself");
                    }
                    if (key == 'n' || key == 'N') {
                        open_visualizer("N", "scripts/network_visualizer.py",
                                        "network visualizer opening: the latest champion's neural network, live");
                    }
                    if (key == 'c' || key == 'C') {
                        open_visualizer("C", "tools/character_ranking.py",
                                        "ranking the champion's characters in a new window (about a minute)", true);
                    }
                    if ((key == 'q' || key == 'Q') && !quit_requested.exchange(true)) {
                        std::ofstream(stop_request) << "save and quit\n";
                        std::cout << "\n[Q] saving a checkpoint after this update, then quitting...\n" << std::flush;
                    }
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
        });
        std::cout << "Controls: [V] watch the champion   [N] its neural network   [C] best characters   [Q] save and quit   [Ctrl+C] stop without saving\n"
                  << "========================================\n\n";
#else
        std::cout << "Ctrl+C stops training; run T8Train again to resume.\n\n";
#endif
        keep_awake(true);
        const auto started = std::chrono::steady_clock::now();
        exit_code = run_and_log(command, *repository, *repository / run_dir / "launcher.log");
        keep_awake(false);
#ifdef _WIN32
        training = false;
        keys.join();
        if (dashboard) {
            TerminateProcess(dashboard->hProcess, 0);
            CloseHandle(dashboard->hProcess);
            CloseHandle(dashboard->hThread);
        }
        // One last dashboard refresh with the final results.
        if (get("dashboard", "true") == "true" && fs::exists(python)) {
            if (auto final = start_background({python.string(), (*repository / "scripts" / "training.py").string(),
                                               "report", "--run-dir", run_dir}, *repository)) {
                WaitForSingleObject(final->hProcess, 120'000);
                CloseHandle(final->hProcess);
                CloseHandle(final->hThread);
            }
        }
#endif
        const double minutes = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() / 60.0;
#ifdef _WIN32
        const bool stopped_by_user = interrupted.load();
#else
        const bool stopped_by_user = false;
#endif
        std::cout << "\n"
                  << (quit_requested.load() && exit_code == 0 ? "Saved and stopped"
                      : stopped_by_user ? "Training stopped with Ctrl+C (progress since the last checkpoint is lost)"
                      : exit_code == 0 ? "Training finished"
                                       : "Training stopped (exit code " + std::to_string(exit_code) + ")")
                  << " after " << static_cast<int>(minutes) << " min. Results: " << (*repository / run_dir).string()
                  << "\nRun T8Train again to continue from the last checkpoint.\n";
        quit_now = quit_requested.load() && exit_code == 0;
    } catch (const std::exception& error) {
        std::cerr << "T8Train error: " << error.what() << '\n';
        exit_code = 1;
    }
    if (!no_pause && !quit_now) wait_for_enter();
    return exit_code;
}
