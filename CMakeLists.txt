cmake_minimum_required(VERSION 3.24)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # Avoid CMake's `native` probe as the default: it can fail on headless builders and
    # during Visual Studio regeneration. Ship native code for NVIDIA's supported
    # Turing-through-Blackwell generations, with PTX in the newest image.
    set(CMAKE_CUDA_ARCHITECTURES "75-real;86-real;89-real;120" CACHE STRING
        "CUDA architectures to build (for example: 120, 89-real;120, or native)")
endif()

project(tekken_lite_v2 LANGUAGES CXX)

option(T8_V2_ENABLE_CUDA
    "Build the CUDA simulator, PPO trainer, and GPU tests (OFF builds the CPU oracle, full-combat engine, roster, and data tests only)"
    ON)
option(T8_V2_BUILD_TESTS "Build CPU-oracle and CUDA parity tests" ON)
option(T8_V2_BUILD_BENCHMARKS "Build GPU throughput benchmarks" ON)
option(T8_V2_PPO_FAST_MATH "Use approximate CUDA math in PPO probability calculations" OFF)

if(T8_V2_ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
else()
    set(T8_V2_BUILD_BENCHMARKS OFF)
    message(STATUS "T8_V2_ENABLE_CUDA=OFF: building the CPU targets and tests only")
endif()

# This target is deliberately an oracle, not the production environment.
add_library(t8_v2_cpu_oracle STATIC src/sim.cpp)
target_include_directories(t8_v2_cpu_oracle PUBLIC include)
target_compile_features(t8_v2_cpu_oracle PUBLIC cxx_std_20)

add_library(t8_v2_full_combat STATIC src/full_combat.cpp src/full_combat_engine.cpp src/full_combat_binding.cpp
    src/full_combat_routes.cpp src/full_move_roster.cpp)
target_include_directories(t8_v2_full_combat PUBLIC include)
target_compile_features(t8_v2_full_combat PUBLIC cxx_std_20)

add_library(t8_v2_roster STATIC src/roster.cpp)
target_include_directories(t8_v2_roster PUBLIC include)
target_compile_features(t8_v2_roster PUBLIC cxx_std_20)

if(T8_V2_ENABLE_CUDA)
add_library(t8_v2_gpu STATIC src/gpu_sim.cu)
target_include_directories(t8_v2_gpu PUBLIC include)
target_compile_features(t8_v2_gpu PUBLIC cxx_std_20)
target_link_libraries(t8_v2_gpu PUBLIC CUDA::cudart)
set_target_properties(t8_v2_gpu PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION OFF
)
target_compile_options(t8_v2_gpu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;-lineinfo>
)

add_library(t8_v2_ppo STATIC src/ppo.cu src/training_router.cu)
target_include_directories(t8_v2_ppo PUBLIC include)
target_compile_features(t8_v2_ppo PUBLIC cxx_std_20)
target_link_libraries(t8_v2_ppo PUBLIC CUDA::cudart CUDA::cublas)
set_target_properties(t8_v2_ppo PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION OFF
)
target_compile_options(t8_v2_ppo PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)
if(T8_V2_PPO_FAST_MATH)
    target_compile_options(t8_v2_ppo PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
endif()

add_library(t8_v2_opponents STATIC src/opponents.cu src/temporal.cu)
target_include_directories(t8_v2_opponents PUBLIC include)
target_compile_features(t8_v2_opponents PUBLIC cxx_std_20)
target_link_libraries(t8_v2_opponents PUBLIC CUDA::cudart)
set_target_properties(t8_v2_opponents PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION OFF
)
target_compile_options(t8_v2_opponents PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;-lineinfo>
)

# Full-move simulator: the scalar engine's frame step (full_combat_core.hpp)
# compiled for the GPU. No fused multiply-add, so results match the CPU bit for bit.
add_library(t8_v2_full_move STATIC src/full_move_gpu.cu)
target_include_directories(t8_v2_full_move PUBLIC include)
target_compile_features(t8_v2_full_move PUBLIC cxx_std_20)
target_link_libraries(t8_v2_full_move PUBLIC t8_v2_full_combat CUDA::cudart)
set_target_properties(t8_v2_full_move PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION OFF
)
target_compile_options(t8_v2_full_move PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;-lineinfo;--expt-relaxed-constexpr>
)
endif()

if(T8_V2_BUILD_TESTS)
    enable_testing()

    add_executable(t8_v2_parity_tests tests/parity_tests.cpp)
    target_link_libraries(t8_v2_parity_tests PRIVATE t8_v2_cpu_oracle)
    add_test(NAME t8_v2_cpu_contract COMMAND t8_v2_parity_tests)

    add_executable(t8_v2_full_combat_tests tests/full_combat_tests.cpp)
    target_link_libraries(t8_v2_full_combat_tests PRIVATE t8_v2_full_combat)
    add_test(NAME t8_v2_full_combat_oracle COMMAND t8_v2_full_combat_tests)

    add_executable(t8_v2_full_combat_engine_tests tests/full_combat_engine_tests.cpp)
    target_link_libraries(t8_v2_full_combat_engine_tests PRIVATE t8_v2_full_combat)
    add_test(NAME t8_v2_full_combat_engine COMMAND t8_v2_full_combat_engine_tests)

    add_executable(t8_v2_full_combat_binding_tests tests/full_combat_binding_tests.cpp)
    target_link_libraries(t8_v2_full_combat_binding_tests PRIVATE t8_v2_full_combat)
    add_test(NAME t8_v2_full_combat_binding COMMAND t8_v2_full_combat_binding_tests
             ${CMAKE_CURRENT_SOURCE_DIR}/data/generated/full_combat_bindings.csv)

    # Randomized state traces; identical output before and after an engine refactor.
    add_executable(t8_v2_full_combat_binding_report tests/full_combat_binding_report.cpp)
    target_link_libraries(t8_v2_full_combat_binding_report PRIVATE t8_v2_full_combat)

    add_executable(t8_v2_full_combat_trace tests/full_combat_trace.cpp)
    target_link_libraries(t8_v2_full_combat_trace PRIVATE t8_v2_full_combat)

    add_executable(t8_v2_full_combat_route_tests tests/full_combat_route_tests.cpp)
    target_link_libraries(t8_v2_full_combat_route_tests PRIVATE t8_v2_full_combat)
    add_test(NAME t8_v2_full_combat_routes COMMAND t8_v2_full_combat_route_tests
             ${CMAKE_CURRENT_SOURCE_DIR}/data/generated)

    add_executable(t8_v2_roster_tests tests/roster_tests.cpp)
    target_link_libraries(t8_v2_roster_tests PRIVATE t8_v2_roster)
    add_test(NAME t8_v2_roster_catalog COMMAND t8_v2_roster_tests
             ${CMAKE_CURRENT_SOURCE_DIR}/data/generated/opponent_profiles.csv
             ${CMAKE_CURRENT_SOURCE_DIR}/data/generated/character_move_specs.csv)

    add_executable(t8_v2_full_move_catalog_tests tests/full_move_catalog_tests.cpp)
    target_link_libraries(t8_v2_full_move_catalog_tests PRIVATE t8_v2_roster)
    add_test(NAME t8_v2_full_move_catalog COMMAND t8_v2_full_move_catalog_tests
             ${CMAKE_CURRENT_SOURCE_DIR}/data/generated/full_move_catalog.csv)

    add_executable(t8_v2_roster_scheduler_tests tests/roster_scheduler_tests.cpp)
    target_link_libraries(t8_v2_roster_scheduler_tests PRIVATE t8_v2_roster)
    add_test(NAME t8_v2_roster_scheduler COMMAND t8_v2_roster_scheduler_tests)

    if(T8_V2_ENABLE_CUDA)
        add_executable(t8_v2_gpu_parity_tests tests/gpu_parity_tests.cpp)
        target_link_libraries(t8_v2_gpu_parity_tests PRIVATE t8_v2_gpu t8_v2_cpu_oracle)
        add_test(NAME t8_v2_gpu_parity COMMAND t8_v2_gpu_parity_tests)

        add_executable(t8_v2_policy_tests tests/policy_tests.cpp)
        target_link_libraries(t8_v2_policy_tests PRIVATE t8_v2_ppo t8_v2_gpu t8_v2_opponents)
        add_test(NAME t8_v2_gpu_policy COMMAND t8_v2_policy_tests)

        add_executable(t8_v2_screen_observation_tests tests/screen_observation_tests.cpp)
        target_link_libraries(t8_v2_screen_observation_tests PRIVATE t8_v2_gpu t8_v2_opponents)
        add_test(NAME t8_v2_screen_observations COMMAND t8_v2_screen_observation_tests)

        add_executable(t8_v2_rollout_tests tests/rollout_tests.cpp)
        target_link_libraries(t8_v2_rollout_tests PRIVATE t8_v2_ppo t8_v2_gpu)
        add_test(NAME t8_v2_gpu_rollout COMMAND t8_v2_rollout_tests)

        add_executable(t8_v2_full_move_gpu_tests tests/full_move_gpu_tests.cpp)
        target_link_libraries(t8_v2_full_move_gpu_tests PRIVATE t8_v2_full_move)
        add_test(NAME t8_v2_full_move_gpu COMMAND t8_v2_full_move_gpu_tests
                 ${CMAKE_CURRENT_SOURCE_DIR}/data/generated)

        add_executable(t8_v2_normalization_pipeline_tests tests/normalization_pipeline_tests.cpp)
        target_link_libraries(t8_v2_normalization_pipeline_tests PRIVATE t8_v2_full_move t8_v2_ppo CUDA::cudart)
        add_test(NAME t8_v2_normalization_pipeline COMMAND t8_v2_normalization_pipeline_tests
                 ${CMAKE_CURRENT_SOURCE_DIR}/data/generated)

        add_executable(t8_v2_full_roster_smoke tests/full_roster_smoke.cpp)
        target_link_libraries(t8_v2_full_roster_smoke PRIVATE t8_v2_full_move t8_v2_roster CUDA::cudart)
        add_test(NAME t8_v2_full_roster_smoke COMMAND t8_v2_full_roster_smoke ${CMAKE_CURRENT_SOURCE_DIR}/data/generated)

        add_executable(t8_v2_feature_tables_tests tests/feature_tables_tests.cpp)
        target_link_libraries(t8_v2_feature_tables_tests PRIVATE t8_v2_ppo CUDA::cudart)
        add_test(NAME t8_v2_feature_tables COMMAND t8_v2_feature_tables_tests)

        add_executable(t8_v2_ppo_math_tests tests/ppo_math_tests.cpp)
        target_link_libraries(t8_v2_ppo_math_tests PRIVATE t8_v2_ppo)
        add_test(NAME t8_v2_ppo_math COMMAND t8_v2_ppo_math_tests)

    endif()
endif()

if(T8_V2_BUILD_BENCHMARKS)
    add_executable(t8_v2_gpu_benchmark benchmarks/gpu_throughput.cpp)
    target_link_libraries(t8_v2_gpu_benchmark PRIVATE t8_v2_gpu CUDA::cudart)

    add_executable(t8_v2_full_move_benchmark benchmarks/full_move_throughput.cpp)
    target_link_libraries(t8_v2_full_move_benchmark PRIVATE t8_v2_full_move t8_v2_ppo CUDA::cudart)
    target_compile_definitions(t8_v2_full_move_benchmark PRIVATE T8_V2_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

    add_executable(t8_v2_training_benchmark benchmarks/training_throughput.cpp)
    target_link_libraries(t8_v2_training_benchmark PRIVATE
        t8_v2_ppo t8_v2_gpu t8_v2_opponents CUDA::cudart)
endif()

if(T8_V2_ENABLE_CUDA)
    add_executable(t8_v2_full_move_train src/full_move_train.cpp)
    target_link_libraries(t8_v2_full_move_train PRIVATE t8_v2_full_move t8_v2_ppo t8_v2_roster)
    target_compile_definitions(t8_v2_full_move_train PRIVATE T8_V2_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

    add_executable(t8_v2_full_move_inspect src/full_move_inspect.cpp)
    target_link_libraries(t8_v2_full_move_inspect PRIVATE t8_v2_full_move)
    target_compile_definitions(t8_v2_full_move_inspect PRIVATE T8_V2_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
    add_executable(t8_v2_full_move_arena src/full_move_arena.cpp)
    target_link_libraries(t8_v2_full_move_arena PRIVATE t8_v2_full_move t8_v2_ppo)
    target_compile_definitions(t8_v2_full_move_arena PRIVATE T8_V2_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

    add_executable(t8_v2_full_move_feed src/full_move_feed.cpp)
    target_link_libraries(t8_v2_full_move_feed PRIVATE t8_v2_full_combat t8_v2_ppo CUDA::cudart)
    target_compile_definitions(t8_v2_full_move_feed PRIVATE T8_V2_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

    # One-click launcher, copied to the repository root: double-click T8Train to train.
    add_executable(T8Train src/train_launcher.cpp)
    target_compile_features(T8Train PRIVATE cxx_std_20)
    add_dependencies(T8Train t8_v2_full_move_train t8_v2_train)
    add_custom_command(TARGET T8Train POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:T8Train> ${CMAKE_CURRENT_SOURCE_DIR}/$<TARGET_FILE_NAME:T8Train>
        COMMENT "Copying T8Train to the repository root")

    add_executable(t8_v2_train src/train.cpp)
    target_link_libraries(t8_v2_train PRIVATE t8_v2_ppo t8_v2_gpu t8_v2_opponents t8_v2_roster)
    # Last-resort location of data/generated for builds outside the source tree.
    target_compile_definitions(t8_v2_train PRIVATE T8_V2_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
endif()

if(T8_V2_BUILD_TESTS)
    # Prefer the repository virtual environment (Windows or POSIX layout).
    if(WIN32 AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.venv/Scripts/python.exe")
        set(Python3_EXECUTABLE "${CMAKE_CURRENT_SOURCE_DIR}/.venv/Scripts/python.exe")
    elseif(NOT WIN32 AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.venv/bin/python")
        set(Python3_EXECUTABLE "${CMAKE_CURRENT_SOURCE_DIR}/.venv/bin/python")
    endif()
    find_package(Python3 COMPONENTS Interpreter)
    if(Python3_Interpreter_FOUND)
        execute_process(COMMAND ${Python3_EXECUTABLE} -c "import pytest, numpy, yaml"
                        RESULT_VARIABLE T8_V2_PYTEST_MISSING OUTPUT_QUIET ERROR_QUIET)
        if(T8_V2_PYTEST_MISSING)
            message(STATUS "pytest, numpy, or pyyaml not found for ${Python3_EXECUTABLE}: skipping t8_v2_live_policy")
        endif()
        if(T8_V2_ENABLE_CUDA)
            add_test(
                NAME t8_v2_exact_resume
                COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/resume_smoke.py
                        $<TARGET_FILE:t8_v2_train>
            )
            set_tests_properties(t8_v2_exact_resume PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
            add_test(
                NAME t8_v2_full_move_resume
                COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/full_move_resume_smoke.py
                        $<TARGET_FILE:t8_v2_full_move_train>
            )
            set_tests_properties(t8_v2_full_move_resume PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
        endif()
        if(NOT T8_V2_PYTEST_MISSING)
        add_test(
            NAME t8_v2_live_policy
            COMMAND ${CMAKE_COMMAND} -E env
                    PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/src
                    ${Python3_EXECUTABLE} -m pytest -q
                    -p no:cacheprovider --basetemp=${CMAKE_CURRENT_BINARY_DIR}/pytest-live-policy
                    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_v2_live_policy.py
                    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_live_backends.py
                    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_live_runtime.py
                    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_vision.py
                    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_roster_system.py
                    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_full_move_catalog.py
                    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_patch_update.py
                    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_training_cli.py
                    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_full_move_live.py
                    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_run_report.py
                    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_contract_fixtures.py
        )
        endif()
        if(T8_V2_ENABLE_CUDA)
            # The visualizer draws a fight without a window.
            add_test(
                NAME t8_v2_visualizer_headless
                COMMAND ${Python3_EXECUTABLE}
                        ${CMAKE_CURRENT_SOURCE_DIR}/scripts/visualizer.py
                        --headless-frames 120 --character jun --opponent king --seed 1
            )
        endif()
    endif()
endif()

# Tests that need an NVIDIA GPU at run time; CPU-only machines (CI) run `ctest -LE gpu`.
if(T8_V2_BUILD_TESTS AND T8_V2_ENABLE_CUDA)
    set(T8_V2_GPU_TESTS
        t8_v2_gpu_parity t8_v2_gpu_policy t8_v2_screen_observations t8_v2_gpu_rollout t8_v2_ppo_math
        t8_v2_full_move_gpu t8_v2_normalization_pipeline t8_v2_full_roster_smoke
        t8_v2_feature_tables)
    if(TARGET t8_v2_train AND Python3_Interpreter_FOUND)
        list(APPEND T8_V2_GPU_TESTS t8_v2_exact_resume t8_v2_full_move_resume t8_v2_visualizer_headless)
    endif()
    set_tests_properties(${T8_V2_GPU_TESTS} PROPERTIES LABELS gpu)
endif()

function(t8_v2_enable_warnings target)
    if(MSVC)
        target_compile_options(${target} PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:/W4;/permissive->
            # nvcc 13.1 emits C stub declarations that trigger MSVC C4211;
            # suppress that generated-code warning while retaining /W4.
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd4211>
        )
    else()
        target_compile_options(${target} PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
            # Not -Wpedantic for CUDA: nvcc's generated host code uses GCC line directives.
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>
        )
    endif()
endfunction()

set(T8_V2_WARNING_TARGETS t8_v2_cpu_oracle t8_v2_full_combat t8_v2_roster)
if(T8_V2_ENABLE_CUDA)
    list(APPEND T8_V2_WARNING_TARGETS
        t8_v2_gpu t8_v2_ppo t8_v2_opponents t8_v2_train t8_v2_full_move
        t8_v2_full_move_train t8_v2_full_move_arena t8_v2_full_move_feed T8Train)
endif()
if(T8_V2_BUILD_TESTS)
    list(APPEND T8_V2_WARNING_TARGETS
        t8_v2_parity_tests t8_v2_full_combat_tests t8_v2_roster_tests t8_v2_full_move_catalog_tests
        t8_v2_full_combat_engine_tests t8_v2_full_combat_binding_tests t8_v2_full_combat_route_tests
        t8_v2_roster_scheduler_tests t8_v2_full_combat_trace t8_v2_full_combat_binding_report)
    if(T8_V2_ENABLE_CUDA)
        list(APPEND T8_V2_WARNING_TARGETS
            t8_v2_gpu_parity_tests t8_v2_policy_tests t8_v2_rollout_tests t8_v2_ppo_math_tests
            t8_v2_screen_observation_tests t8_v2_full_move_gpu_tests t8_v2_normalization_pipeline_tests
            t8_v2_full_roster_smoke t8_v2_feature_tables_tests)
    endif()
endif()
if(T8_V2_BUILD_BENCHMARKS)
    list(APPEND T8_V2_WARNING_TARGETS
        t8_v2_gpu_benchmark t8_v2_training_benchmark t8_v2_full_move_benchmark)
endif()
foreach(target IN LISTS T8_V2_WARNING_TARGETS)
    t8_v2_enable_warnings(${target})
endforeach()
