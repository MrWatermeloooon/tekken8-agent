cmake_minimum_required(VERSION 3.24)

project(tekken_lite_v2 LANGUAGES CXX)

option(T8_V2_BUILD_TESTS "Build V1 parity tests" ON)
option(T8_V2_BUILD_BENCHMARKS "Build CPU throughput benchmarks" ON)

add_library(t8_v2_sim STATIC
    src/sim.cpp
)
target_include_directories(t8_v2_sim PUBLIC include)
target_compile_features(t8_v2_sim PUBLIC cxx_std_20)

if(MSVC)
    target_compile_options(t8_v2_sim PRIVATE /W4 /permissive-)
else()
    target_compile_options(t8_v2_sim PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(T8_V2_BUILD_TESTS)
    enable_testing()
    add_executable(t8_v2_parity_tests tests/parity_tests.cpp)
    target_link_libraries(t8_v2_parity_tests PRIVATE t8_v2_sim)
    add_test(NAME t8_v2_parity COMMAND t8_v2_parity_tests)
endif()

if(T8_V2_BUILD_BENCHMARKS)
    add_executable(t8_v2_benchmark benchmarks/cpu_throughput.cpp)
    target_link_libraries(t8_v2_benchmark PRIVATE t8_v2_sim)
endif()
