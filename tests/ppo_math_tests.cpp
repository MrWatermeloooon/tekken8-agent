#include "t8_v2/ppo.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, std::string_view message) {
    if (!condition) {
        ++failures;
        std::cerr << "FAIL: " << message << '\n';
    }
}

double objective(
    const std::vector<float>& output,
    const std::vector<std::uint8_t>& mask,
    std::int64_t selected,
    float old_log_probability,
    float old_value,
    float advantage,
    float return_value,
    const t8::v2::PpoUpdateConfig& config) {
    const std::size_t action_count = mask.size();
    double maximum = -1e300;
    for (std::size_t action = 0; action < action_count; ++action) {
        if (mask[action]) maximum = std::max(maximum, static_cast<double>(output[action]));
    }
    double sum = 0.0;
    for (std::size_t action = 0; action < action_count; ++action) {
        if (mask[action]) sum += std::exp(static_cast<double>(output[action]) - maximum);
    }
    const double log_sum = std::log(sum) + maximum;
    const double new_log_probability = output[static_cast<std::size_t>(selected)] - log_sum;
    const double ratio = std::exp(new_log_probability - old_log_probability);
    const double clipped_ratio = std::clamp(
        ratio, 1.0 - static_cast<double>(config.clip_range),
        1.0 + static_cast<double>(config.clip_range));
    const double policy_loss = -std::min(ratio * advantage, clipped_ratio * advantage);

    double entropy = 0.0;
    for (std::size_t action = 0; action < action_count; ++action) {
        if (!mask[action]) continue;
        const double log_probability = output[action] - log_sum;
        const double probability = std::exp(log_probability);
        entropy -= probability * log_probability;
    }

    const double value = output[action_count];
    const double value_error = value - return_value;
    double value_loss = 0.5 * value_error * value_error;
    if (config.value_clip_range > 0.0F) {
        const double clipped_value = old_value + std::clamp(
            value - old_value,
            -static_cast<double>(config.value_clip_range),
            static_cast<double>(config.value_clip_range));
        const double clipped_error = clipped_value - return_value;
        value_loss = std::max(value_loss, 0.5 * clipped_error * clipped_error);
    }
    return policy_loss + config.value_coefficient * value_loss -
        config.entropy_coefficient * entropy;
}

void test_cuda_gradient_matches_finite_difference() {
    constexpr std::size_t actions = 6;
    std::vector<float> output = {-0.20F, 0.12F, -0.04F, 0.31F, 0.08F, -0.11F, 0.15F};
    std::vector<std::uint8_t> mask = {1, 1, 0, 1, 1, 1};
    constexpr std::int64_t selected = 3;
    t8::v2::PpoUpdateConfig config{};
    config.entropy_coefficient = 0.017F;
    config.value_coefficient = 0.5F;
    config.clip_range = 0.2F;
    config.value_clip_range = 0.2F;

    double maximum = -1e300;
    for (std::size_t action = 0; action < actions; ++action)
        if (mask[action]) maximum = std::max(maximum, static_cast<double>(output[action]));
    double sum = 0.0;
    for (std::size_t action = 0; action < actions; ++action)
        if (mask[action]) sum += std::exp(static_cast<double>(output[action]) - maximum);
    const float old_log_probability = static_cast<float>(
        output[selected] - (std::log(sum) + maximum) - 0.03);
    constexpr float old_value = 0.10F;
    constexpr float advantage = 0.70F;
    constexpr float return_value = 0.35F;
    const auto cuda_gradient = t8::v2::debug_ppo_objective_gradient(
        output, mask, selected, old_log_probability, old_value,
        advantage, return_value, config);

    constexpr float epsilon = 1e-3F;
    for (std::size_t index = 0; index < output.size(); ++index) {
        auto plus = output;
        auto minus = output;
        plus[index] += epsilon;
        minus[index] -= epsilon;
        const double numerical = (
            objective(plus, mask, selected, old_log_probability, old_value,
                      advantage, return_value, config) -
            objective(minus, mask, selected, old_log_probability, old_value,
                      advantage, return_value, config)) / (2.0 * epsilon);
        check(std::abs(cuda_gradient[index] - numerical) < 2e-3,
              "CUDA PPO output gradient matches finite difference");
    }
    check(cuda_gradient[2] == 0.0F, "masked action has exactly zero gradient");

    config.learning_rate = std::numeric_limits<float>::quiet_NaN();
    bool rejected_non_finite = false;
    try {
        static_cast<void>(t8::v2::debug_ppo_objective_gradient(
            output, mask, selected, old_log_probability, old_value,
            advantage, return_value, config));
    } catch (const std::invalid_argument&) {
        rejected_non_finite = true;
    }
    check(rejected_non_finite, "PPO rejects non-finite optimizer coefficients");
}

}  // namespace

int main() {
    test_cuda_gradient_matches_finite_difference();
    if (failures != 0) return EXIT_FAILURE;
    std::cout << "CUDA PPO objective gradient matches finite differences\n";
    return EXIT_SUCCESS;
}
