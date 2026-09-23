#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>

#if defined(__CUDACC__)
#define T8_HOST_DEVICE __host__ __device__
#else
#define T8_HOST_DEVICE
#endif

namespace t8::v2 {

// Screen-only observation contract ("screen-matchup-95-v1").
//
// Every feature is something the live screen pipeline can measure: HUD
// health, fighter screen positions (and distance/velocity derived from them),
// health-drop hit events, and two visual detections per fighter: "activity"
// (the fighter is animating) and "attack cue" (an attack is visibly coming
// out). Nothing is derived from the hidden move ID, hit level, stance, frame
// data, or remaining recovery.
//
// Measurement error is simulated: each environment lane has its own position
// noise level and event-detection error rate (domain randomization), applied
// as Gaussian position/health noise and independent detection flips. The same
// two levels are given to the policy as inputs, normalized by the scales
// below. Live play supplies measured values instead (config/live_screen.yaml,
// see scripts/calibrate_screen_uncertainty.py).
struct ScreenObservationNoise {
    float position_sigma_min = 0.0F;  // stage units (stage is 7.2 wide)
    float position_sigma_max = 1.0F;
    float health_sigma = 0.005F;      // fraction of max health
    float event_error_min = 0.0F;     // probability a detection is flipped
    float event_error_max = 0.30F;
    std::uint32_t salt = 0x5C1EE7U;
};

inline constexpr float kScreenPositionSigmaScale = 1.0F;
inline constexpr float kScreenEventErrorScale = 0.5F;
inline constexpr float kScreenActivitySpeed = 0.01F;  // stage units per decision

T8_HOST_DEVICE inline std::uint64_t screen_mix(std::uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

// Uniform in (0, 1), from the top 24 bits.
T8_HOST_DEVICE inline float screen_uniform(std::uint64_t key) {
    return (static_cast<float>(screen_mix(key) >> 40U) + 0.5F) * (1.0F / 16777216.0F);
}

T8_HOST_DEVICE inline float screen_gaussian(std::uint64_t key) {
    const float radius = sqrtf(-2.0F * logf(screen_uniform(key)));
    return radius * cosf(6.283185307F * screen_uniform(key ^ 0xa5a5a5a5a5a5a5a5ULL));
}

T8_HOST_DEVICE inline std::uint32_t screen_float_bits(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

T8_HOST_DEVICE inline float screen_lane_position_sigma(const ScreenObservationNoise& noise, std::uint64_t lane) {
    const float unit = screen_uniform((static_cast<std::uint64_t>(noise.salt) << 32U) ^ (lane * 2ULL));
    return noise.position_sigma_min + (noise.position_sigma_max - noise.position_sigma_min) * unit;
}

T8_HOST_DEVICE inline float screen_lane_event_error(const ScreenObservationNoise& noise, std::uint64_t lane) {
    const float unit = screen_uniform((static_cast<std::uint64_t>(noise.salt) << 32U) ^ (lane * 2ULL + 1ULL));
    return noise.event_error_min + (noise.event_error_max - noise.event_error_min) * unit;
}

// Key for one measurement of one fighter (1 or 2) in one simulator state. The
// state is identified by lane, frame, and both fighters' exact positions and
// health, so re-deriving the previous decision's measurement (for velocity)
// reproduces it exactly and different episodes of a lane differ.
T8_HOST_DEVICE inline std::uint64_t screen_measurement_key(
    const ScreenObservationNoise& noise,
    std::uint64_t lane,
    int frame,
    float p1_x,
    float p2_x,
    float p1_health,
    float p2_health,
    int fighter,
    int channel) {
    std::uint64_t key = screen_mix((static_cast<std::uint64_t>(noise.salt) << 32U) ^ lane);
    key = screen_mix(key ^ static_cast<std::uint32_t>(frame));
    key = screen_mix(key ^ ((static_cast<std::uint64_t>(screen_float_bits(p1_x)) << 32U) |
                            screen_float_bits(p2_x)));
    key = screen_mix(key ^ ((static_cast<std::uint64_t>(screen_float_bits(p1_health)) << 32U) |
                            screen_float_bits(p2_health)));
    return key ^ (static_cast<std::uint64_t>(fighter) << 8U) ^ static_cast<std::uint64_t>(channel);
}

T8_HOST_DEVICE inline float screen_normalized_position_sigma(float sigma) {
    const float value = sigma / kScreenPositionSigmaScale;
    return value < 0.0F ? 0.0F : (value > 1.0F ? 1.0F : value);
}

T8_HOST_DEVICE inline float screen_normalized_event_error(float error) {
    const float value = error / kScreenEventErrorScale;
    return value < 0.0F ? 0.0F : (value > 1.0F ? 1.0F : value);
}

// Channels for screen_measurement_key.
enum ScreenChannel : int {
    ScreenChannelX = 0,
    ScreenChannelHealth = 1,
    ScreenChannelActivity = 2,
    ScreenChannelAttackCue = 3,
};

}  // namespace t8::v2
