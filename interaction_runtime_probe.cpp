#include "interaction_runtime.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {

constexpr float kDt = 1.0F / 25.0F;
constexpr int kMaximumPickupUpdates = 1200;
constexpr int kCarryUpdates = 50;
constexpr float kForwardSpeedMps = 0.50F;

vec3 read_vec3(const std::vector<float>& values, size_t index) {
    const size_t offset = index * 3U;
    return vec3(
        values.at(offset),
        values.at(offset + 1U),
        values.at(offset + 2U));
}

quat read_quat(const std::vector<float>& values, size_t index) {
    const size_t offset = index * 4U;
    return quat(
        values.at(offset),
        values.at(offset + 1U),
        values.at(offset + 2U),
        values.at(offset + 3U));
}

int32_t first_phase_frame(
    const interaction::Database& database,
    uint32_t clip,
    interaction::Phase phase) {
    const int32_t start = database.range_starts.at(clip);
    const int32_t stop = database.range_stops.at(clip);
    for (int32_t frame = start; frame < stop; ++frame) {
        if (database.phases.at(static_cast<size_t>(frame)) ==
            static_cast<uint8_t>(phase)) {
            return frame;
        }
    }
    throw std::invalid_argument("clip does not contain required phase");
}

interaction::LocomotionSnapshot locomotion_at_entry(
    const interaction::Database& database,
    uint32_t clip,
    int32_t entry) {
    interaction::LocomotionSnapshot locomotion{};
    locomotion.pose = interaction::pose_at_frame(database, entry);
    constexpr std::array<int32_t, 3> kFutureOffsets = {8, 17, 25};
    const int32_t stop = database.range_stops.at(clip);
    for (size_t sample = 0; sample < kFutureOffsets.size(); ++sample) {
        const interaction::Pose future = interaction::pose_at_frame(
            database,
            std::min(entry + kFutureOffsets[sample], stop - 1));
        locomotion.future_root_positions[sample] =
            future.positions[g1_skeleton::Simulation];
        locomotion.future_root_rotations[sample] =
            future.rotations[g1_skeleton::Simulation];
    }
    return locomotion;
}

interaction::InteractionTarget demo_target(
    const interaction::Database& database,
    uint32_t clip,
    int32_t stable_object_frame) {
    const size_t clip_index = static_cast<size_t>(clip);
    const size_t frame_index = static_cast<size_t>(stable_object_frame);
    interaction::InteractionTarget target{};
    target.handle = {1U, 1U};
    target.object_world = {
        read_vec3(database.object_positions, frame_index),
        read_quat(database.object_rotations, frame_index),
    };
    target.object_profile_id = 1U;
    target.object_dimensions = read_vec3(
        database.object_dimensions, clip_index);
    target.object_bounds = {
        vec3(), target.object_dimensions * 0.5F};
    target.table_world = {
        read_vec3(database.table_positions, clip_index),
        read_quat(database.table_rotations, clip_index),
    };
    target.table_size = read_vec3(database.table_sizes, clip_index);
    target.affordances = {{
        1U,
        database.active_hands.at(clip_index) == 0U
            ? interaction::Hand::Left
            : interaction::Hand::Right,
        {
            read_vec3(database.grasp_positions_object, clip_index),
            read_quat(database.grasp_rotations_object, clip_index),
        },
        read_vec3(database.approach_directions_object, clip_index),
        0.04F,
    }};
    return target;
}

interaction::RuntimeInput runtime_input(
    const interaction::LocomotionSnapshot& locomotion) {
    interaction::RuntimeInput input{};
    input.dt = kDt;
    input.locomotion = locomotion;
    return input;
}

std::string_view state_name(interaction::RuntimeState state) {
    using interaction::RuntimeState;
    switch (state) {
    case RuntimeState::Disabled: return "Disabled";
    case RuntimeState::Locomotion: return "Locomotion";
    case RuntimeState::Preflight: return "Preflight";
    case RuntimeState::Align: return "Align";
    case RuntimeState::PickupReplay: return "PickupReplay";
    case RuntimeState::Hold: return "Hold";
    case RuntimeState::Carry: return "Carry";
    }
    throw std::logic_error("invalid runtime state");
}

std::string_view result_name(interaction::ResultCode result) {
    using interaction::ResultCode;
    switch (result) {
    case ResultCode::None: return "None";
    case ResultCode::Accepted: return "Accepted";
    case ResultCode::Succeeded: return "Succeeded";
    case ResultCode::Rejected: return "Rejected";
    case ResultCode::Cancelled: return "Cancelled";
    case ResultCode::Failed: return "Failed";
    case ResultCode::Reset: return "Reset";
    }
    throw std::logic_error("invalid result code");
}

std::string_view reason_name(interaction::Reason reason) {
    using interaction::Reason;
    switch (reason) {
    case Reason::None: return "None";
    case Reason::PackUnavailable: return "PackUnavailable";
    case Reason::TargetUnavailable: return "TargetUnavailable";
    case Reason::TargetChanged: return "TargetChanged";
    case Reason::OutOfRange: return "OutOfRange";
    case Reason::NoCandidate: return "NoCandidate";
    case Reason::PoorMatch: return "PoorMatch";
    case Reason::BlockedPath: return "BlockedPath";
    case Reason::CorrectionLimit: return "CorrectionLimit";
    case Reason::Cancelled: return "Cancelled";
    case Reason::ContactPosition: return "ContactPosition";
    case Reason::ContactOrientation: return "ContactOrientation";
    case Reason::JointLimit: return "JointLimit";
    case Reason::LostContact: return "LostContact";
    case Reason::ClipEnded: return "ClipEnded";
    case Reason::Reset: return "Reset";
    case Reason::SurfaceUnavailable: return "SurfaceUnavailable";
    case Reason::SurfaceChanged: return "SurfaceChanged";
    case Reason::PlacementOutOfBounds: return "PlacementOutOfBounds";
    case Reason::ReleasePosition: return "ReleasePosition";
    case Reason::ReleaseOrientation: return "ReleaseOrientation";
    }
    throw std::logic_error("invalid reason");
}

void append_state(
    std::vector<interaction::RuntimeState>& states,
    interaction::RuntimeState state) {
    if (states.empty() || states.back() != state) states.push_back(state);
}

float planar_displacement(vec3 first, vec3 second) {
    return std::hypot(second.x - first.x, second.z - first.z);
}

void update_forward_locomotion(
    interaction::LocomotionSnapshot& locomotion,
    const interaction::Pose& previous_output) {
    locomotion.pose = previous_output;
    const size_t root = g1_skeleton::Simulation;
    locomotion.pose.positions[root].z += kForwardSpeedMps * kDt;
    constexpr std::array<float, 3> kFutureSeconds = {
        0.32F, 0.68F, 1.00F};
    for (size_t sample = 0; sample < kFutureSeconds.size(); ++sample) {
        locomotion.future_root_positions[sample] =
            locomotion.pose.positions[root] +
            vec3(0.0F, 0.0F, kForwardSpeedMps * kFutureSeconds[sample]);
        locomotion.future_root_rotations[sample] =
            locomotion.pose.rotations[root];
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3 || std::string_view(argv[2]) != "--json") {
        std::cerr << "usage: interaction_runtime_probe <pack> --json\n";
        return 2;
    }

    try {
        const std::filesystem::path pack = argv[1];
        const interaction::Database database = interaction::load_database(
            pack / "interaction_database.bin");
        const interaction::Features features = interaction::load_features(
            pack / "interaction_features.bin");
        if (database.frame_count != features.frame_count) {
            throw interaction::FormatError(
                "feature/database frame count mismatch");
        }

        constexpr uint32_t kDemoClip = 0U;
        const int32_t entry = first_phase_frame(
            database, kDemoClip, interaction::Phase::Reach);
        const int32_t contact = first_phase_frame(
            database, kDemoClip, interaction::Phase::Contact);
        interaction::LocomotionSnapshot locomotion = locomotion_at_entry(
            database, kDemoClip, entry);
        interaction::TargetRegistry registry;
        registry.upsert(demo_target(
            database, kDemoClip, contact - 1));
        const interaction::RuntimeConfig config{};
        const vec3 entry_root = locomotion.pose.positions[
            g1_skeleton::Simulation];
        const std::optional<interaction::TargetHandle> resolved =
            registry.resolve_single_target(
                entry_root, config.matcher.maximum_approach_m);
        if (!resolved.has_value()) {
            throw std::runtime_error("demo target is not uniquely resolvable");
        }
        const interaction::InteractionTarget* target = registry.find(
            *resolved);
        if (target == nullptr || target->affordances.size() != 1U) {
            throw std::runtime_error(
                "demo target must have exactly one affordance");
        }
        const interaction::PickRequest request{
            *resolved, target->affordances.front().id, 1U};

        interaction::InteractionRuntime runtime(
            database, features, registry, config);
        std::vector<interaction::RuntimeState> states{
            interaction::RuntimeState::Locomotion};
        interaction::RuntimeInput input = runtime_input(locomotion);
        input.interact_pressed = true;
        input.pick_request = request;
        interaction::RuntimeOutput output = runtime.update(input);
        append_state(states, output.diagnostics.state);

        int32_t attach_frame = -1;
        int32_t selected_clip = -1;
        float held_seconds = 0.0F;
        const float lift_threshold =
            target->object_world.position.y +
            config.attachment.required_lift_m;
        for (int update = 0;
             update < kMaximumPickupUpdates &&
             output.diagnostics.state != interaction::RuntimeState::Carry;
             ++update) {
            output = runtime.update(runtime_input(locomotion));
            append_state(states, output.diagnostics.state);
            if (selected_clip < 0 && output.diagnostics.clip >= 0) {
                selected_clip = output.diagnostics.clip;
            }
            if (attach_frame < 0 && output.diagnostics.attached) {
                attach_frame = output.diagnostics.frame;
            }
            if (output.diagnostics.attached &&
                output.object_world.position.y >= lift_threshold) {
                held_seconds = std::min(
                    config.attachment.required_hold_seconds,
                    held_seconds + kDt);
            } else if (!output.diagnostics.attached ||
                       output.object_world.position.y < lift_threshold) {
                held_seconds = 0.0F;
            }
        }
        if (output.diagnostics.state != interaction::RuntimeState::Carry) {
            throw std::runtime_error(
                "pickup did not reach Carry: " +
                std::string(reason_name(output.diagnostics.reason)));
        }

        const vec3 carry_start = output.pose.positions[
            g1_skeleton::Simulation];
        locomotion.pose = output.pose;
        for (int update = 0; update < kCarryUpdates; ++update) {
            update_forward_locomotion(locomotion, output.pose);
            output = runtime.update(runtime_input(locomotion));
            append_state(states, output.diagnostics.state);
        }
        const vec3 carry_stop = output.pose.positions[
            g1_skeleton::Simulation];
        const float carry_displacement = planar_displacement(
            carry_start, carry_stop);

        std::cout << std::setprecision(9)
                  << "{\"attach_frame\":" << attach_frame
                  << ",\"attached\":"
                  << (output.diagnostics.attached ? "true" : "false")
                  << ",\"carry_mode\":\""
                  << (output.diagnostics.recorded_carry
                          ? "recorded"
                          : "layered")
                  << "\",\"carry_root_displacement_m\":"
                  << carry_displacement
                  << ",\"final_reason\":\""
                  << reason_name(output.diagnostics.reason)
                  << "\",\"final_result\":\""
                  << result_name(output.diagnostics.result)
                  << "\",\"held_time_seconds\":" << held_seconds
                  << ",\"selected_clip\":" << selected_clip
                  << ",\"state_sequence\":[";
        for (size_t index = 0; index < states.size(); ++index) {
            if (index != 0U) std::cout << ',';
            std::cout << '\"' << state_name(states[index]) << '\"';
        }
        std::cout << "]}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "interaction_runtime_probe: " << error.what() << '\n';
        return 1;
    }
}
