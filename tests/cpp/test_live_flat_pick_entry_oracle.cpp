#include "database.h"
#include "interaction_controller_adapter.h"
#include "interaction_pick_assist.h"
#include "interaction_pick_approach.h"
#include "locomotion_controller_update.h"
#include "stationary_motion_matching.h"
#include "tests/cpp/pick_entry_oracle_roots.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iostream>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace interaction {

struct InteractionRuntimeTestAccess {
    static const std::optional<MatchCandidate>& candidate(
        const InteractionRuntime& runtime) {
        return runtime.candidate_;
    }

    static matcher_detail::PickEvaluationInput pick_evaluation_input(
        const InteractionRuntime& runtime,
        const LocomotionSnapshot& locomotion,
        TargetHandle target,
        uint32_t affordance_id) {
        const InteractionRuntime::PickEvaluationBuild built =
            runtime.build_pick_evaluation(
                locomotion, target, affordance_id, true);
        if (!built.accepted) {
            throw std::runtime_error(
                "live-flat oracle could not build a pick evaluation");
        }
        return built.input;
    }
};

}  // namespace interaction

namespace {

constexpr size_t kExpectedStationaryRows = 186U;
constexpr size_t kRootCount = 3U;
constexpr size_t kWorkerCount = 64U;
constexpr size_t kMaximumExecutionCases = 96U;
constexpr int kMaximumRuntimeTicks = 600;
constexpr int32_t kExpectedLongApproachEntry = 11;
constexpr int32_t kExpectedDirectReachEntry = 114;
constexpr int32_t kExpectedContactFrame = 139;
constexpr uint64_t kRequestId = 1U;
constexpr float kThirtyDegreesRadians =
    3.14159265358979323846F / 6.0F;
constexpr std::array<int, 4> kStationarySpanEndpoints{0, 92, 118, 210};

static_assert(
    interaction::kControllerStepSeconds == 1.0F / 25.0F,
    "live-flat oracle must use the native 25 Hz controller step");
static_assert(
    interaction::kInteractionRuntimeStepSeconds == 1.0F / 25.0F,
    "live-flat oracle must use the native 25 Hz runtime step");

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error(message);
}

void require(bool condition, const std::string& message) {
    if (!condition) fail(message);
}

const char* root_label(size_t root_index) {
    switch (root_index) {
    case 0U: return "R";
    case 1U: return "Minus";
    case 2U: return "Plus";
    default: fail("live-flat oracle has an invalid root index");
    }
}

std::string case_label(int frame, size_t root_index) {
    return "frame=" + std::to_string(frame) +
        " root=" + root_label(root_index);
}

int containing_clip_stop(const database& flat_database, int frame) {
    for (int range = 0; range < flat_database.nranges(); ++range) {
        if (frame >= flat_database.range_starts(range) &&
            frame < flat_database.range_stops(range)) {
            return flat_database.range_stops(range);
        }
    }
    return -1;
}

void validate_flat_database_shape(const database& flat_database) {
    require(
        flat_database.nframes() > 0,
        "flat database has no frames");
    require(
        flat_database.nbones() ==
            static_cast<int>(interaction::kFlatControllerBoneCount),
        "flat database bone count does not match the controller");
    require(
        flat_database.bone_velocities.rows == flat_database.nframes() &&
            flat_database.bone_velocities.cols == flat_database.nbones() &&
            flat_database.bone_rotations.rows == flat_database.nframes() &&
            flat_database.bone_rotations.cols == flat_database.nbones() &&
            flat_database.bone_angular_velocities.rows ==
                flat_database.nframes() &&
            flat_database.bone_angular_velocities.cols ==
                flat_database.nbones() &&
            flat_database.contact_states.rows == flat_database.nframes() &&
            flat_database.contact_states.cols >= 2,
        "flat database pose arrays have inconsistent shapes");
    require(
        flat_database.bone_parents.size ==
            static_cast<int>(interaction::kFlatControllerBoneCount),
        "flat database parent count does not match the controller");
    for (size_t bone = 0;
         bone < interaction::kFlatControllerBoneCount;
         ++bone) {
        require(
            flat_database.bone_parents(static_cast<int>(bone)) ==
                interaction::kFlatControllerParents[bone],
            "flat database parent tree does not match the controller");
    }
}

interaction::Pose make_interaction_reference(
    const interaction::Database& interaction_database) {
    require(
        interaction_database.clip_count > 0U &&
            !interaction_database.range_starts.empty(),
        "interaction pack has no reference frame");
    interaction::Pose reference = interaction::pose_at_frame(
        interaction_database, interaction_database.range_starts.at(0U));
    reference.velocities.fill(vec3());
    reference.angular_velocities.fill(vec3());
    reference.hand_dof = interaction::kFlatControllerRestHandDof;
    reference.hand_dof_velocities =
        interaction::kFlatControllerRestHandDofVelocities;
    reference.foot_contacts = {};
    return reference;
}

interaction::FlatControllerPose flat_pose_at(
    const database& flat_database,
    int frame) {
    require(
        frame >= 0 && frame < flat_database.nframes(),
        "flat pose frame is out of range");
    interaction::FlatControllerPose pose{};
    for (size_t bone = 0;
         bone < interaction::kFlatControllerBoneCount;
         ++bone) {
        const int index = static_cast<int>(bone);
        pose.positions[bone] = flat_database.bone_positions(frame, index);
        pose.velocities[bone] = flat_database.bone_velocities(frame, index);
        pose.rotations[bone] = flat_database.bone_rotations(frame, index);
        pose.angular_velocities[bone] =
            flat_database.bone_angular_velocities(frame, index);
    }
    pose.foot_contacts[0] =
        flat_database.contact_states(frame, 0) ? 1U : 0U;
    pose.foot_contacts[1] =
        flat_database.contact_states(frame, 1) ? 1U : 0U;
    return pose;
}

struct FixedControllerPoseBridge {
    interaction::FlatControllerPose flat_reference{};
    interaction::Pose interaction_reference{};
};

FixedControllerPoseBridge make_fixed_controller_pose_bridge(
    const database& flat_database,
    const interaction::Database& interaction_database) {
    require(
        flat_database.nranges() > 0,
        "flat database has no fixed controller reference frame");
    FixedControllerPoseBridge bridge{};
    bridge.flat_reference = flat_pose_at(
        flat_database, flat_database.range_starts(0));
    bridge.flat_reference.velocities.fill(vec3());
    bridge.flat_reference.angular_velocities.fill(vec3());
    bridge.flat_reference.foot_contacts = {};

    bridge.interaction_reference =
        make_interaction_reference(interaction_database);
    constexpr size_t root = g1_skeleton::Simulation;
    bridge.interaction_reference.positions[root] =
        bridge.flat_reference.positions[0];
    bridge.interaction_reference.velocities[root] =
        bridge.flat_reference.velocities[0];
    bridge.interaction_reference.rotations[root] =
        bridge.flat_reference.rotations[0];
    bridge.interaction_reference.angular_velocities[root] =
        bridge.flat_reference.angular_velocities[0];
    return bridge;
}

interaction::LocomotionSnapshot make_stationary_snapshot(
    const database& flat_database,
    int frame,
    const FixedControllerPoseBridge& bridge) {
    const int clip_stop = containing_clip_stop(flat_database, frame);
    require(clip_stop > frame, "stationary frame is outside a clip");
    interaction::LocomotionSnapshot snapshot{};
    snapshot.pose = interaction::expand_flat_controller_pose(
        flat_pose_at(flat_database, frame),
        bridge.interaction_reference,
        bridge.flat_reference);
    for (size_t index = 0;
         index < locomotion_timing::kTrajectoryFrameOffsets.size();
         ++index) {
        const int future = frame +
            locomotion_timing::kTrajectoryFrameOffsets[index];
        require(
            future < clip_stop,
            "stationary frame lacks the production trajectory horizon");
        snapshot.future_root_positions[index] =
            flat_database.bone_positions(future, 0);
        snapshot.future_root_rotations[index] =
            flat_database.bone_rotations(future, 0);
    }
    return snapshot;
}

void require_stationary_snapshot_uses_fixed_calibration(
    const database& flat_database,
    const FixedControllerPoseBridge& bridge) {
    const int reference_frame = flat_database.range_starts(0);
    const interaction::FlatControllerPose current =
        flat_pose_at(flat_database, reference_frame);
    const interaction::Pose expected =
        interaction::expand_flat_controller_pose(
            current,
            bridge.interaction_reference,
            bridge.flat_reference);
    const interaction::Pose legacy =
        interaction::expand_flat_controller_pose(
            current, bridge.interaction_reference);
    const interaction::LocomotionSnapshot actual =
        make_stationary_snapshot(
            flat_database, reference_frame, bridge);
    const interaction::WorldPose expected_world =
        interaction::world_pose(expected);
    const interaction::WorldPose legacy_world =
        interaction::world_pose(legacy);
    const interaction::WorldPose actual_world =
        interaction::world_pose(actual.pose);

    float legacy_maximum_error_m = 0.0F;
    float actual_maximum_error_m = 0.0F;
    for (const interaction::FlatControllerAnchor anchor :
         interaction::kFlatControllerAnchors) {
        legacy_maximum_error_m = std::max(
            legacy_maximum_error_m,
            length(
                legacy_world.positions[anchor.g1_bone] -
                expected_world.positions[anchor.g1_bone]));
        actual_maximum_error_m = std::max(
            actual_maximum_error_m,
            length(
                actual_world.positions[anchor.g1_bone] -
                expected_world.positions[anchor.g1_bone]));
    }
    require(
        legacy_maximum_error_m > 0.20F,
        "fixed-reference witness no longer distinguishes the legacy bridge");
    require(
        actual_maximum_error_m <= 1.0e-4F,
        "stationary snapshot retained the legacy absolute-anchor bridge: " +
            std::to_string(actual_maximum_error_m) + " m versus calibrated, " +
            std::to_string(legacy_maximum_error_m) + " m legacy witness");
}

bool same_transform(
    const interaction::Transform& left,
    const interaction::Transform& right) {
    return left.position.x == right.position.x &&
        left.position.y == right.position.y &&
        left.position.z == right.position.z &&
        left.rotation.w == right.rotation.w &&
        left.rotation.x == right.rotation.x &&
        left.rotation.y == right.rotation.y &&
        left.rotation.z == right.rotation.z;
}

bool same_candidate(
    const interaction::MatchCandidate& left,
    const interaction::MatchCandidate& right) {
    return left.clip == right.clip &&
        left.entry_frame == right.entry_frame &&
        left.contact_frame == right.contact_frame &&
        left.lift_frame == right.lift_frame &&
        left.hold_frame == right.hold_frame &&
        same_transform(left.scene_from_source, right.scene_from_source) &&
        left.entry_root_offset.x == right.entry_root_offset.x &&
        left.entry_root_offset.y == right.entry_root_offset.y &&
        left.entry_root_offset.z == right.entry_root_offset.z &&
        left.entry_yaw_offset == right.entry_yaw_offset &&
        left.total_cost == right.total_cost &&
        left.group_costs == right.group_costs;
}

struct ExecutionObservation {
    int32_t entry_frame = -1;
    int32_t contact_frame = -1;
};

ExecutionObservation execute_ready_case(
    interaction::TargetRegistry& registry,
    interaction::TargetHandle target_handle,
    uint32_t affordance_id,
    interaction::InteractionRuntime& runtime,
    const interaction::PickEntryPreview& preview,
    const interaction::LocomotionSnapshot& live_snapshot,
    interaction::PickEntryRoot root,
    int stationary_frame,
    size_t root_index) {
    const std::string label = case_label(stationary_frame, root_index);
    const interaction::InteractionTarget* target = registry.find(target_handle);
    require(target != nullptr, label + " target registration failed");
    require(
        target->affordances.size() == 1U &&
            target->affordances.front().id == affordance_id,
        label + " target affordance count changed");
    require(preview.match_ready, label + " execution preview is not ready");
    require(preview.path_feasible, label + " ready preview lost its path");
    require(
        preview.path_reason == interaction::Reason::None &&
            preview.match_reason == interaction::Reason::None,
        label + " ready preview retained a rejection reason");

    const interaction::runtime_detail::PickSnapshotMap mapped =
        interaction::runtime_detail::map_pick_entry_snapshot(
            live_snapshot, root);
    require(mapped.accepted, label + " entry-root mapping failed");

    interaction::ControllerInteractionScheduler scheduler{};
    int provider_calls = 0;
    int resolver_calls = 0;
    int runtime_calls = 0;
    int submitted_interacts = 0;
    const auto locomotion_provider = [&]() {
        ++provider_calls;
        return mapped.snapshot;
    };
    const auto request_resolver = [&](
        const interaction::LocomotionSnapshot& snapshot)
        -> std::optional<interaction::PickRequest> {
        ++resolver_calls;
        const std::optional<interaction::TargetHandle> resolved =
            registry.resolve_single_target(
                snapshot.pose.positions[g1_skeleton::Simulation],
                interaction::RuntimeConfig{}.matcher.maximum_approach_m);
        require(
            resolved.has_value() && *resolved == target_handle,
            label + " scheduler resolver did not find the target");
        return interaction::PickRequest{
            target_handle, affordance_id, kRequestId};
    };
    const auto runtime_update = [&](const interaction::RuntimeInput& input) {
        ++runtime_calls;
        require(
            input.dt == 1.0F / 25.0F,
            label + " scheduler escaped exact 25 Hz");
        if (input.interact_pressed) ++submitted_interacts;
        return runtime.update(input);
    };

    interaction::ControllerInteractionEdges edges{};
    edges.interact_pressed = true;
    interaction::RuntimeOutput output = scheduler.tick(
        edges, locomotion_provider, request_resolver, runtime_update);
    require(
        output.diagnostics.state == interaction::RuntimeState::Preflight,
        label + " one Interact did not enter Preflight");
    require(
        resolver_calls == 1 && submitted_interacts == 1,
        label + " scheduler did not submit exactly one Interact");

    output = scheduler.tick(
        {}, locomotion_provider, request_resolver, runtime_update);
    require(
        output.diagnostics.state == interaction::RuntimeState::Align,
        label + " ready Preflight did not enter Align");
    const std::optional<interaction::MatchCandidate>& committed =
        interaction::InteractionRuntimeTestAccess::candidate(runtime);
    require(committed.has_value(), label + " committed candidate is missing");
    require(
        same_candidate(*committed, preview.match_candidate),
        label + " preview/commit candidate mismatch");

    for (int tick = 0; tick < kMaximumRuntimeTicks; ++tick) {
        if (output.diagnostics.state == interaction::RuntimeState::Carry) {
            break;
        }
        output = scheduler.tick(
            {}, locomotion_provider, request_resolver, runtime_update);
        require(
            output.diagnostics.result != interaction::ResultCode::Rejected &&
                output.diagnostics.result != interaction::ResultCode::Failed,
            label + " failed after a ready preview: result=" +
                std::to_string(
                    static_cast<int>(output.diagnostics.result)) +
                " reason=" +
                std::to_string(
                    static_cast<int>(output.diagnostics.reason)));
    }
    require(
        output.diagnostics.state == interaction::RuntimeState::Carry,
        label + " did not reach Carry within the tick budget");
    require(
        output.diagnostics.result == interaction::ResultCode::Succeeded &&
            output.diagnostics.reason == interaction::Reason::None &&
            output.diagnostics.attached &&
            output.diagnostics.object_state == interaction::ObjectState::Held,
        label + " Carry did not publish successful attachment");
    const interaction::InteractionTarget* held = registry.find(target_handle);
    require(
        held != nullptr && held->state == interaction::ObjectState::Held &&
            held->owner_request == kRequestId,
        label + " registry did not retain the held object");
    require(
        resolver_calls == 1 && submitted_interacts == 1,
        label + " submitted more than one Interact");
    require(
        provider_calls == runtime_calls && runtime_calls >= 3,
        label + " scheduler/runtime cadence diverged");
    return {committed->entry_frame, committed->contact_frame};
}

struct Summary {
    size_t sampled = 0U;
    size_t cases = 0U;
    size_t ready = 0U;
    std::array<size_t, kRootCount> ready_by_root{};
    size_t executed = 0U;
    std::array<size_t, kRootCount> executed_by_root{};
    bool first_present = false;
    int first_frame = -1;
    size_t first_root = 0U;
    int32_t first_entry = -1;
    int32_t first_contact = -1;
    int32_t direct_entry = -1;
};

std::vector<int> select_stationary_rows(
    const std::vector<int>& stationary,
    bool exhaustive) {
    require(
        stationary.size() == kExpectedStationaryRows,
        "stationary candidate count changed: expected=186 actual=" +
            std::to_string(stationary.size()));
    require(
        stationary.front() == 0 && stationary.at(92U) == 92 &&
            stationary.at(93U) == 118 && stationary.back() == 210,
        "stationary candidate spans changed");
    for (size_t index = 0U; index < 93U; ++index) {
        require(
            stationary[index] == static_cast<int>(index) &&
                stationary[93U + index] ==
                    118 + static_cast<int>(index),
            "stationary candidate span is not contiguous");
    }
    if (exhaustive) return stationary;
    return {stationary.front()};
}

struct CaseObservation {
    bool ready = false;
    interaction::MatchCandidate candidate{};
};

CaseObservation preview_case(
    const interaction::Database& interaction_database,
    const interaction::Features& interaction_features,
    const interaction::InteractionTarget& authored_target,
    const interaction::LocomotionSnapshot& snapshot,
    interaction::PickEntryRoot root,
    int stationary_frame,
    size_t root_index) {
    interaction::TargetRegistry registry{};
    const interaction::TargetHandle target_handle =
        registry.upsert(authored_target);
    const interaction::InteractionTarget* target = registry.find(target_handle);
    require(
        target != nullptr && target->affordances.size() == 1U,
        case_label(stationary_frame, root_index) +
            " preview target registration failed");
    const uint32_t affordance_id = target->affordances.front().id;
    interaction::InteractionRuntime runtime(
        interaction_database,
        interaction_features,
        registry,
        interaction::RuntimeConfig{});
    const interaction::PickEntryPreview preview = runtime.preview_pick(
        snapshot, root, target_handle, affordance_id);
    if (!preview.match_ready) return {};
    require(
        preview.path_feasible &&
            preview.path_reason == interaction::Reason::None &&
            preview.match_reason == interaction::Reason::None,
        case_label(stationary_frame, root_index) +
            " ready preview retained a rejection");
    return {true, preview.match_candidate};
}

ExecutionObservation execute_coverage_case(
    const interaction::Database& interaction_database,
    const interaction::Features& interaction_features,
    const interaction::InteractionTarget& authored_target,
    const interaction::LocomotionSnapshot& snapshot,
    interaction::PickEntryRoot root,
    const CaseObservation& certified,
    int stationary_frame,
    size_t root_index) {
    interaction::TargetRegistry registry{};
    const interaction::TargetHandle target_handle =
        registry.upsert(authored_target);
    const interaction::InteractionTarget* target = registry.find(target_handle);
    require(
        target != nullptr && target->affordances.size() == 1U,
        case_label(stationary_frame, root_index) +
            " execution target registration failed");
    const uint32_t affordance_id = target->affordances.front().id;
    interaction::InteractionRuntime runtime(
        interaction_database,
        interaction_features,
        registry,
        interaction::RuntimeConfig{});
    const interaction::PickEntryPreview preview = runtime.preview_pick(
        snapshot, root, target_handle, affordance_id);
    require(
        preview.match_ready &&
            same_candidate(preview.match_candidate, certified.candidate),
        case_label(stationary_frame, root_index) +
            " coverage preview changed after certification");

    const ExecutionObservation execution = execute_ready_case(
        registry,
        target_handle,
        affordance_id,
        runtime,
        preview,
        snapshot,
        root,
        stationary_frame,
        root_index);
    require(
        execution.entry_frame == preview.match_candidate.entry_frame &&
            execution.contact_frame == preview.match_candidate.contact_frame,
        case_label(stationary_frame, root_index) +
            " outer preview/commit candidate mismatch");
    return execution;
}

std::vector<size_t> select_execution_coverage(
    const std::vector<int>& stationary,
    const std::vector<CaseObservation>& observations) {
    const size_t case_count = stationary.size() * kRootCount;
    require(
        observations.size() == case_count,
        "live-flat coverage input size changed");
    std::vector<uint8_t> selected(case_count, 0U);
    const auto select = [&](size_t case_index) {
        require(case_index < case_count, "coverage index is out of range");
        selected[case_index] = 1U;
    };

    for (size_t root_index = 0U; root_index < kRootCount; ++root_index) {
        std::optional<size_t> first;
        std::optional<size_t> last;
        std::set<int32_t> selected_entries;
        for (size_t stationary_index = 0U;
             stationary_index < stationary.size();
             ++stationary_index) {
            const size_t case_index =
                stationary_index * kRootCount + root_index;
            const CaseObservation& observation = observations[case_index];
            if (!observation.ready) continue;
            if (!first.has_value()) first = case_index;
            last = case_index;
            if (selected_entries.insert(
                    observation.candidate.entry_frame).second) {
                select(case_index);
            }
        }
        if (first.has_value()) {
            select(*first);
            select(*last);
        }
    }

    for (int endpoint : kStationarySpanEndpoints) {
        const auto found = std::find(
            stationary.begin(), stationary.end(), endpoint);
        require(
            found != stationary.end(),
            "stationary span endpoint is missing: " +
                std::to_string(endpoint));
        const size_t stationary_index = static_cast<size_t>(
            found - stationary.begin());
        for (size_t root_index = 0U;
             root_index < kRootCount;
             ++root_index) {
            const size_t case_index =
                stationary_index * kRootCount + root_index;
            if (observations[case_index].ready) select(case_index);
        }
    }

    std::vector<size_t> coverage;
    for (size_t case_index = 0U; case_index < selected.size(); ++case_index) {
        if (selected[case_index] != 0U) coverage.push_back(case_index);
    }
    require(
        !coverage.empty(),
        "live-flat oracle selected no execution coverage");
    require(
        coverage.size() <= kMaximumExecutionCases,
        "live-flat execution coverage exceeded its pinned cap");
    for (size_t root_index = 0U; root_index < kRootCount; ++root_index) {
        bool any_ready = false;
        bool any_selected = false;
        for (size_t stationary_index = 0U;
             stationary_index < stationary.size();
             ++stationary_index) {
            const size_t case_index =
                stationary_index * kRootCount + root_index;
            any_ready = any_ready || observations[case_index].ready;
            any_selected = any_selected || selected[case_index] != 0U;
        }
        require(
            !any_ready || any_selected,
            std::string("ready root lacks execution coverage: ") +
                root_label(root_index));
    }
    return coverage;
}

interaction::Phase entry_phase(
    const interaction::Database& database,
    const interaction::MatchCandidate& candidate) {
    require(
        candidate.entry_frame >= 0 &&
            static_cast<size_t>(candidate.entry_frame) <
                database.phases.size(),
        "live-flat candidate entry frame is out of range");
    const uint8_t phase = database.phases.at(
        static_cast<size_t>(candidate.entry_frame));
    require(
        phase <= static_cast<uint8_t>(interaction::Phase::Hold),
        "live-flat candidate entry phase is invalid");
    return static_cast<interaction::Phase>(phase);
}

enum class PositionMatrixRoot : uint8_t { Reach, Plus };

struct PositionMatrixRow {
    const char* label = "";
    float translation_x = 0.0F;
    float translation_y = 0.0F;
    float translation_z = 0.0F;
    float yaw_radians = 0.0F;
    PositionMatrixRoot root = PositionMatrixRoot::Reach;
};

constexpr std::array<PositionMatrixRow, 3> kPositionMatrixRows{{
    {"baseline_reach", 0.0F, 0.0F, 0.0F, 0.0F,
     PositionMatrixRoot::Reach},
    {"positive_plus", 0.60F, 0.0F, -0.40F,
     kThirtyDegreesRadians, PositionMatrixRoot::Plus},
    {"negative_reach", -0.60F, 0.0F, 0.40F,
     -kThirtyDegreesRadians, PositionMatrixRoot::Reach},
}};

bool same_vec3(vec3 left, vec3 right) {
    return left.x == right.x && left.y == right.y && left.z == right.z;
}

void require_local_target_data_unchanged(
    const interaction::InteractionTarget& authored,
    const interaction::InteractionTarget& transformed,
    const std::string& label) {
    require(
        transformed.handle == authored.handle &&
            transformed.object_profile_id == authored.object_profile_id &&
            same_vec3(
                transformed.object_bounds.center_object,
                authored.object_bounds.center_object) &&
            same_vec3(
                transformed.object_bounds.half_extents_object,
                authored.object_bounds.half_extents_object) &&
            same_vec3(
                transformed.object_dimensions,
                authored.object_dimensions) &&
            same_vec3(transformed.table_size, authored.table_size) &&
            transformed.state == authored.state &&
            transformed.owner_request == authored.owner_request &&
            transformed.affordances.size() == authored.affordances.size(),
        label + " changed target dimensions or identity");
    for (size_t index = 0U; index < authored.affordances.size(); ++index) {
        const interaction::GraspAffordance& expected =
            authored.affordances[index];
        const interaction::GraspAffordance& actual =
            transformed.affordances[index];
        require(
            actual.id == expected.id && actual.hand == expected.hand &&
                same_transform(
                    actual.hand_in_object, expected.hand_in_object) &&
                same_vec3(
                    actual.approach_direction_object,
                    expected.approach_direction_object) &&
                actual.clearance_radius == expected.clearance_radius,
            label + " changed a local grasp affordance");
    }
}

interaction::InteractionTarget transform_pick_target(
    const interaction::InteractionTarget& authored,
    const PositionMatrixRow& row) {
    interaction::InteractionTarget transformed = authored;
    if (row.yaw_radians != 0.0F || row.translation_x != 0.0F ||
        row.translation_y != 0.0F || row.translation_z != 0.0F) {
        const interaction::Transform scene_from_authored{
            vec3(
                row.translation_x,
                row.translation_y,
                row.translation_z),
            quat_from_angle_axis(
                row.yaw_radians, vec3(0.0F, 1.0F, 0.0F)),
        };
        transformed.table_world = interaction::compose(
            scene_from_authored, authored.table_world);
        transformed.object_world = interaction::compose(
            scene_from_authored, authored.object_world);
    }
    require_local_target_data_unchanged(authored, transformed, row.label);
    return transformed;
}

interaction::PickEntryRoot position_matrix_root(
    const pick_entry_oracle::OracleRoots& roots,
    PositionMatrixRoot selected) {
    switch (selected) {
    case PositionMatrixRoot::Reach: return roots.reach;
    case PositionMatrixRoot::Plus: return roots.plus;
    }
    fail("pickup position matrix has an invalid root selection");
}

void run_position_matrix_oracle(
    const database& flat_database,
    const interaction::Database& interaction_database,
    const interaction::Features& interaction_features,
    const FixedControllerPoseBridge& bridge) {
    const std::vector<int> authoritative_stationary =
        stationary_motion_matching::derive_candidates(flat_database);
    const std::vector<int> stationary = select_stationary_rows(
        authoritative_stationary, false);
    require(
        stationary.size() == 1U && stationary.front() == 0,
        "pickup position matrix must use stationary frame 0");

    const interaction::LocomotionSnapshot snapshot =
        make_stationary_snapshot(
            flat_database, stationary.front(), bridge);
    const interaction::InteractionTarget authored_target =
        interaction::make_controller_demo_target(interaction_database);

    {
        const pick_entry_oracle::OracleRoots roots =
            pick_entry_oracle::make_oracle_roots(
                interaction_database, authored_target);
        interaction::TargetRegistry registry{};
        const interaction::TargetHandle target_handle =
            registry.upsert(authored_target);
        const interaction::InteractionTarget* target =
            registry.find(target_handle);
        require(
            target != nullptr && target->affordances.size() == 1U,
            "baseline_minus control target registration failed");
        interaction::InteractionRuntime runtime(
            interaction_database,
            interaction_features,
            registry,
            interaction::RuntimeConfig{});
        const interaction::PickEntryPreview control = runtime.preview_pick(
            snapshot,
            roots.minus,
            target_handle,
            target->affordances.front().id);
        require(
            !control.path_feasible && !control.match_ready &&
                control.path_reason == interaction::Reason::BlockedPath &&
                control.match_reason == interaction::Reason::BlockedPath,
            "baseline_minus control was not preview-only BlockedPath");
    }

    for (const PositionMatrixRow& row : kPositionMatrixRows) {
        const interaction::InteractionTarget transformed_target =
            transform_pick_target(authored_target, row);
        const pick_entry_oracle::OracleRoots roots =
            pick_entry_oracle::make_oracle_roots(
                interaction_database, transformed_target);
        const interaction::PickEntryRoot root =
            position_matrix_root(roots, row.root);

        interaction::TargetRegistry registry{};
        const interaction::TargetHandle target_handle =
            registry.upsert(transformed_target);
        const interaction::InteractionTarget* target =
            registry.find(target_handle);
        require(
            target != nullptr && target->affordances.size() == 1U,
            std::string(row.label) + " target registration failed");
        const uint32_t affordance_id = target->affordances.front().id;
        interaction::InteractionRuntime runtime(
            interaction_database,
            interaction_features,
            registry,
            interaction::RuntimeConfig{});
        const interaction::PickEntryPreview preview = runtime.preview_pick(
            snapshot, root, target_handle, affordance_id);
        require(
            preview.path_feasible && preview.match_ready &&
                preview.path_reason == interaction::Reason::None &&
                preview.match_reason == interaction::Reason::None,
            std::string(row.label) + " did not produce a ready preview");

        const size_t root_index = row.root == PositionMatrixRoot::Reach
            ? 0U
            : 2U;
        const ExecutionObservation execution = execute_ready_case(
            registry,
            target_handle,
            affordance_id,
            runtime,
            preview,
            snapshot,
            root,
            stationary.front(),
            root_index);
        require(
            execution.entry_frame == preview.match_candidate.entry_frame &&
                execution.contact_frame ==
                    preview.match_candidate.contact_frame,
            std::string(row.label) +
                " lost full preview/commit candidate agreement");
    }
}

void run_manual_pick_assist_oracle(
    const database& flat_database,
    const interaction::Database& interaction_database,
    const interaction::Features& interaction_features,
    const FixedControllerPoseBridge& bridge) {
    const auto planar_distance = [](vec3 left, vec3 right) {
        return static_cast<float>(std::hypot(
            static_cast<double>(left.x) - static_cast<double>(right.x),
            static_cast<double>(left.z) - static_cast<double>(right.z)));
    };
    const auto planar_yaw = [](quat rotation) {
        const vec3 forward = quat_mul_vec3(
            rotation, vec3(0.0F, 0.0F, 1.0F));
        return std::atan2(forward.x, forward.z);
    };

    const interaction::LocomotionSnapshot base_snapshot =
        make_stationary_snapshot(flat_database, 0, bridge);
    interaction::TargetRegistry registry{};
    const interaction::TargetHandle target_handle = registry.upsert(
        interaction::make_controller_demo_target(interaction_database));
    const interaction::InteractionTarget* target =
        registry.find(target_handle);
    require(
        target != nullptr && target->affordances.size() == 1U,
        "manual assist oracle target registration failed");
    const interaction::GraspAffordance affordance =
        target->affordances.front();
    const interaction::Transform reach_waypoint =
        interaction::make_pick_reach_waypoint(
            interaction_database, *target);
    const interaction::PickEntrySlots slots =
        interaction::make_pick_entry_slots(reach_waypoint, *target);
    const interaction::PickAssistConfig assist_config{};

    vec3 reach_facing = quat_mul_vec3(
        reach_waypoint.rotation, vec3(0.0F, 0.0F, 1.0F));
    reach_facing.y = 0.0F;
    require(
        length(reach_facing) > 1.0e-5F,
        "manual assist Reach waypoint has no planar facing");
    reach_facing = normalize(reach_facing);
    const vec3 common_entry = reach_waypoint.position -
        assist_config.reach_entry_distance_m * reach_facing;
    const float entry_yaw = planar_yaw(reach_waypoint.rotation);
    const float camera_azimuth = std::atan2(
        std::sin(entry_yaw - PIf), std::cos(entry_yaw - PIf));

    const auto snapshot_at = [&](interaction::PickEntryRoot root) {
        const interaction::runtime_detail::PickSnapshotMap mapped =
            interaction::runtime_detail::map_pick_entry_snapshot(
                base_snapshot, root);
        require(mapped.accepted, "manual assist snapshot mapping failed");
        return mapped.snapshot;
    };
    const auto preview_slots = [&](
        interaction::InteractionRuntime& runtime,
        const interaction::LocomotionSnapshot& snapshot) {
        std::array<interaction::PickEntryPreview, 2> previews{};
        for (size_t index = 0U; index < previews.size(); ++index) {
            previews[index] = runtime.preview_pick(
                snapshot,
                slots.ordered[index].prospective_root,
                target_handle,
                affordance.id);
        }
        return previews;
    };

    interaction::InteractionRuntime runtime(
        interaction_database,
        interaction_features,
        registry,
        interaction::RuntimeConfig{});
    const interaction::LocomotionSnapshot common_snapshot = snapshot_at({
        common_entry.x, common_entry.z, entry_yaw});
    const std::array<interaction::PickEntryPreview, 2> common_previews =
        preview_slots(runtime, common_snapshot);
    float minimum_eligible_entry_to_slot_m =
        assist_config.maximum_assisted_path_m + 1.0F;
    bool preview_eligible = false;
    for (size_t index = 0U; index < common_previews.size(); ++index) {
        const interaction::PickEntryPreview& preview =
            common_previews[index];
        const bool eligible = preview.path_feasible && preview.match_ready;
        if (eligible) {
            preview_eligible = true;
            minimum_eligible_entry_to_slot_m = std::min(
                minimum_eligible_entry_to_slot_m,
                planar_distance(
                    common_entry,
                    slots.ordered[index].waypoint.position));
        }
    }
    require(preview_eligible, "manual assist found no preview-eligible slot");

    vec3 away_from_object =
        common_entry - target->object_world.position;
    away_from_object.y = 0.0F;
    require(
        length(away_from_object) > 1.0e-5F,
        "manual assist common entry coincides with the object");
    away_from_object = normalize(away_from_object);
    const float maximum_start_to_entry_m =
        assist_config.maximum_assisted_path_m -
        minimum_eligible_entry_to_slot_m;
    require(
        maximum_start_to_entry_m > 0.0F,
        "manual assist eligible slot consumes the whole route budget");

    vec3 start_position{};
    float root_to_object_m = 0.0F;
    bool start_found = false;
    constexpr uint32_t kStartSamples = 400U;
    for (uint32_t sample = 0U; sample <= kStartSamples; ++sample) {
        const float alpha = static_cast<float>(sample) /
            static_cast<float>(kStartSamples);
        const float start_to_entry_m =
            alpha * maximum_start_to_entry_m;
        const vec3 candidate =
            common_entry + start_to_entry_m * away_from_object;
        const float candidate_object_distance_m = planar_distance(
            candidate, target->object_world.position);
        const float candidate_route_m =
            planar_distance(candidate, common_entry) +
            minimum_eligible_entry_to_slot_m;
        if (start_to_entry_m >
                assist_config.reach_entry_tolerance_m &&
            candidate_object_distance_m > 1.00F &&
            candidate_object_distance_m <= 1.45F &&
            candidate_route_m <=
                assist_config.maximum_assisted_path_m) {
            start_position = candidate;
            root_to_object_m = candidate_object_distance_m;
            start_found = true;
            break;
        }
    }
    require(
        start_found && root_to_object_m > 1.00F &&
            root_to_object_m <= 1.45F,
        "manual assist found no >1.00 m acquisition witness");

    const std::optional<interaction::TargetHandle> acquired_target =
        registry.resolve_single_target(
            start_position, 1.45F);
    require(
        acquired_target.has_value() && *acquired_target == target_handle,
        "manual assist 1.45 m acquisition did not resolve the witness");

    interaction::PickAssistStart start{};
    start.target = *acquired_target;
    start.affordance_id = affordance.id;
    start.hand = affordance.hand;
    start.object_world = target->object_world;
    start.root_world = {
        start_position, reach_waypoint.rotation};
    start.reach_waypoint = reach_waypoint;
    start.slots = slots;
    interaction::ControllerPickAssist assist(assist_config);

    constexpr float velocity_halflife = 0.27F;
    constexpr float rotation_halflife = 0.27F;
    constexpr float forward_speed_mps = 0.9F;
    constexpr float side_speed_mps = 0.6F;
    constexpr float backward_speed_mps = 0.6F;
    constexpr float maximum_tick_displacement_m = 0.10F;
    constexpr uint32_t maximum_assist_ticks = 600U;
    static_assert(
        interaction::kControllerStepSeconds == 1.0F / 25.0F,
        "manual assist driver must remain at native 25 Hz");

    vec3 simulation_position = start_position;
    vec3 simulation_velocity{};
    vec3 simulation_acceleration{};
    quat simulation_rotation = reach_waypoint.rotation;
    quat desired_rotation = simulation_rotation;
    vec3 simulation_angular_velocity{};
    array1d<vec3> obstacles_positions;
    array1d<vec3> obstacles_scales;
    interaction::LocomotionSnapshot live_flat_snapshot = snapshot_at({
        simulation_position.x,
        simulation_position.z,
        planar_yaw(simulation_rotation)});
    uint64_t live_flat_snapshot_fingerprint =
        interaction::runtime_detail::locomotion_snapshot_fingerprint(
            live_flat_snapshot);

    interaction::ControllerInteractionScheduler scheduler{};
    uint32_t provider_calls = 0U;
    uint32_t resolver_calls = 0U;
    uint32_t runtime_calls = 0U;
    uint32_t submission_count = 0U;
    uint64_t next_request_id = kRequestId;
    uint64_t certified_provider_snapshot_fingerprint = 0U;
    const auto scheduler_tick = [&] (
        interaction::ControllerInteractionEdges edges) {
        return interaction::RuntimeOutput(scheduler.tick(
            edges,
            [&]() {
                ++provider_calls;
                certified_provider_snapshot_fingerprint =
                    interaction::runtime_detail::
                        locomotion_snapshot_fingerprint(live_flat_snapshot);
                require(
                    certified_provider_snapshot_fingerprint ==
                        live_flat_snapshot_fingerprint,
                    "manual assist provider did not return the live snapshot");
                return live_flat_snapshot;
            },
            [&](const interaction::LocomotionSnapshot& snapshot)
                -> std::optional<interaction::PickRequest> {
                ++resolver_calls;
                require(
                    interaction::runtime_detail::
                            locomotion_snapshot_fingerprint(snapshot) ==
                        live_flat_snapshot_fingerprint,
                    "manual assist resolver received a different snapshot");
                const std::optional<interaction::PickRequest> request =
                    assist.take_submission(next_request_id);
                if (request.has_value()) {
                    ++next_request_id;
                    ++submission_count;
                }
                return request;
            },
            [](const interaction::LocomotionSnapshot&)
                -> std::optional<interaction::ControllerPlaceTarget> {
                return std::nullopt;
            },
            [](interaction::SurfaceHandle, uint32_t) {
                return interaction::PlaceStagingPreview{};
            },
            [&](const interaction::RuntimeInput& input) {
                ++runtime_calls;
                require(
                    input.dt == 1.0F / 25.0F,
                    "manual assist scheduler escaped exact 25 Hz");
                return runtime.update(input);
            }));
    };

    interaction::RuntimeOutput scheduler_output = scheduler_tick({});
    require(
        scheduler_output.diagnostics.state ==
            interaction::RuntimeState::Locomotion,
        "manual assist scheduler did not prime Locomotion");
    require(assist.begin(start), "manual assist rejected the route witness");
    const interaction::PickAssistDiagnostics& diagnostics =
        assist.diagnostics();
    require(
        *std::min_element(
            diagnostics.slot_route_lengths_m.begin(),
            diagnostics.slot_route_lengths_m.end()) <=
            assist_config.maximum_assisted_path_m,
        "manual assist route witness exceeds the configured cap");

    interaction::PickAssistOutput prior_assist_output{};
    bool prior_stationary_constraint = false;
    bool saw_preview = false;
    bool saw_final_approach = false;
    bool saw_settling = false;
    bool saw_final_preview = false;
    bool certified = false;
    uint32_t observation_count = 0U;
    uint32_t materialized_snapshot_count = 0U;
    uint32_t settled_observation_count = 0U;
    uint32_t stationary_constraint_edges = 0U;
    uint32_t stationary_output_applied_ticks = 0U;

    for (uint32_t assist_tick = 0U;
         assist_tick < maximum_assist_ticks;
         ++assist_tick) {
        const bool activation_tick = assist_tick == 0U;
        vec3 applied_left_stick{};
        vec3 applied_right_stick{};
        bool desired_strafe = false;
        if (!activation_tick &&
            prior_assist_output.override_steering) {
            applied_left_stick = prior_assist_output.left_stick;
            applied_right_stick = prior_assist_output.right_stick;
        }
        if (!activation_tick && prior_assist_output.force_strafe) {
            desired_strafe = true;
        }
        if (!activation_tick &&
            prior_assist_output.stationary_constraint) {
            ++stationary_output_applied_ticks;
        }

        const vec3 desired_velocity = desired_velocity_update(
            applied_left_stick,
            camera_azimuth,
            simulation_rotation,
            forward_speed_mps,
            side_speed_mps,
            backward_speed_mps);
        desired_rotation = desired_rotation_update(
            desired_rotation,
            applied_left_stick,
            applied_right_stick,
            camera_azimuth,
            desired_strafe,
            desired_velocity);
        const vec3 previous_position = simulation_position;
        simulation_positions_update(
            simulation_position,
            simulation_velocity,
            simulation_acceleration,
            desired_velocity,
            velocity_halflife,
            interaction::kControllerStepSeconds,
            obstacles_positions,
            obstacles_scales);
        simulation_rotations_update(
            simulation_rotation,
            simulation_angular_velocity,
            desired_rotation,
            rotation_halflife,
            interaction::kControllerStepSeconds);
        const float tick_displacement_m = planar_distance(
            previous_position, simulation_position);
        require(
            tick_displacement_m <= maximum_tick_displacement_m,
            "manual assist ordinary locomotion teleported the root");

        live_flat_snapshot = snapshot_at({
            simulation_position.x,
            simulation_position.z,
            planar_yaw(simulation_rotation)});
        ++materialized_snapshot_count;
        live_flat_snapshot_fingerprint =
            interaction::runtime_detail::locomotion_snapshot_fingerprint(
                live_flat_snapshot);

        std::optional<std::array<interaction::PickEntryPreview, 2>> previews;
        uint64_t preview_snapshot_fingerprint = 0U;
        if (prior_assist_output.needs_preview) {
            previews = preview_slots(runtime, live_flat_snapshot);
            preview_snapshot_fingerprint =
                live_flat_snapshot_fingerprint;
        }
        const interaction::PickAssistState state_before_observe =
            assist.diagnostics().state;
        if (state_before_observe ==
            interaction::PickAssistState::Settling) {
            ++settled_observation_count;
        }
        interaction::PickAssistObservation observation{};
        observation.runtime_state =
            scheduler.cached_output().diagnostics.state;
        observation.target = registry.find(*acquired_target);
        observation.displayed_root = interaction::Transform{
            live_flat_snapshot.pose.positions[g1_skeleton::Simulation],
            live_flat_snapshot.pose.rotations[g1_skeleton::Simulation]};
        observation.simulation_velocity = simulation_velocity;
        observation.displayed_planar_speed_mps =
            tick_displacement_m / interaction::kControllerStepSeconds;
        observation.camera_azimuth = camera_azimuth;
        observation.snapshot_fingerprint =
            live_flat_snapshot_fingerprint;
        observation.preview_snapshot_fingerprint =
            preview_snapshot_fingerprint;
        observation.previews = previews;
        const interaction::PickAssistOutput observed_output =
            assist.observe(observation);
        ++observation_count;

        if (activation_tick) {
            require(
                tick_displacement_m <= 1.0e-6F &&
                    assist.diagnostics().state ==
                        interaction::PickAssistState::CoarseApproach &&
                    observed_output.override_steering,
                "manual assist activation did not observe after zero input");
        }
        require(
            assist.diagnostics().state !=
                interaction::PickAssistState::Failed,
            std::string("manual assist ordinary driver failed: ") +
                interaction::pick_assist_reason_name(
                    assist.diagnostics().reason) +
                " tick=" + std::to_string(assist_tick) +
                " error=" + std::to_string(
                    assist.diagnostics().root_error_m) +
                " yaw=" + std::to_string(
                    assist.diagnostics().yaw_error_radians) +
                " displayed_speed=" + std::to_string(
                    assist.diagnostics().speed_mps) +
                " simulation_speed=" + std::to_string(
                    planar_distance(vec3(), simulation_velocity)) +
                " standoff=" + std::to_string(planar_distance(
                    simulation_position, target->object_world.position)) +
                " simulation_yaw=" + std::to_string(
                    planar_yaw(simulation_rotation)) +
                " desired_yaw=" + std::to_string(
                    planar_yaw(desired_rotation)) +
                " selected_yaw=" + std::to_string(
                    assist.diagnostics().selected_slot >= 0
                    ? slots.ordered[static_cast<size_t>(
                          assist.diagnostics().selected_slot)]
                          .prospective_root.world_yaw_radians
                    : 0.0F) +
                " right_x=" + std::to_string(applied_right_stick.x) +
                " right_z=" + std::to_string(applied_right_stick.z));
        saw_preview = saw_preview ||
            assist.diagnostics().state ==
                interaction::PickAssistState::Preview;
        saw_final_approach = saw_final_approach ||
            assist.diagnostics().state ==
                interaction::PickAssistState::FinalApproach;
        saw_settling = saw_settling ||
            assist.diagnostics().state ==
                interaction::PickAssistState::Settling;
        saw_final_preview = saw_final_preview ||
            assist.diagnostics().state ==
                interaction::PickAssistState::FinalPreview;
        if (observed_output.stationary_constraint &&
            !prior_stationary_constraint) {
            ++stationary_constraint_edges;
        }
        prior_stationary_constraint =
            observed_output.stationary_constraint;
        prior_assist_output = observed_output;

        interaction::ControllerInteractionEdges edges{};
        edges.interact_pressed = observed_output.submit_interact;
        const uint32_t provider_calls_before_certification =
            provider_calls;
        const uint32_t resolver_calls_before_certification =
            resolver_calls;
        const uint32_t runtime_calls_before_certification = runtime_calls;
        const uint32_t submissions_before_certification = submission_count;
        scheduler_output = scheduler_tick(edges);
        if (observed_output.submit_interact) {
            require(
                provider_calls ==
                        provider_calls_before_certification + 1U &&
                    resolver_calls ==
                        resolver_calls_before_certification + 1U &&
                    runtime_calls ==
                        runtime_calls_before_certification + 1U &&
                    submission_count ==
                        submissions_before_certification + 1U &&
                    certified_provider_snapshot_fingerprint ==
                    live_flat_snapshot_fingerprint,
                "manual assist certified tick did not use one live snapshot");
            require(
                scheduler_output.diagnostics.state ==
                        interaction::RuntimeState::Preflight &&
                    scheduler_output.diagnostics.reason !=
                        interaction::Reason::TargetUnavailable,
                "manual assist request did not enter Preflight");
            require(
                !assist.take_submission(next_request_id).has_value(),
                "manual assist submission was not one-shot");
            certified = true;
            break;
        }
        require(
            scheduler_output.diagnostics.state ==
                interaction::RuntimeState::Locomotion,
            "manual assist scheduler left Locomotion before certification");
    }

    require(certified, "manual assist ordinary driver never certified");
    require(
        saw_preview && saw_final_approach && saw_settling &&
            saw_final_preview,
        "manual assist ordinary driver skipped a required state");
    require(
        settled_observation_count == 5U &&
            settled_observation_count ==
                assist_config.required_settle_ticks,
        "manual assist did not preserve five post-latch settled ticks");
    require(
        stationary_constraint_edges == 1U &&
            stationary_output_applied_ticks > 0U,
        "manual assist did not hand off one stationary-output edge");
    require(
        materialized_snapshot_count == observation_count &&
            observation_count > 1U,
        "manual assist did not materialize one snapshot per observation");
    require(
        diagnostics.selected_slot >= 0 &&
            diagnostics.route_length_m <=
                assist_config.maximum_assisted_path_m,
        "manual assist selected an invalid or over-budget slot");
    require(
        resolver_calls == 1U && submission_count == 1U &&
            next_request_id == kRequestId + 1U,
        "manual assist scheduler did not submit exactly one request");

    const uint32_t resolver_calls_before_align = resolver_calls;
    scheduler_output = scheduler_tick({});
    require(
        scheduler_output.diagnostics.state ==
                interaction::RuntimeState::Align &&
            scheduler_output.diagnostics.reason !=
                interaction::Reason::TargetUnavailable &&
            scheduler_output.diagnostics.result !=
                interaction::ResultCode::Rejected &&
            scheduler_output.diagnostics.result !=
                interaction::ResultCode::Failed,
        "manual assist runtime did not progress beyond Preflight");
    require(
        resolver_calls == resolver_calls_before_align &&
            provider_calls == runtime_calls,
        "manual assist scheduler cadence or one-shot resolver diverged");
}

Summary run_default_oracle(
    const database& flat_database,
    const interaction::Database& interaction_database,
    const interaction::Features& interaction_features,
    const FixedControllerPoseBridge& bridge) {
    const std::vector<int> authoritative_stationary =
        stationary_motion_matching::derive_candidates(flat_database);
    const std::vector<int> stationary = select_stationary_rows(
        authoritative_stationary, false);
    require(
        stationary.size() == 1U && stationary.front() == 0,
        "default live-flat reproduction must use stationary frame 0");

    const interaction::InteractionTarget authored_target =
        interaction::make_controller_demo_target(interaction_database);
    interaction::TargetRegistry registry{};
    const interaction::TargetHandle target_handle =
        registry.upsert(authored_target);
    const interaction::InteractionTarget* target = registry.find(target_handle);
    require(
        target != nullptr && target->affordances.size() == 1U,
        "frame=0 root=R target registration failed");
    const uint32_t affordance_id = target->affordances.front().id;
    const pick_entry_oracle::OracleRoots roots =
        pick_entry_oracle::make_oracle_roots(
            interaction_database, *target);
    const interaction::LocomotionSnapshot snapshot =
        make_stationary_snapshot(
            flat_database, 0, bridge);
    const interaction::runtime_detail::PickSnapshotMap mapped =
        interaction::runtime_detail::map_pick_entry_snapshot(
            snapshot, roots.reach);
    require(mapped.accepted, "frame=0 root=R entry-root mapping failed");

    const interaction::RuntimeConfig config{};
    interaction::InteractionRuntime runtime(
        interaction_database,
        interaction_features,
        registry,
        config);
    const interaction::matcher_detail::PickEvaluationInput evaluation_input =
        interaction::InteractionRuntimeTestAccess::pick_evaluation_input(
            runtime,
            mapped.snapshot,
            target_handle,
            affordance_id);
    const interaction::matcher_detail::PickEvaluation unfiltered =
        interaction::matcher_detail::evaluate_pick_entries(
            evaluation_input, config.matcher);
    require(
        unfiltered.path_feasible && unfiltered.match_ready &&
            unfiltered.selection.accepted,
        "frame=0 root=R unfiltered match is unavailable");
    require(
        entry_phase(
            interaction_database,
            unfiltered.selection.candidate) == interaction::Phase::Reach,
        "frame=0 root=R no longer selects the calibrated Reach choice");
    require(
        unfiltered.selection.candidate.entry_frame ==
                kExpectedDirectReachEntry &&
            unfiltered.selection.candidate.contact_frame ==
                kExpectedContactFrame,
        "frame=0 root=R direct Reach frames changed");
    const interaction::runtime_detail::RealizedPickTransitionEvaluation
        direct_reach =
            interaction::runtime_detail::evaluate_realized_pick_transition(
                interaction_database,
                evaluation_input.locomotion.pose,
                unfiltered.selection.candidate,
                evaluation_input.target,
                evaluation_input.affordance,
                config.playback,
                config.ik);
    require(
        direct_reach.feasible &&
            direct_reach.reason == interaction::Reason::None,
        "frame=0 root=R calibrated Reach is not certified safe: feasible=" +
            std::to_string(direct_reach.feasible) + " reason=" +
            std::to_string(static_cast<int>(direct_reach.reason)) +
            " frame=" + std::to_string(direct_reach.frame));
    const size_t direct_hand_bone =
        evaluation_input.affordance.hand == interaction::Hand::Left
        ? interaction::kLeftHandBone
        : interaction::kRightHandBone;
    const interaction::WorldPose direct_contact_world =
        interaction::world_pose(interaction::pose_at_frame(
            interaction_database,
            unfiltered.selection.candidate.contact_frame));
    const interaction::Transform direct_mapped_contact_hand =
        interaction::compose(
            unfiltered.selection.candidate.scene_from_source,
            interaction::Transform{
                direct_contact_world.positions[direct_hand_bone],
                direct_contact_world.rotations[direct_hand_bone]});
    const interaction::Transform direct_target_hand = interaction::compose(
        evaluation_input.target.object_world,
        evaluation_input.affordance.hand_in_object);
    const float direct_root_correction_m = length(
        unfiltered.selection.candidate.entry_root_offset);
    const float direct_hand_correction_m = length(
        direct_mapped_contact_hand.position - direct_target_hand.position);
    const float direct_hand_orientation_radians = quat_angle_between(
        direct_mapped_contact_hand.rotation,
        direct_target_hand.rotation);
    require(
        direct_reach.frame == kExpectedContactFrame &&
            direct_root_correction_m <=
                config.matcher.maximum_root_correction_m &&
            std::abs(unfiltered.selection.candidate.entry_yaw_offset) <=
                config.matcher.maximum_yaw_correction_radians &&
            direct_hand_correction_m <=
                config.matcher.maximum_hand_correction_m &&
            direct_hand_orientation_radians <=
                config.matcher.maximum_hand_orientation_radians,
        "direct Reach geometry exceeded production limits: root_m=" +
            std::to_string(direct_root_correction_m) + " yaw_rad=" +
            std::to_string(std::abs(
                unfiltered.selection.candidate.entry_yaw_offset)) +
            " hand_m=" + std::to_string(direct_hand_correction_m) +
            " hand_rad=" +
            std::to_string(direct_hand_orientation_radians) +
            " clearance_radius_m=" +
            std::to_string(evaluation_input.affordance.clearance_radius) +
            " clear_through_frame=" + std::to_string(direct_reach.frame));

    struct EntryElevenCaptured {};
    std::optional<interaction::MatchCandidate> entry_eleven;
    try {
        (void)interaction::matcher_detail::evaluate_pick_entries(
            evaluation_input,
            config.matcher,
            [&](const interaction::MatchCandidate& candidate) {
                if (candidate.entry_frame == kExpectedLongApproachEntry) {
                    entry_eleven = candidate;
                    throw EntryElevenCaptured{};
                }
                return interaction::Reason::BlockedPath;
            });
    } catch (const EntryElevenCaptured&) {
    }
    require(
        entry_eleven.has_value() &&
            entry_eleven->contact_frame == kExpectedContactFrame &&
            entry_phase(interaction_database, *entry_eleven) ==
                interaction::Phase::Approach,
        "frame=0 root=R long Approach regression entry is unavailable");
    const interaction::runtime_detail::RealizedPickTransitionEvaluation
        entry_eleven_evaluation =
            interaction::runtime_detail::evaluate_realized_pick_transition(
                interaction_database,
                evaluation_input.locomotion.pose,
                *entry_eleven,
                evaluation_input.target,
                evaluation_input.affordance,
                config.playback,
                config.ik);
    require(
        entry_eleven_evaluation.frame == kExpectedContactFrame &&
            (entry_eleven_evaluation.feasible
                ? entry_eleven_evaluation.reason == interaction::Reason::None
                : entry_eleven_evaluation.reason ==
                    interaction::Reason::BlockedPath),
        "frame=0 root=R long Approach did not reach contact safely");

    const interaction::PickEntryPreview preview = runtime.preview_pick(
        snapshot, roots.reach, target_handle, affordance_id);
    require(
        preview.path_feasible && preview.match_ready &&
            preview.path_reason == interaction::Reason::None &&
            preview.match_reason == interaction::Reason::None,
        "frame=0 root=R did not preserve a ready calibrated match");
    require(
        entry_phase(interaction_database, preview.match_candidate) ==
            interaction::Phase::Reach,
        "frame=0 root=R calibrated preview did not select Reach");
    require(
        preview.match_candidate.entry_frame ==
                kExpectedDirectReachEntry &&
            preview.match_candidate.contact_frame == kExpectedContactFrame,
        "frame=0 root=R calibrated Reach frames changed");
    const interaction::runtime_detail::RealizedPickTransitionEvaluation
        preview_reach =
            interaction::runtime_detail::evaluate_realized_pick_transition(
                interaction_database,
                evaluation_input.locomotion.pose,
                preview.match_candidate,
                evaluation_input.target,
                evaluation_input.affordance,
                config.playback,
                config.ik);
    require(
        preview_reach.feasible &&
            preview_reach.reason == interaction::Reason::None,
        "frame=0 root=R preview Reach is not certified safe");
    require(
        same_candidate(
            preview.match_candidate,
            unfiltered.selection.candidate),
        "frame=0 root=R calibrated preview changed the direct Reach choice");

    const ExecutionObservation execution = execute_ready_case(
        registry,
        target_handle,
        affordance_id,
        runtime,
        preview,
        snapshot,
        roots.reach,
        0,
        0U);
    require(
        execution.entry_frame == preview.match_candidate.entry_frame &&
            execution.contact_frame == preview.match_candidate.contact_frame,
        "frame=0 root=R execution lost preview/commit agreement");

    Summary summary{};
    summary.sampled = 1U;
    summary.cases = 1U;
    summary.ready = 1U;
    summary.ready_by_root[0] = 1U;
    summary.executed = 1U;
    summary.executed_by_root[0] = 1U;
    summary.first_present = true;
    summary.first_frame = 0;
    summary.first_root = 0U;
    summary.first_entry = execution.entry_frame;
    summary.first_contact = execution.contact_frame;
    summary.direct_entry = unfiltered.selection.candidate.entry_frame;
    return summary;
}

Summary run_exhaustive_oracle(
    const database& flat_database,
    const interaction::Database& interaction_database,
    const interaction::Features& interaction_features,
    const FixedControllerPoseBridge& bridge) {
    const std::vector<int> authoritative_stationary =
        stationary_motion_matching::derive_candidates(flat_database);
    const std::vector<int> stationary = select_stationary_rows(
        authoritative_stationary, true);
    const interaction::InteractionTarget authored_target =
        interaction::make_controller_demo_target(interaction_database);
    const pick_entry_oracle::OracleRoots roots =
        pick_entry_oracle::make_oracle_roots(
            interaction_database, authored_target);
    const std::array<interaction::PickEntryRoot, kRootCount> root_values{
        roots.reach, roots.minus, roots.plus};

    std::vector<interaction::LocomotionSnapshot> snapshots;
    snapshots.reserve(stationary.size());
    for (int stationary_frame : stationary) {
        snapshots.push_back(make_stationary_snapshot(
            flat_database, stationary_frame, bridge));
    }

    const size_t case_count = stationary.size() * kRootCount;
    std::vector<CaseObservation> observations(case_count);
    std::vector<std::exception_ptr> errors(case_count);
    std::atomic<size_t> next_case{0U};
    std::array<std::thread, kWorkerCount> workers;
    for (std::thread& worker : workers) {
        worker = std::thread([&]() {
            for (;;) {
                const size_t case_index = next_case.fetch_add(
                    1U, std::memory_order_relaxed);
                if (case_index >= case_count) return;
                const size_t stationary_index = case_index / kRootCount;
                const size_t root_index = case_index % kRootCount;
                try {
                    observations[case_index] = preview_case(
                        interaction_database,
                        interaction_features,
                        authored_target,
                        snapshots[stationary_index],
                        root_values[root_index],
                        stationary[stationary_index],
                        root_index);
                } catch (...) {
                    errors[case_index] = std::current_exception();
                }
            }
        });
    }
    for (std::thread& worker : workers) worker.join();
    for (const std::exception_ptr& error : errors) {
        if (error != nullptr) std::rethrow_exception(error);
    }

    const std::vector<size_t> coverage = select_execution_coverage(
        stationary, observations);
    std::vector<ExecutionObservation> executions(coverage.size());
    std::vector<std::exception_ptr> execution_errors(coverage.size());
    std::atomic<size_t> next_execution{0U};
    std::array<std::thread, kWorkerCount> execution_workers;
    for (std::thread& worker : execution_workers) {
        worker = std::thread([&]() {
            for (;;) {
                const size_t coverage_index = next_execution.fetch_add(
                    1U, std::memory_order_relaxed);
                if (coverage_index >= coverage.size()) return;
                const size_t case_index = coverage[coverage_index];
                const size_t stationary_index = case_index / kRootCount;
                const size_t root_index = case_index % kRootCount;
                try {
                    executions[coverage_index] = execute_coverage_case(
                        interaction_database,
                        interaction_features,
                        authored_target,
                        snapshots[stationary_index],
                        root_values[root_index],
                        observations[case_index],
                        stationary[stationary_index],
                        root_index);
                } catch (...) {
                    execution_errors[coverage_index] =
                        std::current_exception();
                }
            }
        });
    }
    for (std::thread& worker : execution_workers) worker.join();
    for (const std::exception_ptr& error : execution_errors) {
        if (error != nullptr) std::rethrow_exception(error);
    }

    Summary summary{};
    summary.sampled = stationary.size();
    for (size_t stationary_index = 0U;
         stationary_index < stationary.size();
         ++stationary_index) {
        for (size_t root_index = 0U;
             root_index < root_values.size();
             ++root_index) {
            ++summary.cases;
            const size_t case_index =
                stationary_index * kRootCount + root_index;
            const CaseObservation& observation = observations[case_index];
            if (!observation.ready) continue;
            ++summary.ready;
            ++summary.ready_by_root[root_index];
            if (!summary.first_present) {
                summary.first_present = true;
                summary.first_frame = stationary[stationary_index];
                summary.first_root = root_index;
                summary.first_entry = observation.candidate.entry_frame;
                summary.first_contact = observation.candidate.contact_frame;
            }
        }
    }
    for (size_t coverage_index = 0U;
         coverage_index < coverage.size();
         ++coverage_index) {
        const size_t case_index = coverage[coverage_index];
        const size_t root_index = case_index % kRootCount;
        require(
            executions[coverage_index].entry_frame ==
                    observations[case_index].candidate.entry_frame &&
                executions[coverage_index].contact_frame ==
                    observations[case_index].candidate.contact_frame,
            "execution coverage lost candidate agreement");
        ++summary.executed;
        ++summary.executed_by_root[root_index];
    }
    require(
        summary.cases == stationary.size() * kRootCount,
        "live-flat oracle did not visit every selected case");
    require(summary.first_present, "live-flat oracle found no ready cases");
    return summary;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3 && argc != 4) {
        std::cerr
            << "usage: test_live_flat_pick_entry_oracle "
               "<flat-database.bin> <interaction-pack-directory> "
               "[--exhaustive|--position-matrix]\n";
        return 2;
    }
    try {
        const std::string option = argc == 4 ? argv[3] : "";
        const bool exhaustive = option == "--exhaustive";
        const bool position_matrix = option == "--position-matrix";
        require(
            option.empty() || exhaustive || position_matrix,
            "unknown live-flat oracle option");
        const std::filesystem::path flat_path = argv[1];
        const std::filesystem::path interaction_pack = argv[2];
        require(
            std::filesystem::is_regular_file(flat_path),
            "flat database file is unavailable");
        require(
            std::filesystem::is_regular_file(
                interaction_pack / "interaction_database.bin") &&
                std::filesystem::is_regular_file(
                    interaction_pack / "interaction_features.bin"),
            "interaction pack files are unavailable");

        database flat_database{};
        const std::string flat_filename = flat_path.string();
        database_load(flat_database, flat_filename.c_str());
        validate_flat_database_shape(flat_database);
        const interaction::Database interaction_database =
            interaction::load_database(
                interaction_pack / "interaction_database.bin");
        const interaction::Features interaction_features =
            interaction::load_features(
                interaction_pack / "interaction_features.bin");
        interaction::validate_controller_interaction_pack(
            interaction_database, interaction_features);
        const FixedControllerPoseBridge bridge =
            make_fixed_controller_pose_bridge(
                flat_database, interaction_database);
        require_stationary_snapshot_uses_fixed_calibration(
            flat_database, bridge);

        if (position_matrix) {
            run_position_matrix_oracle(
                flat_database,
                interaction_database,
                interaction_features,
                bridge);
            std::cout
                << "position_cases=3"
                << " rows=baseline_reach,positive_plus,negative_reach"
                << " control=baseline_minus/BlockedPath\n";
            return 0;
        }

        run_manual_pick_assist_oracle(
            flat_database, interaction_database, interaction_features, bridge);

        const Summary summary = exhaustive
            ? run_exhaustive_oracle(
                flat_database,
                interaction_database,
                interaction_features,
                bridge)
            : run_default_oracle(
                flat_database,
                interaction_database,
                interaction_features,
                bridge);
        std::cout
            << "stationary=" << kExpectedStationaryRows
            << " sampled=" << summary.sampled
            << " cases=" << summary.cases
            << " ready=" << summary.ready
            << " reach=" << summary.ready_by_root[0]
            << " minus=" << summary.ready_by_root[1]
            << " plus=" << summary.ready_by_root[2]
            << " executed=" << summary.executed
            << " ex_reach=" << summary.executed_by_root[0]
            << " ex_minus=" << summary.executed_by_root[1]
            << " ex_plus=" << summary.executed_by_root[2]
            << " first=" << summary.first_frame << '/'
            << root_label(summary.first_root) << '/'
            << summary.first_entry << '/'
            << summary.first_contact;
        if (!exhaustive) {
            std::cout
                << " selection=Reach"
                << " direct_entry=" << summary.direct_entry;
        }
        std::cout << '\n';
    } catch (const std::exception& error) {
        std::cerr << "live-flat pick-entry oracle: "
                  << error.what() << '\n';
        return 1;
    }
    return 0;
}
