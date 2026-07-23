#include "interaction_episode.h"

#include "g1_arm_joint_metadata.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr float kTick = 1.0F / 25.0F;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

episode::FrozenAttempt make_attempt(
    const interaction::Pose& pose,
    reach::Hand hand,
    uint64_t generation,
    uint64_t request_id,
    size_t clip) {
    const size_t wrist = hand == reach::Hand::Left
        ? g1_skeleton::LeftWrist
        : g1_skeleton::RightWrist;
    const interaction::WorldPose world = interaction::world_pose(pose);

    episode::FrozenAttempt attempt{};
    attempt.object.generation = generation;
    attempt.object.world = {
        world.positions[wrist] + vec3(0.0F, -0.05F, 0.0F),
        world.rotations[wrist],
    };
    attempt.object.dimensions = vec3(0.08F, 0.10F, 0.08F);
    attempt.grasp.hand_world = {
        world.positions[wrist],
        world.rotations[wrist],
    };
    attempt.grasp.approach_world = vec3(1.0F, 0.0F, 0.0F);
    attempt.grasp.grasp_id = 11U;
    attempt.plan.grasp = attempt.grasp;
    attempt.plan.hand = hand;
    attempt.plan.entry_root_world = {
        pose.positions[g1_skeleton::Simulation],
        pose.rotations[g1_skeleton::Simulation],
    };
    attempt.plan.reach.candidate.clip = clip;
    attempt.plan.reach.rejection = reach::Rejection::None;
    attempt.plan.reach.poses.assign(4U, pose);
    attempt.plan.return_poses.assign(3U, pose);
    attempt.request_id = request_id;
    return attempt;
}

episode::EpisodeConfig fast_config() {
    episode::EpisodeConfig config{};
    config.entry_position_m = 0.20F;
    config.entry_yaw_radians = 3.14159265F;
    config.stable_entry_ticks = 1;
    config.bridge_seconds = kTick;
    config.place_bridge_seconds = kTick;
    config.carry_blend_seconds = kTick;
    config.attachment.required_lift_m = 0.0F;
    config.attachment.required_hold_seconds = 0.0F;
    config.attachment.require_lift_for_hold = false;
    return config;
}

void advance_until(
    episode::InteractionEpisode& runtime,
    episode::EpisodeState desired,
    uint64_t generation,
    int maximum_ticks = 80,
    uint64_t destination_generation = 0U) {
    for (int tick = 0; tick < maximum_ticks; ++tick) {
        runtime.update({
            kTick,
            episode::LocomotionCommand{},
            generation,
            false,
            false,
            destination_generation,
        });
        if (runtime.state() == desired) return;
    }
    throw std::runtime_error("episode did not reach requested state");
}

size_t wrist_bone(interaction::Hand hand) {
    return hand == interaction::Hand::Left
        ? g1_skeleton::LeftWrist
        : g1_skeleton::RightWrist;
}

interaction::Transform hand_world(
    const interaction::Pose& pose,
    interaction::Hand hand) {
    const interaction::WorldPose world = interaction::world_pose(pose);
    const size_t wrist = wrist_bone(hand);
    return {world.positions[wrist], world.rotations[wrist]};
}

void require_pose_root(
    const interaction::Pose& actual,
    const interaction::Pose& expected,
    const std::string& message) {
    require(
        length(
            actual.positions[g1_skeleton::Simulation] -
            expected.positions[g1_skeleton::Simulation]) < 1.0e-5F,
        message);
}

void require_return_sample(
    const episode::EpisodeOutput& output,
    const interaction::Pose& expected,
    interaction::Hand hand,
    interaction::Transform hand_in_object,
    const std::string& label) {
    require(
        output.state == episode::EpisodeState::Return,
        label + " was not observable in Return");
    require(output.attached, label + " detached the object");
    require_pose_root(output.pose, expected, label + " root differs");
    const interaction::Transform expected_object = interaction::compose(
        hand_world(output.pose, hand),
        interaction::inverse(hand_in_object));
    require(
        length(output.object_world.position - expected_object.position) <
            1.0e-5F,
        label + " object position did not follow the displayed wrist");
    require(
        quat_angle_between(
            output.object_world.rotation,
            expected_object.rotation) < 0.002F,
        label + " object rotation did not follow the displayed wrist");
}

void require_selected_arm_layer(
    const interaction::Pose& pose,
    const interaction::Pose& nominal,
    interaction::Hand hand,
    const std::string& label) {
    const std::array<interaction::HingeJoint, 7>& arm =
        hand == interaction::Hand::Left
            ? interaction::kLeftArm
            : interaction::kRightArm;
    for (const interaction::HingeJoint& joint : arm) {
        const size_t bone = static_cast<size_t>(joint.bone);
        require(
            quat_angle_between(
                pose.rotations[bone], nominal.rotations[bone]) < 0.002F,
            label + " did not retain the final recorded arm layer");
    }
}

void test_commit_rejects_nonfinite_return_pose_before_state_mutation() {
    const std::filesystem::path pack("build/g1-episode");
    episode::InteractionEpisode runtime(
        pack / "walking_database.bin",
        pack / "carry_left_database.bin",
        pack / "carry_right_database.bin",
        fast_config());
    auto malformed = make_attempt(
        runtime.output().pose, reach::Hand::Left, 60U, 69U, 14U);
    malformed.plan.return_poses[1U]
        .positions[g1_skeleton::Simulation].x =
            std::numeric_limits<float>::quiet_NaN();
    malformed.plan.return_poses[2U]
        .rotations[g1_skeleton::LeftWrist].w =
            std::numeric_limits<float>::infinity();
    require(
        !runtime.commit(malformed),
        "commit accepted a non-finite recorded return pose");
    require(
        runtime.state() == episode::EpisodeState::FreeLocomotion &&
            !runtime.attempt().has_value() && !runtime.output().attached,
        "malformed return plan mutated episode state before rejection");
}

void test_return_replays_every_authored_sample_before_carry() {
    const std::filesystem::path pack("build/g1-episode");
    episode::InteractionEpisode runtime(
        pack / "walking_database.bin",
        pack / "carry_left_database.bin",
        pack / "carry_right_database.bin",
        fast_config());
    auto attempt = make_attempt(
        runtime.output().pose, reach::Hand::Left, 61U, 70U, 14U);
    interaction::Pose return0 = runtime.output().pose;
    interaction::Pose return1 = return0;
    interaction::Pose return2 = return0;
    return0.positions[g1_skeleton::Simulation] =
        return0.positions[g1_skeleton::Simulation] +
        vec3(0.02F, 0.0F, 0.01F);
    return1.positions[g1_skeleton::Simulation] =
        return1.positions[g1_skeleton::Simulation] +
        vec3(0.04F, 0.0F, 0.02F);
    return2.positions[g1_skeleton::Simulation] =
        return2.positions[g1_skeleton::Simulation] +
        vec3(0.06F, 0.0F, 0.03F);
    return0.rotations[g1_skeleton::LeftElbow] = quat_mul(
        quat_from_angle_axis(0.10F, vec3(0.0F, 0.0F, 1.0F)),
        return0.rotations[g1_skeleton::LeftElbow]);
    return1.rotations[g1_skeleton::LeftElbow] = quat_mul(
        quat_from_angle_axis(0.20F, vec3(0.0F, 0.0F, 1.0F)),
        return1.rotations[g1_skeleton::LeftElbow]);
    return2.rotations[g1_skeleton::LeftElbow] = quat_mul(
        quat_from_angle_axis(0.30F, vec3(0.0F, 0.0F, 1.0F)),
        return2.rotations[g1_skeleton::LeftElbow]);
    attempt.plan.return_poses = {return0, return1, return2};
    const interaction::Transform hand_in_object = interaction::compose(
        interaction::inverse(attempt.object.world),
        attempt.grasp.hand_world);
    require(runtime.commit(attempt), "return fixture did not commit");
    advance_until(runtime, episode::EpisodeState::Reach, 61U);
    advance_until(runtime, episode::EpisodeState::Return, 61U);

    require_return_sample(
        runtime.output(), return0, interaction::Hand::Left, hand_in_object,
        "return sample 0");
    const float large_dt = 3.0F * kTick;
    const episode::LocomotionCommand ignored_command{
        vec3(0.0F, 0.0F, 1.0F), quat()};
    const episode::EpisodeOutput& sample1 = runtime.update({
        large_dt, ignored_command, 61U, false, false});
    require_return_sample(
        sample1, return1, interaction::Hand::Left, hand_in_object,
        "return sample 1");
    const episode::EpisodeOutput& sample2 = runtime.update({
        large_dt, ignored_command, 61U, false, false});
    require_return_sample(
        sample2, return2, interaction::Hand::Left, hand_in_object,
        "return sample 2");
    const episode::EpisodeOutput& carry_start = runtime.update({
        0.0F, ignored_command, 61U, false, false});
    require(
        runtime.state() == episode::EpisodeState::Carry &&
            carry_start.state == episode::EpisodeState::Carry,
        "recorded return did not lead into carry after its final sample");
    require_pose_root(
        carry_start.pose, return2,
        "carry matcher rebased with a root discontinuity");
    require_selected_arm_layer(
        carry_start.pose, return2, interaction::Hand::Left,
        "first carry pose");

    const vec3 carry_root =
        carry_start.pose.positions[g1_skeleton::Simulation];
    interaction::Pose previous = carry_start.pose;
    float maximum_knee_change = 0.0F;
    for (int tick = 0; tick < 100; ++tick) {
        const episode::EpisodeOutput& output = runtime.update({
            kTick, ignored_command, 61U, false, false});
        maximum_knee_change = std::max(
            maximum_knee_change,
            quat_angle_between(
                output.pose.rotations[g1_skeleton::LeftKnee],
                previous.rotations[g1_skeleton::LeftKnee]));
        require_selected_arm_layer(
            output.pose, return2, interaction::Hand::Left,
            "walking carry pose");
        previous = output.pose;
    }
    require(
        length(
            previous.positions[g1_skeleton::Simulation] - carry_root) >
            0.20F,
        "walking carry did not move the rebased root");
    require(
        maximum_knee_change > 0.005F,
        "walking carry did not retain leg motion");
}

void test_freeze_attach_and_selected_carry_hand(reach::Hand hand) {
    const std::filesystem::path pack("build/g1-episode");
    episode::InteractionEpisode runtime(
        pack / "walking_database.bin",
        pack / "carry_left_database.bin",
        pack / "carry_right_database.bin",
        fast_config());
    const auto original = make_attempt(
        runtime.output().pose, hand, 7U, 100U, 12U);
    auto replacement = original;
    replacement.plan.reach.candidate.clip = 99U;

    require(runtime.commit(original), "F did not commit a valid plan");
    require(
        runtime.state() == episode::EpisodeState::Approach,
        "committed plan did not enter approach");
    require(
        !runtime.commit(replacement),
        "repeated F replaced an active attempt");
    require(
        runtime.attempt()->plan.reach.candidate.clip == 12U,
        "active plan was not immutable");

    advance_until(runtime, episode::EpisodeState::Reach, 7U);
    advance_until(runtime, episode::EpisodeState::Carry, 7U);
    require(runtime.output().attached, "object did not attach at contact");
    const interaction::Hand expected = hand == reach::Hand::Left
        ? interaction::Hand::Left
        : interaction::Hand::Right;
    require(
        runtime.output().selected_hand == expected,
        "selected reach hand was not propagated into carry");
}

void test_layered_carry_does_not_require_full_pose_carry_databases(
    reach::Hand hand) {
    const std::filesystem::path pack("build/g1-episode");
    episode::InteractionEpisode runtime(
        pack / "walking_database.bin",
        pack / "missing-carry-left.bin",
        pack / "missing-carry-right.bin",
        fast_config());
    const auto attempt = make_attempt(
        runtime.output().pose, hand, 17U, 101U, 13U);
    require(
        runtime.commit(attempt),
        "layered carry fixture did not commit");
    advance_until(runtime, episode::EpisodeState::Carry, 17U);
    require(
        runtime.output().attached,
        "layered carry fixture did not retain attachment");
    require(
        runtime.output().diagnostic.empty(),
        "successful layered carry reported an IK fallback");
}

void test_generation_change_and_cancel_fail_before_contact() {
    const std::filesystem::path pack("build/g1-episode");
    episode::InteractionEpisode changed(
        pack / "walking_database.bin",
        pack / "carry_left_database.bin",
        pack / "carry_right_database.bin",
        fast_config());
    require(
        changed.commit(make_attempt(
            changed.output().pose, reach::Hand::Left, 4U, 10U, 1U)),
        "generation fixture did not commit");
    changed.update({
        kTick, episode::LocomotionCommand{}, 5U, false, false});
    require(
        changed.state() == episode::EpisodeState::Failed,
        "object generation change did not fail the attempt");

    episode::InteractionEpisode cancelled(
        pack / "walking_database.bin",
        pack / "carry_left_database.bin",
        pack / "carry_right_database.bin",
        fast_config());
    require(
        cancelled.commit(make_attempt(
            cancelled.output().pose, reach::Hand::Right, 9U, 11U, 2U)),
        "cancel fixture did not commit");
    cancelled.update({
        kTick, episode::LocomotionCommand{}, 9U, true, false});
    require(
        cancelled.state() == episode::EpisodeState::FreeLocomotion,
        "cancel before contact did not return locomotion control");
}

void test_contact_rejection_timeout_and_reset() {
    const std::filesystem::path pack("build/g1-episode");
    episode::InteractionEpisode rejected(
        pack / "walking_database.bin",
        pack / "carry_left_database.bin",
        pack / "carry_right_database.bin",
        fast_config());
    auto bad_contact = make_attempt(
        rejected.output().pose, reach::Hand::Left, 12U, 20U, 3U);
    bad_contact.grasp.hand_world.position.x += 0.20F;
    bad_contact.plan.grasp = bad_contact.grasp;
    require(rejected.commit(bad_contact), "bad-contact fixture did not commit");
    advance_until(rejected, episode::EpisodeState::Failed, 12U);
    require(
        !rejected.output().attached,
        "geometrically invalid contact attached the object");
    rejected.update({
        kTick, episode::LocomotionCommand{}, 12U, false, true});
    require(
        rejected.state() == episode::EpisodeState::FreeLocomotion &&
            !rejected.attempt().has_value(),
        "reset did not clear failed attempt");

    episode::EpisodeConfig timeout_config = fast_config();
    timeout_config.approach_timeout_seconds = kTick;
    episode::InteractionEpisode timed_out(
        pack / "walking_database.bin",
        pack / "carry_left_database.bin",
        pack / "carry_right_database.bin",
        timeout_config);
    auto unreachable = make_attempt(
        timed_out.output().pose, reach::Hand::Right, 13U, 21U, 4U);
    unreachable.plan.entry_root_world.position.x += 20.0F;
    require(timed_out.commit(unreachable), "timeout fixture did not commit");
    timed_out.update({
        kTick, episode::LocomotionCommand{}, 13U, false, false});
    require(
        timed_out.state() == episode::EpisodeState::Failed,
        "approach timeout did not fail");
}

void test_native_walking_converges_to_a_distant_entry() {
    const std::filesystem::path pack("build/g1-episode");
    episode::EpisodeConfig config{};
    config.bridge_seconds = 10.0F;
    episode::InteractionEpisode runtime(
        pack / "walking_database.bin",
        pack / "carry_left_database.bin",
        pack / "carry_right_database.bin",
        config);
    auto attempt = make_attempt(
        runtime.output().pose, reach::Hand::Left, 21U, 30U, 8U);
    attempt.plan.entry_root_world.position =
        runtime.output().pose.positions[g1_skeleton::Simulation] +
        vec3(-0.514F, 0.0F, 0.948F);
    attempt.plan.entry_root_world.rotation = quat_mul(
        quat_from_angle_axis(
            0.529F, vec3(0.0F, 1.0F, 0.0F)),
        runtime.output().pose.rotations[g1_skeleton::Simulation]);
    require(runtime.commit(attempt), "distant-entry fixture did not commit");
    for (int tick = 0;
         tick < 400 && runtime.state() == episode::EpisodeState::Approach;
         ++tick) {
        runtime.update({
            kTick, episode::LocomotionCommand{}, 21U, false, false});
    }
    require(
        runtime.state() == episode::EpisodeState::Bridge,
        "flat walking missed entry: distance " +
            std::to_string(runtime.output().approach_distance_m) +
            " yaw " +
            std::to_string(runtime.output().approach_yaw_error_radians) +
            " speed " +
            std::to_string(runtime.output().locomotion_speed_mps));
    runtime.update({
        kTick, episode::LocomotionCommand{}, 21U, false, false});
    require(
        runtime.state() == episode::EpisodeState::Bridge &&
            !runtime.output().flat_locomotion.valid &&
            runtime.output().bridge_alpha > 0.0F &&
            runtime.output().bridge_alpha < 1.0F,
        "native G1 bridge exposed a second skeleton or skipped blending");
}

void test_native_walking_reach_and_carry_preserve_contact_root() {
    const std::filesystem::path pack("build/g1-episode");
    episode::InteractionEpisode runtime(
        pack / "walking_database.bin",
        pack / "carry_left_database.bin",
        pack / "carry_right_database.bin",
        fast_config());
    const interaction::Pose start = runtime.output().pose;
    interaction::Pose contact = start;
    contact.positions[g1_skeleton::Simulation].x += 0.30F;
    auto attempt = make_attempt(
        start, reach::Hand::Left, 31U, 40U, 9U);
    attempt.plan.reach.poses = {start, start, contact, contact};
    attempt.plan.return_poses = {contact, contact, contact};
    const interaction::WorldPose contact_world =
        interaction::world_pose(contact);
    const size_t wrist = g1_skeleton::LeftWrist;
    attempt.grasp.hand_world = {
        contact_world.positions[wrist],
        contact_world.rotations[wrist],
    };
    attempt.grasp.approach_world = vec3(1.0F, 0.0F, 0.0F);
    attempt.plan.grasp = attempt.grasp;
    attempt.object.world = {
        contact_world.positions[wrist] + vec3(0.0F, -0.05F, 0.0F),
        contact_world.rotations[wrist],
    };

    require(runtime.commit(attempt), "flat full-episode fixture did not commit");
    advance_until(runtime, episode::EpisodeState::Carry, 31U);
    require(runtime.output().attached, "flat full episode did not attach");
    require(
        length(
            runtime.output().pose.positions[g1_skeleton::Simulation] -
            contact.positions[g1_skeleton::Simulation]) < 0.15F,
        "carry rebased away from the reach contact root");

    const vec3 carry_root =
        runtime.output().pose.positions[g1_skeleton::Simulation];
    const vec3 carry_object = runtime.output().object_world.position;
    const episode::LocomotionCommand command{
        vec3(0.0F, 0.0F, 0.60F), quat()};
    for (int tick = 0; tick < 100; ++tick) {
        runtime.update({kTick, command, 31U, false, false});
    }
    require(
        length(
            runtime.output().pose.positions[g1_skeleton::Simulation] -
            carry_root) > 0.20F,
        "flat carry did not respond to locomotion");
    require(
        length(runtime.output().object_world.position - carry_object) >
            0.20F,
        "attached object did not follow flat carry locomotion");
}

void test_same_hand_place_releases_and_returns_to_locomotion() {
    const std::filesystem::path pack("build/g1-episode");
    episode::InteractionEpisode runtime(
        pack / "walking_database.bin",
        pack / "carry_left_database.bin",
        pack / "carry_right_database.bin",
        fast_config());
    const auto pickup = make_attempt(
        runtime.output().pose, reach::Hand::Left, 41U, 50U, 10U);
    require(runtime.commit(pickup), "place pickup fixture did not commit");
    advance_until(runtime, episode::EpisodeState::Carry, 41U);
    for (int tick = 0;
         tick < 100 && !runtime.output().place_ready;
         ++tick) {
        runtime.update({
            kTick,
            episode::LocomotionCommand{},
            41U,
            false,
            false,
            0U,
        });
    }
    require(
        runtime.output().place_ready,
        "held object was not place-ready: object y " +
            std::to_string(runtime.output().object_world.position.y) +
            " pre-lift y " +
            std::to_string(pickup.object.world.position.y));

    const interaction::Pose start = runtime.output().pose;
    interaction::Pose contact = start;
    const vec3 placement_delta(0.25F, 0.0F, 0.10F);
    contact.positions[g1_skeleton::Simulation] =
        contact.positions[g1_skeleton::Simulation] + placement_delta;
    const interaction::WorldPose contact_world =
        interaction::world_pose(contact);
    const size_t wrist = g1_skeleton::LeftWrist;
    episode::FrozenPlaceAttempt place{};
    place.destination = {
        51U,
        {
            runtime.output().object_world.position + placement_delta,
            runtime.output().object_world.rotation,
        },
        pickup.object.dimensions,
    };
    place.grasp = {
        {
            contact_world.positions[wrist],
            contact_world.rotations[wrist],
        },
        vec3(1.0F, 0.0F, 0.0F),
        interaction::Hand::Left,
        12U,
    };
    place.plan.grasp = place.grasp;
    place.plan.hand = reach::Hand::Left;
    place.plan.entry_root_world = {
        start.positions[g1_skeleton::Simulation],
        start.rotations[g1_skeleton::Simulation],
    };
    place.plan.reach.candidate.clip = 10U;
    place.plan.reach.rejection = reach::Rejection::None;
    place.plan.reach.poses = {start, start, contact, contact};
    place.support = {
        {vec3(0.0F, 0.65F, 0.0F), quat()},
        vec3(1.20F, 0.06F, 0.75F),
    };
    place.request_id = 52U;

    require(runtime.commit_place(place), "valid place attempt did not commit");
    require(
        runtime.state() == episode::EpisodeState::PlaceApproach,
        "place attempt did not enter approach");
    for (int tick = 0;
         tick < 200 &&
         runtime.state() != episode::EpisodeState::FreeLocomotion;
         ++tick) {
        runtime.update({
            kTick,
            episode::LocomotionCommand{},
            41U,
            false,
            false,
            51U,
        });
    }
    require(
        runtime.state() == episode::EpisodeState::FreeLocomotion,
        "place playback did not return locomotion control");
    require(
        runtime.output().placed && !runtime.output().attached,
        "place playback did not release the object");
    require(
        length(
            runtime.output().object_world.position -
            place.destination.world.position) < 1.0e-5F,
        "placed object did not retain the frozen destination");
    require(
        !runtime.place_attempt().has_value(),
        "completed place attempt remained active");
}

}  // namespace

int main() {
    try {
        test_freeze_attach_and_selected_carry_hand(reach::Hand::Left);
        test_freeze_attach_and_selected_carry_hand(reach::Hand::Right);
        test_layered_carry_does_not_require_full_pose_carry_databases(
            reach::Hand::Left);
        test_layered_carry_does_not_require_full_pose_carry_databases(
            reach::Hand::Right);
        test_commit_rejects_nonfinite_return_pose_before_state_mutation();
        test_return_replays_every_authored_sample_before_carry();
        test_generation_change_and_cancel_fail_before_contact();
        test_contact_rejection_timeout_and_reset();
        test_native_walking_converges_to_a_distant_entry();
        test_native_walking_reach_and_carry_preserve_contact_root();
        test_same_hand_place_releases_and_returns_to_locomotion();
        std::cout << "interaction episode PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "interaction episode FAILED: " << error.what() << '\n';
        return 1;
    }
}
