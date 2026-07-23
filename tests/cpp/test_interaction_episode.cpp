#include "interaction_episode.h"

#include <cmath>
#include <filesystem>
#include <iostream>
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

void test_return_replays_attached_frozen_poses_before_carry() {
    const std::filesystem::path pack("build/g1-episode");
    episode::InteractionEpisode runtime(
        pack / "walking_database.bin",
        pack / "carry_left_database.bin",
        pack / "carry_right_database.bin",
        fast_config());
    const auto attempt = make_attempt(
        runtime.output().pose, reach::Hand::Left, 61U, 70U, 14U);
    require(runtime.commit(attempt), "return fixture did not commit");
    advance_until(runtime, episode::EpisodeState::Reach, 61U);
    advance_until(runtime, episode::EpisodeState::Return, 61U);
    require(
        runtime.output().attached,
        "object detached at the first recorded return frame");
    require(
        length(
            runtime.output().pose.positions[g1_skeleton::Simulation] -
            attempt.plan.return_poses.front()
                .positions[g1_skeleton::Simulation]) < 1.0e-5F,
        "return did not begin at the first frozen pose");

    for (size_t frame = 1U;
         frame < attempt.plan.return_poses.size();
         ++frame) {
        const episode::EpisodeOutput& output = runtime.update({
            kTick,
            {vec3(0.0F, 0.0F, 1.0F), quat()},
            61U,
            false,
            false,
        });
        require(output.attached, "object detached during recorded return");
        require(
            length(
                output.pose.positions[g1_skeleton::Simulation] -
                attempt.plan.return_poses[frame]
                    .positions[g1_skeleton::Simulation]) < 1.0e-5F,
            "return accepted locomotion input instead of playback");
    }
    require(
        runtime.state() == episode::EpisodeState::Carry,
        "recorded return did not lead into carry");
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
        test_return_replays_attached_frozen_poses_before_carry();
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
