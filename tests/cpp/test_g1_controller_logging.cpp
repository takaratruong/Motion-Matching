#define G1_FRAME_TRANSACTION_ENABLE_TEST_SEAM
#define main g1_controller_logging_oracle_embedded_main
#include "test_g1_frame_transaction_production.cpp"
#undef main
#undef G1_FRAME_TRANSACTION_ENABLE_TEST_SEAM

#define g1_frame_transaction_run(runtime, runner, external, error, capacity) \
    g1_frame_transaction_run(                                           \
        runtime, runner, external, nullptr, error, capacity)
#define main g1_controller_logging_embedded_main
#include "controller.cpp"
#undef main
#undef g1_frame_transaction_run

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>

static void logging_check(bool condition, const char* message)
{
    if (!condition) {
        std::fprintf(stderr, "G1 controller logging test failed: %s\n", message);
        std::exit(1);
    }
}

template<unsigned char Fill>
struct padding_poisoned_state
{
    alignas(g1_controller_state)
        unsigned char storage[sizeof(g1_controller_state)];
    g1_controller_state* state = nullptr;

    padding_poisoned_state()
    {
        std::memset(storage, Fill, sizeof(storage));
        state = new (storage) g1_controller_state;
    }

    ~padding_poisoned_state()
    {
        state->~g1_controller_state();
    }

    padding_poisoned_state(const padding_poisoned_state&) = delete;
    padding_poisoned_state& operator=(const padding_poisoned_state&) = delete;
};

static void test_accepted_state_digest_ignores_padding_and_tracks_values()
{
    padding_poisoned_state<0x2a> first;
    padding_poisoned_state<0xd5> second;
    logging_check(
        std::memcmp(first.storage, second.storage, sizeof(first.storage)) != 0,
        "fixture gives equal logical states different object representations");

    const uint64_t first_digest =
        g1_log_accepted_state_digest(*first.state);
    const uint64_t second_digest =
        g1_log_accepted_state_digest(*second.state);
    logging_check(
        first_digest == state_logical_digest(*first.state),
        "accepted-state digest matches the independent logical oracle");
    logging_check(
        first_digest == second_digest,
        "accepted-state digest ignores struct padding and storage history");

    second.state->frame_index = 1;
    logging_check(
        g1_log_accepted_state_digest(*second.state) != first_digest,
        "accepted-state digest authenticates scalar logical values");
    second.state->frame_index = first.state->frame_index;
    second.state->ik_frame.feet[0].target.desired_sole_normal =
        vec3(0.0f, 0.8f, 0.6f);
    logging_check(
        g1_log_accepted_state_digest(*second.state) != first_digest,
        "accepted-state digest authenticates nested logical values");

    fixture rich;
    g1_controller_state& rich_state = rich.runtime.accepted_state;
    const auto check_oracle = [&rich_state](const char* message) {
        logging_check(
            g1_log_accepted_state_digest(rich_state) ==
                state_logical_digest(rich_state),
            message);
    };
    check_oracle("rich accepted state matches the independent logical oracle");
    rich_state.curr_bone_positions(0).x += 0.125f;
    check_oracle("array element mutation remains aligned with the oracle");
    rich_state.footprint.feet[0].probes[0]
        .predicted_surface_status[0] = G1SurfaceQueryInvalid;
    check_oracle("footprint mutation remains aligned with the oracle");
    ++rich_state.ik.feet[0].lock.release_frames;
    check_oracle("IK-state mutation remains aligned with the oracle");
    rich_state.ik_frame.feet[0].orientation.correction_radians += 0.01f;
    check_oracle("IK-frame mutation remains aligned with the oracle");
    rich_state.ik_clearance.left.knee.witness.body_x = 0.03125;
    check_oracle("clearance mutation remains aligned with the oracle");
    rich_state.command.predicted_root_positions[0].z += 0.0625f;
    check_oracle("command mutation remains aligned with the oracle");
    rich_state.support_observation_now.delta[1] += 0.0078125f;
    check_oracle("support mutation remains aligned with the oracle");
}

struct accepted_row_fixture
{
    g1_controller_state state;
    G1FrameAcceptedDiagnostic diagnostic;
    G1FramePublication publication;
    motion_source_record sources[2];
    const char* source_names[2] = {"source-zero", "source-one"};
    const char* source_terrains[2] = {"flat", "stairs"};
    const char* scene_ids[1] = {"logging-scene"};
    int active_scene_index = 0;
    int scene_generation = 1;
    int scene_reset_count = 1;
    bool scene_switch_failed = false;
    int motion_pack_load_count = 1;
    int model_load_count = 1;
    int model_unload_count = 0;
    G1AcceptedLogContext context;

    accepted_row_fixture()
    {
        sources[0].range_start = 0;
        sources[0].range_stop = 10;
        sources[1].range_start = 10;
        sources[1].range_stop = 20;
        state.frame_index = 5;
        diagnostic.ready = true;
        diagnostic.query_database_frame = 5;
        diagnostic.query_range = 0;
        diagnostic.selected_database_frame = 5;
        context.sources = sources;
        context.source_names = source_names;
        context.source_terrains = source_terrains;
        context.source_count = 2;
        context.scene_ids = scene_ids;
        context.active_scene_index = &active_scene_index;
        context.scene_generation = &scene_generation;
        context.scene_reset_count = &scene_reset_count;
        context.scene_switch_failed = &scene_switch_failed;
        context.motion_pack_load_count = &motion_pack_load_count;
        context.model_load_count = &model_load_count;
        context.model_unload_count = &model_unload_count;
    }

    bool build(motion_match_log_row& row, char* error, int capacity)
    {
        return g1_build_accepted_log_row(
            row, state, diagnostic, publication, context, error, capacity);
    }
};

static void test_accepted_row_uses_authoritative_names()
{
    const support_source sources[] = {
        support_root,
        support_left,
        support_right,
        support_both,
        support_held,
        support_airborne_root,
    };
    for (const support_source source : sources) {
        accepted_row_fixture fixture;
        fixture.state.support.source = source;
        motion_match_log_row row;
        char error[256] = {};
        logging_check(
            fixture.build(row, error, sizeof(error)),
            "accepted row builds for every certified support source");
        logging_check(
            std::strcmp(row.support_source, support_source_name(source)) == 0,
            "accepted row uses the authoritative support-source name");
    }

    const walkability_reason reasons[] = {
        walkability_clear,
        walkability_blocked_cell,
        walkability_out_of_bounds,
        walkability_nonfinite,
    };
    for (const walkability_reason reason : reasons) {
        accepted_row_fixture fixture;
        fixture.diagnostic.traversal.reason = reason;
        motion_match_log_row row;
        char error[256] = {};
        logging_check(
            fixture.build(row, error, sizeof(error)),
            "accepted row builds for every traversal reason");
        logging_check(
            std::strcmp(row.blocked_reason,
                        walkability_reason_name(reason)) == 0,
            "accepted row uses the authoritative walkability-reason name");
    }
}

static void test_first_rejected_row_uses_canonical_accepted_baseline()
{
    accepted_row_fixture fixture;
    fixture.diagnostic = G1FrameAcceptedDiagnostic{};
    fixture.publication.rejection.rejected = true;
    motion_match_log_row row;
    char error[256] = {};
    logging_check(
        fixture.build(row, error, sizeof(error)),
        "first finite rejection uses the immutable accepted-state baseline");
    logging_check(
        row.query_database_frame == fixture.state.frame_index &&
            row.selected_database_frame == fixture.state.frame_index &&
            row.database_frame == fixture.state.frame_index &&
            row.query_range == 0 && row.source_range == 0 &&
            row.range == 0 && row.source_index == 0,
        "first finite rejection resolves all provenance from the accepted frame");
    logging_check(
        terrain_float_bits(row.terrain[0]) == 0U &&
            terrain_float_bits(row.terrain[1]) == 0U &&
            terrain_float_bits(row.terrain[2]) == 0U &&
            terrain_float_bits(row.terrain[3]) == 0U &&
            terrain_float_bits(row.raw_selected.hips_y) == 0U &&
            terrain_float_bits(row.inertialized.hips_y) == 0U &&
            terrain_float_bits(row.rendered.hips_y) == 0U,
        "first finite rejection retains canonical zero diagnostics");

    accepted_row_fixture invalid;
    invalid.diagnostic = G1FrameAcceptedDiagnostic{};
    invalid.publication.rejection.rejected = false;
    motion_match_log_row untouched;
    untouched.frame = 1234;
    error[0] = '\0';
    logging_check(
        !invalid.build(untouched, error, sizeof(error)) && error[0] != '\0',
        "unready diagnostics are rejected without a finite rejection");
    logging_check(
        untouched.frame == 1234,
        "invalid unready diagnostics leave the base row untouched");
}

static void test_accepted_row_rejects_ambiguous_or_missing_provenance()
{
    {
        accepted_row_fixture fixture;
        fixture.state.frame_index = 25;
        motion_match_log_row row;
        row.frame = 1234;
        char error[256] = {};
        logging_check(
            !fixture.build(row, error, sizeof(error)) && error[0] != '\0',
            "accepted row rejects a database frame with no source owner");
        logging_check(row.frame == 1234,
                      "failed provenance validation leaves the row untouched");
    }
    {
        accepted_row_fixture fixture;
        fixture.diagnostic.selected_database_frame = 25;
        motion_match_log_row row;
        char error[256] = {};
        logging_check(
            !fixture.build(row, error, sizeof(error)) && error[0] != '\0',
            "accepted row rejects a selected frame with no source owner");
    }
    {
        accepted_row_fixture fixture;
        fixture.diagnostic.query_database_frame = 25;
        motion_match_log_row row;
        char error[256] = {};
        logging_check(
            !fixture.build(row, error, sizeof(error)) && error[0] != '\0',
            "accepted row rejects a query frame with no source owner");
    }
    {
        accepted_row_fixture fixture;
        fixture.diagnostic.query_range = 1;
        motion_match_log_row row;
        char error[256] = {};
        logging_check(
            !fixture.build(row, error, sizeof(error)) && error[0] != '\0',
            "accepted row rejects a query range that contradicts its frame");
    }
    {
        accepted_row_fixture fixture;
        fixture.diagnostic.selected_database_frame = 15;
        motion_match_log_row row;
        char error[256] = {};
        logging_check(
            !fixture.build(row, error, sizeof(error)) && error[0] != '\0',
            "accepted row rejects selected and accepted frames in different ranges");
    }
    {
        accepted_row_fixture fixture;
        fixture.sources[0].range_stop = 15;
        fixture.sources[1].range_start = 5;
        fixture.state.frame_index = 7;
        fixture.diagnostic.selected_database_frame = 7;
        motion_match_log_row row;
        char error[256] = {};
        logging_check(
            !fixture.build(row, error, sizeof(error)) && error[0] != '\0',
            "accepted row rejects multiply-owned provenance");
    }
}

static void test_cross_source_transition_maps_each_provenance_owner()
{
    accepted_row_fixture fixture;
    fixture.state.frame_index = 15;
    fixture.state.transitioned = true;
    fixture.diagnostic.query_database_frame = 5;
    fixture.diagnostic.query_range = 0;
    fixture.diagnostic.selected_database_frame = 15;
    motion_match_log_row row;
    char error[256] = {};
    logging_check(
        fixture.build(row, error, sizeof(error)),
        "a valid cross-source transition row materializes");
    logging_check(
        row.query_database_frame == 5 && row.query_range == 0 &&
            row.selected_database_frame == 15 &&
            row.database_frame == 15 && row.source_range == 1 &&
            row.range == 1 && row.source_index == 1 &&
            std::strcmp(row.source_name, "source-one") == 0 &&
            std::strcmp(row.source_terrain, "stairs") == 0,
        "query, selected, and accepted provenance follow their independent immutable owners");
}

static void seed_ik_logging_state(g1_controller_state& state)
{
    state.ik_global_bone_positions.resize(G1_BoneCount);
    state.ik_global_bone_rotations.resize(G1_BoneCount);
    state.ik_global_bone_positions.set(vec3());
    state.ik_global_bone_rotations.set(quat());
    const G1LegConfig configs[2] = {
        g1_left_leg_config(), g1_right_leg_config()
    };
    for (int foot = 0; foot < 2; ++foot) {
        G1FootFrameResult& result = state.ik_frame.feet[foot];
        const vec3 sole_normal = quat_mul_vec3(
            state.ik_global_bone_rotations(configs[foot].contact),
            configs[foot].sole_normal_local);
        result.target.surface.normal = vec3(0.0f, 0.8f, 0.6f);
        result.target.desired_sole_normal = sole_normal;
        result.position.applied = false;
        result.position.reachable = foot == 1;
    }
}

static void test_ik_logging_uses_desired_normal_and_authoritative_reachability()
{
    g1_controller_state state;
    seed_ik_logging_state(state);
    motion_match_ik_diagnostic diagnostic;
    char error[256] = {};
    logging_check(
        g1_log_accepted_ik(diagnostic, state, error, sizeof(error)),
        "accepted IK diagnostic materializes");
    for (int foot = 0; foot < 2; ++foot) {
        logging_check(
            g1_ik_vec3_bits_equal(
                diagnostic.target_normal[foot],
                state.ik_frame.feet[foot].target.desired_sole_normal),
            "accepted IK target normal is the requested sole normal");
        logging_check(
            std::fabs(diagnostic.sole_normal_alignment[foot] - 1.0f) <
                1.0e-6f,
            "sole alignment compares against the requested sole normal");
        logging_check(
            diagnostic.reachable[foot] ==
                state.ik_frame.feet[foot].position.reachable,
            "accepted IK reachability uses the authoritative solver field");
    }

    G1FrameRejectionDiagnostic rejection;
    rejection.attempted_ik_available = true;
    rejection.ik_frame.feet[0].target.surface.normal =
        vec3(0.0f, 0.8f, 0.6f);
    rejection.ik_frame.feet[0].target.desired_sole_normal =
        vec3(0.0f, 1.0f, 0.0f);
    rejection.ik_frame.feet[0].position.applied = true;
    rejection.ik_frame.feet[0].position.reachable = false;
    motion_match_rejected_foot_diagnostic rejected;
    g1_log_rejected_foot(rejected, rejection, 0);
    logging_check(
        !rejected.reachable,
        "rejected IK reachability uses the authoritative solver field");
    logging_check(
        g1_ik_vec3_bits_equal(
            rejected.target_normal,
            rejection.ik_frame.feet[0].target.desired_sole_normal),
        "rejected IK target normal is the requested sole normal");
}

static void test_heading_error_checked_normalizes_inputs()
{
    const float half_sqrt_two = 0.7071067811865475f;
    const quat unit_yaw(half_sqrt_two, 0.0f, half_sqrt_two, 0.0f);
    const quat scaled_yaw = unit_yaw * 2.0f;
    logging_check(
        std::fabs(g1_log_heading_error_degrees(scaled_yaw, unit_yaw)) <
            1.0e-5,
        "heading error checked-normalizes the desired quaternion");
    logging_check(
        std::fabs(g1_log_heading_error_degrees(unit_yaw, scaled_yaw)) <
            1.0e-5,
        "heading error checked-normalizes the actual quaternion");
    logging_check(
        !terrain_double_is_finite(
            g1_log_heading_error_degrees(quat(0.0f, 0.0f, 0.0f, 0.0f),
                                         unit_yaw)) &&
            !terrain_double_is_finite(
                g1_log_heading_error_degrees(
                    unit_yaw,
                    quat(std::numeric_limits<float>::infinity(),
                         0.0f, 0.0f, 0.0f))),
        "heading error fails closed for zero and nonfinite quaternions");
}

static void test_directional_footprint_reason_uses_authoritative_name()
{
    g1_controller_state state;
    state.ik_global_bone_rotations.resize(1);
    state.ik_global_bone_rotations(0) = quat();
    G1FramePublication publication;
    publication.requested_intent.desired_heading = quat();
    const walkability_reason reasons[] = {
        walkability_clear,
        walkability_blocked_cell,
        walkability_out_of_bounds,
        walkability_nonfinite,
    };
    for (const walkability_reason reason : reasons) {
        state.footprint.blocked_reason = reason;
        motion_match_directional_diagnostic diagnostic;
        g1_log_directional(diagnostic, state, publication);
        logging_check(
            std::strcmp(
                diagnostic.footprint_blocked_reason,
                walkability_reason_name(reason)) == 0,
            "directional footprint reason uses the authoritative name");
    }
}

int main()
{
    test_controller_source_and_no_main_contract();
    test_accepted_state_digest_ignores_padding_and_tracks_values();
    test_accepted_row_uses_authoritative_names();
    test_first_rejected_row_uses_canonical_accepted_baseline();
    test_accepted_row_rejects_ambiguous_or_missing_provenance();
    test_cross_source_transition_maps_each_provenance_owner();
    test_ik_logging_uses_desired_normal_and_authoritative_reachability();
    test_heading_error_checked_normalizes_inputs();
    test_directional_footprint_reason_uses_authoritative_name();
    return 0;
}
