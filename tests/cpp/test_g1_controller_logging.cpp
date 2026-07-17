#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-result"
#endif

#define G1_FRAME_TRANSACTION_ENABLE_TEST_SEAM
#define main g1_controller_logging_oracle_embedded_main
#include "test_g1_frame_transaction_production.cpp"
#undef main
#undef G1_FRAME_TRANSACTION_ENABLE_TEST_SEAM

static G1FrameTransactionStatus
g1_controller_logging_frame_transaction_run(
    G1FrameRuntime& runtime,
    G1FrameStageRunner runner,
    G1RecoveryProvider provider,
    const G1FrameExternalInputs& external,
    char* error,
    int capacity)
{
    return ::g1_frame_transaction_run(
        runtime,
        runner,
        provider,
        external,
        nullptr,
        error,
        capacity);
}

#define g1_frame_transaction_run \
    g1_controller_logging_frame_transaction_run
#define main g1_controller_logging_embedded_main
#include "controller.cpp"
#undef main
#undef g1_frame_transaction_run

#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#include "sha256.h"

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <string>
#include <unistd.h>
#include <vector>

static void logging_check(bool condition, const char* message)
{
    if (!condition) {
        std::fprintf(stderr, "G1 controller logging test failed: %s\n", message);
        std::exit(1);
    }
}

static bool logging_work_equal(
    const motion_match_clearance_work_diagnostic& first,
    const motion_match_clearance_work_diagnostic& second)
{
    return first.point_queries == second.point_queries &&
           first.cells_visited == second.cells_visited &&
           first.primitive_triangle_pairs ==
               second.primitive_triangle_pairs &&
           first.face_patches == second.face_patches &&
           first.candidate_tests == second.candidate_tests &&
           first.subdivision_nodes == second.subdivision_nodes;
}

static bool logging_swing_equal(
    const motion_match_swing_diagnostic& first,
    const motion_match_swing_diagnostic& second)
{
    return first.candidates_evaluated == second.candidates_evaluated &&
           first.selected_index == second.selected_index &&
           first.selected_lift_bits == second.selected_lift_bits &&
           first.materialized_command_y_bits ==
               second.materialized_command_y_bits &&
           std::strcmp(
               first.actual_sphere_center_bits_hex,
               second.actual_sphere_center_bits_hex) == 0 &&
           std::strcmp(
               first.selected_clearance_status,
               second.selected_clearance_status) == 0 &&
           first.selected_controller_constraints_passed ==
               second.selected_controller_constraints_passed &&
           first.selected_clearance_certified ==
               second.selected_clearance_certified &&
           g1_frame_double_bits_equal(
               first.lower_margin, second.lower_margin) &&
           g1_frame_double_bits_equal(
               first.witness_upper_margin,
               second.witness_upper_margin) &&
           logging_work_equal(first.selected_work, second.selected_work) &&
           logging_work_equal(first.total_work, second.total_work);
}

static bool logging_ik_equal(
    const motion_match_ik_diagnostic& first,
    const motion_match_ik_diagnostic& second)
{
    if (first.applied != second.applied ||
        first.safe_stop_requested != second.safe_stop_requested ||
        std::strcmp(first.stop_reason, second.stop_reason) != 0 ||
        !g1_frame_float_bits_equal(
            first.max_correction, second.max_correction) ||
        !g1_frame_float_bits_equal(
            first.actual_simulation_speed,
            second.actual_simulation_speed) ||
        first.candidate_rejected != second.candidate_rejected ||
        std::strcmp(
            first.candidate_clearance_status,
            second.candidate_clearance_status) != 0 ||
        !g1_frame_double_bits_equal(
            first.candidate_minimum_clearance,
            second.candidate_minimum_clearance) ||
        !g1_frame_double_bits_equal(
            first.hips_clearance, second.hips_clearance) ||
        !g1_frame_double_bits_equal(
            first.minimum_clearance, second.minimum_clearance)) {
        return false;
    }
    for (int foot = 0; foot < 2; ++foot) {
        if (!g1_frame_double_bits_equal(
                first.candidate_toe_clearance[foot],
                second.candidate_toe_clearance[foot]) ||
            !g1_frame_double_bits_equal(
                first.candidate_foot_clearance[foot],
                second.candidate_foot_clearance[foot]) ||
            first.recorded_contact[foot] !=
                second.recorded_contact[foot] ||
            first.locked[foot] != second.locked[foot] ||
            !g1_frame_float_bits_equal(
                first.observed_lock_drift[foot],
                second.observed_lock_drift[foot]) ||
            !g1_frame_float_bits_equal(
                first.lock_drift[foot], second.lock_drift[foot]) ||
            !g1_frame_float_bits_equal(
                first.sole_normal_alignment[foot],
                second.sole_normal_alignment[foot]) ||
            !g1_frame_float_bits_equal(
                first.contact_residual[foot],
                second.contact_residual[foot]) ||
            !g1_frame_float_bits_equal(
                first.target_height[foot],
                second.target_height[foot]) ||
            !g1_frame_vec3_bits_equal(
                first.target_normal[foot],
                second.target_normal[foot]) ||
            !logging_swing_equal(
                first.swing[foot], second.swing[foot]) ||
            first.reachable[foot] != second.reachable[foot] ||
            !g1_frame_double_bits_equal(
                first.knee_clearance[foot],
                second.knee_clearance[foot]) ||
            !g1_frame_double_bits_equal(
                first.ankle_clearance[foot],
                second.ankle_clearance[foot]) ||
            !g1_frame_double_bits_equal(
                first.toe_clearance[foot],
                second.toe_clearance[foot]) ||
            !g1_frame_double_bits_equal(
                first.foot_clearance[foot],
                second.foot_clearance[foot]) ||
            !g1_frame_double_bits_equal(
                first.shin_clearance[foot],
                second.shin_clearance[foot]) ||
            !g1_frame_double_bits_equal(
                first.thigh_clearance[foot],
                second.thigh_clearance[foot])) {
            return false;
        }
    }
    return true;
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
    const G1RootReachPlan canonical_plan =
        rich_state.ik_frame.root_reach;
    const uint64_t canonical_plan_digest =
        g1_log_accepted_state_digest(rich_state);
    const auto check_plan_mutation = [
        &rich_state,
        &check_oracle,
        &canonical_plan,
        canonical_plan_digest](
        const G1RootReachPlan& plan,
        const char* message) {
        rich_state.ik_frame.root_reach = plan;
        check_oracle(message);
        logging_check(
            g1_log_accepted_state_digest(rich_state) !=
                canonical_plan_digest,
            message);
        rich_state.ik_frame.root_reach = canonical_plan;
    };
    G1RootReachPlan mutated_plan = canonical_plan;
    mutated_plan.active = !mutated_plan.active;
    check_plan_mutation(
        mutated_plan,
        "production and independent digests own root-plan active");
    mutated_plan = canonical_plan;
    mutated_plan.common_interval_found =
        !mutated_plan.common_interval_found;
    check_plan_mutation(
        mutated_plan,
        "production and independent digests own root-plan common interval");
    mutated_plan = canonical_plan;
    mutated_plan.applied = !mutated_plan.applied;
    check_plan_mutation(
        mutated_plan,
        "production and independent digests own root-plan applied");
    mutated_plan = canonical_plan;
    mutated_plan.root_y_delta_m = 0.03125f;
    check_plan_mutation(
        mutated_plan,
        "production and independent digests own exact root-plan delta bits");
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

enum bounded_fixture_kind
{
    bounded_fixture_legacy,
    bounded_fixture_recovery,
    bounded_fixture_incumbent,
};

static void prepare_bounded_fixture(
    fixture& value,
    G1CandidateCertificationTrace& trace,
    bounded_fixture_kind kind,
    bool ik_enabled = false)
{
    if (kind == bounded_fixture_legacy) {
        configure_real_candidate_fixture(value, false, false);
    } else if (kind == bounded_fixture_recovery) {
        configure_real_candidate_fixture(value, true, false);
    } else {
        configure_real_candidate_fixture(value, true, true, 159, false);
    }
    value.external.input.presentation_frame = 0;
    value.external.tuning.ik_enabled = ik_enabled;
    char error[1024] = {};
    const G1FrameTransactionStatus status = run_real_candidate_fixture(
        value,
        trace,
        ::g1_recovery_candidates_build,
        kind == bounded_fixture_recovery
            ? real_candidate_c_guard
            : nullptr,
        error,
        static_cast<int>(sizeof(error)));
    logging_check(
        status == G1FrameTransactionAccepted,
        error[0] != '\0'
            ? error
            : "authentic bounded logging fixture is accepted");
    logging_check(
        value.runtime.accepted_diagnostic.ready &&
            value.runtime.accepted_diagnostic.presentation_frame == 0 &&
            value.runtime.publication.presentation_frame == 0,
        "authentic bounded logging fixture owns frame-zero provenance");
}

static bool build_bounded_row(
    motion_match_log_row& row,
    fixture& value,
    bool ik_enabled,
    char* error,
    int error_capacity)
{
    motion_source_record source;
    source.range_start = 0;
    source.range_stop = value.db.nframes();
    const char* source_names[1] = {"bounded-source"};
    const char* source_terrains[1] = {"flat"};
    const char* scene_ids[1] = {"bounded-logging-scene"};
    int active_scene_index = 0;
    int scene_generation = 1;
    int scene_reset_count = 1;
    bool scene_switch_failed = false;
    int motion_pack_load_count = 1;
    int model_load_count = 1;
    int model_unload_count = 0;
    G1AcceptedLogContext context;
    context.mode = "flat";
    context.route = "manual";
    context.ik_enabled = ik_enabled;
    context.db = &value.db;
    context.sources = &source;
    context.source_names = source_names;
    context.source_terrains = source_terrains;
    context.source_count = 1;
    context.scene_ids = scene_ids;
    context.active_scene_index = &active_scene_index;
    context.scene_generation = &scene_generation;
    context.scene_reset_count = &scene_reset_count;
    context.scene_switch_failed = &scene_switch_failed;
    context.motion_pack_load_count = &motion_pack_load_count;
    context.model_load_count = &model_load_count;
    context.model_unload_count = &model_unload_count;
    return g1_build_accepted_log_row(
               row,
               value.runtime.accepted_state,
               value.runtime.accepted_diagnostic,
               value.runtime.publication,
               context,
               error,
               error_capacity) &&
           g1_build_task7_log_suffix(
               row,
               value.runtime.accepted_state,
               value.runtime.accepted_diagnostic,
               value.runtime.publication,
               ik_enabled,
               error,
               error_capacity);
}

static std::vector<std::string> split_csv_line(
    const std::string& line)
{
    std::vector<std::string> fields;
    std::size_t begin = 0U;
    for (;;) {
        const std::size_t comma = line.find(',', begin);
        fields.push_back(line.substr(
            begin,
            comma == std::string::npos
                ? std::string::npos
                : comma - begin));
        if (comma == std::string::npos) break;
        begin = comma + 1U;
    }
    return fields;
}

static std::string serialized_ik_columns(
    const motion_match_log_row& row)
{
    char path[] = "/tmp/g1_task4_logging_row_XXXXXX";
    const int fd = mkstemp(path);
    logging_check(fd >= 0 && close(fd) == 0,
                  "temporary serialized row path is available");
    motion_match_log log;
    char error[512] = {};
    logging_check(log.open(path, error, sizeof(error)), error);
    logging_check(log.write(row, error, sizeof(error)), error);
    logging_check(log.close(error, sizeof(error)), error);
    const std::string serialized = read_source_file(path);
    logging_check(std::remove(path) == 0,
                  "temporary serialized row is removed");
    const std::size_t header_end = serialized.find('\n');
    const std::size_t row_end = serialized.find('\n', header_end + 1U);
    logging_check(header_end != std::string::npos &&
                      row_end != std::string::npos,
                  "production writer emits one header and one row");
    const std::vector<std::string> header = split_csv_line(
        serialized.substr(0U, header_end));
    const std::vector<std::string> values = split_csv_line(
        serialized.substr(
            header_end + 1U,
            row_end - header_end - 1U));
    logging_check(header.size() == values.size(),
                  "serialized row has the production header width");
    std::size_t begin = header.size();
    std::size_t end = header.size();
    for (std::size_t index = 0U; index < header.size(); ++index) {
        if (header[index] == "ik_applied") begin = index;
        if (header[index] == "requested_velocity_x") end = index;
    }
    logging_check(begin < end && end <= values.size(),
                  "serialized IK suffix has exact production boundaries");
    std::string output;
    for (std::size_t index = begin; index < end; ++index) {
        if (index != begin) output.push_back(',');
        output += values[index];
    }
    return output;
}

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

static void poison_clearance_work(
    G1ClearanceWork& work,
    uint32_t seed)
{
    work.point_queries = seed + 1U;
    work.cells_visited = seed + 2U;
    work.primitive_triangle_pairs = seed + 3U;
    work.face_patches = seed + 4U;
    work.candidate_tests = seed + 5U;
    work.subdivision_nodes = seed + 6U;
}

static void poison_clearance_result(
    G1ClearanceResult& result,
    uint32_t seed)
{
    result.lower_bound_m = 0.25 + static_cast<double>(seed);
    result.witness_upper_m = 0.50 + static_cast<double>(seed);
    result.witness.body_x = 1.0 + seed;
    result.witness.body_y = 2.0 + seed;
    result.witness.body_z = 3.0 + seed;
    result.witness.surface_x = 4.0 + seed;
    result.witness.surface_y = 5.0 + seed;
    result.witness.surface_z = 6.0 + seed;
    result.witness.segment_parameter = 0.125;
    result.witness.terrain_weight_0 = 0.25;
    result.witness.terrain_weight_1 = 0.50;
    result.witness.terrain_weight_2 = 0.25;
    result.witness.primitive_index = seed + 7U;
    result.witness.cell_x = static_cast<int32_t>(seed + 8U);
    result.witness.cell_z = -static_cast<int32_t>(seed + 9U);
    result.witness.terrain_triangle_index = seed + 10U;
    result.witness.patch_index = seed + 11U;
    result.witness.candidate_kind = seed + 12U;
    result.witness.candidate_subindex = seed + 13U;
    poison_clearance_work(result.work, seed + 14U);
}

static void poison_hidden_ik_owner(g1_controller_state& state)
{
    state.ik.initialized = !state.ik.initialized;
    for (int foot = 0; foot < 2; ++foot) {
        G1FootIkState& foot_state = state.ik.feet[foot];
        G1FootLockState& lock = foot_state.lock;
        lock.initialized = !lock.initialized;
        lock.contact = !lock.contact;
        lock.locked = !lock.locked;
        lock.position_active = !lock.position_active;
        lock.releasing = !lock.releasing;
        lock.release_frames = 17 + foot;
        lock.previous_input = vec3(1.0f + foot, 2.0f, 3.0f);
        lock.lock_point = vec3(4.0f + foot, 5.0f, 6.0f);
        lock.output_position = vec3(7.0f + foot, 8.0f, 9.0f);
        lock.output_velocity = vec3(10.0f + foot, 11.0f, 12.0f);
        lock.offset_position = vec3(13.0f + foot, 14.0f, 15.0f);
        lock.offset_velocity = vec3(16.0f + foot, 17.0f, 18.0f);
        foot_state.swing.initialized = !foot_state.swing.initialized;
        for (int probe = 0; probe < 4; ++probe) {
            foot_state.swing.previous_sphere_centers[probe] = vec3(
                20.0f + foot,
                21.0f + probe,
                22.0f + foot + probe);
        }
        foot_state.baseline_sole_normal =
            vec3(0.125f + foot, 0.25f, 0.5f);

        G1FootFrameResult& result = state.ik_frame.feet[foot];
        result.recorded_contact = !result.recorded_contact;
        result.target.locked = !result.target.locked;
        result.target.position_active = !result.target.position_active;
        result.target.releasing = !result.target.releasing;
        result.target.drift_limit_exceeded =
            !result.target.drift_limit_exceeded;
        result.target.surface.point = vec3(23.0f + foot, 24.0f, 25.0f);
        result.target.surface.normal = vec3(0.0f, 0.6f, 0.8f);
        result.target.desired_sole_normal = vec3(0.8f, 0.6f, 0.0f);
        result.target.sole_center = vec3(26.0f + foot, 27.0f, 28.0f);
        result.target.horizontal_drift_m = 0.375f + foot;
        result.swing_selection.candidates_evaluated = 31U + foot;
        result.swing_selection.selected_index = 7U + foot;
        result.swing_selection.selected.candidate_index = 7U + foot;
        result.swing_selection.selected.lift_bits =
            UINT32_C(0x3e800000) + static_cast<uint32_t>(foot);
        result.swing_selection.selected.materialized_command_y_bits =
            UINT32_C(0x3f400000) + static_cast<uint32_t>(foot);
        result.swing_selection.selected.actual_sphere_center_bits[0][0] =
            UINT32_C(0x3f800000) + static_cast<uint32_t>(foot);
        result.swing_selection.selected.clearance_status = G1ClearanceOk;
        result.swing_selection.selected.controller_constraints_passed = true;
        result.swing_selection.selected.clearance_certified = true;
        result.swing_selection.selected.lower_margin_m = 0.0625 + foot;
        result.swing_selection.selected.witness_upper_margin_m =
            0.03125 + foot;
        poison_clearance_work(
            result.swing_selection.selected.clearance_work,
            40U + static_cast<uint32_t>(foot));
        poison_clearance_work(
            result.swing_selection.total_clearance_work,
            50U + static_cast<uint32_t>(foot));
        result.defensive_swing.lower_margin_m = 0.75 + foot;
        result.defensive_swing.witness_upper_m = 0.875 + foot;
        result.defensive_swing.sweep_evaluated = true;
        poison_clearance_work(
            result.defensive_swing.work,
            60U + static_cast<uint32_t>(foot));
        result.position.applied = true;
        result.position.reachable = foot == 0;
        result.position.correction_limited = true;
        result.position.safe_stop_requested = true;
        result.position.iterations = 3 + foot;
        result.position.iteration_provenance = G1LegIterationContact3;
        result.position.requested_ankle_target = vec3(29.0f, 30.0f, 31.0f);
        result.position.clamped_ankle_target = vec3(32.0f, 33.0f, 34.0f);
        result.position.hinge_axis_world = vec3(1.0f, 0.0f, 0.0f);
        result.position.bend_direction = vec3(0.0f, 0.0f, 1.0f);
        result.position.bend_used_current_projection = true;
        result.position.bend_used_hinge_fallback = true;
        result.position.bend_used_safe_perpendicular = true;
        result.position.bend_sign_flipped = true;
        result.position.raw_distance_m = 1.25f + foot;
        result.position.clamped_distance_m = 1.5f + foot;
        result.position.max_correction_radians = 0.5f + foot;
        result.position.contact_residual_m = 0.25f + foot;
        result.orientation.applied = true;
        result.orientation.correction_limited = true;
        result.orientation.safe_stop_requested = true;
        result.orientation.target_global_rotation =
            quat(0.5f, 0.5f, 0.5f, 0.5f);
        result.orientation.requested_correction_radians = 0.625f + foot;
        result.orientation.correction_radians = 0.5f + foot;
    }
    state.ik_frame.applied = !state.ik_frame.applied;
    state.ik_frame.safe_stop_requested = !state.ik_frame.safe_stop_requested;
    state.ik_frame.stop_reason = G1IkStopPoseClearanceRejected;
    state.ik_frame.max_correction_radians = 0.875f;
    state.ik_frame.root_reach.active = true;
    state.ik_frame.root_reach.common_interval_found = true;
    state.ik_frame.root_reach.applied = true;
    state.ik_frame.root_reach.root_y_delta_m = 0.03125f;

    state.ik_candidate_rejected = !state.ik_candidate_rejected;
    state.ik_candidate_clearance_status = G1ClearanceOk;
    poison_clearance_result(state.ik_candidate_clearance.left.toe, 70U);
    poison_clearance_result(state.ik_candidate_clearance.left.foot, 71U);
    poison_clearance_result(state.ik_candidate_clearance.right.toe, 72U);
    poison_clearance_result(state.ik_candidate_clearance.right.foot, 73U);
    poison_clearance_result(state.ik_candidate_clearance.minimum, 74U);
    poison_clearance_result(state.ik_clearance.hips, 80U);
    poison_clearance_result(state.ik_clearance.left.knee, 81U);
    poison_clearance_result(state.ik_clearance.left.ankle, 82U);
    poison_clearance_result(state.ik_clearance.left.toe, 83U);
    poison_clearance_result(state.ik_clearance.left.foot, 84U);
    poison_clearance_result(state.ik_clearance.left.shin, 85U);
    poison_clearance_result(state.ik_clearance.left.thigh, 86U);
    poison_clearance_result(state.ik_clearance.right.knee, 87U);
    poison_clearance_result(state.ik_clearance.right.ankle, 88U);
    poison_clearance_result(state.ik_clearance.right.toe, 89U);
    poison_clearance_result(state.ik_clearance.right.foot, 90U);
    poison_clearance_result(state.ik_clearance.right.shin, 91U);
    poison_clearance_result(state.ik_clearance.right.thigh, 92U);
    poison_clearance_result(state.ik_clearance.minimum, 93U);

    state.ik_bone_positions(0) = vec3(35.0f, 36.0f, 37.0f);
    state.ik_bone_rotations(0) = quat(0.5f, 0.5f, 0.5f, 0.5f);
    state.ik_global_bone_positions(0) = vec3(38.0f, 39.0f, 40.0f);
    state.ik_global_bone_rotations(0) = quat(0.5f, -0.5f, 0.5f, 0.5f);
    state.ik_candidate_bone_positions(0) = vec3(41.0f, 42.0f, 43.0f);
    state.ik_candidate_bone_rotations(0) =
        quat(-0.5f, 0.5f, 0.5f, 0.5f);
    state.ik_candidate_global_bone_positions(0) =
        vec3(44.0f, 45.0f, 46.0f);
    state.ik_candidate_global_bone_rotations(0) =
        quat(0.5f, 0.5f, -0.5f, 0.5f);
}

static void test_disabled_projection_is_exact_default_under_hidden_poison()
{
    fixture clean;
    fixture poisoned;
    G1CandidateCertificationTrace clean_trace;
    G1CandidateCertificationTrace poisoned_trace;
    prepare_bounded_fixture(
        clean, clean_trace, bounded_fixture_recovery, false);
    prepare_bounded_fixture(
        poisoned, poisoned_trace, bounded_fixture_recovery, false);
    logging_check(
        g1_frame_controller_states_equal(
            clean.runtime.accepted_state,
            poisoned.runtime.accepted_state) &&
            g1_frame_accepted_diagnostic_equal(
                clean.runtime.accepted_diagnostic,
                poisoned.runtime.accepted_diagnostic) &&
            clean.runtime.publication.presentation_frame ==
                poisoned.runtime.publication.presentation_frame &&
            g1_frame_intent_bits_equal(
                clean.runtime.publication.requested_intent,
                poisoned.runtime.publication.requested_intent),
        "valid disabled fixtures begin with identical raw/public products");

    poison_hidden_ik_owner(poisoned.runtime.accepted_state);
    motion_match_log_row clean_row;
    motion_match_log_row poisoned_row;
    char error[1024] = {};
    logging_check(
        build_bounded_row(
            clean_row, clean, false, error, static_cast<int>(sizeof(error))),
        error);
    error[0] = '\0';
    logging_check(
        build_bounded_row(
            poisoned_row,
            poisoned,
            false,
            error,
            static_cast<int>(sizeof(error))),
        error);
    const motion_match_ik_diagnostic canonical{};
    motion_match_ik_diagnostic direct_canonical;
    g1_log_canonical_disabled_ik(direct_canonical);
    logging_check(
        logging_ik_equal(direct_canonical, canonical) &&
            logging_ik_equal(clean_row.ik, canonical) &&
            logging_ik_equal(poisoned_row.ik, canonical),
        "disabled IK projection is the exact independent default value");
    logging_check(
        std::strcmp(clean_row.ik.candidate_clearance_status,
                    "invalid-input") == 0 &&
            std::strcmp(poisoned_row.ik.candidate_clearance_status,
                        "invalid-input") == 0 &&
            std::strcmp(clean_row.ik.stop_reason, "none") == 0 &&
            std::strcmp(poisoned_row.ik.stop_reason, "none") == 0 &&
            clean_row.ik.swing[0].selected_index == UINT32_MAX &&
            clean_row.ik.swing[1].selected_index == UINT32_MAX &&
            poisoned_row.ik.swing[0].selected_index == UINT32_MAX &&
            poisoned_row.ik.swing[1].selected_index == UINT32_MAX &&
            std::strcmp(clean_row.ik.swing[0].selected_clearance_status,
                        "invalid-input") == 0 &&
            std::strcmp(clean_row.ik.swing[1].selected_clearance_status,
                        "invalid-input") == 0 &&
            std::strcmp(
                poisoned_row.ik.swing[0].selected_clearance_status,
                "invalid-input") == 0 &&
            std::strcmp(
                poisoned_row.ik.swing[1].selected_clearance_status,
                "invalid-input") == 0,
        "disabled IK projection retains exact strings and sentinels");
    logging_check(
        serialized_ik_columns(clean_row) ==
            serialized_ik_columns(poisoned_row),
        "disabled serialized IK suffix is byte-identical under poison");
}

static void test_enabled_projection_still_uses_certified_ik_products()
{
    fixture value;
    G1CandidateCertificationTrace trace;
    prepare_bounded_fixture(value, trace, bounded_fixture_recovery, true);
    motion_match_log_row row;
    motion_match_ik_diagnostic expected;
    char error[1024] = {};
    logging_check(
        build_bounded_row(
            row, value, true, error, static_cast<int>(sizeof(error))),
        error);
    error[0] = '\0';
    logging_check(
        g1_log_accepted_ik(
            expected,
            value.runtime.accepted_state,
            error,
            static_cast<int>(sizeof(error))),
        error);
    logging_check(
        logging_ik_equal(row.ik, expected),
        "enabled projection still equals the certified IK logger");
    logging_check(
        row.ik.applied &&
            (row.ik.swing[0].selected_index != UINT32_MAX ||
             row.ik.swing[1].selected_index != UINT32_MAX ||
             row.ik.minimum_clearance != 0.0),
        "enabled projection retains nondefault certified IK products");
}

static void test_disabled_projection_has_no_hidden_owner_source_path()
{
    const std::string source = read_source_file("controller.cpp");
    std::string token_error;
    const std::vector<CppToken> tokens =
        tokenize_cpp_source(source, token_error);
    logging_check(token_error.empty(), "controller source tokenizes");

    ExactFunctionRange canonical;
    logging_check(
        cpp_exact_function_range(
            tokens,
            {"static", "void", "g1_log_canonical_disabled_ik", "(",
             "motion_match_ik_diagnostic", "&", "output", ")"},
            canonical),
        "state-free canonical disabled IK logger has the exact signature");
    logging_check(
        cpp_identifier_count(
            tokens,
            canonical.body_begin + 1U,
            canonical.body_end,
            "output") == 1U,
        "canonical disabled logger has exactly one output reference");
    const char* forbidden[] = {
        "ik", "state", "lock", "swing", "root_reach", "clearance",
        "g1_log_accepted_ik",
    };
    for (const char* identifier : forbidden) {
        logging_check(
            cpp_identifier_count(
                tokens,
                canonical.signature,
                canonical.body_end,
                identifier) == 0U,
            "canonical disabled logger has no hidden owner identifier");
    }
    logging_check(
        cpp_identifier_count(
            tokens,
            canonical.signature,
            canonical.body_begin,
            "g1_controller_state") == 0U,
        "canonical disabled logger has no controller-state parameter");

    ExactFunctionRange suffix;
    logging_check(
        cpp_exact_function_range(
            tokens,
            {"static", "bool", "g1_build_task7_log_suffix", "(",
             "motion_match_log_row", "&", "log_row", ",",
             "const", "g1_controller_state", "&", "accepted_state", ",",
             "const", "G1FrameAcceptedDiagnostic", "&",
             "accepted_diagnostic", ",", "const", "G1FramePublication",
             "&", "publication", ",", "bool", "ik_enabled", ",",
             "char", "*", "error", ",", "int", "error_capacity", ")"},
            suffix),
        "log suffix builder receives the explicit immutable IK mode");
    logging_check(
        cpp_identifier_count(
            tokens,
            suffix.body_begin + 1U,
            suffix.body_end,
            "g1_log_accepted_ik") == 1U &&
            cpp_identifier_count(
                tokens,
                suffix.body_begin + 1U,
                suffix.body_end,
                "g1_log_canonical_disabled_ik") == 1U,
        "suffix contains exactly one enabled and one disabled IK logger");
    const std::string compact = cpp_compact_tokens(
        tokens, suffix.body_begin + 1U, suffix.body_end);
    const std::size_t enabled_if = compact.find("if(ik_enabled){");
    const std::size_t accepted_call = compact.find("g1_log_accepted_ik(");
    const std::size_t opposite = compact.find("}else{");
    const std::size_t canonical_call = compact.find(
        "g1_log_canonical_disabled_ik(");
    logging_check(
        enabled_if != std::string::npos &&
            accepted_call > enabled_if &&
            opposite > accepted_call &&
            canonical_call > opposite,
        "accepted and canonical loggers are confined to opposite mode branches");
}

static void test_log_schema_and_runtime_checker_hashes_are_unchanged()
{
    struct HashExpectation
    {
        const char* path;
        const char* expected;
    };
    const HashExpectation expectations[] = {
        {"motion_match_log.h",
         "470dcd5978fe6f03e9398dc845c5454f70d5a6f1958e54014dc0eda69d9c5586"},
        {"resources/check_g1_runtime_log.py",
         "1a45700fd1066de1f6b06e0defd1ecddb4b8fe1a067d739755e85fac8df1e565"},
    };
    for (const HashExpectation& expectation : expectations) {
        std::string observed;
        char error[512] = {};
        logging_check(
            sha256_file_hex(
                observed,
                expectation.path,
                error,
                static_cast<int>(sizeof(error))),
            error);
        logging_check(
            observed == expectation.expected,
            "immutable log schema/checker hash matches the Task 1 baseline");
    }
}

static void test_recovery_selected_cost_uses_strict_word_and_checker_rule()
{
    fixture value;
    G1CandidateCertificationTrace trace;
    prepare_bounded_fixture(value, trace, bounded_fixture_recovery, false);
    logging_check(
        trace.attempt_count >= 2U &&
            trace.attempts[1].score_owner == G1CandidateScoreStrictRecovery &&
            trace.attempts[1].common == G1CandidateDispositionAccepted &&
            trace.attempts[1].raw == G1CandidateDispositionAccepted,
        "recovery fixture admits the authentic strict scored record");
    motion_match_log_row row;
    char error[1024] = {};
    logging_check(
        build_bounded_row(
            row, value, false, error, static_cast<int>(sizeof(error))),
        error);
    logging_check(
        terrain_float_bits(row.selected_cost) ==
                terrain_float_bits(trace.attempts[1].candidate.selected_cost) &&
            terrain_float_bits(row.incumbent_cost) !=
                terrain_float_bits(FLT_MAX) &&
            row.selected_cost < row.incumbent_cost &&
            row.searched && row.transitioned,
        "recovery row publishes the strict word that numerically beats incumbent");
    const std::string checker =
        read_source_file("resources/check_g1_runtime_log.py");
    logging_check(
        checker.find("if transitioned and not selected < incumbent:") !=
                std::string::npos &&
            checker.find("if cost_ulps > 4:") != std::string::npos,
        "unchanged checker retains strict comparison and bounded legacy exception");
}

static void test_end_of_animation_incumbent_keeps_public_sentinels()
{
    fixture value;
    G1CandidateCertificationTrace trace;
    prepare_bounded_fixture(value, trace, bounded_fixture_incumbent, false);
    logging_check(
        trace.attempt_count == 4U &&
            trace.attempts[3].candidate.kind == G1CandidateIncumbent,
        "end-of-animation fixture reaches the authentic incumbent fallback");
    motion_match_log_row row;
    char error[1024] = {};
    logging_check(
        build_bounded_row(
            row, value, false, error, static_cast<int>(sizeof(error))),
        error);
    logging_check(
        terrain_float_bits(row.incumbent_cost) ==
                terrain_float_bits(FLT_MAX) &&
            terrain_float_bits(row.selected_cost) ==
                terrain_float_bits(FLT_MAX),
        "end-of-animation incumbent keeps both exact public sentinels");
}

static void emit_bounded_fixtures(const char* directory)
{
    struct FixtureOutput
    {
        bounded_fixture_kind kind;
        const char* name;
    };
    const FixtureOutput outputs[] = {
        {bounded_fixture_legacy, "legacy.csv"},
        {bounded_fixture_recovery, "recovery.csv"},
        {bounded_fixture_incumbent, "incumbent.csv"},
    };
    for (const FixtureOutput& output : outputs) {
        fixture value;
        G1CandidateCertificationTrace trace;
        prepare_bounded_fixture(value, trace, output.kind, false);
        motion_match_log_row row;
        char error[1024] = {};
        logging_check(
            build_bounded_row(
                row, value, false, error, static_cast<int>(sizeof(error))),
            error);
        const std::string path =
            std::string(directory) + "/" + output.name;
        motion_match_log log;
        logging_check(log.open(path.c_str(), error, sizeof(error)), error);
        logging_check(log.write(row, error, sizeof(error)), error);
        logging_check(log.close(error, sizeof(error)), error);
    }
}

int main(int argc, char** argv)
{
    if (argc == 3 &&
        std::strcmp(argv[1], "--emit-bounded-fixtures") == 0) {
        emit_bounded_fixtures(argv[2]);
        return 0;
    }
    if (argc != 1) return 2;
    test_controller_source_and_no_main_contract();
    test_accepted_state_digest_ignores_padding_and_tracks_values();
    test_accepted_row_uses_authoritative_names();
    test_first_rejected_row_uses_canonical_accepted_baseline();
    test_accepted_row_rejects_ambiguous_or_missing_provenance();
    test_cross_source_transition_maps_each_provenance_owner();
    test_ik_logging_uses_desired_normal_and_authoritative_reachability();
    test_heading_error_checked_normalizes_inputs();
    test_directional_footprint_reason_uses_authoritative_name();
    test_disabled_projection_is_exact_default_under_hidden_poison();
    test_enabled_projection_still_uses_certified_ik_products();
    test_disabled_projection_has_no_hidden_owner_source_path();
    test_log_schema_and_runtime_checker_hashes_are_unchanged();
    test_recovery_selected_cost_uses_strict_word_and_checker_rule();
    test_end_of_animation_incumbent_keeps_public_sentinels();
    return 0;
}
