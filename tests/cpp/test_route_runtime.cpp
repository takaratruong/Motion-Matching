#include "route_runtime.h"
#include "g1_runtime_diagnostics.h"
#include "motion_match_log.h"

#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>

static void check(bool value, const char* message)
{
    if (!value) {
        std::fprintf(stderr, "route runtime test failed: %s\n", message);
        std::exit(1);
    }
}

static uint32_t bits(float value)
{
    uint32_t out;
    std::memcpy(&out, &value, sizeof(out));
    return out;
}

static std::string read_source(const char* path)
{
    std::ifstream file(path, std::ios::binary);
    check(file.good(), "open source contract input");
    return std::string(
        std::istreambuf_iterator<char>(file),
        std::istreambuf_iterator<char>());
}

static size_t source_occurrence_count(
    const std::string& source, const std::string& needle)
{
    size_t count = 0;
    size_t position = 0;
    while ((position = source.find(needle, position)) != std::string::npos) {
        ++count;
        position += needle.size();
    }
    return count;
}

static std::string compact_source(const std::string& source)
{
    std::string output;
    output.reserve(source.size());
    for (const char value : source) {
        if (value != ' ' && value != '\t' &&
            value != '\r' && value != '\n') {
            output.push_back(value);
        }
    }
    return output;
}

static size_t task6_main_signature_position(const std::string& compact)
{
    const size_t void_signature = compact.find("intmain(void)");
    if (void_signature != std::string::npos) {
        return void_signature;
    }
    return compact.find("intmain(intargc,char**argv)");
}

static void test_task6_main_signature_forms()
{
    check(task6_main_signature_position(
              compact_source("int main(void) {}")) != std::string::npos,
          "publication audit recognizes conventional main(void)");
    check(task6_main_signature_position(
              compact_source("int main(int argc, char** argv) {}")) !=
              std::string::npos,
          "publication audit recognizes conventional main(argc, argv)");
}

static void check_task6_publication_contract(const char* path)
{
    const std::string source = read_source(path);
    const std::string compact = compact_source(source);
    const size_t main_signature = task6_main_signature_position(compact);
    check(main_signature != std::string::npos,
          "controller has one concrete production main");
    const size_t main_open = compact.find("{", main_signature);
    check(main_open != std::string::npos,
          "production main body opens");
    int depth = 0;
    size_t main_close = std::string::npos;
    for (size_t cursor = main_open; cursor < compact.size(); ++cursor) {
        if (compact[cursor] == '{') {
            ++depth;
        } else if (compact[cursor] == '}' && --depth == 0) {
            main_close = cursor;
            break;
        }
    }
    check(main_close != std::string::npos,
          "production main body closes");
    const std::string main_body = compact.substr(
        main_open, main_close - main_open + 1);

    const size_t coordinator = main_body.find(
        "g1_frame_transaction_run(frame_runtime,"
        "::g1_controller_frame_stage_run,frame_external,");
    check(coordinator != std::string::npos &&
              source_occurrence_count(
                  main_body, "g1_frame_transaction_run(") == 1,
          "Task6 outer loop must call the typed frame coordinator exactly once");
    check(main_body.find("autoupdate_func=[&]()") == std::string::npos,
          "Task6 outer loop contains no capturing update lambda");
    check(source_occurrence_count(
              main_body, "G1FrameRuntimeframe_runtime;") == 1 &&
          source_occurrence_count(
              main_body, "G1FrameExternalInputsframe_external;") == 1,
          "main owns one frame runtime and one immutable external snapshot");

    const size_t status = main_body.rfind(
        "constG1FrameTransactionStatusframe_status=", coordinator);
    const size_t global_error = main_body.find(
        "G1FrameTransactionGlobalError", coordinator);
    const size_t log_build = main_body.find(
        "g1_build_accepted_log_row(", coordinator);
    const size_t log_write = main_body.find(
        "deterministic_log.write(", log_build);
    const size_t camera = main_body.find(
        "update_g1_camera_from_accepted(", log_write);
    const size_t render = main_body.find("draw_g1_skeleton(", camera);
    check(status != std::string::npos && status <= coordinator &&
              global_error != std::string::npos &&
              log_build != std::string::npos &&
              log_write != std::string::npos &&
              camera != std::string::npos && render != std::string::npos &&
              coordinator < global_error && global_error < log_build &&
              log_build < log_write && log_write < camera && camera < render,
          "coordinator publication and status handling precede accepted-owner log, camera, and render");

    const std::string log_call = main_body.substr(
        log_build, log_write - log_build);
    check(log_call.find("frame_runtime.accepted_state") !=
              std::string::npos &&
          log_call.find("frame_runtime.accepted_diagnostic") !=
              std::string::npos &&
          log_call.find("frame_runtime.publication") !=
              std::string::npos &&
          log_call.find("frame_runtime.working_state") ==
              std::string::npos,
          "log rows are built only from accepted state/diagnostic and publication");

    const std::string camera_to_render = main_body.substr(
        camera, render - camera);
    check(camera_to_render.find(
              "frame_runtime.accepted_state.camera_azimuth") !=
              std::string::npos &&
          camera_to_render.find(
              "frame_runtime.accepted_state.camera_altitude") !=
              std::string::npos &&
          camera_to_render.find(
              "frame_runtime.accepted_state.camera_distance") !=
              std::string::npos &&
          camera_to_render.find(
              "frame_runtime.accepted_state.ik_global_bone_positions(0)") !=
              std::string::npos,
          "camera derives only from accepted camera scalars and final-FK root");
    const std::string render_tail = main_body.substr(render);
    check(render_tail.find(
              "frame_runtime.accepted_state.ik_global_bone_positions") !=
              std::string::npos &&
          render_tail.find(
              "frame_runtime.accepted_state.ik_global_bone_rotations") !=
              std::string::npos &&
          render_tail.find("frame_runtime.working_state") ==
              std::string::npos,
          "skeleton renders only the accepted final IK pose");

    check(main_body.find("working_state") == std::string::npos &&
          source_occurrence_count(main_body, "accepted_state") ==
              source_occurrence_count(
                  main_body, "frame_runtime.accepted_state") &&
          source_occurrence_count(main_body, "accepted_diagnostic") ==
              source_occurrence_count(
                  main_body, "frame_runtime.accepted_diagnostic") &&
          source_occurrence_count(main_body, "publication") ==
              source_occurrence_count(
                  main_body, "frame_runtime.publication"),
          "outer main has no accepted/working/publication aliases or alternate owners");

    const size_t move_snapshot = main_body.rfind(
        "frame_external.input.move_stick=", coordinator);
    const size_t look_snapshot = main_body.rfind(
        "frame_external.input.look_stick=", coordinator);
    const size_t ik_lowering = main_body.find(
        "frame_external.tuning.ik_enabled=process_config.ik_enabled;");
    check(move_snapshot != std::string::npos &&
              look_snapshot != std::string::npos &&
              ik_lowering != std::string::npos &&
              move_snapshot < coordinator && look_snapshot < coordinator &&
              ik_lowering < coordinator,
          "device input and startup-only IK are lowered before the coordinator");
}

static void check_route_sample_cursor_contract(const char* path)
{
    const std::string source = read_source(path);
    check(source.find("#include \"g1_command_runtime.h\"") !=
              std::string::npos,
          "route runtime directly includes the command contract");
    check(source.find(
              "for (int step = 0; step <= steps; ++step)") ==
              std::string::npos,
          "route target sampling does not increment INT_MAX int cursor");
    const size_t loop = source.find("for (int64_t sample_index = 0;");
    const size_t cast = source.find(
        "const int step = static_cast<int>(sample_index);", loop);
    check(loop != std::string::npos && cast != std::string::npos && loop < cast,
          "route target sampling uses a safe 64-bit cursor");
}

static deterministic_route_prediction poisoned_prediction()
{
    deterministic_route_prediction output;
    for (int index = 0; index < G1CommandTrajectorySampleCount; ++index) {
        output.commands[index] = vec3(
            90.0f + static_cast<float>(index),
            80.0f + static_cast<float>(index),
            70.0f + static_cast<float>(index));
        output.sampled_frames[index] = 60 + index;
    }
    output.force_search = true;
    return output;
}

static bool same_prediction(
    const deterministic_route_prediction& first,
    const deterministic_route_prediction& second)
{
    if (first.force_search != second.force_search) return false;
    for (int index = 0; index < G1CommandTrajectorySampleCount; ++index) {
        if (bits(first.commands[index].x) != bits(second.commands[index].x) ||
            bits(first.commands[index].y) != bits(second.commands[index].y) ||
            bits(first.commands[index].z) != bits(second.commands[index].z) ||
            first.sampled_frames[index] != second.sampled_frames[index]) {
            return false;
        }
    }
    return true;
}

static void check_prediction_failure(
    deterministic_route_prediction& output,
    const deterministic_route_prediction& before,
    bool result,
    const char* message)
{
    check(!result, message);
    check(same_prediction(output, before), message);
}

static scene_route route_fixture(
    const char* id,
    std::initializer_list<std::pair<float, float>> waypoints,
    float hold = 0.0f)
{
    scene_route route;
    route.id = id;
    route.expected_outcome = "traverse";
    route.walkability_class = 1;
    route.landing_hold_seconds = hold;
    route.waypoints_xz.assign(waypoints.begin(), waypoints.end());
    return route;
}

static void check_command_bits(
    vec3 command,
    float x,
    float y,
    float z,
    const char* message)
{
    check(bits(command.x) == bits(x) &&
              bits(command.y) == bits(y) &&
              bits(command.z) == bits(z),
          message);
}

static void test_four_horizon_route_predictions()
{
    static_assert(G1CommandTrajectorySampleCount == 4,
                  "route command horizon remains exactly four");
    const scene_route route = route_fixture(
        "corner", {{0.0f, 0.0f}, {0.1f, 0.0f}, {0.1f, 1.0f}});
    deterministic_route_prediction prediction = poisoned_prediction();
    char error[256] = {};
    check(deterministic_route_predict_commands(
              prediction,
              route,
              0,
              vec3(0.125f, -0.0f, -0.25f),
              0.04f,
              0.50f,
              1.0f / 3.0f,
              1.0f,
              false,
              error,
              sizeof(error)),
          error);
    const int expected_frames[] = {0, 9, 17, 25};
    for (int index = 0; index < G1CommandTrajectorySampleCount; ++index) {
        check(prediction.sampled_frames[index] == expected_frames[index],
              "four horizons use checked ceil route-time offsets");
    }
    check_command_bits(
        prediction.commands[0], 0.125f, 0.0f, -0.25f,
        "sample zero is current terrain-applied velocity");
    for (int index = 1; index < G1CommandTrajectorySampleCount; ++index) {
        check_command_bits(
            prediction.commands[index], 0.0f, 0.0f, 0.50f,
            "future corner samples follow deterministic route time");
    }
    check(!prediction.force_search,
          "ordinary route prediction does not force a search");
}

static void test_tangent_level_boundary_exact_schedule()
{
    const scene_route route = route_fixture(
        "tangent-level-boundary",
        {{0.0f, 0.0f}, {0.62f, 2.0f}, {0.62f, 6.0f}});
    check(deterministic_route_motion_frames(route) == 305,
          "tangent-level-boundary has the exact 305-frame schedule");

    char error[256] = {};
    deterministic_route_sample before_boundary;
    deterministic_route_sample at_boundary;
    check(deterministic_route_command(
              before_boundary, route, 104, 0.04f, 0.50f,
              error, sizeof(error)),
          error);
    check(deterministic_route_command(
              at_boundary, route, 105, 0.04f, 0.50f,
              error, sizeof(error)),
          error);
    check(before_boundary.waypoint == 1 && at_boundary.waypoint == 2,
          "tangent route changes segment at exact frame 105");

    deterministic_route_sample current;
    check(deterministic_route_command(
              current, route, 100, 0.04f, 0.50f,
              error, sizeof(error)),
          error);
    deterministic_route_prediction prediction = poisoned_prediction();
    check(deterministic_route_predict_commands(
              prediction, route, 100, current.command,
              0.04f, 0.50f, 1.0f / 3.0f, 1.0f, false,
              error, sizeof(error)),
          error);
    const int expected_frames[] = {100, 109, 117, 125};
    for (int index = 0; index < G1CommandTrajectorySampleCount; ++index) {
        check(prediction.sampled_frames[index] == expected_frames[index],
              "tangent route prediction publishes exact sampled frames");
        deterministic_route_sample oracle;
        check(deterministic_route_command(
                  oracle, route, expected_frames[index],
                  0.04f, 0.50f, error, sizeof(error)),
              error);
        check_command_bits(
            prediction.commands[index],
            oracle.command.x,
            oracle.command.y,
            oracle.command.z,
            "tangent route prediction is bit-exact at every horizon");
    }
}

static void test_hold_completion_and_speed_scales()
{
    const scene_route held = route_fixture(
        "positive-hold",
        {{0.0f, 0.0f}, {1.0f, 0.0f}, {1.0f, 1.0f}, {2.0f, 1.0f}},
        0.4f);
    check(deterministic_route_motion_frames(held) == 160,
          "positive hold has exact schedule");
    deterministic_route_prediction prediction = poisoned_prediction();
    char error[256] = {};
    check(deterministic_route_predict_commands(
              prediction, held, 90, vec3(0.0f, 0.0f, 0.50f),
              0.04f, 0.50f, 1.0f / 3.0f, 1.0f, false,
              error, sizeof(error)),
          error);
    const int hold_frames[] = {90, 99, 107, 115};
    for (int index = 0; index < G1CommandTrajectorySampleCount; ++index) {
        check(prediction.sampled_frames[index] == hold_frames[index],
              "hold fixture uses route-time horizons");
    }
    check_command_bits(
        prediction.commands[0], 0.0f, 0.0f, 0.50f,
        "current command is moving before hold");
    check_command_bits(
        prediction.commands[1], 0.0f, 0.0f, 0.50f,
        "future command is moving immediately before hold");
    check_command_bits(
        prediction.commands[2], 0.0f, 0.0f, 0.0f,
        "future command is canonical zero inside hold");
    check_command_bits(
        prediction.commands[3], 0.50f, 0.0f, 0.0f,
        "future command resumes after hold");

    check(deterministic_route_predict_commands(
              prediction, held, 160, vec3(-0.0f, -0.0f, -0.0f),
              0.04f, 0.50f, 1.0f / 3.0f, 1.0f, false,
              error, sizeof(error)),
          error);
    for (int index = 0; index < G1CommandTrajectorySampleCount; ++index) {
        check_command_bits(
            prediction.commands[index], 0.0f, 0.0f, 0.0f,
            "completed route publishes canonical positive zeros");
    }

    const scene_route straight = route_fixture(
        "straight", {{0.0f, 0.0f}, {0.0f, 2.0f}});
    const float scales[] = {1.0f, 0.5f, 0.0f};
    const float expected_z[] = {0.50f, 0.25f, 0.0f};
    for (int scale_index = 0; scale_index < 3; ++scale_index) {
        check(deterministic_route_predict_commands(
                  prediction, straight, 0, vec3(0.125f, -0.0f, -0.25f),
                  0.04f, 0.50f, 1.0f / 3.0f,
                  scales[scale_index], false, error, sizeof(error)),
              error);
        check_command_bits(
            prediction.commands[0], 0.125f, 0.0f, -0.25f,
            "future scale never changes sample-zero applied velocity");
        for (int index = 1; index < G1CommandTrajectorySampleCount; ++index) {
            check_command_bits(
                prediction.commands[index], 0.0f, 0.0f,
                expected_z[scale_index],
                "future route command has exact checked speed-scale bits");
        }
    }
}

static void test_safe_stop_and_prediction_failures_are_transactional()
{
    scene_route route = route_fixture(
        "safe-stop", {{0.0f, 0.0f}, {1.0f, 0.0f}});
    deterministic_route_prediction output = poisoned_prediction();
    char error[256] = {};
    check(deterministic_route_predict_commands(
              output, route, 4, vec3(0.5f, 0.0f, -0.25f),
              0.04f, 0.50f, 1.0f / 3.0f, 1.0f, true,
              error, sizeof(error)),
          error);
    check(output.force_search,
          "consumed safe-stop latch forces a matcher search");
    for (int index = 0; index < G1CommandTrajectorySampleCount; ++index) {
        check_command_bits(
            output.commands[index], 0.0f, 0.0f, 0.0f,
            "safe-stop canonicalizes every command to positive zero");
    }

    const float infinity = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float wrong_dt = std::nextafter(0.04f, infinity);
    struct InvalidInputs
    {
        int frame;
        vec3 current;
        float dt;
        float speed;
        float sample_time;
        float scale;
        bool safe_stop;
        const char* message;
    } invalid[] = {
        {-1, vec3(), 0.04f, 0.50f, 1.0f / 3.0f, 1.0f, false,
         "negative current frame is transactional"},
        {INT_MAX, vec3(), 0.04f, 0.50f, 1.0f / 3.0f, 1.0f, false,
         "future frame addition overflow is transactional"},
        {0, vec3(infinity, 0.0f, 0.0f), 0.04f, 0.50f,
         1.0f / 3.0f, 1.0f, false,
         "nonfinite current applied velocity is transactional"},
        {0, vec3(), wrong_dt, 0.50f, 1.0f / 3.0f, 1.0f, false,
         "non-exact 25 Hz dt is transactional"},
        {0, vec3(), 0.04f, 0.0f, 1.0f / 3.0f, 1.0f, false,
         "zero speed is transactional"},
        {0, vec3(), 0.04f, infinity, 1.0f / 3.0f, 1.0f, false,
         "nonfinite speed is transactional"},
        {0, vec3(), 0.04f, 0.50f, 0.0f, 1.0f, false,
         "zero trajectory sample time is transactional"},
        {0, vec3(), 0.04f, 0.50f, infinity, 1.0f, false,
         "nonfinite trajectory sample time is transactional"},
        {0, vec3(), 0.04f, 0.50f, 1.0e20f, 1.0f, false,
         "horizon integer overflow is transactional"},
        {0, vec3(), 0.04f, 0.50f, 1.0f / 3.0f, -0.01f, false,
         "negative future speed scale is transactional"},
        {0, vec3(), 0.04f, 0.50f, 1.0f / 3.0f, 1.01f, false,
         "future speed scale above one is transactional"},
        {0, vec3(), 0.04f, 0.50f, 1.0f / 3.0f, nan, false,
         "nonfinite future speed scale is transactional"},
        {0, vec3(), wrong_dt, 0.50f, 1.0f / 3.0f, 1.0f, true,
         "safe-stop still validates exact dt transactionally"},
    };
    for (const InvalidInputs& input : invalid) {
        output = poisoned_prediction();
        const deterministic_route_prediction before = output;
        check_prediction_failure(
            output,
            before,
            deterministic_route_predict_commands(
                output, route, input.frame, input.current, input.dt,
                input.speed, input.sample_time, input.scale, input.safe_stop,
                error, sizeof(error)),
            input.message);
    }

    route.waypoints_xz[1] = route.waypoints_xz[0];
    output = poisoned_prediction();
    const deterministic_route_prediction before = output;
    check_prediction_failure(
        output,
        before,
        deterministic_route_predict_commands(
            output, route, 0, vec3(), 0.04f, 0.50f,
            1.0f / 3.0f, 1.0f, false, error, sizeof(error)),
        "invalid route is rejected with poisoned output unchanged");

    output = poisoned_prediction();
    const deterministic_route_prediction safe_stop_before = output;
    check_prediction_failure(
        output,
        safe_stop_before,
        deterministic_route_predict_commands(
            output, route, 0, vec3(), 0.04f, 0.50f,
            1.0f / 3.0f, 1.0f, true, error, sizeof(error)),
        "safe-stop cannot bypass malformed route validation");
}

struct frame_prediction_counts
{
    int route = 0;
    int heading = 0;
    int rotation = 0;
    int live_velocity = 0;
    int position = 0;
};

static bool same_quat_bits(quat first, quat second)
{
    return bits(first.w) == bits(second.w) &&
           bits(first.x) == bits(second.x) &&
           bits(first.y) == bits(second.y) &&
           bits(first.z) == bits(second.z);
}

static bool same_command_snapshot_bits(
    const G1CommandSnapshot& first,
    const G1CommandSnapshot& second)
{
    if (bits(first.intent.requested_velocity.x) !=
            bits(second.intent.requested_velocity.x) ||
        bits(first.intent.requested_velocity.y) !=
            bits(second.intent.requested_velocity.y) ||
        bits(first.intent.requested_velocity.z) !=
            bits(second.intent.requested_velocity.z) ||
        !same_quat_bits(
            first.intent.desired_heading, second.intent.desired_heading) ||
        bits(first.applied_velocity.x) != bits(second.applied_velocity.x) ||
        bits(first.applied_velocity.y) != bits(second.applied_velocity.y) ||
        bits(first.applied_velocity.z) != bits(second.applied_velocity.z)) {
        return false;
    }
    for (int index = 0; index < G1CommandTrajectorySampleCount; ++index) {
        const vec3 first_values[] = {
            first.predicted_desired_velocities[index],
            first.predicted_root_positions[index]
        };
        const vec3 second_values[] = {
            second.predicted_desired_velocities[index],
            second.predicted_root_positions[index]
        };
        for (int field = 0; field < 2; ++field) {
            if (bits(first_values[field].x) != bits(second_values[field].x) ||
                bits(first_values[field].y) != bits(second_values[field].y) ||
                bits(first_values[field].z) != bits(second_values[field].z)) {
                return false;
            }
        }
        if (!same_quat_bits(
                first.predicted_root_rotations[index],
                second.predicted_root_rotations[index]) ||
            !same_quat_bits(
                first.predicted_desired_headings[index],
                second.predicted_desired_headings[index])) {
            return false;
        }
    }
    return true;
}

static G1CommandFramePrediction frame_prediction_seed(float offset = 0.0f)
{
    G1CommandFramePrediction seed;
    vec3 desired_velocities[G1CommandTrajectorySampleCount];
    vec3 root_positions[G1CommandTrajectorySampleCount];
    quat root_rotations[G1CommandTrajectorySampleCount];
    quat desired_headings[G1CommandTrajectorySampleCount];
    for (int index = 0; index < G1CommandTrajectorySampleCount; ++index) {
        const float sample = static_cast<float>(index);
        desired_velocities[index] =
            vec3(offset + 0.05f * sample, 0.0f, offset + 0.10f);
        root_positions[index] =
            vec3(offset + sample, 0.0f, offset + 2.0f * sample);
        root_rotations[index] = quat(1.0f, 0.0f, 0.0f, 0.0f);
        desired_headings[index] = quat(1.0f, 0.0f, 0.0f, 0.0f);
        seed.predicted_root_velocities[index] =
            vec3(offset + 0.01f * sample, 0.0f, 0.0f);
        seed.predicted_root_accelerations[index] =
            vec3(0.0f, 0.0f, offset + 0.02f * sample);
        seed.predicted_root_angular_velocities[index] = vec3();
    }
    G1CommandIntent intent;
    intent.requested_velocity = vec3(offset + 0.25f, 0.0f, 0.0f);
    intent.desired_heading = quat(1.0f, 0.0f, 0.0f, 0.0f);
    char error[256] = {};
    check(g1_command_snapshot_build(
              seed.command,
              intent,
              vec3(offset + 0.125f, 0.0f, 0.0f),
              slice1d<vec3>(G1CommandTrajectorySampleCount,
                            desired_velocities),
              slice1d<vec3>(G1CommandTrajectorySampleCount, root_positions),
              slice1d<quat>(G1CommandTrajectorySampleCount, root_rotations),
              slice1d<quat>(G1CommandTrajectorySampleCount, desired_headings),
              error,
              sizeof(error)),
          error);
    return seed;
}

static bool same_frame_prediction_bits(
    const G1CommandFramePrediction& first,
    const G1CommandFramePrediction& second)
{
    if (first.force_search != second.force_search ||
        !same_command_snapshot_bits(first.command, second.command)) {
        return false;
    }
    for (int index = 0; index < G1CommandTrajectorySampleCount; ++index) {
        const vec3 first_values[] = {
            first.predicted_root_velocities[index],
            first.predicted_root_accelerations[index],
            first.predicted_root_angular_velocities[index]
        };
        const vec3 second_values[] = {
            second.predicted_root_velocities[index],
            second.predicted_root_accelerations[index],
            second.predicted_root_angular_velocities[index]
        };
        for (int field = 0; field < 3; ++field) {
            if (bits(first_values[field].x) != bits(second_values[field].x) ||
                bits(first_values[field].y) != bits(second_values[field].y) ||
                bits(first_values[field].z) != bits(second_values[field].z)) {
                return false;
            }
        }
    }
    return true;
}

static void run_route_frame_prediction_case(
    const scene_route& route,
    int current_frame,
    vec3 current_applied_velocity,
    bool safe_stop,
    const char* message)
{
    G1CommandFramePrediction seed = frame_prediction_seed();
    G1CommandFramePrediction output = frame_prediction_seed(10.0f);
    G1CommandFramePredictionRequest request;
    request.route_mode = true;
    request.heading_override.active = true;
    request.heading_override.heading =
        quat(0.707106769f, 0.0f, 0.707106769f, 0.0f);
    request.intent.requested_velocity = current_applied_velocity;
    request.intent.desired_heading = request.heading_override.heading;
    request.applied_velocity = current_applied_velocity;
    frame_prediction_counts counts;
    deterministic_route_prediction route_oracle = poisoned_prediction();
    bool route_ready = false;
    char error[256] = {};

    const auto route_predictor = [&](slice1d<vec3> velocities,
                                     bool& force_search,
                                     char* callback_error,
                                     int callback_capacity) {
        ++counts.route;
        if (!deterministic_route_predict_commands(
                route_oracle,
                route,
                current_frame,
                current_applied_velocity,
                0.04f,
                0.50f,
                1.0f / 3.0f,
                1.0f,
                safe_stop,
                callback_error,
                callback_capacity)) {
            return false;
        }
        if (velocities.size != G1CommandTrajectorySampleCount ||
            velocities.data == NULL) {
            return terrain_error(
                callback_error, callback_capacity,
                "route callback received invalid velocity storage");
        }
        for (int index = 0; index < G1CommandTrajectorySampleCount; ++index) {
            velocities(index) = route_oracle.commands[index];
        }
        force_search = route_oracle.force_search;
        route_ready = true;
        return true;
    };
    const auto heading_predictor = [&](slice1d<quat>,
                                       const slice1d<vec3>,
                                       char*, int) {
        ++counts.heading;
        return true;
    };
    const auto rotation_predictor = [&](slice1d<quat> root_rotations,
                                        slice1d<vec3> angular_velocities,
                                        const slice1d<quat> desired_headings,
                                        const slice1d<vec3> desired_velocities,
                                        char* callback_error,
                                        int callback_capacity) {
        ++counts.rotation;
        if (!route_ready || counts.route != 1 || counts.heading != 0) {
            return terrain_error(
                callback_error, callback_capacity,
                "route/override ordering was not established before rotation");
        }
        for (int index = 0; index < G1CommandTrajectorySampleCount; ++index) {
            if (!same_quat_bits(
                    desired_headings(index),
                    request.heading_override.heading) ||
                bits(desired_velocities(index).x) !=
                    bits(route_oracle.commands[index].x) ||
                bits(desired_velocities(index).y) !=
                    bits(route_oracle.commands[index].y) ||
                bits(desired_velocities(index).z) !=
                    bits(route_oracle.commands[index].z)) {
                return terrain_error(
                    callback_error, callback_capacity,
                    "rotation callback did not observe route/override bits");
            }
            root_rotations(index) = desired_headings(index);
            angular_velocities(index) = vec3();
        }
        return true;
    };
    const auto live_velocity_predictor = [&](slice1d<vec3> velocities,
                                             const slice1d<quat>,
                                             char*, int) {
        ++counts.live_velocity;
        velocities.set(vec3(9.0f, 0.0f, 9.0f));
        return true;
    };
    const auto position_predictor = [&](slice1d<vec3> positions,
                                        slice1d<vec3> velocities,
                                        slice1d<vec3> accelerations,
                                        const slice1d<vec3> desired_velocities,
                                        char* callback_error,
                                        int callback_capacity) {
        ++counts.position;
        if (counts.rotation != 1 || counts.live_velocity != 0) {
            return terrain_error(
                callback_error, callback_capacity,
                "position callback did not follow route rotation ordering");
        }
        for (int index = 0; index < G1CommandTrajectorySampleCount; ++index) {
            if (bits(desired_velocities(index).x) !=
                    bits(route_oracle.commands[index].x) ||
                bits(desired_velocities(index).z) !=
                    bits(route_oracle.commands[index].z)) {
                return terrain_error(
                    callback_error, callback_capacity,
                    "route velocities changed before position prediction");
            }
            positions(index) = vec3(
                static_cast<float>(index), 0.0f,
                2.0f * static_cast<float>(index));
            velocities(index) = vec3();
            accelerations(index) = vec3();
        }
        return true;
    };

    check(g1_command_frame_prediction_build(
              output,
              seed,
              request,
              route_predictor,
              heading_predictor,
              rotation_predictor,
              live_velocity_predictor,
              position_predictor,
              error,
              sizeof(error)),
          error);
    check(counts.route == 1 && counts.heading == 0 &&
              counts.rotation == 1 && counts.live_velocity == 0 &&
              counts.position == 1,
          message);
    check(output.force_search == route_oracle.force_search,
          message);
    check(same_quat_bits(
              output.command.intent.desired_heading,
              request.heading_override.heading),
          message);
    for (int index = 0; index < G1CommandTrajectorySampleCount; ++index) {
        check_command_bits(
            output.command.predicted_desired_velocities[index],
            route_oracle.commands[index].x,
            route_oracle.commands[index].y,
            route_oracle.commands[index].z,
            message);
        check(same_quat_bits(
                  output.command.predicted_desired_headings[index],
                  request.heading_override.heading),
              message);
    }
}

static void test_executable_frame_prediction_route_and_heading_policy()
{
    const scene_route corner = route_fixture(
        "corner-seam", {{0.0f, 0.0f}, {0.1f, 0.0f}, {0.1f, 1.0f}});
    run_route_frame_prediction_case(
        corner, 0, vec3(0.125f, 0.0f, -0.25f), false,
        "frame seam retains corner route velocities and override headings");

    const scene_route held = route_fixture(
        "hold-seam",
        {{0.0f, 0.0f}, {1.0f, 0.0f}, {1.0f, 1.0f}, {2.0f, 1.0f}},
        0.4f);
    run_route_frame_prediction_case(
        held, 90, vec3(0.0f, 0.0f, 0.50f), false,
        "frame seam retains before/inside/after hold route velocities");
    run_route_frame_prediction_case(
        held, 160, vec3(-0.0f, -0.0f, -0.0f), false,
        "frame seam retains completion canonical zeros and heading bits");
    run_route_frame_prediction_case(
        held, 90, vec3(0.0f, 0.0f, 0.50f), true,
        "frame seam retains safe-stop zeros, force-search, and heading bits");
}

static void test_executable_frame_prediction_live_policy_and_order()
{
    G1CommandFramePrediction seed = frame_prediction_seed();
    G1CommandFramePrediction output = frame_prediction_seed(10.0f);
    G1CommandFramePredictionRequest request;
    request.route_mode = false;
    request.intent.requested_velocity = vec3(0.5f, 0.0f, 0.0f);
    request.intent.desired_heading = quat(0.0f, 0.0f, 1.0f, 0.0f);
    request.applied_velocity = vec3(0.25f, 0.0f, 0.0f);
    frame_prediction_counts counts;
    char error[256] = {};

    const auto route_predictor = [&](slice1d<vec3>, bool&, char*, int) {
        ++counts.route;
        return true;
    };
    const auto heading_predictor = [&](slice1d<quat> headings,
                                       const slice1d<vec3>,
                                       char*, int) {
        ++counts.heading;
        headings.set(request.intent.desired_heading);
        return true;
    };
    const auto rotation_predictor = [&](slice1d<quat> rotations,
                                        slice1d<vec3> angular_velocities,
                                        const slice1d<quat> headings,
                                        const slice1d<vec3>,
                                        char* callback_error,
                                        int callback_capacity) {
        ++counts.rotation;
        if (counts.route != 0 || counts.heading != 1) {
            return terrain_error(
                callback_error, callback_capacity,
                "live rotation predictor ordering failed");
        }
        for (int index = 0; index < G1CommandTrajectorySampleCount; ++index) {
            rotations(index) = headings(index);
            angular_velocities(index) = vec3();
        }
        return true;
    };
    const auto live_velocity_predictor = [&](slice1d<vec3> velocities,
                                             const slice1d<quat>,
                                             char* callback_error,
                                             int callback_capacity) {
        ++counts.live_velocity;
        if (counts.rotation != 1) {
            return terrain_error(
                callback_error, callback_capacity,
                "live velocity predictor preceded root rotation");
        }
        for (int index = 0; index < G1CommandTrajectorySampleCount; ++index) {
            velocities(index) = vec3(
                0.25f * static_cast<float>(index + 1), 0.0f, -0.25f);
        }
        return true;
    };
    const auto position_predictor = [&](slice1d<vec3> positions,
                                        slice1d<vec3> velocities,
                                        slice1d<vec3> accelerations,
                                        const slice1d<vec3> desired_velocities,
                                        char* callback_error,
                                        int callback_capacity) {
        ++counts.position;
        if (counts.live_velocity != 1) {
            return terrain_error(
                callback_error, callback_capacity,
                "live position predictor preceded desired velocity");
        }
        for (int index = 0; index < G1CommandTrajectorySampleCount; ++index) {
            positions(index) = desired_velocities(index);
            velocities(index) = vec3();
            accelerations(index) = vec3();
        }
        return true;
    };

    check(g1_command_frame_prediction_build(
              output,
              seed,
              request,
              route_predictor,
              heading_predictor,
              rotation_predictor,
              live_velocity_predictor,
              position_predictor,
              error,
              sizeof(error)),
          error);
    check(counts.route == 0 && counts.heading == 1 &&
              counts.rotation == 1 && counts.live_velocity == 1 &&
              counts.position == 1,
          "frame seam invokes only ordinary live predictors in live mode");
    for (int index = 0; index < G1CommandTrajectorySampleCount; ++index) {
        check_command_bits(
            output.command.predicted_desired_velocities[index],
            0.25f * static_cast<float>(index + 1), 0.0f, -0.25f,
            "frame seam snapshots final live desired velocities");
        check(same_quat_bits(
                  output.command.predicted_desired_headings[index],
                  request.intent.desired_heading),
              "frame seam snapshots ordinary predicted headings");
    }
}

static void test_executable_frame_prediction_failures_are_transactional()
{
    G1CommandFramePrediction seed = frame_prediction_seed();
    G1CommandFramePrediction output = frame_prediction_seed(10.0f);
    const G1CommandFramePrediction before = output;
    G1CommandFramePredictionRequest request;
    request.route_mode = true;
    request.heading_override.active = true;
    request.heading_override.heading = quat(1.0f, 0.0f, 0.0f, 0.0f);
    request.intent.requested_velocity = vec3(0.5f, 0.0f, 0.0f);
    request.intent.desired_heading = request.heading_override.heading;
    request.applied_velocity = vec3(0.25f, 0.0f, 0.0f);
    char error[256] = {};

    const auto route_failure = [](slice1d<vec3>, bool&,
                                  char* callback_error,
                                  int callback_capacity) {
        return terrain_error(
            callback_error, callback_capacity, "injected route failure");
    };
    const auto heading_success = [](slice1d<quat>, const slice1d<vec3>,
                                    char*, int) { return true; };
    const auto rotation_success = [](slice1d<quat>, slice1d<vec3>,
                                     const slice1d<quat>,
                                     const slice1d<vec3>,
                                     char*, int) { return true; };
    const auto live_success = [](slice1d<vec3>, const slice1d<quat>,
                                 char*, int) { return true; };
    const auto position_success = [](slice1d<vec3>, slice1d<vec3>,
                                     slice1d<vec3>, const slice1d<vec3>,
                                     char*, int) { return true; };
    check(!g1_command_frame_prediction_build(
              output, seed, request, route_failure, heading_success,
              rotation_success, live_success, position_success,
              error, sizeof(error)),
          "frame seam propagates route callback failure");
    check(same_frame_prediction_bits(output, before),
          "route callback failure preserves poisoned frame output");

    request.route_mode = false;
    request.heading_override.active = false;
    const auto route_success = [](slice1d<vec3>, bool&, char*, int) {
        return true;
    };
    const auto heading_failure = [](slice1d<quat>, const slice1d<vec3>,
                                    char* callback_error,
                                    int callback_capacity) {
        return terrain_error(
            callback_error, callback_capacity, "injected heading failure");
    };
    check(!g1_command_frame_prediction_build(
              output, seed, request, route_success, heading_failure,
              rotation_success, live_success, position_success,
              error, sizeof(error)),
          "frame seam propagates heading callback failure");
    check(same_frame_prediction_bits(output, before),
          "heading callback failure preserves poisoned frame output");

    const auto rotation_failure = [](slice1d<quat>, slice1d<vec3>,
                                     const slice1d<quat>,
                                     const slice1d<vec3>,
                                     char* callback_error,
                                     int callback_capacity) {
        return terrain_error(
            callback_error, callback_capacity, "injected rotation failure");
    };
    check(!g1_command_frame_prediction_build(
              output, seed, request, route_success, heading_success,
              rotation_failure, live_success, position_success,
              error, sizeof(error)),
          "frame seam propagates rotation callback failure");
    check(same_frame_prediction_bits(output, before),
          "rotation callback failure preserves poisoned frame output");

    const auto live_failure = [](slice1d<vec3>, const slice1d<quat>,
                                 char* callback_error,
                                 int callback_capacity) {
        return terrain_error(
            callback_error, callback_capacity,
            "injected live-velocity failure");
    };
    check(!g1_command_frame_prediction_build(
              output, seed, request, route_success, heading_success,
              rotation_success, live_failure, position_success,
              error, sizeof(error)),
          "frame seam propagates live-velocity callback failure");
    check(same_frame_prediction_bits(output, before),
          "live-velocity callback failure preserves poisoned frame output");

    const auto position_failure = [](slice1d<vec3>, slice1d<vec3>,
                                     slice1d<vec3>, const slice1d<vec3>,
                                     char* callback_error,
                                     int callback_capacity) {
        return terrain_error(
            callback_error, callback_capacity, "injected position failure");
    };
    check(!g1_command_frame_prediction_build(
              output, seed, request, route_success, heading_success,
              rotation_success, live_success, position_failure,
              error, sizeof(error)),
          "frame seam propagates position callback failure");
    check(same_frame_prediction_bits(output, before),
          "position callback failure preserves poisoned frame output");

    const auto invalid_rotation = [](slice1d<quat> rotations,
                                     slice1d<vec3>,
                                     const slice1d<quat>,
                                     const slice1d<vec3>,
                                     char*, int) {
        rotations.set(quat(2.0f, 0.0f, 0.0f, 0.0f));
        return true;
    };
    check(!g1_command_frame_prediction_build(
              output, seed, request, route_success, heading_success,
              invalid_rotation, live_success, position_success,
              error, sizeof(error)),
          "frame seam rejects invalid callback publication");
    check(same_frame_prediction_bits(output, before),
          "invalid callback publication preserves poisoned frame output");

    check(!g1_command_frame_prediction_build(
              output, output, request, route_success, heading_success,
              rotation_success, live_success, position_success,
              error, sizeof(error)),
          "frame seam rejects output/seed alias");
    check(same_frame_prediction_bits(output, before),
          "output/seed alias preserves poisoned frame output");

    check(!g1_command_frame_prediction_build(
              output, seed, request, route_success, heading_success,
              rotation_success, live_success, position_success,
              reinterpret_cast<char*>(&output),
              static_cast<int>(sizeof(output))),
          "frame seam rejects diagnostic/output alias");
    check(same_frame_prediction_bits(output, before),
          "diagnostic/output alias preserves poisoned frame output");
}

int main(int argc, char** argv)
{
    if (argc == 3 && std::strcmp(argv[1], "--controller") == 0) {
        check_task6_publication_contract(argv[2]);
        return 0;
    }
    if (argc == 3 && std::strcmp(argv[1], "--route-header") == 0) {
        check_route_sample_cursor_contract(argv[2]);
        return 0;
    }
    check(argc == 1,
          "usage: test_route_runtime [--controller path|--route-header path]");

    test_task6_main_signature_forms();
    test_four_horizon_route_predictions();
    test_tangent_level_boundary_exact_schedule();
    test_hold_completion_and_speed_scales();
    test_safe_stop_and_prediction_failures_are_transactional();
    test_executable_frame_prediction_route_and_heading_policy();
    test_executable_frame_prediction_live_policy_and_order();
    test_executable_frame_prediction_failures_are_transactional();

    scene_route route;
    route.id = "corner";
    route.expected_outcome = "traverse";
    route.walkability_class = 1;
    route.landing_hold_seconds = 2;
    route.waypoints_xz = {{0, 0}, {1, 0}, {1, 1}, {2, 1}};

    deterministic_route_sample a, b;
    char error[256] = {};
    check(deterministic_route_command(
              a, route, 0, 0.04f, 0.50f, error, sizeof(error)),
          error);
    check(a.waypoint == 1 && bits(a.command.x) == bits(0.50f) &&
              a.command.z == 0 && !a.complete,
          "first segment");
    check(deterministic_route_command(
              a, route, 49, 0.04f, 0.50f, error, sizeof(error)),
          error);
    check(a.waypoint == 1 && a.command.x == 0.50f,
          "last frame first segment");
    check(deterministic_route_command(
              a, route, 50, 0.04f, 0.50f, error, sizeof(error)),
          error);
    check(a.waypoint == 2 && a.command.x == 0 && a.command.z == 0.50f,
          "second segment");
    check(deterministic_route_command(
              a, route, 100, 0.04f, 0.50f, error, sizeof(error)),
          error);
    check(!a.complete && a.waypoint == 2 && a.command.x == 0 &&
              a.command.z == 0,
          "landing hold begins");
    check(deterministic_route_command(
              a, route, 149, 0.04f, 0.50f, error, sizeof(error)),
          error);
    check(!a.complete && a.waypoint == 2 && a.command.x == 0 &&
              a.command.z == 0,
          "landing hold lasts two seconds");
    check(deterministic_route_command(
              a, route, 150, 0.04f, 0.50f, error, sizeof(error)),
          error);
    check(!a.complete && a.waypoint == 3 && a.command.x == 0.50f &&
              a.command.z == 0,
          "motion resumes after hold");
    check(deterministic_route_command(
              a, route, 200, 0.04f, 0.50f, error, sizeof(error)),
          error);
    check(a.complete && a.command.x == 0 && a.command.z == 0,
          "route complete");
    check(deterministic_route_motion_frames(route) == 200,
          "motion count includes hold");
    for (int frame = 0; frame < 225; ++frame) {
        check(deterministic_route_command(
                  a, route, frame, 0.04f, 0.50f, error, sizeof(error)),
              error);
        check(deterministic_route_command(
                  b, route, frame, 0.04f, 0.50f, error, sizeof(error)),
              error);
        check(bits(a.command.x) == bits(b.command.x) &&
                  bits(a.command.z) == bits(b.command.z) &&
                  a.waypoint == b.waypoint && a.complete == b.complete,
              "repeatable route input");
    }
    float rounded_sample = 0;
    check(scene_binary32_lerp(
              rounded_sample,
              -4.821664810180664f,
              0.22549442946910858f,
              13,
              23),
          "awkward route interpolation");
    check(bits(rounded_sample) == UINT32_C(0xbffc05aa),
          "route mul then add rounds separately without FMA");
    route.waypoints_xz[1] = route.waypoints_xz[0];
    const deterministic_route_sample preserved = a;
    check(!deterministic_route_command(
              a, route, 0, 0.04f, 0.50f, error, sizeof(error)),
          "zero segment rejected");
    check(bits(a.command.x) == bits(preserved.command.x) &&
              bits(a.command.z) == bits(preserved.command.z) &&
              a.waypoint == preserved.waypoint &&
              a.complete == preserved.complete,
          "route failure preserves output");

    route.waypoints_xz = {{0, 0}, {1, 0}, {1, 1}, {2, 1}};
    check(!deterministic_route_command(
              a, route, -1, 0.04f, 0.50f, error, sizeof(error)),
          "negative frame rejected");
    check(!deterministic_route_command(
              a, route, 0, 0.0f, 0.50f, error, sizeof(error)),
          "zero dt rejected");
    check(!deterministic_route_command(
              a, route, 0, 0.04f, INFINITY, error, sizeof(error)),
          "infinite speed rejected");
    route.landing_hold_seconds = 0x1p31f;
    check(deterministic_route_motion_frames(route, 1.0f, 1.0f) == -1,
          "hold frame int overflow rejected");
    route.landing_hold_seconds = 0.0f;
    route.waypoints_xz = {
        {0.0f, 0.0f}, {20000000.0f, 0.0f},
        {40000000.0f, 0.0f}, {60000000.0f, 0.0f}};
    check(deterministic_route_motion_frames(route) == -1,
          "total frame int overflow rejected");
    check(!deterministic_route_command(
              a, route, 0, 0.04f, 0.50f, error, sizeof(error)),
          "overflowing route schedule rejected");

    heightfield terrain;
    terrain.version = 2;
    terrain.nx = 3;
    terrain.nz = 2;
    terrain.origin_x = 0.0f;
    terrain.origin_z = 0.0f;
    terrain.cell_size = 1.0f;
    terrain.exterior_height = 0.0f;
    terrain.heights.resize(6);
    terrain.heights(0) = 0.0f;
    terrain.heights(1) = 1.0f;
    terrain.heights(2) = 1.0f;
    terrain.heights(3) = 0.0f;
    terrain.heights(4) = 1.0f;
    terrain.heights(5) = 1.0f;
    scene_route height_route;
    height_route.id = "height";
    height_route.waypoints_xz = {{0.0f, 0.0f}, {2.0f, 0.0f}};
    check(bits(deterministic_route_target_height(height_route, terrain)) ==
              bits(1.0f),
          "route target chooses persistent elevated height");
    height_route.waypoints_xz.clear();
    check(!terrain_float_is_finite(
              deterministic_route_target_height(height_route, terrain)),
          "invalid route target input rejected");

    motion_pack_manifest manifest;
    manifest.sources.push_back(
        motion_source_record{"source", "flat", 10, 20});
    g1_controller_state state;
    state.frame_index = 12;
    state.scene_frame = 7;
    state.incumbent_cost = 1.25f;
    state.support_observation_now.source_height[0] = 0.1f;
    state.support_observation_now.source_height[1] = 0.2f;
    state.support_observation_now.source_height[2] = 0.3f;
    state.support_observation_now.runtime_height[0] = 0.4f;
    state.support_observation_now.runtime_height[1] = 0.5f;
    state.support_observation_now.runtime_height[2] = 0.6f;
    state.support_observation_now.delta[0] = 0.3f;
    state.support_observation_now.delta[1] = 0.3f;
    state.support_observation_now.delta[2] = 0.3f;
    state.support_observation_now.contact[0] = true;
    state.support_observation_now.contact[1] = false;
    state.support.height = 0.3f;
    state.support.velocity = 0.01f;
    state.support.nominal_height = 0.3f;
    state.support.nominal_velocity = 0.01f;
    state.support.offset_height = 0.0f;
    state.support.offset_velocity = 0.0f;
    state.support.airborne_frames = 0;
    state.support.source = support_left;
    state.support.initialized = true;
    state.global_bone_positions.resize(G1_BoneCount);
    state.global_bone_positions.set(vec3());
    state.global_bone_positions(G1_Hips).y = 0.9f;
    state.simulation_position = vec3(1.0f, 0.0f, 2.0f);
    state.walkability_class = 1;
    traversability_diagnostics traversal;
    traversal.walkability_class = 1;
    traversal.reason = walkability_clear;
    traversal.distance = FLT_MAX;
    traversal.commanded_speed = 0.50f;
    traversal.applied_speed = 0.40f;
    traversal.point = vec3(3.0f, 0.0f, 4.0f);
    deterministic_route_sample diagnostic_route;
    diagnostic_route.command = vec3(0.50f, 0.0f, 0.0f);
    diagnostic_route.waypoint = 2;
    g1_runtime_diagnostic_snapshot snapshot;
    check(g1_runtime_diagnostics_build(
              snapshot, manifest, state, traversal, diagnostic_route, 1.0f,
              0, 1, false, 1, 1, 0, error, sizeof(error)),
          error);
    check(std::strcmp(snapshot.source_name, "source") == 0 &&
              std::strcmp(snapshot.source_terrain, "flat") == 0 &&
              snapshot.source_index == 0 &&
              bits(snapshot.continuation_cost) == bits(1.25f) &&
              bits(snapshot.source_left_toe_height) == bits(0.2f) &&
              bits(snapshot.runtime_support_right_toe_height) == bits(0.6f) &&
              bits(snapshot.support_root_delta) == bits(0.3f) &&
              std::strcmp(snapshot.support_source, "left") == 0 &&
              snapshot.left_contact && !snapshot.right_contact &&
              bits(snapshot.support_retargeted_hips_y) == bits(0.9f) &&
              bits(snapshot.ik_adjusted_hips_y) == bits(0.9f) &&
              snapshot.route_waypoint == 2 && !snapshot.route_complete &&
              snapshot.scene_frame == 7 && snapshot.live_model_count == 1,
          "diagnostic snapshot maps one shared runtime state");

    const g1_runtime_diagnostic_snapshot preserved_snapshot = snapshot;
    state.support.nominal_height = INFINITY;
    check(!g1_runtime_diagnostics_build(
              snapshot, manifest, state, traversal, diagnostic_route, 1.0f,
              0, 1, false, 1, 1, 0, error, sizeof(error)),
          "non-finite support state rejected");
    check(snapshot.source_index == preserved_snapshot.source_index &&
              bits(snapshot.support_height) ==
                  bits(preserved_snapshot.support_height),
          "diagnostic failure preserves output");
    state.support.nominal_height = 0.3f;
    state.walkability_class = 3;
    check(!g1_runtime_diagnostics_build(
              snapshot, manifest, state, traversal, diagnostic_route, 1.0f,
              0, 1, false, 1, 1, 0, error, sizeof(error)),
          "invalid current walkability class rejected");
    state.walkability_class = 1;
    check(!g1_runtime_diagnostics_build(
              snapshot, manifest, state, traversal, diagnostic_route, 1.0f,
              0, 1, false, 1, 2, 0, error, sizeof(error)),
          "more than one live model rejected");

    const char* log_path = "/tmp/test_route_runtime_log.csv";
    (void)std::remove(log_path);
    motion_match_log log;
    check(log.open(log_path, error, sizeof(error)), error);
    motion_match_log_row row;
    row.source_name = "s";
    row.source_terrain = "t";
    row.source_index = 3;
    row.continuation_cost = 1.0f;
    row.source_root_height = 2.0f;
    row.source_left_toe_height = 3.0f;
    row.source_right_toe_height = 4.0f;
    row.runtime_support_root_height = 5.0f;
    row.runtime_support_left_toe_height = 6.0f;
    row.runtime_support_right_toe_height = 7.0f;
    row.support_root_delta = 8.0f;
    row.support_left_toe_delta = 9.0f;
    row.support_right_toe_delta = 10.0f;
    row.support_height = 11.0f;
    row.support_velocity = 12.0f;
    row.support_source = "both";
    row.airborne_frames = 13;
    row.left_contact = true;
    row.right_contact = false;
    row.support_retargeted_hips_y = 14.0f;
    row.ik_adjusted_hips_y = 15.0f;
    row.simulation_x = 16.0f;
    row.simulation_z = 17.0f;
    row.walkability_class = 2;
    row.blocked = true;
    row.blocked_reason = "blocked-cell";
    row.blocked_distance = 18.0f;
    row.blocked_point_x = 19.0f;
    row.blocked_point_z = 20.0f;
    row.commanded_speed = 0.5f;
    row.applied_speed = 0.25f;
    row.route_waypoint = 4;
    row.route_complete = true;
    row.route_target_height = 21.0f;
    row.scene_generation = 5;
    row.scene_frame = 6;
    row.scene_reset_count = 7;
    row.scene_switch_failed = true;
    row.motion_pack_load_count = 1;
    row.model_load_count = 8;
    row.model_unload_count = 7;
    row.live_model_count = 1;
    check(log.write(row, error, sizeof(error)), error);
    check(log.close(error, sizeof(error)), error);
    FILE* log_file = std::fopen(log_path, "rb");
    check(log_file != NULL, "open exact runtime log");
    char header[8192] = {};
    char data[8192] = {};
    check(std::fgets(header, sizeof(header), log_file) != NULL,
          "read runtime header");
    check(std::fgets(data, sizeof(data), log_file) != NULL,
          "read runtime row");
    check(std::fclose(log_file) == 0, "close exact runtime log");
    const char* expected_header_suffix =
        ",source_name,source_terrain,source_index,continuation_cost,"
        "source_root_height,source_left_toe_height,source_right_toe_height,"
        "runtime_support_root_height,runtime_support_left_toe_height,"
        "runtime_support_right_toe_height,support_root_delta,"
        "support_left_toe_delta,support_right_toe_delta,support_height,"
        "support_velocity,support_source,airborne_frames,left_contact,"
        "right_contact,support_retargeted_hips_y,ik_adjusted_hips_y,"
        "simulation_x,simulation_z,walkability_class,blocked,blocked_reason,"
        "blocked_distance,blocked_point_x,blocked_point_z,commanded_speed,"
        "applied_speed,route_waypoint,route_complete,route_target_height,"
        "scene_generation,scene_frame,scene_reset_count,scene_switch_failed,"
        "motion_pack_load_count,model_load_count,model_unload_count,"
        "live_model_count,ik_applied,ik_safe_stop_requested,ik_stop_reason,";
    check(std::strstr(header, expected_header_suffix) != NULL,
          "exact append-only runtime header suffix");
    check(std::strstr(
              data,
              ",s,t,3,1,2,3,4,5,6,7,8,9,10,11,12,both,13,1,0,14,15,"
              "16,17,2,1,blocked-cell,18,19,20,0.5,0.25,4,1,21,5,6,7,1,"
              "1,8,7,1,0,0,none,") != NULL,
          "runtime row suffix matches header order");
    check(std::strstr(
              header,
              "rejected_right_selected_witness_upper,"
              "accepted_state_digest_hex\n") != NULL,
          "directional suffix closes with accepted-state digest");
    check(std::remove(log_path) == 0, "remove exact runtime log");
    return 0;
}
