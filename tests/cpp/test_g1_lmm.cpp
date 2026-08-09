#include "lmm.h"
#include "g1_controller_state.h"
#include "sonic/cpp/g1_runtime.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

static void check(bool value, const char* message)
{
    if (!value) {
        std::fprintf(stderr, "G1 LMM test failed: %s\n", message);
        std::exit(1);
    }
}

static void test_dimensions_and_normalized_projector_cost()
{
    check(g1_lmm_dimensions_valid(31, 32, 31, 2),
          "production G1 LMM dimensions are accepted");
    check(!g1_lmm_dimensions_valid(31, 31, 31, 2),
          "incorrect latent dimensions are rejected");

    array1d<float> query(31);
    array1d<float> projected(31);
    query.zero();
    projected.zero();
    query(0) = 3.0f;
    projected(0) = 1.0f;
    query(30) = -4.0f;
    projected(30) = -1.0f;
    check(std::fabs(projector_cost_normalized(query, projected) -
                    std::sqrt(13.0f)) <= 1.0e-6f,
          "projector cost is computed in one normalized unit system");
}

static void test_controller_state_owns_lmm_transaction_state()
{
    g1_controller_state state;
    check(state.lmm_features.size == 0 && state.lmm_latent.size == 0,
          "default state has no implicit LMM allocation");
    check(state.lmm_commit_count == 0 && state.lmm_stepper_count == 0,
          "default state starts with zero LMM ownership counters");
}

static void test_missing_bundle_fails_before_evaluation_allocation()
{
    motion_pack_manifest data_manifest;
    data_manifest.flat_lmm_bundle = true;
    data_manifest.database_frames = 3853;
    data_manifest.database.path = "database.bin";
    data_manifest.database.sha256 = std::string(64, '0');
    data_manifest.matching_features.path = "features.bin";
    data_manifest.matching_features.sha256 = std::string(64, '1');
    g1_lmm_model_bundle model;
    char error[512] = {};
    check(!g1_lmm_model_load_and_verify(
              model,
              "/definitely/missing/model",
              "/definitely/missing/data",
              data_manifest,
              error,
              static_cast<int>(sizeof(error))),
          "missing authenticated model bundle is rejected");
    check(model.evaluation_allocation_count == 0,
          "rejection precedes network evaluation allocation");
}

static void write_u32(std::ofstream& output, const std::uint32_t value)
{
    const unsigned char bytes[4] = {
        static_cast<unsigned char>(value),
        static_cast<unsigned char>(value >> 8),
        static_cast<unsigned char>(value >> 16),
        static_cast<unsigned char>(value >> 24)};
    output.write(reinterpret_cast<const char*>(bytes), 4);
}

static void write_f32(std::ofstream& output, const float value)
{
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    write_u32(output, bits);
}

static void write_f32_values(
    std::ofstream& output,
    const std::vector<float>& values)
{
    write_u32(output, static_cast<std::uint32_t>(values.size()));
    for (float value : values) write_f32(output, value);
}

static void write_zero_floats(std::ofstream& output, std::size_t count)
{
    const std::vector<char> zeros(4096, '\0');
    while (count > 0) {
        const std::size_t values = std::min(
            count, zeros.size() / sizeof(float));
        output.write(
            zeros.data(),
            static_cast<std::streamsize>(values * sizeof(float)));
        count -= values;
    }
}

static void write_network(
    const std::filesystem::path& path,
    const int input,
    const int output_count,
    const std::vector<std::pair<int, int>>& layers,
    const std::vector<float>& output_mean = {})
{
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    check(output.good(), "synthetic network opens");
    write_f32_values(output, std::vector<float>(input, 0.0f));
    write_f32_values(output, std::vector<float>(input, 1.0f));
    write_f32_values(
        output,
        output_mean.empty()
            ? std::vector<float>(output_count, 0.0f)
            : output_mean);
    write_f32_values(output, std::vector<float>(output_count, 1.0f));
    write_u32(output, static_cast<std::uint32_t>(layers.size()));
    for (const auto& layer : layers) {
        write_u32(output, static_cast<std::uint32_t>(layer.first));
        write_u32(output, static_cast<std::uint32_t>(layer.second));
        write_zero_floats(
            output,
            static_cast<std::size_t>(layer.first) *
                static_cast<std::size_t>(layer.second));
        write_u32(output, static_cast<std::uint32_t>(layer.second));
        write_zero_floats(output, static_cast<std::size_t>(layer.second));
    }
    output.close();
    check(output.good(), "synthetic network writes completely");
}

static std::string file_sha(const std::filesystem::path& path)
{
    std::string digest;
    char error[512] = {};
    check(sha256_file_hex(
              digest,
              path.c_str(),
              error,
              static_cast<int>(sizeof(error))),
          error);
    return digest;
}

struct synthetic_lmm_fixture
{
    std::filesystem::path root = "/tmp/test_g1_lmm_bundle";
    std::filesystem::path data = root / "data";
    std::filesystem::path model = root / "model";
    motion_pack_manifest data_manifest;
    std::string data_manifest_sha;
    std::string model_names[6] = {
        "latent.bin", "decompressor.bin", "stepper.bin", "projector.bin",
        "training.json", "evaluation.json"};
    std::string model_sha[6];
    int model_size[6] = {};

    synthetic_lmm_fixture()
    {
        namespace fs = std::filesystem;
        fs::remove_all(root);
        fs::create_directories(data);
        fs::create_directories(model);

        {
            std::ofstream output(data / "manifest.json", std::ios::binary);
            output << "synthetic authenticated flat data identity\n";
        }
        {
            std::ofstream output(data / "database.bin", std::ios::binary);
            output << "database";
        }
        {
            std::ofstream output(data / "features.bin", std::ios::binary);
            output << "features";
        }
        data_manifest_sha = file_sha(data / "manifest.json");
        data_manifest.flat_lmm_bundle = true;
        data_manifest.database_frames = 3853;
        data_manifest.database.path = "database.bin";
        data_manifest.database.size_bytes = static_cast<int>(
            fs::file_size(data / "database.bin"));
        data_manifest.database.sha256 = file_sha(data / "database.bin");
        data_manifest.matching_features.path = "features.bin";
        data_manifest.matching_features.size_bytes = static_cast<int>(
            fs::file_size(data / "features.bin"));
        data_manifest.matching_features.sha256 = file_sha(
            data / "features.bin");

        {
            std::ofstream output(model / "latent.bin", std::ios::binary);
            write_u32(output, 3853);
            write_u32(output, 32);
            write_zero_floats(output, 3853u * 32u);
        }
        std::vector<float> decompressor_mean(458, 0.0f);
        decompressor_mean[1] = 1.0f;
        for (int bone = 0; bone < 30; ++bone) {
            const int offset = 90 + bone * 6;
            decompressor_mean[offset + 0] = 1.0f;
            decompressor_mean[offset + 3] = 1.0f;
        }
        write_network(
            model / "decompressor.bin", 63, 458,
            {{63, 512}, {512, 458}}, decompressor_mean);
        write_network(
            model / "stepper.bin", 63, 63,
            {{63, 512}, {512, 512}, {512, 63}});
        write_network(
            model / "projector.bin", 31, 63,
            {{31, 512}, {512, 512}, {512, 512}, {512, 512}, {512, 63}});
        {
            std::ofstream output(model / "training.json", std::ios::binary);
            output << "{}\n";
        }
        {
            std::ofstream output(model / "evaluation.json", std::ios::binary);
            output << "{}\n";
        }
        for (int index = 0; index < 6; ++index) {
            model_sha[index] = file_sha(model / model_names[index]);
            model_size[index] = static_cast<int>(
                fs::file_size(model / model_names[index]));
        }
        write_manifest();
    }

    ~synthetic_lmm_fixture()
    {
        std::filesystem::remove_all(root);
    }

    void write_manifest(
        const std::string& bound_data_sha = std::string(),
        const int corrupt_artifact = -1,
        const int corrupt_data_artifact = -1)
    {
        std::ofstream output(
            model / "manifest.json", std::ios::binary | std::ios::trunc);
        check(output.good(), "synthetic model manifest opens");
        output << "{\n  \"artifacts\": {\n";
        for (int index = 0; index < 6; ++index) {
            std::string digest = model_sha[index];
            if (index == corrupt_artifact) digest[0] = digest[0] == '0' ? '1' : '0';
            output << "    \"" << model_names[index] << "\": {"
                   << "\"path\": \"" << model_names[index] << "\", "
                   << "\"size_bytes\": " << model_size[index] << ", "
                   << "\"sha256\": \"" << digest << "\"}"
                   << (index == 5 ? "\n" : ",\n");
        }
        std::string database_sha = data_manifest.database.sha256;
        std::string features_sha = data_manifest.matching_features.sha256;
        if (corrupt_data_artifact == 0)
            database_sha[0] = database_sha[0] == '0' ? '1' : '0';
        if (corrupt_data_artifact == 1)
            features_sha[0] = features_sha[0] == '0' ? '1' : '0';
        output << "  },\n"
               << "  \"data_artifacts\": {\"database.bin\": \""
               << database_sha << "\", \"features.bin\": \""
               << features_sha << "\"},\n"
               << "  \"data_manifest_schema\": \"g1-lmm-flat-data/v2\",\n"
               << "  \"data_manifest_sha256\": \""
               << (bound_data_sha.empty() ? data_manifest_sha : bound_data_sha)
               << "\",\n"
               << "  \"dimensions\": {\"bones\": 31, \"contacts\": 2, "
                  "\"features\": 31, \"latent\": 32},\n"
               << "  \"output_fps\": 60.0,\n"
               << "  \"schema\": \"g1-lmm-model/v1\",\n"
               << "  \"status\": \"accepted\"\n}\n";
        output.close();
        check(output.good(), "synthetic model manifest writes completely");
    }
};

static void make_lmm_database(database& db)
{
    static const int parents[G1_BoneCount] = {
        -1, 0, 1, 2, 3, 4, 5, 6, 1, 8, 9, 10, 11, 12, 1, 14,
        15, 16, 17, 18, 19, 20, 21, 22, 16, 24, 25, 26, 27, 28, 29};
    const int frames = 3853;
    db.bone_positions.resize(frames, G1_BoneCount);
    db.bone_velocities.resize(frames, G1_BoneCount);
    db.bone_rotations.resize(frames, G1_BoneCount);
    db.bone_angular_velocities.resize(frames, G1_BoneCount);
    db.bone_parents.resize(G1_BoneCount);
    db.contact_states.resize(frames, 2);
    db.range_starts.resize(1);
    db.range_stops.resize(1);
    db.range_starts(0) = 0;
    db.range_stops(0) = frames;
    for (int bone = 0; bone < G1_BoneCount; ++bone)
        db.bone_parents(bone) = parents[bone];
    for (int frame = 0; frame < frames; ++frame) {
        for (int bone = 0; bone < G1_BoneCount; ++bone) {
            db.bone_positions(frame, bone) =
                bone == G1_Hips ? vec3(0.0f, 1.0f, 0.0f) : vec3();
            db.bone_velocities(frame, bone) = vec3();
            db.bone_rotations(frame, bone) = quat();
            db.bone_angular_velocities(frame, bone) = vec3();
        }
    }
    db.contact_states.zero();
    db.features.resize(frames, 31);
    db.features.zero();
    db.features_offset.resize(31);
    db.features_offset.zero();
    db.features_scale.resize(31);
    db.features_scale.set(1.0f);
    db.terrain_features.resize(frames, 4);
    db.terrain_features.zero();
    database_build_bounds(db);
}

static scene_pack make_lmm_scene()
{
    scene_pack scene;
    scene.metadata.id = "lmm-flat-fixture";
    scene.metadata.spawn_position = vec3();
    scene.metadata.spawn_yaw = 0.0f;
    scene.metadata.playable_bounds = {-4.0f, -4.0f, 4.0f, 4.0f};
    scene.metadata.lookahead_bounds = scene.metadata.playable_bounds;
    scene.terrain.version = 2;
    scene.terrain.nx = 65;
    scene.terrain.nz = 65;
    scene.terrain.origin_x = -8.0f;
    scene.terrain.origin_z = -8.0f;
    scene.terrain.cell_size = 0.25f;
    scene.terrain.exterior_height = 0.0f;
    scene.terrain.heights.resize(scene.terrain.nx * scene.terrain.nz);
    scene.terrain.heights.zero();
    scene.walkability.nx = scene.terrain.nx;
    scene.walkability.nz = scene.terrain.nz;
    scene.walkability.cells.resize(scene.walkability.nx * scene.walkability.nz);
    scene.walkability.cells.set(1);
    return scene;
}

template<typename T>
static bool same_array_bits(const array1d<T>& first, const array1d<T>& second)
{
    return first.size == second.size &&
           (first.size == 0 || std::memcmp(
               first.data, second.data,
               static_cast<std::size_t>(first.size) * sizeof(T)) == 0);
}

static void test_transactional_lmm_tick(
    g1_lmm_model_bundle& model)
{
    database db;
    make_lmm_database(db);
    const scene_pack scene = make_lmm_scene();
    terrain_support_set support;
    support.values.resize(db.nframes(), 3);
    support.values.zero();
    g1_controller_state state;
    char error[512] = {};
    check(g1_controller_state_reset(
              state, db, support, scene, error,
              static_cast<int>(sizeof(error))),
          error);
    state.lmm_latent = model.latent(0);
    float query_normalized[31] = {};
    query_normalized[27] = 2.0f;
    model.stepper.weights[0](27, 0) = 1.0f;
    model.stepper.weights[1](0, 0) = 1.0f;
    model.stepper.weights[2](0, 0) = 1.0f;
    model.decompressor.weights[0](27, 0) = 1.0f;
    model.decompressor.weights[1](0, 450) = 0.02f;
    g1_runtime_step_result result;
    check(g1_runtime_step_lmm_normalized(
              result,
              state,
              db,
              scene,
              model,
              slice1d<float>(31, query_normalized),
              error,
              static_cast<int>(sizeof(error))),
          error);
    check(state.lmm_commit_count == 1 && state.lmm_stepper_count == 1,
          "one accepted LMM tick commits and steps exactly once");
    check(state.frame_index == 0 && result.selected_database_frame == -1,
          "LMM tick does not publish an ordinary database identity");
    check(result.engine == G1LocomotionLMM,
          "accepted tick diagnostics identify the LMM owner");
    check(std::fabs(state.lmm_features(0) - 2.0f / 60.0f) <= 1.0e-7f,
          "projected terrain is overwritten before the one stepper call");
    check(state.lmm_features(27) == 0.0f,
          "candidate-root terrain overwrites the stepped recurrent feature");
    check(result.lmm_provisional_root_position.x > 0.0f &&
              std::fabs(result.lmm_provisional_root_position.x) < 0.001f &&
              result.lmm_final_root_position.x == 0.0f &&
              state.bone_positions(G1_Simulation).x == 0.0f,
          "provisional decode is scratch-only and final decode uses candidate terrain");

    const auto prediction_builder = [](
        G1CommandFramePrediction& output,
        const G1CommandFramePrediction& seed,
        const G1CommandFramePredictionRequest& request,
        g1_controller_state&,
        const g1_runtime_config&,
        char*,
        int)
    {
        output = seed;
        output.command.intent = request.intent;
        output.command.applied_velocity = request.applied_velocity;
        return true;
    };
    g1_runtime_step_request request;
    request.mode = G1RuntimeDirect;
    request.requested_velocity_holden = vec3();
    request.desired_heading_holden = quat();
    request.matching_enabled = true;
    const g1_runtime_config config;
    g1_runtime_step_result prepared_result;
    check(g1_runtime_step_lmm(
              prepared_result,
              state,
              db,
              scene,
              model,
              request,
              config,
              prediction_builder,
              error,
              static_cast<int>(sizeof(error))),
          error);
    check(state.lmm_commit_count == 2 && state.lmm_stepper_count == 2,
          "high-level command/query LMM path still steps and commits once");
    for (int feature = 0; feature < 31; ++feature)
        check(std::fabs(
                  normalize_query_feature(
                      prepared_result.query[feature],
                      db.features_offset(feature),
                      db.features_scale(feature)) -
                  prepared_result.query_normalized[feature]) <= 1.0e-6f,
              "high-level LMM query preserves raw/normalized parity");

    const g1_controller_state before(state);
    g1_runtime_step_result sentinel = {};
    sentinel.query_database_frame = 90;
    sentinel.selected_database_frame = 91;
    sentinel.query_range = 92;
    sentinel.query[0] = 93.0f;
    sentinel.query_normalized[30] = 94.0f;
    sentinel.lmm_projector_cost = 95.0f;
    const g1_runtime_step_result sentinel_before = sentinel;
    model.projector.output_mean(0) = std::numeric_limits<float>::quiet_NaN();
    error[0] = '\0';
    check(!g1_runtime_step_lmm(
              sentinel,
              state,
              db,
              scene,
              model,
              request,
              config,
              prediction_builder,
              error,
              static_cast<int>(sizeof(error))),
          "non-finite projector output is rejected");
    check(state.lmm_commit_count == before.lmm_commit_count &&
              state.lmm_stepper_count == before.lmm_stepper_count &&
              same_array_bits(state.lmm_features, before.lmm_features) &&
              same_array_bits(state.lmm_latent, before.lmm_latent) &&
              same_array_bits(state.bone_positions, before.bone_positions) &&
              same_array_bits(state.bone_rotations, before.bone_rotations) &&
              same_array_bits(state.curr_bone_contacts,
                              before.curr_bone_contacts) &&
              std::memcmp(
                  &state.command, &before.command, sizeof(state.command)) == 0,
          "failed LMM tick leaves command, recurrent, pose, and contact state bitwise unchanged");
    check(std::memcmp(&sentinel, &sentinel_before, sizeof(sentinel)) == 0,
          "failed LMM tick leaves output diagnostics bitwise unchanged");
    model.projector.output_mean(0) = 0.0f;
}

static void test_authenticated_bundle_and_digest_tampers()
{
    synthetic_lmm_fixture fixture;
    char error[512] = {};
    g1_lmm_model_bundle accepted;
    check(g1_lmm_model_load_and_verify(
              accepted,
              fixture.model.c_str(),
              fixture.data.c_str(),
              fixture.data_manifest,
              error,
              static_cast<int>(sizeof(error))),
          error);
    check(accepted.authenticated && accepted.evaluation_allocation_count == 3,
          "accepted model allocates exactly three evaluation states");
    check(accepted.latent.rows == 3853 && accepted.latent.cols == 32,
          "accepted latent table matches the bound database");
    test_transactional_lmm_tick(accepted);

    std::string wrong_data_sha = fixture.data_manifest_sha;
    wrong_data_sha[0] = wrong_data_sha[0] == '0' ? '1' : '0';
    fixture.write_manifest(wrong_data_sha);
    g1_lmm_model_bundle rejected_data_manifest;
    error[0] = '\0';
    check(!g1_lmm_model_load_and_verify(
              rejected_data_manifest,
              fixture.model.c_str(),
              fixture.data.c_str(),
              fixture.data_manifest,
              error,
              static_cast<int>(sizeof(error))) &&
              rejected_data_manifest.evaluation_allocation_count == 0,
          "data manifest digest tamper rejects before evaluation allocation");

    for (int artifact = 0; artifact < 2; ++artifact) {
        fixture.write_manifest(std::string(), -1, artifact);
        g1_lmm_model_bundle rejected;
        error[0] = '\0';
        check(!g1_lmm_model_load_and_verify(
                  rejected,
                  fixture.model.c_str(),
                  fixture.data.c_str(),
                  fixture.data_manifest,
                  error,
                  static_cast<int>(sizeof(error))) &&
                  rejected.evaluation_allocation_count == 0,
              "data artifact digest tamper rejects before evaluation allocation");
    }
    for (int artifact = 0; artifact < 6; ++artifact) {
        fixture.write_manifest(std::string(), artifact);
        g1_lmm_model_bundle rejected;
        error[0] = '\0';
        check(!g1_lmm_model_load_and_verify(
                  rejected,
                  fixture.model.c_str(),
                  fixture.data.c_str(),
                  fixture.data_manifest,
                  error,
                  static_cast<int>(sizeof(error))) &&
                  rejected.evaluation_allocation_count == 0,
              "model artifact digest tamper rejects before evaluation allocation");
    }

    const std::filesystem::path real_data =
        "sonic/runs/g1-lmm-flat-60hz/data";
    motion_pack_manifest real_data_manifest;
    error[0] = '\0';
    check(motion_manifest_load_and_verify(
              real_data_manifest,
              real_data.c_str(),
              error,
              static_cast<int>(sizeof(error))),
          error);
    const motion_pack_manifest synthetic_data_manifest = fixture.data_manifest;
    const std::string synthetic_data_sha = fixture.data_manifest_sha;
    fixture.data_manifest = real_data_manifest;
    fixture.data_manifest_sha = file_sha(real_data / "manifest.json");
    fixture.write_manifest();
    g1_lmm_model_bundle real_bound_model;
    check(g1_lmm_model_load_and_verify(
              real_bound_model,
              fixture.model.c_str(),
              real_data.c_str(),
              real_data_manifest,
              error,
              static_cast<int>(sizeof(error))) &&
              real_bound_model.authenticated,
          "production-shaped synthetic model binds to the canonical Task 1 v2 bundle");
    fixture.data_manifest = synthetic_data_manifest;
    fixture.data_manifest_sha = synthetic_data_sha;

    fixture.write_manifest();
    {
        std::ofstream trailing(
            fixture.model / "projector.bin",
            std::ios::binary | std::ios::app);
        trailing.put('\0');
    }
    fixture.model_sha[3] = file_sha(fixture.model / "projector.bin");
    fixture.model_size[3] = static_cast<int>(std::filesystem::file_size(
        fixture.model / "projector.bin"));
    fixture.write_manifest();
    g1_lmm_model_bundle rejected_abi;
    error[0] = '\0';
    check(!g1_lmm_model_load_and_verify(
              rejected_abi,
              fixture.model.c_str(),
              fixture.data.c_str(),
              fixture.data_manifest,
              error,
              static_cast<int>(sizeof(error))) &&
              rejected_abi.evaluation_allocation_count == 0 &&
              std::strstr(error, "trailing") != nullptr,
          "self-consistent network trailing bytes fail the exact Orange Duck ABI");

    {
        std::ofstream tampered(
            fixture.data / "database.bin",
            std::ios::binary | std::ios::app);
        tampered.put('\0');
    }
    g1_lmm_model_bundle rejected_live_data;
    error[0] = '\0';
    check(!g1_lmm_model_load_and_verify(
              rejected_live_data,
              fixture.model.c_str(),
              fixture.data.c_str(),
              fixture.data_manifest,
              error,
              static_cast<int>(sizeof(error))) &&
              rejected_live_data.evaluation_allocation_count == 0,
          "live data artifact tamper is reauthenticated before model evaluation allocation");
}

int main()
{
    test_dimensions_and_normalized_projector_cost();
    test_controller_state_owns_lmm_transaction_state();
    test_missing_bundle_fails_before_evaluation_allocation();
    test_authenticated_bundle_and_digest_tampers();
    return 0;
}
