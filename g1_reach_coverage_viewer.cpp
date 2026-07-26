#include "g1_mesh_renderer.h"
#include "g1_skeleton.h"
#include "reach_search.h"
#include "reach_straight_approach.h"
#include "raylib.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr float kPi = 3.14159265358979323846F;
constexpr float kCameraTargetLimit = 5.0F;
constexpr float kWristContactOffset = 0.04F;
constexpr float kInboundPlaybackSeconds = 3.6F;
constexpr float kTableTop = 0.65F;
constexpr float kTableThickness = 0.06F;
constexpr float kTableCenterY = 0.62F;
constexpr vec3 kTableDimensions(1.20F, kTableThickness, 0.75F);
constexpr const char* kCapturedLabel = "CAPTURED LEFT";
constexpr const char* kMirroredLabel = "MIRRORED RIGHT";

enum class ApproachDisplayMode : uint8_t {
    Raw = 0U,
    Preferred = 1U,
    Retargeted = 2U,
};

const char* approach_mode_label(ApproachDisplayMode mode) {
    switch (mode) {
        case ApproachDisplayMode::Raw: return "RAW";
        case ApproachDisplayMode::Preferred: return "PREFERRED";
        case ApproachDisplayMode::Retargeted: return "RETARGETED";
    }
    return "RAW";
}

// RAW indexes the raw accepted order; PREFERRED and RETARGETED index the
// straight-in preferred order. Both vectors hold the same candidate identities.
const std::vector<size_t>& active_order(
    const std::optional<reach::SearchResult>& results,
    ApproachDisplayMode mode) {
    return mode == ApproachDisplayMode::Raw
        ? results->accepted
        : results->preferred;
}

struct Options {
    std::filesystem::path pack;
    vec3 object_size{0.10F, 0.10F, 0.10F};
};

struct ViewDiagnostics {
    std::array<size_t, reach::kRejectionCount> rejections{};
    std::array<size_t, 2U> accepted_hands{};
    size_t joint_limit_saturated = 0U;
};

struct OrbitCameraState {
    vec3 target{0.0F, 0.70F, 0.0F};
    float azimuth = -0.80F;
    float altitude = 0.32F;
    float distance = 3.50F;
};

Vector3 ray(vec3 value) {
    return {value.x, value.y, value.z};
}

void update_orbit_camera(
    Camera3D& camera,
    OrbitCameraState& state) {
    const Vector2 mouse_delta = GetMouseDelta();
    const bool pan_down =
        IsMouseButtonDown(MOUSE_BUTTON_RIGHT) ||
        IsMouseButtonDown(MOUSE_BUTTON_MIDDLE);
    if (IsMouseButtonDown(MOUSE_BUTTON_LEFT) && !pan_down) {
        state.azimuth -= 0.006F * mouse_delta.x;
        state.azimuth = std::remainder(state.azimuth, 2.0F * kPi);
        state.altitude = std::clamp(
            state.altitude + 0.006F * mouse_delta.y,
            -1.30F,
            1.30F);
    }
    state.distance = std::clamp(
        state.distance * std::exp(-0.16F * GetMouseWheelMove()),
        0.60F,
        8.0F);

    const float cos_altitude = std::cos(state.altitude);
    const vec3 offset_direction(
        cos_altitude * std::cos(state.azimuth),
        std::sin(state.altitude),
        cos_altitude * std::sin(state.azimuth));
    const vec3 forward = -offset_direction;
    const vec3 world_up(0.0F, 1.0F, 0.0F);
    const vec3 right = normalize(cross(forward, world_up));
    const vec3 view_up = normalize(cross(right, forward));
    if (pan_down) {
        const float pan_scale = 0.0015F * state.distance;
        state.target = state.target +
            pan_scale * (mouse_delta.x * right + mouse_delta.y * view_up);
    }
    state.target.x = std::clamp(
        state.target.x, -kCameraTargetLimit, kCameraTargetLimit);
    state.target.y = std::clamp(
        state.target.y, -kCameraTargetLimit, kCameraTargetLimit);
    state.target.z = std::clamp(
        state.target.z, -kCameraTargetLimit, kCameraTargetLimit);
    const vec3 position = state.target + state.distance * offset_direction;
    camera.position = ray(position);
    camera.target = ray(state.target);
    camera.up = ray(world_up);
}

Options parse_options(int argc, char** argv) {
    if (argc < 2) {
        throw std::invalid_argument(
            "usage: g1_reach_coverage_viewer PACK "
            "[--object-size X Y Z]");
    }
    Options options{};
    options.pack = argv[1];
    for (int arg = 2; arg < argc; ++arg) {
        const std::string value = argv[arg];
        if (value == "--object-size" && arg + 3 < argc) {
            const float x = std::stof(argv[++arg]);
            const float y = std::stof(argv[++arg]);
            const float z = std::stof(argv[++arg]);
            options.object_size = vec3(x, y, z);
        } else {
            throw std::invalid_argument(
                "unknown or incomplete argument: " + value);
        }
    }
    if (!(options.object_size.x > 0.0F &&
          options.object_size.y > 0.0F &&
          options.object_size.z > 0.0F)) {
        throw std::invalid_argument("object dimensions must be positive");
    }
    return options;
}

size_t wrist_bone(reach::Hand hand) {
    return hand == reach::Hand::Left
        ? static_cast<size_t>(g1_skeleton::LeftWrist)
        : static_cast<size_t>(g1_skeleton::RightWrist);
}

reach::Hand evaluation_hand(
    const reach::Pack& pack,
    const reach::Evaluation& evaluation) {
    return static_cast<reach::Hand>(
        pack.database.active_hands.at(evaluation.candidate.clip));
}

const char* provenance(const reach::Pack& pack, size_t clip) {
    return pack.database.augmentations.at(clip) ==
        static_cast<uint8_t>(reach::Augmentation::Captured)
        ? kCapturedLabel
        : kMirroredLabel;
}

size_t inbound_pose_count(
    const reach::Database& database,
    size_t clip) {
    const int32_t start = database.range_starts.at(clip);
    const int32_t contact = database.contact_frames.at(clip);
    if (start < 0 || contact < start) {
        throw std::runtime_error("invalid inbound reach frame boundary");
    }
    return static_cast<size_t>(contact - start) + 1U;
}

size_t inbound_playback_sample(
    size_t pose_count,
    float animation_seconds) {
    if (pose_count == 0U) {
        throw std::runtime_error("cannot play an empty inbound reach");
    }
    const float progress = std::clamp(
        animation_seconds / kInboundPlaybackSeconds,
        0.0F,
        1.0F);
    return std::min(
        pose_count - 1U,
        static_cast<size_t>(
            progress * static_cast<float>(pose_count - 1U)));
}

interaction::Transform desired_grasp(
    const interaction::Transform& object,
    vec3 dimensions) {
    return interaction::compose(
        object,
        {
            vec3(
                0.5F * dimensions.x + kWristContactOffset,
                0.0F,
                0.0F),
            quat_from_angle_axis(kPi, vec3(0.0F, 1.0F, 0.0F)),
        });
}

reach::ExhaustiveQuery make_query(
    const interaction::Transform& object,
    vec3 dimensions,
    vec3 grasp_approach_local) {
    return {
        desired_grasp(object, dimensions),
        normalize(quat_mul_vec3(object.rotation, grasp_approach_local)),
    };
}

interaction::Transform reset_object(vec3 dimensions) {
    return {
        vec3(0.0F, kTableTop + 0.5F * dimensions.y, 0.0F),
        quat(),
    };
}

vec3 reset_grasp_approach_local() {
    return vec3(-1.0F, 0.0F, 0.0F);
}

ViewDiagnostics summarize(
    const reach::Pack& pack,
    const reach::SearchResult& results) {
    ViewDiagnostics diagnostics{};
    for (const reach::CompactEvaluation& compact : results.evaluations) {
        const reach::Evaluation& evaluation = compact.evaluation;
        ++diagnostics.rejections.at(
            static_cast<size_t>(evaluation.rejection));
        if (evaluation.joint_limit_saturated) {
            ++diagnostics.joint_limit_saturated;
        }
        if (evaluation.rejection == reach::Rejection::None) {
            ++diagnostics.accepted_hands.at(
                pack.database.active_hands.at(evaluation.candidate.clip));
        }
    }
    return diagnostics;
}

std::vector<size_t> rejected_indices(
    const reach::SearchResult& results) {
    std::vector<size_t> rejected;
    rejected.reserve(results.evaluations.size() - results.accepted.size());
    for (size_t index = 0U; index < results.evaluations.size(); ++index) {
        if (results.evaluations[index].evaluation.rejection !=
            reach::Rejection::None) {
            rejected.push_back(index);
        }
    }
    return rejected;
}

Color trajectory_color(const reach::Pack& pack, size_t clip, bool accepted) {
    const bool mirrored = pack.database.augmentations.at(clip) ==
        static_cast<uint8_t>(reach::Augmentation::Mirrored);
    const Color base = mirrored ? MAGENTA : SKYBLUE;
    return accepted ? base : Fade(base, 0.28F);
}

void draw_oriented_box(
    const interaction::OrientedBox& box,
    Color color) {
    std::array<vec3, 8U> corners{};
    for (size_t corner = 0U; corner < corners.size(); ++corner) {
        const vec3 local(
            (corner & 1U ? 0.5F : -0.5F) * box.dimensions.x,
            (corner & 2U ? 0.5F : -0.5F) * box.dimensions.y,
            (corner & 4U ? 0.5F : -0.5F) * box.dimensions.z);
        corners[corner] = box.world.position +
            quat_mul_vec3(box.world.rotation, local);
    }
    for (size_t corner = 0U; corner < corners.size(); ++corner) {
        for (const size_t bit : {1U, 2U, 4U}) {
            const size_t other = corner ^ bit;
            if (corner < other) {
                DrawLine3D(ray(corners[corner]), ray(corners[other]), color);
            }
        }
    }
}

void draw_axes(
    const interaction::Transform& transform,
    float scale,
    unsigned char alpha) {
    const std::array<vec3, 3U> axes = {
        vec3(scale, 0, 0), vec3(0, scale, 0), vec3(0, 0, scale)};
    const std::array<Color, 3U> colors = {
        Color{230, 65, 65, alpha}, Color{65, 220, 90, alpha},
        Color{65, 110, 240, alpha}};
    for (size_t axis = 0U; axis < axes.size(); ++axis) {
        DrawLine3D(
            ray(transform.position),
            ray(transform.position +
                quat_mul_vec3(transform.rotation, axes[axis])),
            colors[axis]);
    }
}

void draw_pose(
    const interaction::WorldPose& world,
    Color joint_color,
    Color bone_color) {
    for (size_t bone = 0U; bone < g1_skeleton::BoneCount; ++bone) {
        DrawSphereWires(
            ray(world.positions[bone]), 0.024F, 4, 8, joint_color);
        const int32_t parent = g1_skeleton::kParents[bone];
        if (parent < 0) continue;
        DrawCylinderEx(
            ray(world.positions[static_cast<size_t>(parent)]),
            ray(world.positions[bone]),
            0.014F,
            0.014F,
            6,
            bone_color);
    }
}

void draw_path(
    const std::vector<vec3>& path,
    Color color,
    size_t stride,
    size_t sample_count,
    bool emphasized = false) {
    sample_count = std::min(sample_count, path.size());
    if (sample_count < 2U) return;
    vec3 previous = path.front();
    for (size_t sample = stride; sample < sample_count; sample += stride) {
        const vec3 current = path[sample];
        if (emphasized) {
            DrawCylinderEx(
                ray(previous), ray(current), 0.009F, 0.009F, 6, color);
            DrawSphere(ray(current), 0.014F, color);
        } else {
            DrawLine3D(ray(previous), ray(current), color);
        }
        previous = current;
    }
    const vec3 inbound_end = path[sample_count - 1U];
    DrawLine3D(ray(previous), ray(inbound_end), color);
    DrawSphere(ray(inbound_end), emphasized ? 0.018F : 0.010F, color);
}

std::vector<vec3> active_wrist_path(
    const reach::Evaluation& evaluation,
    reach::Hand hand,
    size_t sample_count) {
    sample_count = std::min(sample_count, evaluation.poses.size());
    std::vector<vec3> path;
    path.reserve(sample_count);
    const size_t wrist = wrist_bone(hand);
    for (size_t sample = 0U; sample < sample_count; ++sample) {
        path.push_back(
            interaction::world_pose(evaluation.poses[sample]).positions[wrist]);
    }
    return path;
}

void draw_hud(
    const reach::Pack& pack,
    const std::optional<reach::SearchResult>& results,
    size_t selected,
    const std::vector<size_t>& rejected,
    size_t selected_rejected_option,
    bool selected_rejected,
    bool stale,
    bool search_incomplete,
    bool coverage_environment,
    bool show_rejected,
    bool show_g1_mesh,
    bool show_g1_bones,
    ApproachDisplayMode approach_mode,
    const std::optional<reach::CorridorRetargetResult>& selected_retarget,
    const char* g1_mesh_error) {
    DrawRectangle(14, 14, 800, 400, Color{255, 255, 255, 230});
    DrawText(
        "Arrow X/Z  W/S Y | Q/E yaw R/F pitch Z/C roll | Enter search",
        26, 24, 17, DARKGRAY);
    DrawText(
        "G environment | A mode | [ ] / cycle | < > rejected | V paths | M mesh | B bones",
        26, 48, 16, DARKGRAY);
    DrawText(
        TextFormat("%s ENVIRONMENT | MODE %s | rejected paths %s",
            coverage_environment ? "COVERAGE" : "OPEN",
            approach_mode_label(approach_mode),
            show_rejected ? "ON" : "OFF"),
        26, 74, 18, DARKGRAY);

    if (!results.has_value()) {
        DrawText("PRESS ENTER TO SEARCH", 26, 106, 24, DARKBLUE);
    } else {
        const ViewDiagnostics diagnostics = summarize(pack, *results);
        const size_t displayed = results->accepted.size();
        DrawText(
            TextFormat("RAW INSTANCES %zu | PROCESSED %zu | ACCEPTED %zu",
                results->total, results->processed, results->accepted.size()),
            26, 102, 18, DARKGRAY);
        DrawText(
            TextFormat(
                "displayed %zu / accepted %zu | LEFT %zu | RIGHT %zu | joint limit %zu",
                displayed,
                results->accepted.size(),
                diagnostics.accepted_hands[0],
                diagnostics.accepted_hands[1],
                diagnostics.joint_limit_saturated),
            26, 126, 16, DARKGRAY);
        static constexpr std::array<const char*, 8U> labels = {
            "ACCEPTED", "OUTSIDE ENVELOPE", "INVALID SOLVER", "POSITION ERROR",
            "APPROACH AXIS", "FULL ORIENTATION", "OBJECT COLLISION",
            "ENVIRONMENT COLLISION"};
        for (size_t reason = 0U; reason < labels.size(); ++reason) {
            const int column = reason % 2U == 0U ? 26 : 390;
            const int row = 151 + static_cast<int>(reason / 2U) * 20;
            DrawText(
                TextFormat("%s: %zu", labels[reason],
                    diagnostics.rejections[reason]),
                column,
                row,
                15,
                reason == 0U ? DARKGREEN : DARKGRAY);
        }
        const std::vector<size_t>& order =
            active_order(results, approach_mode);
        const bool has_selected = selected_rejected
            ? selected_rejected_option < rejected.size()
            : selected < order.size();
        if (has_selected) {
            const size_t selected_evaluation = selected_rejected
                ? rejected[selected_rejected_option]
                : order[selected];
            const reach::Evaluation& evaluation =
                results->evaluations[selected_evaluation].evaluation;
            const bool has_path_quality =
                !results->evaluations[selected_evaluation].hand_path.empty() &&
                std::isfinite(evaluation.directness_cost) &&
                std::isfinite(evaluation.backtrack_ratio) &&
                std::isfinite(evaluation.excess_path_ratio);
            const size_t clip = evaluation.candidate.clip;
            const auto raw_rank_it = std::find(
                results->accepted.begin(),
                results->accepted.end(),
                selected_evaluation);
            const auto preferred_rank_it = std::find(
                results->preferred.begin(),
                results->preferred.end(),
                selected_evaluation);
            const size_t raw_rank = raw_rank_it == results->accepted.end()
                ? 0U
                : static_cast<size_t>(
                    raw_rank_it - results->accepted.begin()) + 1U;
            const size_t preferred_rank =
                preferred_rank_it == results->preferred.end()
                ? 0U
                : static_cast<size_t>(
                    preferred_rank_it - results->preferred.begin()) + 1U;
            DrawText(
                TextFormat(
                    "%s option %zu/%zu | RAW %zu/%zu | PREF %zu/%zu | "
                    "YAW PLACEMENT %u/12 | %s | %s",
                    selected_rejected ? "REJECTED" : approach_mode_label(
                        approach_mode),
                    (selected_rejected ? selected_rejected_option : selected) + 1U,
                    selected_rejected ? rejected.size() : order.size(),
                    raw_rank,
                    results->accepted.size(),
                    preferred_rank,
                    results->accepted.size(),
                    static_cast<unsigned>(evaluation.candidate.yaw_index) + 1U,
                    provenance(pack, clip),
                    pack.database.source_names.at(
                        pack.database.source_indices.at(clip)).c_str()),
                26,
                239,
                16,
                selected_rejected ? MAROON : DARKGREEN);
            DrawText(
                TextFormat(
                    "%s | final %.2f cm | approach %.1f deg | orientation %.1f deg | deform %.4f",
                    reach::rejection_name(evaluation.rejection),
                    evaluation.position_error_m * 100.0F,
                    evaluation.approach_error_radians * 180.0F / kPi,
                    evaluation.orientation_error_radians * 180.0F / kPi,
                    evaluation.active_arm_deformation),
                26,
                264,
                16,
                DARKGRAY);
            if (has_path_quality) {
                DrawText(
                    TextFormat(
                        "DIRECT %.3f | BACKTRACK %.1f%% | EXCESS PATH %.1f%%",
                        evaluation.directness_cost,
                        100.0F * evaluation.backtrack_ratio,
                        100.0F * evaluation.excess_path_ratio),
                    26,
                    289,
                    16,
                    evaluation.rejection == reach::Rejection::None &&
                            evaluation.directness_cost <= 0.10F
                        ? DARKGREEN
                        : DARKGRAY);
            } else {
                DrawText(
                    "DIRECT N/A | BACKTRACK N/A | EXCESS PATH N/A",
                    26,
                    289,
                    16,
                    DARKGRAY);
            }
            DrawText(
                TextFormat(
                    "OBSERVED OBJECT COLLISION: %s | OBSERVED ENVIRONMENT COLLISION: %s",
                    evaluation.object_collision_observed ? "YES" : "NO",
                    evaluation.environment_collision_observed ? "YES" : "NO"),
                26,
                314,
                16,
                evaluation.object_collision_observed ||
                        evaluation.environment_collision_observed
                    ? MAROON
                    : DARKGREEN);
            const reach::StraightApproachQuality& quality =
                results->evaluations[selected_evaluation].straight_approach;
            DrawText(
                TextFormat(
                    "MAX LATERAL %.2f cm | RMS LATERAL %.2f cm | "
                    "MAX ANGLE %.1f deg | RMS ANGLE %.1f deg | BACKWARD %.1f%%",
                    quality.maximum_lateral_m * 100.0F,
                    quality.rms_lateral_m * 100.0F,
                    quality.maximum_angle_radians * 180.0F / kPi,
                    quality.rms_angle_radians * 180.0F / kPi,
                    100.0F * quality.backward_ratio),
                26,
                339,
                15,
                quality.reaches_pregrasp_plane ? DARKGRAY : MAROON);
            if (!quality.reaches_pregrasp_plane) {
                DrawText("NO PREGRASP COVERAGE", 26, 359, 16, MAROON);
            } else if (approach_mode == ApproachDisplayMode::Retargeted) {
                if (selected_retarget.has_value() &&
                    selected_retarget->accepted) {
                    DrawText(
                        TextFormat(
                            "RETARGET ACCEPTED | active arm deform %.4f",
                            selected_retarget->active_arm_deformation),
                        26,
                        359,
                        16,
                        DARKGREEN);
                } else {
                    const char* reason = "INVALID";
                    if (selected_retarget.has_value()) {
                        switch (selected_retarget->failure) {
                            case reach::CorridorRetargetFailure::
                                NoPregraspCoverage:
                                reason = "NO PREGRASP COVERAGE"; break;
                            case reach::CorridorRetargetFailure::OutsideCorridor:
                                reason = "OUTSIDE CORRIDOR"; break;
                            case reach::CorridorRetargetFailure::BackwardMotion:
                                reason = "BACKWARD MOTION"; break;
                            case reach::CorridorRetargetFailure::ObjectCollision:
                                reason = "OBJECT COLLISION"; break;
                            case reach::CorridorRetargetFailure::
                                EnvironmentCollision:
                                reason = "ENVIRONMENT COLLISION"; break;
                            case reach::CorridorRetargetFailure::InvalidSolver:
                                reason = "INVALID SOLVER"; break;
                            default: reason = "INVALID INPUT"; break;
                        }
                    }
                    DrawText(
                        TextFormat("CORRIDOR RETARGET FAILED: %s", reason),
                        26,
                        359,
                        16,
                        MAROON);
                }
            }
        } else if (order.empty()) {
            DrawText("0 valid motions", 26, 239, 20, MAROON);
        }
    }
    if (show_rejected) {
        DrawText(
            "REJECTED - motion shown for diagnosis",
            26,
            343,
            15,
            MAROON);
    }
    DrawText(
        TextFormat(
            "M mesh %s | B bones %s | left orbit | right/middle pan | wheel zoom",
            show_g1_mesh ? "ON" : "OFF",
            show_g1_bones ? "ON" : "OFF"),
        26,
        367,
        15,
        DARKGRAY);
    if (g1_mesh_error != nullptr && g1_mesh_error[0] != '\0') {
        DrawText(
            TextFormat("G1 MESH DISABLED: %s", g1_mesh_error),
            820,
            82,
            16,
            MAROON);
    }
    if (stale) {
        DrawText("SEARCH STALE - press Enter", 830, 22, 24, MAROON);
    }
    if (search_incomplete) {
        DrawText(
            "SEARCH INCOMPLETE - previous complete results retained",
            820,
            52,
            18,
            MAROON);
    }
}

void regenerate_index(
    const reach::Pack& pack,
    const reach::SearchResult& results,
    size_t evaluation_index,
    const reach::ExhaustiveQuery& query,
    const interaction::OrientedBox& object,
    const interaction::EnvironmentGeometry& environment,
    const reach::SearchConfig& config,
    std::optional<reach::Evaluation>& selected_full) {
    selected_full.reset();
    selected_full = reach::regenerate(
        pack,
        results.evaluations.at(evaluation_index),
        query,
        object,
        environment,
        config);
}

// Lazily retargets only the selected regenerated candidate through the
// straight-in corridor. Cached by the caller until selection/mode/object/
// grasp/environment changes; never reruns exhaustive search.
void retarget_selected(
    const reach::Pack& pack,
    const std::optional<reach::Evaluation>& selected_full,
    const reach::ExhaustiveQuery& query,
    const interaction::OrientedBox& object,
    const interaction::EnvironmentGeometry& environment,
    const reach::SearchConfig& config,
    std::optional<reach::CorridorRetargetResult>& selected_retarget) {
    selected_retarget.reset();
    if (!selected_full.has_value() || selected_full->poses.empty()) return;
    const reach::Hand hand = evaluation_hand(pack, *selected_full);
    const float fps =
        static_cast<float>(pack.database.fps_numerator) /
        static_cast<float>(pack.database.fps_denominator);
    selected_retarget = reach::retarget_straight_approach(
        selected_full->poses,
        hand,
        query.target,
        query.approach_world,
        object,
        environment,
        fps,
        config.straight_approach);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        const reach::Pack pack = reach::load_pack(options.pack);
        const interaction::Transform table_world{
            vec3(0.0F, kTableCenterY, 0.0F), quat()};
        const interaction::EnvironmentGeometry coverage =
            interaction::make_coverage_environment(
                table_world, kTableDimensions);
        const interaction::EnvironmentGeometry open{};
        interaction::Transform object = reset_object(options.object_size);
        vec3 grasp_approach_local = reset_grasp_approach_local();
        bool use_coverage_environment = true;
        bool show_rejected = false;
        bool stale = false;
        bool search_incomplete = false;
        size_t selected = 0U;
        std::vector<size_t> rejected;
        size_t selected_rejected_option = 0U;
        bool selected_rejected = false;
        float animation_seconds = 0.0F;
        ApproachDisplayMode approach_mode = ApproachDisplayMode::Raw;
        std::optional<reach::SearchResult> results;
        std::optional<reach::Evaluation> selected_full;
        std::optional<reach::CorridorRetargetResult> selected_retarget;
        reach::SearchConfig search_config{};
        search_config.worker_count = std::max<size_t>(1U, std::min<size_t>(
            32U, std::max(1U, std::thread::hardware_concurrency())));
        reach::ExhaustiveQuery query = make_query(
            object, options.object_size, grasp_approach_local);

        SetConfigFlags(FLAG_VSYNC_HINT | FLAG_MSAA_4X_HINT);
        InitWindow(1280, 800, "G1 contact-anchored reach coverage");
        if (!IsWindowReady()) {
            throw std::runtime_error(
                "G1 reach coverage window failed to initialize");
        }
        SetTargetFPS(60);
        Camera3D camera{};
        camera.fovy = 45.0F;
        camera.projection = CAMERA_PERSPECTIVE;
        OrbitCameraState camera_state{};
        update_orbit_camera(camera, camera_state);
        G1MeshRenderer g1_mesh_renderer{};
        std::array<char, 256U> g1_mesh_error{};
        bool show_g1_mesh = true;
        bool show_g1_bones = false;
        if (!::g1_mesh_renderer_load(
                g1_mesh_renderer,
                "resources/g1_mesh/g1_raylib.glb",
                g1_mesh_error.data(),
                static_cast<int>(g1_mesh_error.size()))) {
            show_g1_mesh = false;
            show_g1_bones = true;
        }
        const auto close_graphics = [&] {
            ::g1_mesh_renderer_unload(g1_mesh_renderer);
            CloseWindow();
        };

        try {
        while (!WindowShouldClose()) {
            const float dt = GetFrameTime();
            update_orbit_camera(camera, camera_state);
            bool target_changed = false;
            const float move = 0.60F * dt;
            if (IsKeyDown(KEY_LEFT)) {
                object.position.x -= move;
                target_changed = true;
            }
            if (IsKeyDown(KEY_RIGHT)) {
                object.position.x += move;
                target_changed = true;
            }
            if (IsKeyDown(KEY_UP)) {
                object.position.z -= move;
                target_changed = true;
            }
            if (IsKeyDown(KEY_DOWN)) {
                object.position.z += move;
                target_changed = true;
            }
            if (IsKeyDown(KEY_W)) {
                object.position.y += move;
                target_changed = true;
            }
            if (IsKeyDown(KEY_S)) {
                object.position.y -= move;
                target_changed = true;
            }
            const std::array<int, 6U> rotation_keys = {
                KEY_Q, KEY_E, KEY_R, KEY_F, KEY_Z, KEY_C};
            const std::array<vec3, 6U> rotation_axes = {
                vec3(0, 1, 0), vec3(0, -1, 0), vec3(0, 0, 1),
                vec3(0, 0, -1), vec3(1, 0, 0), vec3(-1, 0, 0)};
            for (size_t key = 0U; key < rotation_keys.size(); ++key) {
                if (IsKeyDown(rotation_keys[key])) {
                    object.rotation = quat_normalize(quat_mul(
                        quat_from_angle_axis(1.3F * dt, rotation_axes[key]),
                        object.rotation));
                    target_changed = true;
                }
            }
            if (IsKeyPressed(KEY_G)) {
                use_coverage_environment = !use_coverage_environment;
                target_changed = true;
            }
            bool mode_changed = false;
            if (IsKeyPressed(KEY_A)) {
                // Cycle RAW -> PREFERRED -> RETARGETED without touching the
                // cached results or rerunning the exhaustive search.
                approach_mode = static_cast<ApproachDisplayMode>(
                    (static_cast<uint8_t>(approach_mode) + 1U) % 3U);
                selected = 0U;
                selected_rejected = false;
                animation_seconds = 0.0F;
                selected_full.reset();
                selected_retarget.reset();
                mode_changed = true;
            }
            if (IsKeyPressed(KEY_M) && g1_mesh_renderer.loaded) {
                show_g1_mesh = !show_g1_mesh;
            }
            if (IsKeyPressed(KEY_B)) {
                show_g1_bones = !show_g1_bones;
            }
            if (IsKeyPressed(KEY_V)) {
                show_rejected = !show_rejected;
                if (!show_rejected && selected_rejected) {
                    selected_rejected = false;
                    animation_seconds = 0.0F;
                    selected_full.reset();
                    selected_retarget.reset();
                    if (results.has_value() && !results->accepted.empty() && !stale) {
                        regenerate_index(
                            pack,
                            *results,
                            active_order(results, approach_mode)[selected],
                            query,
                            {object, options.object_size},
                            use_coverage_environment ? coverage : open,
                            search_config,
                            selected_full);
                        if (approach_mode ==
                            ApproachDisplayMode::Retargeted) {
                            retarget_selected(
                                pack, selected_full, query,
                                {object, options.object_size},
                                use_coverage_environment ? coverage : open,
                                search_config, selected_retarget);
                        }
                    }
                }
            }
            if (IsKeyPressed(KEY_BACKSPACE)) {
                object = reset_object(options.object_size);
                grasp_approach_local = reset_grasp_approach_local();
                target_changed = true;
            }
            if (target_changed) stale = true;
            query = make_query(
                object, options.object_size, grasp_approach_local);

            const interaction::EnvironmentGeometry& environment =
                use_coverage_environment ? coverage : open;
            const interaction::OrientedBox object_box{
                object, options.object_size};
            if (IsKeyPressed(KEY_ENTER)) {
                reach::SearchResult completed = reach::search_all(
                    pack, query, object_box, environment, search_config);
                if (completed.complete) {
                    results = std::move(completed);
                    selected = 0U;
                    rejected = rejected_indices(*results);
                    selected_rejected_option = 0U;
                    selected_rejected = false;
                    animation_seconds = 0.0F;
                    selected_full.reset();
                    selected_retarget.reset();
                    const std::vector<size_t>& order =
                        active_order(results, approach_mode);
                    if (!order.empty()) {
                        regenerate_index(
                            pack,
                            *results,
                            order[selected],
                            query,
                            object_box,
                            environment,
                            search_config,
                            selected_full);
                        if (approach_mode ==
                            ApproachDisplayMode::Retargeted) {
                            retarget_selected(
                                pack, selected_full, query, object_box,
                                environment, search_config,
                                selected_retarget);
                        }
                    }
                    stale = false;
                    search_incomplete = false;
                } else {
                    search_incomplete = true;
                }
            }
            if (results.has_value() &&
                !active_order(results, approach_mode).empty() && !stale) {
                const std::vector<size_t>& order =
                    active_order(results, approach_mode);
                bool selection_changed = mode_changed;
                if (mode_changed && selected >= order.size()) {
                    selected = 0U;
                }
                // Bracket cycling wraps within the active cached order and
                // never reruns the exhaustive search.
                if (IsKeyPressed(KEY_LEFT_BRACKET)) {
                    selected =
                        (selected + order.size() - 1U) % order.size();
                    selection_changed = true;
                }
                if (IsKeyPressed(KEY_RIGHT_BRACKET) ||
                    IsKeyPressed(KEY_SLASH)) {
                    selected = (selected + 1U) % order.size();
                    selection_changed = true;
                }
                if (selection_changed) {
                    selected_rejected = false;
                    animation_seconds = 0.0F;
                    selected_retarget.reset();
                    regenerate_index(
                        pack,
                        *results,
                        order[selected],
                        query,
                        object_box,
                        environment,
                        search_config,
                        selected_full);
                    if (approach_mode == ApproachDisplayMode::Retargeted) {
                        retarget_selected(
                            pack, selected_full, query, object_box,
                            environment, search_config, selected_retarget);
                    }
                }
            }
            if (results.has_value() && show_rejected &&
                !rejected.empty() && !stale) {
                bool rejected_selection_changed = false;
                if (IsKeyPressed(KEY_COMMA)) {
                    selected_rejected_option =
                        (selected_rejected_option + rejected.size() - 1U) %
                        rejected.size();
                    rejected_selection_changed = true;
                }
                if (IsKeyPressed(KEY_PERIOD)) {
                    selected_rejected_option =
                        (selected_rejected_option + 1U) % rejected.size();
                    rejected_selection_changed = true;
                }
                if (rejected_selection_changed) {
                    selected_rejected = true;
                    animation_seconds = 0.0F;
                    selected_retarget.reset();
                    regenerate_index(
                        pack,
                        *results,
                        rejected[selected_rejected_option],
                        query,
                        object_box,
                        environment,
                        search_config,
                        selected_full);
                }
            }
            if (selected_full.has_value()) animation_seconds += dt;

            size_t selected_evaluation = 0U;
            bool has_selected_evaluation = false;
            if (results.has_value() && selected_rejected &&
                selected_rejected_option < rejected.size()) {
                selected_evaluation = rejected[selected_rejected_option];
                has_selected_evaluation = true;
            } else if (results.has_value() &&
                       !active_order(results, approach_mode).empty()) {
                selected_evaluation =
                    active_order(results, approach_mode)[selected];
                has_selected_evaluation = true;
            }

            // The animated mesh/bones use the retargeted poses only when the
            // retarget is accepted; otherwise the regenerated recorded poses.
            const bool use_retargeted_poses =
                approach_mode == ApproachDisplayMode::Retargeted &&
                selected_retarget.has_value() &&
                selected_retarget->accepted &&
                !selected_retarget->poses.empty();
            const std::vector<interaction::Pose>* animation_poses =
                use_retargeted_poses
                    ? &selected_retarget->poses
                    : (selected_full.has_value()
                        ? &selected_full->poses
                        : nullptr);
            std::optional<interaction::WorldPose> selected_world_pose;
            if (animation_poses != nullptr && !animation_poses->empty() &&
                selected_full.has_value()) {
                const size_t playback_pose_count = std::min(
                    animation_poses->size(),
                    inbound_pose_count(
                        pack.database, selected_full->candidate.clip));
                const size_t sample = inbound_playback_sample(
                    playback_pose_count, animation_seconds);
                selected_world_pose = interaction::world_pose(
                    (*animation_poses)[sample]);
                if (show_g1_mesh && g1_mesh_renderer.loaded) {
                    interaction::WorldPose& mesh_world_pose =
                        *selected_world_pose;
                    if (!::g1_mesh_renderer_update(
                            g1_mesh_renderer,
                            slice1d<vec3>(
                                static_cast<int>(g1_skeleton::BoneCount),
                                mesh_world_pose.positions.data()),
                            slice1d<quat>(
                                static_cast<int>(g1_skeleton::BoneCount),
                                mesh_world_pose.rotations.data()),
                            g1_mesh_error.data(),
                            static_cast<int>(g1_mesh_error.size()))) {
                        show_g1_mesh = false;
                        show_g1_bones = true;
                    }
                }
            }

            BeginDrawing();
            ClearBackground(Color{238, 241, 245, 255});
            BeginMode3D(camera);
            DrawGrid(20, 0.20F);
            for (const interaction::OrientedBox& box : environment.boxes) {
                DrawCubeV(
                    ray(box.world.position),
                    ray(box.dimensions),
                    Color{135, 102, 74, 155});
                draw_oriented_box(box, Color{72, 52, 39, 255});
            }
            draw_oriented_box(object_box, GOLD);
            DrawSphere(ray(query.target.position), 0.028F, GOLD);
            draw_axes(query.target, 0.16F, 255);
            if (approach_mode == ApproachDisplayMode::Retargeted) {
                // Requested pre-grasp point (gold sphere) and 20 cm corridor
                // (gold cylinder) from pre-grasp toward contact.
                const vec3 corridor_dir = normalize(query.approach_world);
                const vec3 pregrasp = query.target.position -
                    search_config.straight_approach.corridor_length_m *
                        corridor_dir;
                DrawSphere(ray(pregrasp), 0.022F, GOLD);
                DrawCylinderEx(
                    ray(pregrasp),
                    ray(query.target.position),
                    search_config.straight_approach.corridor_radius_m,
                    search_config.straight_approach.corridor_radius_m,
                    12,
                    Color{212, 175, 55, 90});
            }
            if (show_g1_mesh && selected_world_pose.has_value()) {
                ::g1_mesh_renderer_draw(g1_mesh_renderer);
            }
            if (results.has_value()) {
                for (size_t index = 0U;
                     index < results->evaluations.size();
                     ++index) {
                    if (has_selected_evaluation && index == selected_evaluation) {
                        continue;
                    }
                    const reach::CompactEvaluation& compact =
                        results->evaluations[index];
                    const reach::Evaluation& evaluation = compact.evaluation;
                    const bool accepted =
                        evaluation.rejection == reach::Rejection::None;
                    if (!accepted && !show_rejected) continue;
                    const size_t inbound_count = inbound_pose_count(
                        pack.database, evaluation.candidate.clip);
                    draw_path(
                        compact.hand_path,
                        accepted
                            ? trajectory_color(
                                  pack, evaluation.candidate.clip, true)
                            : Color{210, 45, 55, 60},
                        accepted ? 5U : 10U,
                        inbound_count);
                }
            }
            if (selected_full.has_value() &&
                !selected_full->poses.empty()) {
                const reach::Evaluation& evaluation = *selected_full;
                const reach::Hand selected_hand =
                    evaluation_hand(pack, evaluation);
                const size_t inbound_count = inbound_pose_count(
                    pack.database, evaluation.candidate.clip);
                const std::vector<vec3> raw_path =
                    active_wrist_path(
                        evaluation, selected_hand, inbound_count);
                if (approach_mode == ApproachDisplayMode::Retargeted) {
                    // Raw wrist path as a faint purple reference.
                    draw_path(
                        raw_path,
                        Color{150, 90, 210, 90},
                        1U,
                        inbound_count,
                        false);
                    if (selected_retarget.has_value() &&
                        !selected_retarget->poses.empty()) {
                        std::vector<vec3> solved_path;
                        const size_t solved_count = std::min(
                            inbound_count, selected_retarget->poses.size());
                        solved_path.reserve(solved_count);
                        const size_t wrist = wrist_bone(selected_hand);
                        for (size_t sample = 0U;
                             sample < solved_count;
                             ++sample) {
                            solved_path.push_back(
                                interaction::world_pose(
                                    selected_retarget->poses[sample])
                                    .positions[wrist]);
                        }
                        // Accepted retarget path emphasized green, failed
                        // retarget path in orange.
                        draw_path(
                            solved_path,
                            selected_retarget->accepted ? GREEN : ORANGE,
                            1U,
                            solved_count,
                            true);
                    }
                }
                const Color selected_color =
                    evaluation.rejection == reach::Rejection::None ? LIME : ORANGE;
                if (approach_mode != ApproachDisplayMode::Retargeted) {
                    draw_path(
                        raw_path,
                        selected_color,
                        1U,
                        inbound_count,
                        true);
                }
                if (show_g1_bones && selected_world_pose.has_value()) {
                    draw_pose(
                        *selected_world_pose,
                        evaluation.rejection == reach::Rejection::None
                            ? DARKBLUE : MAROON,
                        evaluation.rejection == reach::Rejection::None
                            ? SKYBLUE : ORANGE);
                }
                const interaction::WorldPose final = interaction::world_pose(
                    evaluation.poses.at(inbound_count - 1U));
                const size_t wrist = wrist_bone(selected_hand);
                draw_axes(
                    {final.positions[wrist], final.rotations[wrist]},
                    0.13F,
                    150);
            }
            EndMode3D();
            draw_hud(
                pack,
                results,
                selected,
                rejected,
                selected_rejected_option,
                selected_rejected,
                stale,
                search_incomplete,
                use_coverage_environment,
                show_rejected,
                show_g1_mesh,
                show_g1_bones,
                approach_mode,
                selected_retarget,
                g1_mesh_error.data());
            EndDrawing();
        }
        } catch (...) {
            close_graphics();
            throw;
        }
        close_graphics();
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "g1 reach coverage viewer: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
