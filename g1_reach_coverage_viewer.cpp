#include "g1_skeleton.h"
#include "reach_search.h"
#include "raylib.h"

#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr float kPi = 3.14159265358979323846F;
constexpr float kWristContactOffset = 0.04F;
constexpr const char* kCapturedLabel = "CAPTURED LEFT";
constexpr const char* kMirroredLabel = "MIRRORED RIGHT";

struct Options {
    std::filesystem::path pack;
    vec3 object_size{0.10F, 0.10F, 0.10F};
};

struct ViewDiagnostics {
    std::array<size_t, reach::kRejectionCount> rejections{};
    std::array<size_t, 2U> accepted_hands{};
    size_t joint_limit_saturated = 0U;
};

Vector3 ray(vec3 value) {
    return {value.x, value.y, value.z};
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
    constexpr float table_top = 0.77F;
    return {
        vec3(0.0F, table_top + 0.5F * dimensions.y, 0.0F),
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
    const interaction::Pose& pose,
    Color joint_color,
    Color bone_color) {
    const interaction::WorldPose world = interaction::world_pose(pose);
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
    bool emphasized = false) {
    if (path.size() < 2U) return;
    vec3 previous = path.front();
    for (size_t sample = stride; sample < path.size(); sample += stride) {
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
    DrawLine3D(ray(previous), ray(path.back()), color);
    DrawSphere(ray(path.back()), emphasized ? 0.018F : 0.010F, color);
}

std::vector<vec3> active_wrist_path(
    const reach::Evaluation& evaluation,
    reach::Hand hand) {
    std::vector<vec3> path;
    path.reserve(evaluation.poses.size());
    const size_t wrist = wrist_bone(hand);
    for (const interaction::Pose& pose : evaluation.poses) {
        path.push_back(interaction::world_pose(pose).positions[wrist]);
    }
    return path;
}

void draw_hud(
    const reach::Pack& pack,
    const std::optional<reach::SearchResult>& results,
    size_t selected,
    bool stale,
    bool search_incomplete,
    bool coverage_environment,
    bool show_rejected) {
    DrawRectangle(14, 14, 800, 342, Color{255, 255, 255, 230});
    DrawText(
        "Arrow X/Z  W/S Y | Q/E yaw R/F pitch Z/C roll | Enter search",
        26, 24, 17, DARKGRAY);
    DrawText(
        "G open/coverage | [ previous, ] or / next | V rejected | Backspace reset",
        26, 48, 16, DARKGRAY);
    DrawText(
        TextFormat("%s ENVIRONMENT | rejected paths %s",
            coverage_environment ? "COVERAGE" : "OPEN",
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
        if (!results->accepted.empty()) {
            const size_t selected_evaluation = results->accepted[selected];
            const reach::Evaluation& evaluation =
                results->evaluations[selected_evaluation].evaluation;
            const size_t clip = evaluation.candidate.clip;
            DrawText(
                TextFormat(
                    "option %zu/%zu | YAW PLACEMENT %u/12 | %s | %s",
                    selected + 1U,
                    results->accepted.size(),
                    static_cast<unsigned>(evaluation.candidate.yaw_index) + 1U,
                    provenance(pack, clip),
                    pack.database.source_names.at(
                        pack.database.source_indices.at(clip)).c_str()),
                26,
                239,
                16,
                DARKGREEN);
            DrawText(
                TextFormat(
                    "final %.2f cm | approach %.1f deg | orientation %.1f deg | deform %.4f",
                    evaluation.position_error_m * 100.0F,
                    evaluation.approach_error_radians * 180.0F / kPi,
                    evaluation.orientation_error_radians * 180.0F / kPi,
                    evaluation.active_arm_deformation),
                26,
                264,
                16,
                DARKGRAY);
            DrawText(
                TextFormat(
                    "OBSERVED OBJECT COLLISION: %s | OBSERVED ENVIRONMENT COLLISION: %s",
                    evaluation.object_collision_observed ? "YES" : "NO",
                    evaluation.environment_collision_observed ? "YES" : "NO"),
                26,
                289,
                16,
                evaluation.object_collision_observed ||
                        evaluation.environment_collision_observed
                    ? MAROON
                    : DARKGREEN);
        } else {
            DrawText("0 valid motions", 26, 239, 20, MAROON);
        }
    }
    if (show_rejected) {
        DrawText(
            "REJECTED - motion shown for diagnosis",
            26,
            318,
            15,
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

void regenerate_selected(
    const reach::Pack& pack,
    const reach::SearchResult& results,
    size_t selected,
    const reach::ExhaustiveQuery& query,
    const interaction::OrientedBox& object,
    const interaction::EnvironmentGeometry& environment,
    const reach::SearchConfig& config,
    std::optional<reach::Evaluation>& selected_full) {
    selected_full.reset();
    if (results.accepted.empty()) return;
    selected_full = reach::regenerate(
        pack,
        results.evaluations[results.accepted[selected]],
        query,
        object,
        environment,
        config);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        const reach::Pack pack = reach::load_pack(options.pack);
        const interaction::Transform table_world{
            vec3(0.0F, 0.74F, 0.0F), quat()};
        const vec3 table_dimensions(1.20F, 0.06F, 0.75F);
        const interaction::EnvironmentGeometry coverage =
            interaction::make_coverage_environment(
                table_world, table_dimensions);
        const interaction::EnvironmentGeometry open{};
        interaction::Transform object = reset_object(options.object_size);
        vec3 grasp_approach_local = reset_grasp_approach_local();
        bool use_coverage_environment = true;
        bool show_rejected = false;
        bool stale = false;
        bool search_incomplete = false;
        size_t selected = 0U;
        float animation_seconds = 0.0F;
        std::optional<reach::SearchResult> results;
        std::optional<reach::Evaluation> selected_full;
        reach::SearchConfig search_config{};
        reach::ExhaustiveQuery query = make_query(
            object, options.object_size, grasp_approach_local);

        SetConfigFlags(FLAG_VSYNC_HINT | FLAG_MSAA_4X_HINT);
        InitWindow(1280, 800, "G1 contact-anchored reach coverage");
        SetTargetFPS(60);
        Camera3D camera{};
        camera.position = Vector3{2.35F, 1.55F, -2.45F};
        camera.target = Vector3{0.0F, 0.70F, 0.0F};
        camera.up = Vector3{0.0F, 1.0F, 0.0F};
        camera.fovy = 45.0F;
        camera.projection = CAMERA_PERSPECTIVE;

        while (!WindowShouldClose()) {
            const float dt = GetFrameTime();
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
            if (IsKeyPressed(KEY_V)) show_rejected = !show_rejected;
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
                    animation_seconds = 0.0F;
                    regenerate_selected(
                        pack,
                        *results,
                        selected,
                        query,
                        object_box,
                        environment,
                        search_config,
                        selected_full);
                    stale = false;
                    search_incomplete = false;
                } else {
                    search_incomplete = true;
                }
            }
            if (results.has_value() && !results->accepted.empty() && !stale) {
                bool selection_changed = false;
                if (IsKeyPressed(KEY_LEFT_BRACKET)) {
                    selected =
                        (selected + results->accepted.size() - 1U) %
                        results->accepted.size();
                    selection_changed = true;
                }
                if (IsKeyPressed(KEY_RIGHT_BRACKET) ||
                    IsKeyPressed(KEY_SLASH)) {
                    selected = (selected + 1U) % results->accepted.size();
                    selection_changed = true;
                }
                if (selection_changed) {
                    animation_seconds = 0.0F;
                    regenerate_selected(
                        pack,
                        *results,
                        selected,
                        query,
                        object_box,
                        environment,
                        search_config,
                        selected_full);
                }
            }
            if (selected_full.has_value()) animation_seconds += dt;

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
            DrawCubeV(
                ray(object.position),
                ray(options.object_size),
                Color{245, 185, 45, 180});
            draw_oriented_box(object_box, GOLD);
            DrawSphere(ray(query.target.position), 0.028F, GOLD);
            draw_axes(query.target, 0.16F, 255);

            size_t selected_evaluation = 0U;
            bool has_selected_evaluation = false;
            if (results.has_value() && !results->accepted.empty()) {
                selected_evaluation = results->accepted[selected];
                has_selected_evaluation = true;
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
                    draw_path(
                        compact.hand_path,
                        accepted
                            ? trajectory_color(
                                  pack, evaluation.candidate.clip, true)
                            : Color{210, 45, 55, 60},
                        accepted ? 5U : 10U);
                }
            }
            if (selected_full.has_value() &&
                !selected_full->poses.empty()) {
                const reach::Evaluation& evaluation = *selected_full;
                const reach::Hand selected_hand =
                    evaluation_hand(pack, evaluation);
                const Color selected_color =
                    evaluation.rejection == reach::Rejection::None ? LIME : ORANGE;
                draw_path(
                    active_wrist_path(evaluation, selected_hand),
                    selected_color,
                    1U,
                    true);
                const float fps =
                    static_cast<float>(pack.database.fps_numerator) /
                    static_cast<float>(pack.database.fps_denominator);
                const size_t sample = static_cast<size_t>(
                    animation_seconds * fps) % evaluation.poses.size();
                draw_pose(
                    evaluation.poses[sample], DARKBLUE, SKYBLUE);
                const interaction::WorldPose final = interaction::world_pose(
                    evaluation.poses.back());
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
                stale,
                search_incomplete,
                use_coverage_environment,
                show_rejected);
            EndDrawing();
        }
        CloseWindow();
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "g1 reach coverage viewer: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
