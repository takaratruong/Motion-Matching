#include "g1_skeleton.h"
#include "reach_coverage.h"
#include "raylib.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

constexpr float kPi = 3.14159265358979323846F;
constexpr const char* kCapturedLabel = "CAPTURED LEFT";
constexpr const char* kMirroredLabel = "MIRRORED RIGHT";

struct Options {
    std::filesystem::path pack;
    vec3 object_size{0.10F, 0.10F, 0.10F};
};

struct SearchResults {
    std::vector<reach::Evaluation> evaluations;
    std::vector<size_t> accepted;
    reach::Diagnostics diagnostics{};
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
            throw std::invalid_argument("unknown or incomplete argument: " + value);
        }
    }
    if (!(options.object_size.x > 0.0F && options.object_size.y > 0.0F &&
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

interaction::Transform desired_grasp(
    const interaction::Transform& object,
    vec3 dimensions) {
    const interaction::Transform local{
        vec3(0.5F * dimensions.x, 0.0F, 0.0F),
        quat_from_angle_axis(kPi, vec3(0.0F, 1.0F, 0.0F)),
    };
    return interaction::compose(object, local);
}

reach::Query make_query(
    const interaction::Transform& object,
    vec3 dimensions,
    reach::Hand hand) {
    reach::Query query{};
    query.hand = hand;
    query.target = desired_grasp(object, dimensions);
    query.approach_world = normalize(quat_mul_vec3(
        object.rotation, vec3(-1.0F, 0.0F, 0.0F)));
    return query;
}

interaction::Transform reset_object(
    const reach::Pack& pack,
    reach::Hand hand,
    vec3 dimensions) {
    for (size_t clip = 0U; clip < pack.database.clip_count; ++clip) {
        if (pack.database.active_hands.at(clip) !=
            static_cast<uint8_t>(hand)) {
            continue;
        }
        const interaction::Transform endpoint = reach::endpoint_transform(
            pack.database, clip);
        interaction::Transform object{};
        object.rotation = quat_mul(
            endpoint.rotation,
            quat_from_angle_axis(-kPi, vec3(0.0F, 1.0F, 0.0F)));
        object.position = endpoint.position - quat_mul_vec3(
            object.rotation, vec3(0.5F * dimensions.x, 0.0F, 0.0F));
        return object;
    }
    return {{0.65F, 0.85F, 0.0F}, quat()};
}

SearchResults rerun_search(
    const reach::Pack& pack,
    const reach::Query& query,
    const interaction::OrientedBox& object,
    const interaction::EnvironmentGeometry& environment) {
    SearchResults results{};
    const std::vector<reach::Candidate> candidates =
        reach::select_candidates(pack, query);
    std::unordered_set<size_t> selected;
    selected.reserve(candidates.size());
    results.evaluations.reserve(pack.database.clip_count);
    for (const reach::Candidate& candidate : candidates) {
        selected.insert(candidate.clip);
        results.evaluations.push_back(reach::evaluate_candidate(
            pack, candidate, query, object, environment));
    }
    for (size_t clip = 0U; clip < pack.database.clip_count; ++clip) {
        if (pack.database.active_hands.at(clip) !=
                static_cast<uint8_t>(query.hand) ||
            selected.count(clip) != 0U) {
            continue;
        }
        reach::Evaluation outside{};
        outside.candidate.clip = clip;
        outside.rejection = reach::Rejection::OutsideEnvelope;
        results.evaluations.push_back(std::move(outside));
    }
    std::stable_sort(
        results.evaluations.begin(), results.evaluations.end(),
        [](const reach::Evaluation& left, const reach::Evaluation& right) {
            if (left.rejection != right.rejection) {
                return left.rejection == reach::Rejection::None;
            }
            if (left.candidate.cost != right.candidate.cost) {
                return left.candidate.cost < right.candidate.cost;
            }
            return left.candidate.clip < right.candidate.clip;
        });
    for (size_t index = 0U; index < results.evaluations.size(); ++index) {
        if (results.evaluations[index].rejection == reach::Rejection::None) {
            results.accepted.push_back(index);
        }
    }
    results.diagnostics = reach::summarize_evaluations(
        pack, results.evaluations);
    return results;
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

void draw_pose(const interaction::Pose& pose, Color color) {
    const interaction::WorldPose world = interaction::world_pose(pose);
    for (size_t bone = 1U; bone < g1_skeleton::BoneCount; ++bone) {
        const int32_t parent = g1_skeleton::kParents[bone];
        if (parent < 0) continue;
        const Vector3 start = ray(world.positions[static_cast<size_t>(parent)]);
        const Vector3 stop = ray(world.positions[bone]);
        DrawCylinderEx(start, stop, 0.012F, 0.012F, 7, color);
        DrawSphere(stop, 0.018F, color);
    }
}

void draw_path(
    const reach::Evaluation& evaluation,
    reach::Hand hand,
    Color color) {
    if (evaluation.poses.size() < 2U) return;
    const size_t wrist = wrist_bone(hand);
    vec3 previous = interaction::world_pose(
        evaluation.poses.front()).positions[wrist];
    for (size_t sample = 1U; sample < evaluation.poses.size(); ++sample) {
        const vec3 current = interaction::world_pose(
            evaluation.poses[sample]).positions[wrist];
        DrawLine3D(ray(previous), ray(current), color);
        previous = current;
    }
}

const char* provenance(const reach::Pack& pack, size_t clip) {
    return pack.database.augmentations.at(clip) ==
        static_cast<uint8_t>(reach::Augmentation::Captured)
        ? kCapturedLabel
        : kMirroredLabel;
}

void draw_hud(
    const reach::Pack& pack,
    const SearchResults& results,
    size_t selected,
    bool stale,
    bool coverage_environment,
    bool show_rejected,
    reach::Hand hand) {
    DrawRectangle(12, 12, 650, 252, Fade(BLACK, 0.76F));
    DrawText(
        "Arrow X/Z  W/S Y | Q/E yaw R/F pitch Z/C roll | Enter search",
        22, 20, 17, RAYWHITE);
    DrawText(
        "M hand | G open/coverage | [ previous, ] or / next | V rejected | Backspace reset",
        22, 43, 16, LIGHTGRAY);
    DrawText(
        TextFormat("HAND %s | %s ENVIRONMENT | rejected paths %s",
            hand == reach::Hand::Left ? "LEFT" : "RIGHT",
            coverage_environment ? "COVERAGE" : "OPEN",
            show_rejected ? "ON" : "OFF"),
        22, 69, 18, YELLOW);
    DrawText(
        TextFormat("accepted %zu / evaluated %zu | joint-limit diagnostic %zu",
            results.diagnostics.accepted, results.diagnostics.evaluations,
            results.diagnostics.joint_limit_saturated),
        22, 94, 18, RAYWHITE);
    static constexpr std::array<const char*, 8U> labels = {
        "ACCEPTED", "OUTSIDE ENVELOPE", "INVALID SOLVER", "POSITION ERROR",
        "APPROACH AXIS", "FULL ORIENTATION", "OBJECT COLLISION",
        "ENVIRONMENT COLLISION"};
    for (size_t reason = 0U; reason < labels.size(); ++reason) {
        const int column = reason % 2U == 0U ? 22 : 330;
        const int row = 121 + static_cast<int>(reason / 2U) * 21;
        DrawText(
            TextFormat("%s: %zu", labels[reason],
                results.diagnostics.rejection_counts[reason]),
            column, row, 16, reason == 0U ? LIME : LIGHTGRAY);
    }
    if (!results.accepted.empty()) {
        const reach::Evaluation& evaluation =
            results.evaluations[results.accepted[selected]];
        const size_t clip = evaluation.candidate.clip;
        DrawText(
            TextFormat("option %zu/%zu | %s | %s",
                selected + 1U, results.accepted.size(), provenance(pack, clip),
                pack.database.source_names.at(
                    pack.database.source_indices.at(clip)).c_str()),
            22, 210, 17, trajectory_color(pack, clip, true));
        DrawText(
            TextFormat("final: %.1f cm | approach %.1f deg | orientation %.1f deg",
                evaluation.position_error_m * 100.0F,
                evaluation.approach_error_radians * 180.0F / kPi,
                evaluation.orientation_error_radians * 180.0F / kPi),
            22, 233, 16, RAYWHITE);
    }
    if (stale) {
        DrawText("SEARCH STALE - press Enter", 680, 22, 24, ORANGE);
    }
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
        reach::Hand hand = reach::Hand::Left;
        interaction::Transform object = reset_object(
            pack, hand, options.object_size);
        bool use_coverage_environment = true;
        bool show_rejected = true;
        bool stale = false;
        size_t selected = 0U;
        float animation_seconds = 0.0F;
        reach::Query query = make_query(object, options.object_size, hand);
        SearchResults results = rerun_search(
            pack, query, {object, options.object_size}, coverage);

        InitWindow(1500, 900, "G1 retargeted reach coverage");
        SetTargetFPS(60);
        Camera3D camera{};
        camera.position = Vector3{2.35F, 1.65F, 2.35F};
        camera.target = Vector3{0.0F, 0.85F, 0.0F};
        camera.up = Vector3{0.0F, 1.0F, 0.0F};
        camera.fovy = 45.0F;
        camera.projection = CAMERA_PERSPECTIVE;

        while (!WindowShouldClose()) {
            const float dt = GetFrameTime();
            bool target_changed = false;
            const float move = 0.60F * dt;
            if (IsKeyDown(KEY_LEFT)) { object.position.x -= move; target_changed = true; }
            if (IsKeyDown(KEY_RIGHT)) { object.position.x += move; target_changed = true; }
            if (IsKeyDown(KEY_UP)) { object.position.z -= move; target_changed = true; }
            if (IsKeyDown(KEY_DOWN)) { object.position.z += move; target_changed = true; }
            if (IsKeyDown(KEY_W)) { object.position.y += move; target_changed = true; }
            if (IsKeyDown(KEY_S)) { object.position.y -= move; target_changed = true; }
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
            if (IsKeyPressed(KEY_M)) {
                hand = hand == reach::Hand::Left
                    ? reach::Hand::Right : reach::Hand::Left;
                object = reset_object(pack, hand, options.object_size);
                target_changed = true;
            }
            if (IsKeyPressed(KEY_G)) {
                use_coverage_environment = !use_coverage_environment;
                target_changed = true;
            }
            if (IsKeyPressed(KEY_V)) show_rejected = !show_rejected;
            if (IsKeyPressed(KEY_BACKSPACE)) {
                object = reset_object(pack, hand, options.object_size);
                target_changed = true;
            }
            if (target_changed) stale = true;
            query = make_query(object, options.object_size, hand);
            if (IsKeyPressed(KEY_ENTER)) {
                results = rerun_search(
                    pack, query, {object, options.object_size},
                    use_coverage_environment ? coverage : open);
                selected = 0U;
                animation_seconds = 0.0F;
                stale = false;
            }
            if (!results.accepted.empty()) {
                if (IsKeyPressed(KEY_LEFT_BRACKET)) {
                    selected = (selected + results.accepted.size() - 1U) %
                        results.accepted.size();
                    animation_seconds = 0.0F;
                }
                if (IsKeyPressed(KEY_RIGHT_BRACKET) ||
                    IsKeyPressed(KEY_SLASH)) {
                    selected = (selected + 1U) % results.accepted.size();
                    animation_seconds = 0.0F;
                }
                animation_seconds += dt;
            }

            BeginDrawing();
            ClearBackground(Color{24, 27, 35, 255});
            BeginMode3D(camera);
            DrawGrid(20, 0.20F);
            for (const interaction::OrientedBox& box :
                 (use_coverage_environment ? coverage.boxes : open.boxes)) {
                draw_oriented_box(box, Fade(GRAY, 0.72F));
            }
            draw_oriented_box({object, options.object_size}, GOLD);
            draw_axes(query.target, 0.16F, 255);
            for (const reach::Evaluation& evaluation : results.evaluations) {
                if (evaluation.rejection != reach::Rejection::None &&
                    !show_rejected) {
                    continue;
                }
                draw_path(
                    evaluation, hand,
                    trajectory_color(
                        pack, evaluation.candidate.clip,
                        evaluation.rejection == reach::Rejection::None));
            }
            if (!results.accepted.empty()) {
                const reach::Evaluation& evaluation =
                    results.evaluations[results.accepted[selected]];
                const float fps = static_cast<float>(pack.database.fps_numerator) /
                    static_cast<float>(pack.database.fps_denominator);
                const size_t sample = static_cast<size_t>(
                    animation_seconds * fps) % evaluation.poses.size();
                draw_pose(evaluation.poses[sample], RAYWHITE);
                const interaction::WorldPose final = interaction::world_pose(
                    evaluation.poses.back());
                const size_t wrist = wrist_bone(hand);
                draw_axes({final.positions[wrist], final.rotations[wrist]},
                    0.13F, 150);
            }
            EndMode3D();
            draw_hud(
                pack, results, selected, stale, use_coverage_environment,
                show_rejected, hand);
            EndDrawing();
        }
        CloseWindow();
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "g1 reach coverage viewer: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
