#include "interaction_hand_trajectories.h"
#include "interaction_pose.h"
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
#include <vector>

namespace {

using interaction::HandTrajectory;
using interaction::HandTrajectoryQuery;
using interaction::MappedHandTrajectory;
using interaction::OrientedBox;
using interaction::ShelfGeometry;
using interaction::TrajectoryFeasibilityReason;

struct RenderedTrajectory {
    MappedHandTrajectory mapped;
    TrajectoryFeasibilityReason reason = TrajectoryFeasibilityReason::None;
    size_t contact_point = 0U;
    float cost = 0.0F;
};

Vector3 ray_vector(vec3 value) {
    return Vector3{value.x, value.y, value.z};
}

vec3 read_vec3(const std::vector<float>& values, size_t index) {
    const size_t offset = index * 3U;
    return vec3(values.at(offset), values.at(offset + 1U), values.at(offset + 2U));
}

quat read_quat(const std::vector<float>& values, size_t index) {
    const size_t offset = index * 4U;
    return quat_normalize(quat(
        values.at(offset), values.at(offset + 1U),
        values.at(offset + 2U), values.at(offset + 3U)));
}

struct CanonicalGrasp {
    size_t clip = 0U;
    int32_t contact_frame = -1;
    interaction::Hand hand = interaction::Hand::Right;
    vec3 dimensions{};
    interaction::Transform grasp_in_object{};
    quat source_object_rotation{};
};

CanonicalGrasp find_canonical_grasp(const interaction::Database& database) {
    constexpr uint8_t contact_phase = 2U;
    for (size_t clip = 0U; clip < database.clip_count; ++clip) {
        const int32_t start = database.range_starts.at(clip);
        const int32_t stop = database.range_stops.at(clip);
        for (int32_t frame = start; frame < stop; ++frame) {
            if (database.phases.at(static_cast<size_t>(frame)) != contact_phase) {
                continue;
            }
            const int32_t object_frame = std::max(start, frame - 1);
            return {
                clip,
                frame,
                static_cast<interaction::Hand>(database.active_hands.at(clip)),
                read_vec3(database.object_dimensions, clip),
                {
                    read_vec3(database.grasp_positions_object, clip),
                    read_quat(database.grasp_rotations_object, clip),
                },
                read_quat(database.object_rotations, static_cast<size_t>(object_frame)),
            };
        }
    }
    throw std::runtime_error("interaction database has no Contact clip");
}

float vertical_half_extent(vec3 dimensions, quat rotation) {
    const vec3 x_axis = quat_mul_vec3(rotation, vec3(1.0F, 0.0F, 0.0F));
    const vec3 y_axis = quat_mul_vec3(rotation, vec3(0.0F, 1.0F, 0.0F));
    const vec3 z_axis = quat_mul_vec3(rotation, vec3(0.0F, 0.0F, 1.0F));
    return 0.5F * (
        std::abs(x_axis.y) * dimensions.x +
        std::abs(y_axis.y) * dimensions.y +
        std::abs(z_axis.y) * dimensions.z);
}

ShelfGeometry make_shelf() {
    ShelfGeometry shelf{};
    shelf.boxes[0] = {{vec3(0.0F, 0.025F, 0.0F), quat()},
                      vec3(1.40F, 0.05F, 0.80F)};
    shelf.boxes[1] = {{vec3(0.0F, 0.70F, 0.375F), quat()},
                      vec3(1.40F, 1.40F, 0.05F)};
    shelf.boxes[2] = {{vec3(-0.675F, 0.70F, 0.0F), quat()},
                      vec3(0.05F, 1.40F, 0.80F)};
    shelf.boxes[3] = {{vec3(0.675F, 0.70F, 0.0F), quat()},
                      vec3(0.05F, 1.40F, 0.80F)};
    shelf.boxes[4] = {{vec3(0.0F, 1.375F, 0.0F), quat()},
                      vec3(1.40F, 0.05F, 0.80F)};
    return shelf;
}

HandTrajectoryQuery make_query(
    const interaction::Transform& object_world,
    const CanonicalGrasp& canonical,
    bool position_only) {
    const interaction::Transform grasp_world = interaction::compose(
        object_world, canonical.grasp_in_object);
    HandTrajectoryQuery query{};
    query.object_world = object_world;
    query.object_dimensions = canonical.dimensions;
    query.hand = canonical.hand;
    query.grasp_world_position = grasp_world.position;
    query.grasp_world_rotation = position_only
        ? std::nullopt
        : std::optional<quat>(grasp_world.rotation);
    return query;
}

std::vector<RenderedTrajectory> classify_trajectories(
    const std::vector<HandTrajectory>& candidates,
    const HandTrajectoryQuery& query,
    const ShelfGeometry& shelf) {
    const OrientedBox object{query.object_world, query.object_dimensions};
    std::vector<RenderedTrajectory> rendered;
    rendered.reserve(candidates.size());
    for (const HandTrajectory& candidate : candidates) {
        MappedHandTrajectory mapped = interaction::map_hand_trajectory(
            candidate, query);
        const auto feasibility = interaction::evaluate_trajectory_feasibility(
            mapped, candidate.contact_point, object, shelf);
        rendered.push_back({
            std::move(mapped), feasibility.reason,
            candidate.contact_point, candidate.cost});
    }
    return rendered;
}

void draw_oriented_box(const OrientedBox& box, Color color) {
    const vec3 half = box.dimensions * 0.5F;
    const std::array<vec3, 8> local = {
        vec3(-half.x, -half.y, -half.z), vec3(half.x, -half.y, -half.z),
        vec3(half.x, half.y, -half.z), vec3(-half.x, half.y, -half.z),
        vec3(-half.x, -half.y, half.z), vec3(half.x, -half.y, half.z),
        vec3(half.x, half.y, half.z), vec3(-half.x, half.y, half.z),
    };
    std::array<vec3, 8> world{};
    for (size_t corner = 0U; corner < local.size(); ++corner) {
        world[corner] = interaction::compose(
            box.world, interaction::Transform{local[corner], quat()}).position;
    }
    constexpr std::array<std::array<size_t, 2>, 12> edges = {{
        {{0U, 1U}}, {{1U, 2U}}, {{2U, 3U}}, {{3U, 0U}},
        {{4U, 5U}}, {{5U, 6U}}, {{6U, 7U}}, {{7U, 4U}},
        {{0U, 4U}}, {{1U, 5U}}, {{2U, 6U}}, {{3U, 7U}},
    }};
    for (const auto& edge : edges) {
        DrawLine3D(ray_vector(world[edge[0]]), ray_vector(world[edge[1]]), color);
    }
}

void draw_path(const RenderedTrajectory& trajectory, Color color) {
    for (size_t sample = 1U; sample < trajectory.mapped.hands.size(); ++sample) {
        DrawLine3D(
            ray_vector(trajectory.mapped.hands[sample - 1U].position),
            ray_vector(trajectory.mapped.hands[sample].position), color);
    }
    DrawSphere(
        ray_vector(trajectory.mapped.hands[trajectory.contact_point].position),
        0.018F, color);
}

void apply_rotation(
    interaction::Transform& object_world,
    float angle,
    vec3 axis) {
    object_world.rotation = quat_normalize(quat_mul(
        quat_from_angle_axis(angle, axis), object_world.rotation));
}

std::filesystem::path pack_path(int argc, char** argv) {
    if (argc > 2) throw std::invalid_argument("usage: hand_trajectory_viewer [pack]");
    if (argc == 2) return argv[1];
    const char* configured = std::getenv("MM_INTERACTION_PACK");
    return configured != nullptr
        ? std::filesystem::path(configured)
        : std::filesystem::path("build/smart-pickup/full-pack");
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::filesystem::path pack = pack_path(argc, argv);
        const interaction::Database database = interaction::load_database(
            pack / "interaction_database.bin");
        const CanonicalGrasp canonical = find_canonical_grasp(database);
        const ShelfGeometry shelf = make_shelf();

        interaction::Transform initial_object{};
        initial_object.rotation = canonical.source_object_rotation;
        initial_object.position = vec3(
            0.0F,
            0.05F + vertical_half_extent(
                canonical.dimensions, initial_object.rotation),
            0.0F);
        interaction::Transform object_world = initial_object;
        bool position_only = false;
        bool show_rejected = true;
        HandTrajectoryQuery query = make_query(
            object_world, canonical, position_only);
        std::vector<HandTrajectory> candidates =
            interaction::select_hand_trajectories(database, query);
        std::vector<RenderedTrajectory> rendered = classify_trajectories(
            candidates, query, shelf);

        SetConfigFlags(FLAG_VSYNC_HINT | FLAG_MSAA_4X_HINT);
        InitWindow(1280, 800, "Generic grasp trajectory shelf lab");
        if (!IsWindowReady()) {
            throw std::runtime_error("trajectory viewer could not open a window");
        }
        SetTargetFPS(60);
        Camera3D camera{};
        camera.position = Vector3{2.2F, 1.65F, -2.4F};
        camera.target = Vector3{0.0F, 0.65F, 0.0F};
        camera.up = Vector3{0.0F, 1.0F, 0.0F};
        camera.fovy = 45.0F;
        camera.projection = CAMERA_PERSPECTIVE;

        while (!WindowShouldClose()) {
            const float dt = std::min(GetFrameTime(), 0.05F);
            const float translation_step = 0.45F * dt;
            const float rotation_step = 0.9F * dt;
            bool object_changed = false;
            bool selection_changed = false;
            if (IsKeyDown(KEY_LEFT)) {
                object_world.position.x -= translation_step;
                object_changed = true;
            }
            if (IsKeyDown(KEY_RIGHT)) {
                object_world.position.x += translation_step;
                object_changed = true;
            }
            if (IsKeyDown(KEY_UP)) {
                object_world.position.z += translation_step;
                object_changed = true;
            }
            if (IsKeyDown(KEY_DOWN)) {
                object_world.position.z -= translation_step;
                object_changed = true;
            }
            if (IsKeyDown(KEY_PAGE_UP)) {
                object_world.position.y += translation_step;
                object_changed = true;
            }
            if (IsKeyDown(KEY_PAGE_DOWN)) {
                object_world.position.y -= translation_step;
                object_changed = true;
            }
            if (IsKeyDown(KEY_Q)) {
                apply_rotation(object_world, rotation_step, vec3(0.0F, 1.0F, 0.0F));
                object_changed = true;
            }
            if (IsKeyDown(KEY_E)) {
                apply_rotation(object_world, -rotation_step, vec3(0.0F, 1.0F, 0.0F));
                object_changed = true;
            }
            if (IsKeyDown(KEY_R)) {
                apply_rotation(object_world, rotation_step, vec3(1.0F, 0.0F, 0.0F));
                object_changed = true;
            }
            if (IsKeyDown(KEY_F)) {
                apply_rotation(object_world, -rotation_step, vec3(1.0F, 0.0F, 0.0F));
                object_changed = true;
            }
            if (IsKeyDown(KEY_Z)) {
                apply_rotation(object_world, rotation_step, vec3(0.0F, 0.0F, 1.0F));
                object_changed = true;
            }
            if (IsKeyDown(KEY_C)) {
                apply_rotation(object_world, -rotation_step, vec3(0.0F, 0.0F, 1.0F));
                object_changed = true;
            }
            if (IsKeyPressed(KEY_BACKSPACE)) {
                object_world = initial_object;
                object_changed = true;
            }
            if (IsKeyPressed(KEY_V)) show_rejected = !show_rejected;
            if (IsKeyPressed(KEY_P)) {
                position_only = !position_only;
                selection_changed = true;
                object_changed = true;
            }
            if (object_changed) {
                query = make_query(object_world, canonical, position_only);
                if (selection_changed) {
                    candidates = interaction::select_hand_trajectories(
                        database, query);
                }
                rendered = classify_trajectories(candidates, query, shelf);
            }

            size_t accepted = 0U;
            size_t object_rejected = 0U;
            size_t shelf_rejected = 0U;
            for (const RenderedTrajectory& trajectory : rendered) {
                if (trajectory.reason == TrajectoryFeasibilityReason::None) {
                    ++accepted;
                } else if (trajectory.reason ==
                           TrajectoryFeasibilityReason::ObjectCollision) {
                    ++object_rejected;
                } else {
                    ++shelf_rejected;
                }
            }

            BeginDrawing();
            ClearBackground(Color{238, 241, 245, 255});
            BeginMode3D(camera);
            DrawGrid(20, 0.25F);
            for (const OrientedBox& box : shelf.boxes) {
                DrawCubeV(
                    ray_vector(box.world.position), ray_vector(box.dimensions),
                    Color{150, 155, 164, 105});
                DrawCubeWiresV(
                    ray_vector(box.world.position), ray_vector(box.dimensions),
                    Color{75, 80, 90, 255});
            }
            draw_oriented_box(
                {object_world, canonical.dimensions},
                Color{255, 177, 35, 255});
            DrawSphere(ray_vector(query.grasp_world_position), 0.028F, GOLD);
            for (size_t index = 0U; index < rendered.size(); ++index) {
                const RenderedTrajectory& trajectory = rendered[index];
                if (trajectory.reason != TrajectoryFeasibilityReason::None) {
                    if (show_rejected) {
                        draw_path(trajectory, Color{210, 45, 55, 75});
                    }
                    continue;
                }
                const float fraction = accepted <= 1U
                    ? 0.0F
                    : static_cast<float>(index) /
                      static_cast<float>(rendered.size() - 1U);
                draw_path(
                    trajectory,
                    ColorFromHSV(125.0F + 95.0F * fraction, 0.78F, 0.86F));
            }
            EndMode3D();

            DrawRectangle(14, 14, 570, 132, Color{255, 255, 255, 225});
            DrawText("Generic grasp trajectory field", 26, 24, 24, DARKGRAY);
            DrawText(
                TextFormat(
                    "compatible %i | accepted %i | object-rejected %i | shelf-rejected %i",
                    static_cast<int>(candidates.size()),
                    static_cast<int>(accepted),
                    static_cast<int>(object_rejected),
                    static_cast<int>(shelf_rejected)),
                26, 56, 18, DARKGRAY);
            DrawText(
                "Arrows: X/Z  PgUp/PgDn: height  Q/E: yaw  R/F: pitch  Z/C: roll",
                26, 82, 16, DARKGRAY);
            DrawText(
                TextFormat(
                    "V: rejected paths  P: %s  Backspace: reset",
                    position_only ? "position-only grasp" : "full grasp pose"),
                26, 108, 16, DARKGRAY);
            EndDrawing();
        }
        CloseWindow();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "hand trajectory viewer: " << error.what() << '\n';
        return 2;
    }
}
