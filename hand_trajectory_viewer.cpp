#include "interaction_hand_trajectories.h"
#include "interaction_pose.h"
#include "interaction_trajectory_database.h"
#include "raylib.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

using interaction::HandTrajectory;
using interaction::HandTrajectoryQuery;
using interaction::OrientedBox;
using interaction::EnvironmentGeometry;
using interaction::ShapedHandTrajectory;
using interaction::TrajectoryCollisionConfig;
using interaction::TrajectoryFeasibilityReason;

constexpr size_t kBackgroundPathStride = 5U;
constexpr size_t kTargetValidTrajectories = 12U;

struct RenderedTrajectory {
    HandTrajectory source;
    ShapedHandTrajectory shaped;
    TrajectoryFeasibilityReason reason = TrajectoryFeasibilityReason::None;
};

struct TrajectorySet {
    std::vector<RenderedTrajectory> valid;
    std::vector<RenderedTrajectory> rejected;
    size_t exact_compatible = 0U;
    size_t fallback_compatible = 0U;
    size_t exact_valid = 0U;
    size_t fallback_valid = 0U;
    size_t ik_rejected = 0U;
    size_t object_rejected = 0U;
    size_t environment_rejected = 0U;
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
    vec3 approach_in_object{1.0F, 0.0F, 0.0F};
    interaction::Transform source_object{};
    interaction::Transform table_world{};
    vec3 table_dimensions{};
};

EnvironmentGeometry support_geometry(
    interaction::SupportKind support,
    const interaction::Transform& table_world,
    vec3 table_dimensions) {
    if (support == interaction::SupportKind::Ground) {
        return {};
    }
    return interaction::make_coverage_environment(
        table_world, table_dimensions);
}

TrajectoryCollisionConfig viewer_collision_config() {
    TrajectoryCollisionConfig config{};
    config.joint_radius_m += 0.02F;
    config.limb_radius_m += 0.02F;
    config.torso_radius_m += 0.02F;
    return config;
}

CanonicalGrasp canonical_grasp(
    const interaction::Database& database,
    size_t clip,
    int32_t contact_frame) {
    const int32_t start = database.range_starts.at(clip);
    const int32_t object_frame = std::max(start, contact_frame - 1);
    return {
        clip,
        contact_frame,
        static_cast<interaction::Hand>(database.active_hands.at(clip)),
        read_vec3(database.object_dimensions, clip),
        {
            read_vec3(database.grasp_positions_object, clip),
            read_quat(database.grasp_rotations_object, clip),
        },
        read_vec3(database.approach_directions_object, clip),
        {
            read_vec3(
                database.object_positions,
                static_cast<size_t>(object_frame)),
            read_quat(
                database.object_rotations,
                static_cast<size_t>(object_frame)),
        },
        {
            read_vec3(database.table_positions, clip),
            read_quat(database.table_rotations, clip),
        },
        read_vec3(database.table_sizes, clip),
    };
}

CanonicalGrasp find_canonical_grasp(const interaction::Database& database) {
    constexpr uint8_t contact_phase = 2U;
    std::optional<CanonicalGrasp> fallback;
    for (size_t clip = 0U; clip < database.clip_count; ++clip) {
        const int32_t start = database.range_starts.at(clip);
        const int32_t stop = database.range_stops.at(clip);
        for (int32_t frame = start; frame < stop; ++frame) {
            if (database.phases.at(static_cast<size_t>(frame)) != contact_phase) {
                continue;
            }
            const CanonicalGrasp candidate =
                canonical_grasp(database, clip, frame);
            if (!fallback.has_value()) fallback = candidate;
            if (interaction::support_kind(database, clip) ==
                interaction::SupportKind::Table) {
                return candidate;
            }
            break;
        }
    }
    if (fallback.has_value()) return *fallback;
    throw std::runtime_error("interaction database has no Contact clip");
}

HandTrajectoryQuery make_query(
    const interaction::Transform& object_world,
    const CanonicalGrasp& canonical,
    interaction::GraspOrientationMode orientation_mode) {
    const interaction::Transform grasp_world = interaction::compose(
        object_world, canonical.grasp_in_object);
    HandTrajectoryQuery query{};
    query.object_world = object_world;
    query.object_dimensions = canonical.dimensions;
    query.hand = canonical.hand;
    query.grasp_world_position = grasp_world.position;
    query.grasp_world_rotation = grasp_world.rotation;
    query.approach_world_direction = normalize(quat_mul_vec3(
        object_world.rotation, canonical.approach_in_object));
    query.orientation_mode = orientation_mode;
    return query;
}

void process_candidates(
    const interaction::Database& database,
    const std::vector<HandTrajectory>& candidates,
    const HandTrajectoryQuery& query,
    const EnvironmentGeometry& environment,
    const TrajectoryCollisionConfig& collision_config,
    std::unordered_set<int32_t>& accepted_clips,
    size_t stop_after_valid,
    TrajectorySet& result) {
    const OrientedBox object{query.object_world, query.object_dimensions};
    for (const HandTrajectory& candidate : candidates) {
        if (result.valid.size() >= stop_after_valid) break;
        if (accepted_clips.count(candidate.clip) != 0U) continue;
        ShapedHandTrajectory shaped = interaction::shape_hand_trajectory(
            database, candidate, query);
        if (!shaped.contact_accepted) {
            ++result.ik_rejected;
            std::vector<interaction::Pose>{}.swap(shaped.poses);
            result.rejected.push_back({
                candidate, std::move(shaped),
                TrajectoryFeasibilityReason::None});
            continue;
        }
        const auto feasibility =
            interaction::evaluate_shaped_trajectory_feasibility(
                shaped, candidate.contact_point, query.hand, object,
                environment, collision_config);
        std::vector<interaction::Pose>{}.swap(shaped.poses);
        RenderedTrajectory rendered{
            candidate, std::move(shaped), feasibility.reason};
        if (feasibility.reason == TrajectoryFeasibilityReason::None) {
            accepted_clips.insert(candidate.clip);
            if (candidate.match_tier ==
                interaction::TrajectoryMatchTier::AxisFallback) {
                ++result.fallback_valid;
            } else {
                ++result.exact_valid;
            }
            result.valid.push_back(std::move(rendered));
        } else {
            if (feasibility.reason ==
                TrajectoryFeasibilityReason::ObjectCollision) {
                ++result.object_rejected;
            } else {
                ++result.environment_rejected;
            }
            result.rejected.push_back(std::move(rendered));
        }
    }
}

TrajectorySet rebuild_valid_trajectories(
    const interaction::Database& database,
    const HandTrajectoryQuery& base_query,
    const EnvironmentGeometry& environment) {
    TrajectorySet result{};
    const TrajectoryCollisionConfig collision_config =
        viewer_collision_config();
    std::unordered_set<int32_t> accepted_clips;
    HandTrajectoryQuery exact_query = base_query;
    const bool position_only = base_query.orientation_mode ==
        interaction::GraspOrientationMode::PositionOnly;
    exact_query.orientation_mode = position_only
        ? interaction::GraspOrientationMode::PositionOnly
        : interaction::GraspOrientationMode::ExactPose;
    const std::vector<HandTrajectory> exact =
        interaction::select_hand_trajectories(database, exact_query);
    result.exact_compatible = exact.size();
    result.valid.reserve(exact.size());
    result.rejected.reserve(exact.size());
    process_candidates(
        database,
        exact,
        exact_query,
        environment,
        collision_config,
        accepted_clips,
        std::numeric_limits<size_t>::max(),
        result);

    if (!position_only && result.valid.size() < kTargetValidTrajectories) {
        HandTrajectoryQuery fallback_query = base_query;
        fallback_query.orientation_mode =
            interaction::GraspOrientationMode::ApproachAxis;
        const std::vector<HandTrajectory> fallback =
            interaction::select_hand_trajectories(database, fallback_query);
        result.fallback_compatible = fallback.size();
        process_candidates(
            database,
            fallback,
            fallback_query,
            environment,
            collision_config,
            accepted_clips,
            kTargetValidTrajectories,
            result);
    }
    std::stable_sort(
        result.valid.begin(),
        result.valid.end(),
        [](const RenderedTrajectory& left, const RenderedTrajectory& right) {
            const bool left_fallback = left.source.match_tier ==
                interaction::TrajectoryMatchTier::AxisFallback;
            const bool right_fallback = right.source.match_tier ==
                interaction::TrajectoryMatchTier::AxisFallback;
            if (left_fallback != right_fallback) return !left_fallback;
            if (!left_fallback) return false;
            if (left.shaped.achieved_orientation_error_radians !=
                right.shaped.achieved_orientation_error_radians) {
                return left.shaped.achieved_orientation_error_radians <
                    right.shaped.achieved_orientation_error_radians;
            }
            if (left.source.cost != right.source.cost) {
                return left.source.cost < right.source.cost;
            }
            return left.source.clip < right.source.clip;
        });
    return result;
}

ShapedHandTrajectory shape_selected_animation(
    const interaction::Database& database,
    const TrajectorySet& trajectories,
    size_t selected_index,
    const HandTrajectoryQuery& searched_query) {
    if (trajectories.valid.empty()) return {};
    if (selected_index >= trajectories.valid.size()) {
        throw std::out_of_range("selected trajectory index is invalid");
    }
    HandTrajectoryQuery selected_query = searched_query;
    const auto tier = trajectories.valid[selected_index].source.match_tier;
    selected_query.orientation_mode = tier ==
            interaction::TrajectoryMatchTier::AxisFallback
        ? interaction::GraspOrientationMode::ApproachAxis
        : (tier == interaction::TrajectoryMatchTier::PositionOnly
               ? interaction::GraspOrientationMode::PositionOnly
               : interaction::GraspOrientationMode::ExactPose);
    ShapedHandTrajectory shaped = interaction::shape_hand_trajectory(
        database, trajectories.valid[selected_index].source, selected_query);
    if (!shaped.contact_accepted) {
        throw std::runtime_error("validated trajectory no longer passes IK");
    }
    return shaped;
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

void draw_path(
    const RenderedTrajectory& trajectory,
    Color color,
    size_t sample_stride = 1U,
    bool emphasized = false) {
    const auto& hands = trajectory.shaped.path.hands;
    size_t previous = 0U;
    for (size_t sample = sample_stride;
         sample < hands.size();
         sample += sample_stride) {
        const Vector3 start = ray_vector(
            hands[previous].position);
        const Vector3 stop = ray_vector(
            hands[sample].position);
        if (emphasized) {
            DrawCylinderEx(start, stop, 0.009F, 0.009F, 6, color);
            DrawSphere(stop, 0.014F, color);
        } else {
            DrawLine3D(start, stop, color);
        }
        previous = sample;
    }
    if (!hands.empty() && previous != hands.size() - 1U) {
        DrawLine3D(
            ray_vector(hands[previous].position),
            ray_vector(hands.back().position),
            color);
    }
    if (!hands.empty()) {
        DrawSphere(
            ray_vector(hands[trajectory.source.contact_point].position),
            0.018F, color);
    }
}

const char* phase_name(uint8_t phase) {
    switch (phase) {
        case 0U: return "APPROACH";
        case 1U: return "REACH";
        case 2U: return "CONTACT";
        case 3U: return "LIFT";
        default: return "TRANSITION";
    }
}

const char* support_name(interaction::SupportKind support) {
    return support == interaction::SupportKind::Ground
        ? "GROUND" : "TABLE";
}

const char* match_tier_name(interaction::TrajectoryMatchTier tier) {
    switch (tier) {
        case interaction::TrajectoryMatchTier::Exact: return "EXACT";
        case interaction::TrajectoryMatchTier::AxisFallback:
            return "AXIS-FALLBACK";
        case interaction::TrajectoryMatchTier::PositionOnly:
            return "POSITION-ONLY";
    }
    return "UNKNOWN";
}

size_t animated_sample(
    const ShapedHandTrajectory& animation,
    float animation_seconds) {
    if (animation.poses.empty()) {
        throw std::invalid_argument("selected trajectory has no shaped poses");
    }
    return static_cast<size_t>(std::floor(animation_seconds * 25.0F)) %
        animation.poses.size();
}

void draw_selected_skeleton(
    const interaction::Pose& shaped_pose,
    Color joint_color,
    Color bone_color) {
    const interaction::WorldPose shaped = interaction::world_pose(shaped_pose);
    for (size_t bone = 0U; bone < shaped.positions.size(); ++bone) {
        DrawSphereWires(
            ray_vector(shaped.positions[bone]), 0.024F, 4, 8, joint_color);
        const int32_t parent = g1_skeleton::kParents[bone];
        if (parent >= 0) {
            DrawCylinderEx(
                ray_vector(shaped.positions[static_cast<size_t>(parent)]),
                ray_vector(shaped.positions[bone]),
                0.014F, 0.014F, 6, bone_color);
        }
    }
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
        const interaction::Database database = interaction::load_trajectory_database(
            pack / "interaction_database.bin");
        const CanonicalGrasp canonical = find_canonical_grasp(database);
        const interaction::SupportKind canonical_support =
            interaction::support_kind(database, canonical.clip);
        const vec3 scene_offset(
            -canonical.table_world.position.x,
            0.0F,
            -canonical.table_world.position.z);
        interaction::Transform table_world = canonical.table_world;
        table_world.position = table_world.position + scene_offset;
        const EnvironmentGeometry environment = support_geometry(
            canonical_support, table_world, canonical.table_dimensions);

        interaction::Transform initial_object = canonical.source_object;
        initial_object.position = initial_object.position + scene_offset;
        interaction::Transform object_world = initial_object;
        bool position_only = false;
        bool show_rejected = true;
        bool search_stale = false;
        size_t selected_index = 0U;
        float animation_seconds = 0.0F;
        HandTrajectoryQuery query = make_query(
            object_world,
            canonical,
            interaction::GraspOrientationMode::ExactPose);
        TrajectorySet trajectories = rebuild_valid_trajectories(
            database, query, environment);
        HandTrajectoryQuery searched_query = query;
        ShapedHandTrajectory selected_animation = shape_selected_animation(
            database, trajectories, selected_index, searched_query);

        SetConfigFlags(FLAG_VSYNC_HINT | FLAG_MSAA_4X_HINT);
        InitWindow(1280, 800, "Generic grasp kinematic trajectory lab");
        if (!IsWindowReady()) {
            throw std::runtime_error("trajectory viewer could not open a window");
        }
        SetTargetFPS(60);
        Camera3D camera{};
        camera.position = Vector3{2.35F, 1.55F, -2.45F};
        camera.target = Vector3{0.0F, 0.70F, 0.0F};
        camera.up = Vector3{0.0F, 1.0F, 0.0F};
        camera.fovy = 45.0F;
        camera.projection = CAMERA_PERSPECTIVE;

        while (!WindowShouldClose()) {
            const float dt = std::min(GetFrameTime(), 0.05F);
            animation_seconds += dt;
            const float translation_step = 0.45F * dt;
            const float rotation_step = 0.9F * dt;
            bool grasp_changed = false;
            if (IsKeyDown(KEY_LEFT)) {
                object_world.position.x -= translation_step;
                grasp_changed = true;
            }
            if (IsKeyDown(KEY_RIGHT)) {
                object_world.position.x += translation_step;
                grasp_changed = true;
            }
            if (IsKeyDown(KEY_UP)) {
                object_world.position.z += translation_step;
                grasp_changed = true;
            }
            if (IsKeyDown(KEY_DOWN)) {
                object_world.position.z -= translation_step;
                grasp_changed = true;
            }
            if (IsKeyDown(KEY_W) || IsKeyDown(KEY_PAGE_UP)) {
                object_world.position.y += translation_step;
                grasp_changed = true;
            }
            if (IsKeyDown(KEY_S) || IsKeyDown(KEY_PAGE_DOWN)) {
                object_world.position.y -= translation_step;
                grasp_changed = true;
            }
            if (IsKeyDown(KEY_Q)) {
                apply_rotation(object_world, rotation_step, vec3(0.0F, 1.0F, 0.0F));
                grasp_changed = true;
            }
            if (IsKeyDown(KEY_E)) {
                apply_rotation(object_world, -rotation_step, vec3(0.0F, 1.0F, 0.0F));
                grasp_changed = true;
            }
            if (IsKeyDown(KEY_R)) {
                apply_rotation(object_world, rotation_step, vec3(1.0F, 0.0F, 0.0F));
                grasp_changed = true;
            }
            if (IsKeyDown(KEY_F)) {
                apply_rotation(object_world, -rotation_step, vec3(1.0F, 0.0F, 0.0F));
                grasp_changed = true;
            }
            if (IsKeyDown(KEY_Z)) {
                apply_rotation(object_world, rotation_step, vec3(0.0F, 0.0F, 1.0F));
                grasp_changed = true;
            }
            if (IsKeyDown(KEY_C)) {
                apply_rotation(object_world, -rotation_step, vec3(0.0F, 0.0F, 1.0F));
                grasp_changed = true;
            }
            if (IsKeyPressed(KEY_BACKSPACE)) {
                object_world = initial_object;
                grasp_changed = true;
            }
            if (IsKeyPressed(KEY_LEFT_BRACKET)) {
                if (!trajectories.valid.empty()) {
                    selected_index = selected_index == 0U
                        ? trajectories.valid.size() - 1U
                        : selected_index - 1U;
                    selected_animation = shape_selected_animation(
                        database, trajectories, selected_index, searched_query);
                    animation_seconds = 0.0F;
                }
            }
            if (IsKeyPressed(KEY_SLASH) ||
                IsKeyPressed(KEY_RIGHT_BRACKET)) {
                if (!trajectories.valid.empty()) {
                    selected_index =
                        (selected_index + 1U) % trajectories.valid.size();
                    selected_animation = shape_selected_animation(
                        database, trajectories, selected_index, searched_query);
                    animation_seconds = 0.0F;
                }
            }
            if (IsKeyPressed(KEY_V)) show_rejected = !show_rejected;
            if (IsKeyPressed(KEY_P)) {
                position_only = !position_only;
                grasp_changed = true;
            }
            if (grasp_changed) {
                query = make_query(
                    object_world,
                    canonical,
                    position_only
                        ? interaction::GraspOrientationMode::PositionOnly
                        : interaction::GraspOrientationMode::ExactPose);
                search_stale = true;
            }
            if (IsKeyPressed(KEY_ENTER)) {
                trajectories = rebuild_valid_trajectories(
                    database, query, environment);
                selected_index = 0U;
                searched_query = query;
                selected_animation = shape_selected_animation(
                    database, trajectories, selected_index, searched_query);
                animation_seconds = 0.0F;
                search_stale = false;
            }

            BeginDrawing();
            ClearBackground(Color{238, 241, 245, 255});
            BeginMode3D(camera);
            DrawGrid(20, 0.25F);
            for (const OrientedBox& box : environment.boxes) {
                DrawCubeV(
                    ray_vector(box.world.position),
                    ray_vector(box.dimensions),
                    Color{135, 102, 74, 155});
                draw_oriented_box(box, Color{72, 52, 39, 255});
            }
            draw_oriented_box(
                {object_world, canonical.dimensions},
                Color{255, 177, 35, 255});
            DrawSphere(ray_vector(query.grasp_world_position), 0.028F, GOLD);
            if (show_rejected) {
                for (const RenderedTrajectory& rejected :
                     trajectories.rejected) {
                    draw_path(
                        rejected, Color{210, 45, 55, 75},
                        kBackgroundPathStride);
                }
            }
            for (size_t index = 0U; index < trajectories.valid.size(); ++index) {
                if (index == selected_index) continue;
                const float fraction = trajectories.valid.size() <= 1U
                    ? 0.0F
                    : static_cast<float>(index) /
                      static_cast<float>(trajectories.valid.size() - 1U);
                draw_path(
                    trajectories.valid[index],
                    ColorFromHSV(125.0F + 95.0F * fraction, 0.78F, 0.86F),
                    kBackgroundPathStride);
            }
            if (!trajectories.valid.empty()) {
                const RenderedTrajectory& selected =
                    trajectories.valid[selected_index];
                const size_t sample = animated_sample(
                    selected_animation, animation_seconds);
                draw_path(selected, LIME, 1U, true);
                draw_selected_skeleton(
                    selected_animation.poses[sample], DARKBLUE, SKYBLUE);
            }
            EndMode3D();

            DrawRectangle(14, 14, 790, 210, Color{255, 255, 255, 225});
            DrawText("Generic grasp trajectory field", 26, 24, 24, DARKGRAY);
            DrawText(
                TextFormat(
                    "exact compatible %i valid %i | fallback compatible %i valid %i",
                    static_cast<int>(trajectories.exact_compatible),
                    static_cast<int>(trajectories.exact_valid),
                    static_cast<int>(trajectories.fallback_compatible),
                    static_cast<int>(trajectories.fallback_valid)),
                26, 56, 18, DARKGRAY);
            DrawText(
                TextFormat(
                    "IK %i | object %i | environment %i",
                    static_cast<int>(trajectories.ik_rejected),
                    static_cast<int>(trajectories.object_rejected),
                    static_cast<int>(trajectories.environment_rejected)),
                26, 82, 18, DARKGRAY);
            if (search_stale) {
                DrawText(
                    "SEARCH STALE - press Enter",
                    26, 108, 16, MAROON);
            } else if (trajectories.valid.empty()) {
                DrawText(
                    "0 valid motions for this world grasp",
                    26, 108, 16, MAROON);
            } else {
                const RenderedTrajectory& selected =
                    trajectories.valid[selected_index];
                const size_t sample = animated_sample(
                    selected_animation, animation_seconds);
                const int32_t frame = selected.source.start_frame +
                    static_cast<int32_t>(sample);
                constexpr float radians_to_degrees =
                    57.295779513F;
                const float orientation_error_degrees =
                    selected.shaped.achieved_orientation_error_radians *
                    radians_to_degrees;
                DrawText(
                    TextFormat(
                        "option %i/%i | clip %i | %s | orient %.1f deg | SAFE | %s | %s frame %i",
                        static_cast<int>(selected_index + 1U),
                        static_cast<int>(trajectories.valid.size()),
                        selected.source.clip,
                        match_tier_name(selected.source.match_tier),
                        orientation_error_degrees,
                        support_name(selected.source.support),
                        phase_name(database.phases.at(
                            static_cast<size_t>(frame))),
                        frame),
                    26, 108, 16, DARKGRAY);
            }
            DrawText(
                "Arrows: X/Z  W/S: height  Q/E: yaw  R/F: pitch  Z/C: roll",
                26, 134, 16, DARKGRAY);
            DrawText(
                TextFormat(
                    "[: previous  / or ]: next  Enter: rerun  P: %s",
                    position_only ? "position-only grasp" : "full grasp pose"),
                26, 160, 16, DARKGRAY);
            EndDrawing();
        }
        CloseWindow();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "hand trajectory viewer: " << error.what() << '\n';
        return 2;
    }
}
