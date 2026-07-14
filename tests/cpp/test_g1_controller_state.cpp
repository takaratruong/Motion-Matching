#include "g1_controller_state.h"
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>

static void check(bool value, const char* message)
{
    if (!value) { std::fprintf(stderr, "controller reset test failed: %s\n", message); std::exit(1); }
}

static bool identifier_character(const char value)
{
    return (value >= 'a' && value <= 'z') ||
           (value >= 'A' && value <= 'Z') ||
           (value >= '0' && value <= '9') || value == '_';
}

static bool source_has_call(
    const std::string& source, const char* function_name)
{
    const std::string name(function_name);
    std::size_t position = 0;
    while ((position = source.find(name, position)) != std::string::npos) {
        const bool left_boundary =
            position == 0 || !identifier_character(source[position - 1]);
        std::size_t after = position + name.size();
        const bool right_boundary =
            after == source.size() || !identifier_character(source[after]);
        while (after < source.size() &&
               (source[after] == ' ' || source[after] == '\t' ||
                source[after] == '\r' || source[after] == '\n')) {
            ++after;
        }
        if (left_boundary && right_boundary && after < source.size() &&
            source[after] == '(') {
            return true;
        }
        position += name.size();
    }
    return false;
}

static void test_controller_source_uses_checked_v2_queries()
{
    const char* override_path = std::getenv("G1_CONTROLLER_SOURCE");
    const char* path = override_path != NULL ? override_path : "controller.cpp";
    std::ifstream input(path, std::ios::binary);
    check(input.good(), "controller source opens");
    const std::string source(
        (std::istreambuf_iterator<char>(input)),
        std::istreambuf_iterator<char>());
    check(!input.bad(), "controller source reads");

    const char* legacy_calls[] = {
        "heightfield_sample",
        "heightfield_sample_v1_legacy",
        "heightfield_sample_versioned",
        "terrain_centerline_snapshot_compute",
        "terrain_centerline_query"
    };
    for (const char* legacy : legacy_calls) {
        check(!source_has_call(source, legacy),
              "controller source contains a legacy terrain query");
    }
    check(source_has_call(source, "heightfield_sample_v2"),
          "controller source uses checked v2 height queries");
    check(source_has_call(source, "terrain_centerline_snapshot_compute_v2"),
          "controller source uses the v2 centerline snapshot");
}

static void make_database(database& db)
{
    db.bone_positions.resize(2, G1_BoneCount);
    db.bone_velocities.resize(2, G1_BoneCount);
    db.bone_rotations.resize(2, G1_BoneCount);
    db.bone_angular_velocities.resize(2, G1_BoneCount);
    db.contact_states.resize(2, 2);
    db.range_starts.resize(1); db.range_stops.resize(1);
    db.range_starts(0)=0; db.range_stops(0)=2;
    db.bone_positions.set(vec3()); db.bone_velocities.set(vec3());
    db.bone_rotations.set(quat()); db.bone_angular_velocities.set(vec3());
    db.contact_states.zero();
    db.bone_positions(0,G1_Hips)=vec3(0,0.82f,0);
}

static scene_pack make_scene()
{
    scene_pack scene;
    scene.metadata.id="fixture";
    scene.metadata.spawn_position=vec3(1.25f,0.30f,-2.0f);
    scene.metadata.spawn_yaw=0.5f;
    scene.metadata.playable_bounds={1.0f,-2.25f,1.5f,-1.75f};
    scene.terrain.version=2; scene.terrain.nx=2; scene.terrain.nz=2;
    scene.terrain.origin_x=1.0f; scene.terrain.origin_z=-2.25f;
    scene.terrain.cell_size=0.5f; scene.terrain.exterior_height=-10.0f;
    scene.terrain.heights.resize(4); scene.terrain.heights.set(0.30f);
    scene.walkability.nx=2; scene.walkability.nz=2;
    scene.walkability.cells.resize(4); scene.walkability.cells.set(1);
    return scene;
}

static void test_reset_clears_every_dynamic_subsystem()
{
    database db; make_database(db);
    terrain_support_set support; support.values.resize(2,3); support.values.zero();
    scene_pack scene=make_scene();
    g1_controller_state state;
    state.frame_index=999; state.scene_frame=999; state.search_timer=-9;
    state.simulation_position=vec3(9,9,9); state.simulation_velocity=vec3(9,9,9);
    state.desired_velocity=vec3(9,9,9); state.route_waypoint=99;
    state.support.height=99; state.support.velocity=99;
    state.traversal_speed_scale=0; state.blocked=true;
    state.contact_locks.resize(2); state.contact_locks.set(true);
    state.bone_offset_positions.resize(G1_BoneCount);
    state.bone_offset_positions.set(vec3(9,9,9));
    char error[512]={};
    check(g1_controller_state_reset(state,db,support,scene,error,sizeof(error)),error);
    check(state.frame_index==0 && state.scene_frame==0,"frame reset");
    check(state.search_timer==state.search_time && state.force_search_timer==state.search_time,"search reset");
    check(state.simulation_position.x==1.25f && state.simulation_position.y==0.0f && state.simulation_position.z==-2.0f,"planar spawn");
    check(state.simulation_velocity.x==0 && state.simulation_velocity.y==0 && state.simulation_velocity.z==0,"simulation derivatives reset");
    check(state.desired_velocity.x==0 && state.route_waypoint==1,"input/route reset");
    check(state.support.height==0.30f && state.support.velocity==0,"support reset");
    check(state.adjusted_bone_positions(G1_Simulation).y==0.30f,"render support applied");
    check(state.bone_positions(G1_Simulation).y==0.0f,"inertial pose remains support-local");
    check(state.traversal_speed_scale==1.0f && !state.blocked,"traversal reset");
    for(int i=0;i<state.contact_locks.size;++i)check(!state.contact_locks(i),"contact lock reset");
    for(int i=0;i<state.bone_offset_positions.size;++i)
        check(state.bone_offset_positions(i).x==0&&state.bone_offset_positions(i).y==0&&state.bone_offset_positions(i).z==0,"pose offset reset");
}

static void test_failed_reset_preserves_prior_state()
{
    database db; make_database(db);
    terrain_support_set support; support.values.resize(2,3); support.values.zero();
    scene_pack valid=make_scene(), bad=make_scene(); bad.metadata.spawn_position.x=99;
    g1_controller_state active;
    char error[512]={};
    check(g1_controller_state_reset(active,db,support,valid,error,sizeof(error)),error);
    const float prior_x=active.simulation_position.x;
    const float prior_support=active.support.height;
    const int prior_bone_count=active.bone_positions.size;
    vec3* const prior_bone_data=active.bone_positions.data;
    const vec3 prior_root=active.bone_positions(G1_Simulation);
    check(!g1_controller_state_reset(active,db,support,bad,error,sizeof(error)),"bad spawn rejected");
    check(active.simulation_position.x==prior_x&&
          active.support.height==prior_support,"active scalar state preserved");
    check(active.bone_positions.size==prior_bone_count&&
          active.bone_positions.data==prior_bone_data,
          "active owning array preserved");
    check(active.bone_positions(G1_Simulation).x==prior_root.x&&
          active.bone_positions(G1_Simulation).y==prior_root.y&&
          active.bone_positions(G1_Simulation).z==prior_root.z,
          "active array content preserved");
}

int main()
{
    test_controller_source_uses_checked_v2_queries();
    test_reset_clears_every_dynamic_subsystem();
    test_failed_reset_preserves_prior_state();
    return 0;
}
