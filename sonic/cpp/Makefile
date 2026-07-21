CXX ?= c++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -Werror -pedantic
CPPFLAGS ?= -I../..
BUILD_DIR ?= ../build
BUILD_COMMIT := $(shell git -C ../.. rev-parse HEAD 2>/dev/null || echo unknown)
JOINT_CONTRACT := $(abspath ../configs/g1_joint_contract.json)
SCENE_REGISTRY := $(abspath ../configs/scene_registry.json)

.PHONY: all clean mm_chunk_server route_schedule_cli g1_project_pose_cli

all: mm_chunk_server

mm_chunk_server: $(BUILD_DIR)/mm_chunk_server

route_schedule_cli: $(BUILD_DIR)/route_schedule_cli

g1_project_pose_cli: $(BUILD_DIR)/g1_project_pose_cli

$(BUILD_DIR)/mm_chunk_server: mm_chunk_server.cpp mm_chunk_protocol.h \
		mm_chunk_json.h g1_runtime.h g1_movement_model.h \
		g1_joint_projection.h \
		g1_joint_feasibility.h ../../sha256.h \
		g1_database_validation.h g1_joint_contract_io.h \
		sonic_flat_scene.h ../configs/g1_joint_contract.json \
		../configs/scene_registry.json
	mkdir -p $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) \
		-DMM_CHUNK_BUILD_COMMIT=\"$(BUILD_COMMIT)\" \
		-DMM_CHUNK_DEFAULT_JOINT_CONTRACT=\"$(JOINT_CONTRACT)\" \
		-DMM_CHUNK_DEFAULT_SCENE_REGISTRY=\"$(SCENE_REGISTRY)\" \
		mm_chunk_server.cpp -o $@

$(BUILD_DIR)/route_schedule_cli: route_schedule_cli.cpp ../../route_runtime.h \
		../../scene_runtime.h ../../json_runtime.h ../../terrain_runtime.h
	mkdir -p $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) route_schedule_cli.cpp -o $@

$(BUILD_DIR)/g1_project_pose_cli: g1_project_pose_cli.cpp \
		g1_joint_contract_io.h g1_joint_projection.h \
		../configs/g1_joint_contract.json
	mkdir -p $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) g1_project_pose_cli.cpp -o $@

clean:
	rm -f $(BUILD_DIR)/mm_chunk_server $(BUILD_DIR)/route_schedule_cli \
		$(BUILD_DIR)/g1_project_pose_cli
