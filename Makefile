PLATFORM ?= PLATFORM_DESKTOP
BUILD_MODE ?= RELEASE
DEFINES = -D _DEFAULT_SOURCE -D RAYLIB_BUILD_MODE=$(BUILD_MODE) -D $(PLATFORM)

ifeq ($(PLATFORM),PLATFORM_DESKTOP)
    ifeq ($(OS),Windows_NT)
        RAYLIB_DIR = C:/raylib
        INCLUDE_DIR = -I ./ -I $(RAYLIB_DIR)/raylib/src -I $(RAYLIB_DIR)/raygui/src
        LIBRARY_DIR = -L $(RAYLIB_DIR)/raylib/src
    else
        RAYLIB_DIR ?= .deps/raylib
        RAYGUI_DIR ?= .deps/raygui
        INCLUDE_DIR = -I ./ -I $(RAYLIB_DIR)/src -I $(RAYGUI_DIR)/src
        LIBRARY_DIR =
        LINUX_CONTROLLER_DEPS := $(RAYLIB_DIR)/src/libraylib.a \
          $(RAYGUI_DIR)/src/raygui.h
    endif
else
    RAYLIB_DIR = C:/raylib
    INCLUDE_DIR = -I ./ -I $(RAYLIB_DIR)/raylib/src -I $(RAYLIB_DIR)/raygui/src
    LIBRARY_DIR = -L $(RAYLIB_DIR)/raylib/src
endif

ifeq ($(PLATFORM),PLATFORM_DESKTOP)
    CC = g++
    ifeq ($(OS),Windows_NT)
        EXT = .exe
        ifeq ($(BUILD_MODE),RELEASE)
            CFLAGS ?= $(DEFINES) -ffast-math -march=native -D NDEBUG -O3 $(RAYLIB_DIR)/raylib/src/raylib.rc.data $(INCLUDE_DIR) $(LIBRARY_DIR)
		else
            CFLAGS ?= $(DEFINES) -g $(RAYLIB_DIR)/raylib/src/raylib.rc.data $(INCLUDE_DIR) $(LIBRARY_DIR)
		endif
        LIBS = -lraylib -lopengl32 -lgdi32 -lwinmm
    else
        EXT =
        ifeq ($(BUILD_MODE),RELEASE)
            CFLAGS ?= $(DEFINES) -ffast-math -march=native -D NDEBUG -O3 $(INCLUDE_DIR)
		else
            CFLAGS ?= $(DEFINES) -g $(INCLUDE_DIR)
		endif
        LIBS = $(RAYLIB_DIR)/src/libraylib.a -lGL -lm -lpthread -ldl -lrt -lX11
    endif
endif

ifeq ($(PLATFORM),PLATFORM_WEB)
    CC = emcc
    EXT = .html
    CFLAGS ?= $(DEFINES) $(RAYLIB_DIR)/raylib/src/libraylib.bc -ffast-math -D NDEBUG -O3 -s USE_GLFW=3 -s FORCE_FILESYSTEM=1 -s MAX_WEBGL_VERSION=2 -s ALLOW_MEMORY_GROWTH=1 --preload-file $(dir $<)resources@resources --shell-file ./shell.html $(INCLUDE_DIR) $(LIBRARY_DIR)
endif

CONTROLLER_CXXFLAGS := -std=c++17
INTERACTION_PLACE_SOURCES := interaction_place_target.cpp \
  interaction_place_collision.cpp interaction_place.cpp \
  interaction_place_controller.cpp
INTERACTION_PLACE_HEADERS := interaction_place_target.h \
  interaction_place_collision.h interaction_place.h \
  interaction_place_controller.h
INTERACTION_SOURCES := interaction_pose.cpp interaction_target.cpp \
  interaction_features.cpp interaction_matcher.cpp interaction_playback.cpp \
  interaction_ik.cpp interaction_attachment.cpp interaction_carry.cpp \
  interaction_runtime.cpp interaction_controller_adapter.cpp \
  interaction_target_rig_ik.cpp $(INTERACTION_PLACE_SOURCES)
CONTROLLER_LOCOMOTION_SOURCES := interaction_arrival.cpp \
  interaction_pick_assist.cpp \
  interaction_pick_slots.cpp \
  interaction_pick_approach.cpp \
  locomotion_controller_update.cpp
SOURCE := controller.cpp $(INTERACTION_SOURCES) \
  $(CONTROLLER_LOCOMOTION_SOURCES)
HEADER = $(wildcard *.h)

.PHONY: all bootstrap-raylib controller

all: controller

ifeq ($(strip $(LINUX_CONTROLLER_DEPS)),)
bootstrap-raylib:
	./scripts/bootstrap_raylib.sh
else
.PHONY: verify-raylib

bootstrap-raylib: verify-raylib

verify-raylib:
	./scripts/bootstrap_raylib.sh

$(LINUX_CONTROLLER_DEPS): verify-raylib
endif

controller: $(SOURCE) $(HEADER) $(LINUX_CONTROLLER_DEPS)
	$(CC) $(CONTROLLER_CXXFLAGS) -o $@$(EXT) $(SOURCE) $(CFLAGS) $(LIBS)

clean:
	rm controller$(EXT)

CXX ?= g++
CPP_TEST_FLAGS ?= -std=c++17 -Wall -Wextra -Werror -pedantic -I.
CPP_TEST_DIR := build/tests
TASK12_BUILD_DIR := build/task12
override SAFE_INTERACTION_QUERY_PROBE := build/task12/interaction_query_probe_safe
GRAIL_PICKUP_ROOT ?= /home/ubuntu/datasets/GRAIL/data/pickup_table
GRAIL_ROOT ?= $(GRAIL_PICKUP_ROOT)
G1_XML ?= /home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml
G1_INTERACTION_DIR ?= resources/g1_interaction
DEMO_INTERACTION_LIMIT ?= 5
INTERACTION_DEMO_PACK ?= resources/g1_interaction
PLAYABLE_EVIDENCE_DIR ?= playable-evidence
PLAYABLE_LOG_PATH ?= $(PLAYABLE_EVIDENCE_DIR)/pickup.jsonl
PLAYABLE_SCREENSHOT_PATH ?= $(PLAYABLE_EVIDENCE_DIR)/pickup.png
PLAYABLE_FEATURES_OUTPUT ?= $(PLAYABLE_EVIDENCE_DIR)/locomotion-features.bin
PLACEMENT_EVIDENCE_DIR ?= playable-evidence/placement
PLACEMENT_LOG_PATH ?= $(PLACEMENT_EVIDENCE_DIR)/placement.jsonl
PLACEMENT_SCREENSHOT_PATH ?= $(PLACEMENT_EVIDENCE_DIR)/placement.png
PLACEMENT_FEATURES_OUTPUT ?= $(PLACEMENT_EVIDENCE_DIR)/locomotion-features.bin
CPP_TEST_BINS := $(CPP_TEST_DIR)/test_g1_skeleton
CPP_TEST_BINS += $(CPP_TEST_DIR)/test_interaction_database
CPP_TEST_BINS += $(CPP_TEST_DIR)/test_interaction_pose
CPP_TEST_BINS += $(CPP_TEST_DIR)/test_interaction_target
CPP_TEST_BINS += $(CPP_TEST_DIR)/test_interaction_pick_slots
CPP_TEST_BINS += $(CPP_TEST_DIR)/test_interaction_place_target
CPP_TEST_BINS += $(CPP_TEST_DIR)/test_interaction_place_collision
CPP_TEST_BINS += $(CPP_TEST_DIR)/test_interaction_place
CPP_TEST_BINS += $(CPP_TEST_DIR)/test_interaction_place_controller
CPP_TEST_BINS += $(CPP_TEST_DIR)/test_interaction_features
CPP_TEST_BINS += $(CPP_TEST_DIR)/test_interaction_matcher
CPP_TEST_BINS += $(CPP_TEST_DIR)/test_interaction_playback
CPP_TEST_BINS += $(CPP_TEST_DIR)/test_interaction_ik
CPP_TEST_BINS += $(CPP_TEST_DIR)/test_interaction_attachment
CPP_TEST_BINS += $(CPP_TEST_DIR)/test_interaction_carry
CPP_TEST_BINS += $(CPP_TEST_DIR)/test_interaction_runtime
CPP_TEST_BINS += $(CPP_TEST_DIR)/test_interaction_controller_adapter
CPP_TEST_BINS += $(CPP_TEST_DIR)/test_interaction_target_rig_ik
CPP_TEST_BINS += $(CPP_TEST_DIR)/test_interaction_arrival
CPP_TEST_BINS += $(CPP_TEST_DIR)/test_interaction_arrival_controller
CPP_TEST_BINS += $(CPP_TEST_DIR)/test_interaction_pick_approach
CPP_TEST_BINS += $(CPP_TEST_DIR)/test_interaction_pick_assist
CPP_TEST_BINS += $(CPP_TEST_DIR)/test_stationary_motion_matching
RELEASE_FAST_MATH_TARGET_TEST := \
  $(CPP_TEST_DIR)/test_interaction_target_release_fast_math
RELEASE_FAST_MATH_CARRY_TEST := \
  $(CPP_TEST_DIR)/test_interaction_carry_release_fast_math
PICK_ASSIST_RELEASE_FAST_MATH_TEST := \
  $(CPP_TEST_DIR)/test_interaction_pick_assist_release_fast_math
PLACE_SELECTION_FAST_MATH_TEST := \
  $(CPP_TEST_DIR)/test_interaction_place_selection_fast_math
PLACE_RELEASE_FAST_MATH_TEST := \
  $(CPP_TEST_DIR)/test_interaction_place_release_fast_math
PICK_ENTRY_PREVIEW_FAST_MATH_TEST := \
  $(CPP_TEST_DIR)/test_pick_entry_preview_release_fast_math
PICK_ENTRY_ORACLE_TEST := $(CPP_TEST_DIR)/test_pick_entry_oracle
LIVE_FLAT_PICK_ENTRY_ORACLE_TEST := \
  $(CPP_TEST_DIR)/test_live_flat_pick_entry_oracle
LIVE_FLAT_PICK_ENTRY_ORACLE_RELEASE_TEST := \
  $(CPP_TEST_DIR)/test_live_flat_pick_entry_oracle_release_fast_math
LIVE_FLAT_PICK_ASSIST_SOURCES := interaction_pick_assist.cpp \
  interaction_pick_slots.cpp interaction_pick_approach.cpp \
  interaction_arrival.cpp \
  locomotion_controller_update.cpp
LIVE_FLAT_PICK_ASSIST_HEADERS := interaction_pick_assist.h \
  interaction_pick_slots.h interaction_pick_approach.h \
  interaction_arrival.h \
  locomotion_controller_update.h
LIVE_FLAT_PICK_ENTRY_ORACLE_TIMEOUT_SECONDS := 10
LIVE_FLAT_PICK_ENTRY_ORACLE_GOLDEN := stationary=186 sampled=1 cases=1 ready=1 reach=1 minus=0 plus=0 executed=1 ex_reach=1 ex_minus=0 ex_plus=0 first=0/R/114/139 selection=Reach direct_entry=114
LIVE_FLAT_PICK_POSITION_MATRIX_GOLDEN := position_cases=3 rows=baseline_reach,positive_plus,negative_reach control=baseline_minus/BlockedPath
CONTROLLER_RELEASE_PARITY_FLAGS := -O3 -DNDEBUG -ffast-math -march=native
CONTROLLER_SCENE_RELEASE_FAST_MATH_TEST := $(CPP_TEST_DIR)/test_controller_scene_release_fast_math
RELEASE_INTERACTION_PLACE_PROBE := $(CPP_TEST_DIR)/interaction_place_probe_release
STATIONARY_RELEASE_FAST_MATH_TEST := \
  $(CPP_TEST_DIR)/test_stationary_motion_matching_release_fast_math
ARRIVAL_RELEASE_FAST_MATH_TEST := \
  $(CPP_TEST_DIR)/test_interaction_arrival_release_fast_math
ARRIVAL_CONTROLLER_RELEASE_FAST_MATH_TEST := \
  $(CPP_TEST_DIR)/test_interaction_arrival_controller_release_fast_math
INTERACTION_ARRIVAL_BUILD_INPUTS := \
  $(wildcard interaction_arrival.cpp interaction_arrival.h)
LOCOMOTION_CONTROLLER_UPDATE_BUILD_INPUTS := \
  $(wildcard locomotion_controller_update.cpp \
    locomotion_controller_update.h)

INTERACTION_POSE_GATE_DEPENDENTS := \
  interaction_query_probe \
  $(CPP_TEST_DIR)/test_interaction_pose \
  $(CPP_TEST_DIR)/test_interaction_place_target \
  $(CPP_TEST_DIR)/test_interaction_place_collision \
  $(CPP_TEST_DIR)/test_interaction_features \
  $(CPP_TEST_DIR)/test_interaction_matcher \
  $(CPP_TEST_DIR)/test_interaction_playback \
  $(CPP_TEST_DIR)/test_interaction_attachment \
  $(CPP_TEST_DIR)/test_interaction_target_rig_ik \
  $(SAFE_INTERACTION_QUERY_PROBE)

$(INTERACTION_POSE_GATE_DEPENDENTS): interaction_rotation_gate.h

INTERACTION_RUNTIME_SOURCES := interaction_runtime.cpp
INTERACTION_RUNTIME_SOURCES += interaction_carry.cpp interaction_ik.cpp
INTERACTION_RUNTIME_SOURCES += interaction_attachment.cpp interaction_playback.cpp
INTERACTION_RUNTIME_SOURCES += interaction_matcher.cpp interaction_features.cpp
INTERACTION_RUNTIME_SOURCES += interaction_pose.cpp interaction_target.cpp
INTERACTION_RUNTIME_SOURCES += $(INTERACTION_PLACE_SOURCES)

.PHONY: test-python test-cpp test-interaction
.PHONY: retime-flat-database test-flat-database-retime
.PHONY: test-python-interaction-safe test-interaction-safe
.PHONY: test-interaction-target-release-fast-math
.PHONY: test-interaction-carry-release-fast-math
.PHONY: test-interaction-pick-assist-release-fast-math
.PHONY: test-interaction-place-selection-fast-math
.PHONY: test-interaction-place-release-fast-math
.PHONY: test-pick-entry-preview-release-fast-math
.PHONY: test-controller-scene-release-fast-math
.PHONY: test-interaction-arrival-release-fast-math
.PHONY: test-interaction-arrival-controller-release-fast-math
.PHONY: demo-interaction-pack gate-live-flat-pick-entry-oracle
.PHONY: gate-live-flat-pick-position-matrix
.PHONY: test-live-flat-pick-entry-oracle-exhaustive
.PHONY: gate-playable-interaction gate-place-headless
.PHONY: gate-playable-placement
.PHONY: gate1-interaction

$(CPP_TEST_DIR):
	mkdir -p $@

$(TASK12_BUILD_DIR):
	mkdir -p $@

retime-flat-database:
	python -m resources.retime_flat_database

test-flat-database-retime:
	python -m unittest tests.python.test_flat_database_retime -v

$(CPP_TEST_DIR)/test_g1_skeleton: tests/cpp/test_g1_skeleton.cpp g1_skeleton.h | $(CPP_TEST_DIR)
	$(CXX) $(CPP_TEST_FLAGS) $< -o $@

$(CPP_TEST_DIR)/test_interaction_database: tests/cpp/test_interaction_database.cpp interaction_database.h g1_skeleton.h | $(CPP_TEST_DIR)
	$(CXX) $(CPP_TEST_FLAGS) $< -o $@

$(CPP_TEST_DIR)/test_interaction_pose: tests/cpp/test_interaction_pose.cpp interaction_pose.cpp interaction_pose.h interaction_database.h g1_skeleton.h vec.h quat.h | $(CPP_TEST_DIR)
	$(CXX) $(CPP_TEST_FLAGS) tests/cpp/test_interaction_pose.cpp interaction_pose.cpp -o $@

$(CPP_TEST_DIR)/test_interaction_target: tests/cpp/test_interaction_target.cpp interaction_target.cpp interaction_target.h interaction_pose.h vec.h quat.h | $(CPP_TEST_DIR)
	$(CXX) $(CPP_TEST_FLAGS) tests/cpp/test_interaction_target.cpp interaction_target.cpp -o $@

$(CPP_TEST_DIR)/test_interaction_pick_slots: tests/cpp/test_interaction_pick_slots.cpp interaction_pick_slots.cpp interaction_pick_slots.h interaction_target.cpp interaction_target.h interaction_pose.cpp interaction_pose.h | $(CPP_TEST_DIR)
	$(CXX) $(CPP_TEST_FLAGS) tests/cpp/test_interaction_pick_slots.cpp interaction_pick_slots.cpp interaction_target.cpp interaction_pose.cpp -o $@

$(CPP_TEST_DIR)/test_interaction_place_target: tests/cpp/test_interaction_place_target.cpp interaction_place_target.cpp interaction_place_target.h interaction_matcher.h interaction_features.h interaction_target.h interaction_pose.cpp interaction_pose.h interaction_database.h g1_skeleton.h vec.h quat.h | $(CPP_TEST_DIR)
	$(CXX) $(CPP_TEST_FLAGS) tests/cpp/test_interaction_place_target.cpp interaction_place_target.cpp interaction_pose.cpp -o $@

$(CPP_TEST_DIR)/test_interaction_place_collision: tests/cpp/test_interaction_place_collision.cpp interaction_place_collision.cpp interaction_place_collision.h interaction_place_target.h interaction_pose.cpp interaction_pose.h interaction_target.h interaction_database.h g1_skeleton.h vec.h quat.h | $(CPP_TEST_DIR)
	$(CXX) $(CPP_TEST_FLAGS) tests/cpp/test_interaction_place_collision.cpp interaction_place_collision.cpp interaction_pose.cpp -o $@

$(CPP_TEST_DIR)/test_interaction_place: tests/cpp/test_interaction_place.cpp interaction_place.cpp interaction_place.h interaction_place_collision.cpp interaction_place_collision.h interaction_place_target.cpp interaction_place_target.h interaction_ik.cpp interaction_ik.h interaction_rotation_gate.h g1_arm_joint_metadata.h interaction_matcher.cpp interaction_matcher.h interaction_features.cpp interaction_features.h interaction_pose.cpp interaction_pose.h interaction_target.cpp interaction_target.h interaction_database.h g1_skeleton.h vec.h quat.h | $(CPP_TEST_DIR)
	$(CXX) $(CPP_TEST_FLAGS) tests/cpp/test_interaction_place.cpp interaction_place.cpp interaction_place_collision.cpp interaction_place_target.cpp interaction_ik.cpp interaction_matcher.cpp interaction_features.cpp interaction_pose.cpp interaction_target.cpp -o $@

$(CPP_TEST_DIR)/test_interaction_place_controller: tests/cpp/test_interaction_place_controller.cpp interaction_place_controller.cpp interaction_place_controller.h interaction_place.cpp interaction_place.h interaction_place_collision.cpp interaction_place_collision.h interaction_place_target.cpp interaction_place_target.h interaction_ik.cpp interaction_ik.h interaction_rotation_gate.h g1_arm_joint_metadata.h interaction_matcher.cpp interaction_matcher.h interaction_features.cpp interaction_features.h interaction_pose.cpp interaction_pose.h interaction_target.cpp interaction_target.h interaction_database.h g1_skeleton.h vec.h quat.h | $(CPP_TEST_DIR)
	$(CXX) $(CPP_TEST_FLAGS) tests/cpp/test_interaction_place_controller.cpp interaction_place_controller.cpp interaction_place.cpp interaction_place_collision.cpp interaction_place_target.cpp interaction_ik.cpp interaction_matcher.cpp interaction_features.cpp interaction_pose.cpp interaction_target.cpp -o $@

$(RELEASE_FAST_MATH_TARGET_TEST): tests/cpp/test_interaction_target.cpp interaction_target.cpp interaction_target.h interaction_pose.h vec.h quat.h | $(CPP_TEST_DIR)
	$(CXX) $(CPP_TEST_FLAGS) -O3 -DNDEBUG -ffast-math tests/cpp/test_interaction_target.cpp interaction_target.cpp -o $@

$(RELEASE_FAST_MATH_CARRY_TEST): tests/cpp/test_interaction_carry_fast_math.cpp tests/cpp/interaction_runtime_fixture.h interaction_carry.cpp interaction_carry.h interaction_ik.cpp interaction_ik.h interaction_rotation_gate.h g1_arm_joint_metadata.h interaction_matcher.cpp interaction_matcher.h interaction_features.cpp interaction_features.h interaction_pose.cpp interaction_pose.h interaction_target.cpp interaction_target.h interaction_database.h g1_skeleton.h vec.h quat.h | $(CPP_TEST_DIR)
	$(CXX) $(CPP_TEST_FLAGS) -O3 -DNDEBUG -ffast-math tests/cpp/test_interaction_carry_fast_math.cpp interaction_carry.cpp interaction_ik.cpp interaction_matcher.cpp interaction_features.cpp interaction_pose.cpp interaction_target.cpp -o $@

$(CPP_TEST_DIR)/test_interaction_arrival: tests/cpp/test_interaction_arrival.cpp $(INTERACTION_ARRIVAL_BUILD_INPUTS) vec.h quat.h common.h | $(CPP_TEST_DIR)
	$(CXX) $(CPP_TEST_FLAGS) tests/cpp/test_interaction_arrival.cpp interaction_arrival.cpp -o $@

$(ARRIVAL_RELEASE_FAST_MATH_TEST): tests/cpp/test_interaction_arrival.cpp $(INTERACTION_ARRIVAL_BUILD_INPUTS) vec.h quat.h common.h | $(CPP_TEST_DIR)
	$(CXX) $(CPP_TEST_FLAGS) -O3 -DNDEBUG -ffast-math tests/cpp/test_interaction_arrival.cpp interaction_arrival.cpp -o $@

$(CPP_TEST_DIR)/test_interaction_arrival_controller: tests/cpp/test_interaction_arrival_controller.cpp $(INTERACTION_ARRIVAL_BUILD_INPUTS) $(LOCOMOTION_CONTROLLER_UPDATE_BUILD_INPUTS) array.h vec.h quat.h common.h spring.h | $(CPP_TEST_DIR)
	$(CXX) $(CPP_TEST_FLAGS) tests/cpp/test_interaction_arrival_controller.cpp interaction_arrival.cpp locomotion_controller_update.cpp -o $@

$(CPP_TEST_DIR)/test_interaction_pick_approach: tests/cpp/test_interaction_pick_approach.cpp interaction_pick_approach.cpp interaction_pick_approach.h interaction_runtime.h interaction_matcher.h interaction_features.h interaction_pose.cpp interaction_pose.h interaction_target.h interaction_database.h g1_skeleton.h vec.h quat.h | $(CPP_TEST_DIR)
	$(CXX) $(CPP_TEST_FLAGS) tests/cpp/test_interaction_pick_approach.cpp interaction_pick_approach.cpp interaction_pose.cpp -o $@

$(CPP_TEST_DIR)/test_interaction_pick_assist: tests/cpp/test_interaction_pick_assist.cpp interaction_pick_assist.cpp interaction_pick_assist.h interaction_pick_slots.cpp interaction_pick_slots.h interaction_arrival.cpp interaction_arrival.h interaction_target.cpp interaction_target.h interaction_pose.cpp interaction_pose.h | $(CPP_TEST_DIR)
	$(CXX) $(CPP_TEST_FLAGS) tests/cpp/test_interaction_pick_assist.cpp interaction_pick_assist.cpp interaction_pick_slots.cpp interaction_arrival.cpp interaction_target.cpp interaction_pose.cpp -o $@

$(PICK_ASSIST_RELEASE_FAST_MATH_TEST): tests/cpp/test_interaction_pick_assist.cpp interaction_pick_assist.cpp interaction_pick_assist.h interaction_pick_slots.cpp interaction_pick_slots.h interaction_arrival.cpp interaction_arrival.h interaction_target.cpp interaction_target.h interaction_pose.cpp interaction_pose.h | $(CPP_TEST_DIR)
	# GCC 13 misdiagnoses libstdc++'s small-range std::sort as out of bounds.
	$(CXX) $(CPP_TEST_FLAGS) -Wno-array-bounds -O3 -DNDEBUG -ffast-math tests/cpp/test_interaction_pick_assist.cpp interaction_pick_assist.cpp interaction_pick_slots.cpp interaction_arrival.cpp interaction_target.cpp interaction_pose.cpp -o $@

$(ARRIVAL_CONTROLLER_RELEASE_FAST_MATH_TEST): tests/cpp/test_interaction_arrival_controller.cpp $(INTERACTION_ARRIVAL_BUILD_INPUTS) $(LOCOMOTION_CONTROLLER_UPDATE_BUILD_INPUTS) array.h vec.h quat.h common.h spring.h | $(CPP_TEST_DIR)
	$(CXX) $(CPP_TEST_FLAGS) -Wno-unused-parameter -O3 -DNDEBUG -ffast-math tests/cpp/test_interaction_arrival_controller.cpp interaction_arrival.cpp locomotion_controller_update.cpp -o $@

$(PICK_ENTRY_PREVIEW_FAST_MATH_TEST): tests/cpp/test_pick_entry_preview_fast_math.cpp tests/cpp/interaction_runtime_fixture.h $(INTERACTION_RUNTIME_SOURCES) interaction_runtime.h interaction_matcher.h interaction_pose.h | $(CPP_TEST_DIR)
	$(CXX) $(CPP_TEST_FLAGS) -O3 -DNDEBUG -ffast-math tests/cpp/test_pick_entry_preview_fast_math.cpp $(INTERACTION_RUNTIME_SOURCES) -o $@

$(PICK_ENTRY_ORACLE_TEST): tests/cpp/test_pick_entry_oracle.cpp tests/cpp/pick_entry_oracle_roots.h tests/cpp/interaction_runtime_fixture.h $(INTERACTION_RUNTIME_SOURCES) interaction_runtime.h interaction_matcher.h interaction_pose.h | $(CPP_TEST_DIR)
	$(CXX) $(CPP_TEST_FLAGS) tests/cpp/test_pick_entry_oracle.cpp $(INTERACTION_RUNTIME_SOURCES) -o $@

$(LIVE_FLAT_PICK_ENTRY_ORACLE_TEST): tests/cpp/test_live_flat_pick_entry_oracle.cpp tests/cpp/pick_entry_oracle_roots.h stationary_motion_matching.h database.h interaction_controller_adapter.cpp interaction_controller_adapter.h interaction_target_rig_ik.cpp interaction_target_rig_ik.h $(INTERACTION_RUNTIME_SOURCES) $(LIVE_FLAT_PICK_ASSIST_SOURCES) $(LIVE_FLAT_PICK_ASSIST_HEADERS) interaction_runtime.h interaction_matcher.h interaction_pose.h array.h common.h locomotion_timing.h | $(CPP_TEST_DIR)
	$(CXX) $(CPP_TEST_FLAGS) -Wno-unused-parameter -Wno-unused-result -O2 -pthread tests/cpp/test_live_flat_pick_entry_oracle.cpp interaction_controller_adapter.cpp interaction_target_rig_ik.cpp $(INTERACTION_RUNTIME_SOURCES) $(LIVE_FLAT_PICK_ASSIST_SOURCES) -o $@

$(LIVE_FLAT_PICK_ENTRY_ORACLE_RELEASE_TEST): tests/cpp/test_live_flat_pick_entry_oracle.cpp tests/cpp/pick_entry_oracle_roots.h stationary_motion_matching.h database.h interaction_controller_adapter.cpp interaction_controller_adapter.h interaction_target_rig_ik.cpp interaction_target_rig_ik.h $(INTERACTION_RUNTIME_SOURCES) $(LIVE_FLAT_PICK_ASSIST_SOURCES) $(LIVE_FLAT_PICK_ASSIST_HEADERS) interaction_runtime.h interaction_matcher.h interaction_pose.h array.h common.h locomotion_timing.h | $(CPP_TEST_DIR)
	$(CXX) $(CPP_TEST_FLAGS) -Wno-unused-parameter -Wno-unused-variable -Wno-unused-result $(CONTROLLER_RELEASE_PARITY_FLAGS) -pthread tests/cpp/test_live_flat_pick_entry_oracle.cpp interaction_controller_adapter.cpp interaction_target_rig_ik.cpp $(INTERACTION_RUNTIME_SOURCES) $(LIVE_FLAT_PICK_ASSIST_SOURCES) -o $@

$(PLACE_SELECTION_FAST_MATH_TEST): tests/cpp/test_interaction_place.cpp interaction_place.cpp interaction_place.h interaction_place_collision.cpp interaction_place_collision.h interaction_place_target.cpp interaction_place_target.h interaction_ik.cpp interaction_ik.h interaction_rotation_gate.h g1_arm_joint_metadata.h interaction_matcher.cpp interaction_matcher.h interaction_features.cpp interaction_features.h interaction_pose.cpp interaction_pose.h interaction_target.cpp interaction_target.h interaction_database.h g1_skeleton.h vec.h quat.h | $(CPP_TEST_DIR)
	$(CXX) $(CPP_TEST_FLAGS) -O3 -DNDEBUG -ffast-math tests/cpp/test_interaction_place.cpp interaction_place.cpp interaction_place_collision.cpp interaction_place_target.cpp interaction_ik.cpp interaction_matcher.cpp interaction_features.cpp interaction_pose.cpp interaction_target.cpp -o $@

$(PLACE_RELEASE_FAST_MATH_TEST): tests/cpp/test_interaction_place_fast_math.cpp interaction_place_controller.cpp interaction_place_controller.h interaction_place.cpp interaction_place.h interaction_place_collision.cpp interaction_place_collision.h interaction_place_target.cpp interaction_place_target.h interaction_ik.cpp interaction_ik.h interaction_rotation_gate.h g1_arm_joint_metadata.h interaction_matcher.cpp interaction_matcher.h interaction_features.cpp interaction_features.h interaction_pose.cpp interaction_pose.h interaction_target.cpp interaction_target.h interaction_database.h g1_skeleton.h vec.h quat.h | $(CPP_TEST_DIR)
	$(CXX) $(CPP_TEST_FLAGS) -O3 -DNDEBUG -ffast-math tests/cpp/test_interaction_place_fast_math.cpp interaction_place_controller.cpp interaction_place.cpp interaction_place_collision.cpp interaction_place_target.cpp interaction_ik.cpp interaction_matcher.cpp interaction_features.cpp interaction_pose.cpp interaction_target.cpp -o $@

$(CPP_TEST_DIR)/test_interaction_features: tests/cpp/test_interaction_features.cpp interaction_features.cpp interaction_features.h interaction_pose.cpp interaction_pose.h interaction_target.h interaction_database.h g1_skeleton.h vec.h quat.h | $(CPP_TEST_DIR)
	$(CXX) $(CPP_TEST_FLAGS) tests/cpp/test_interaction_features.cpp interaction_features.cpp interaction_pose.cpp -o $@

$(CPP_TEST_DIR)/test_interaction_matcher: tests/cpp/test_interaction_matcher.cpp tests/cpp/interaction_runtime_fixture.h interaction_matcher.cpp interaction_matcher.h interaction_features.cpp interaction_features.h interaction_pose.cpp interaction_pose.h interaction_target.cpp interaction_target.h interaction_database.h g1_skeleton.h vec.h quat.h | $(CPP_TEST_DIR)
	$(CXX) $(CPP_TEST_FLAGS) tests/cpp/test_interaction_matcher.cpp interaction_matcher.cpp interaction_features.cpp interaction_pose.cpp interaction_target.cpp -o $@

$(CPP_TEST_DIR)/test_interaction_playback: tests/cpp/test_interaction_playback.cpp tests/cpp/interaction_runtime_fixture.h interaction_playback.cpp interaction_playback.h interaction_matcher.cpp interaction_matcher.h interaction_features.cpp interaction_features.h interaction_pose.cpp interaction_pose.h interaction_target.cpp interaction_target.h interaction_database.h g1_skeleton.h vec.h quat.h | $(CPP_TEST_DIR)
	$(CXX) $(CPP_TEST_FLAGS) tests/cpp/test_interaction_playback.cpp interaction_playback.cpp interaction_matcher.cpp interaction_features.cpp interaction_pose.cpp interaction_target.cpp -o $@

$(CPP_TEST_DIR)/test_interaction_ik: tests/cpp/test_interaction_ik.cpp interaction_ik.cpp interaction_ik.h interaction_rotation_gate.h g1_arm_joint_metadata.h interaction_pose.cpp interaction_pose.h interaction_matcher.h interaction_features.h interaction_target.h interaction_database.h g1_skeleton.h vec.h quat.h | $(CPP_TEST_DIR)
	$(CXX) $(CPP_TEST_FLAGS) tests/cpp/test_interaction_ik.cpp interaction_ik.cpp interaction_pose.cpp -o $@

$(CPP_TEST_DIR)/test_interaction_attachment: tests/cpp/test_interaction_attachment.cpp interaction_attachment.cpp interaction_attachment.h interaction_pose.cpp interaction_pose.h interaction_target.cpp interaction_target.h interaction_matcher.h interaction_features.h interaction_database.h g1_skeleton.h vec.h quat.h | $(CPP_TEST_DIR)
	$(CXX) $(CPP_TEST_FLAGS) tests/cpp/test_interaction_attachment.cpp interaction_attachment.cpp interaction_pose.cpp interaction_target.cpp -o $@

$(CPP_TEST_DIR)/test_interaction_carry: tests/cpp/test_interaction_carry.cpp tests/cpp/interaction_runtime_fixture.h interaction_carry.cpp interaction_carry.h interaction_ik.cpp interaction_ik.h interaction_rotation_gate.h g1_arm_joint_metadata.h interaction_matcher.cpp interaction_matcher.h interaction_features.cpp interaction_features.h interaction_pose.cpp interaction_pose.h interaction_target.cpp interaction_target.h interaction_database.h g1_skeleton.h vec.h quat.h | $(CPP_TEST_DIR)
	$(CXX) $(CPP_TEST_FLAGS) -c tests/cpp/test_interaction_carry.cpp -o $(CPP_TEST_DIR)/test_interaction_carry.o
	$(CXX) $(CPP_TEST_FLAGS) $(CPP_TEST_DIR)/test_interaction_carry.o interaction_carry.cpp interaction_ik.cpp interaction_matcher.cpp interaction_features.cpp interaction_pose.cpp interaction_target.cpp -o $@

$(CPP_TEST_DIR)/test_interaction_runtime: tests/cpp/test_interaction_runtime.cpp tests/cpp/interaction_runtime_fixture.h interaction_runtime.h $(INTERACTION_RUNTIME_SOURCES) $(INTERACTION_PLACE_HEADERS) interaction_carry.h interaction_ik.h interaction_rotation_gate.h g1_arm_joint_metadata.h interaction_attachment.h interaction_playback.h interaction_matcher.h interaction_features.h interaction_pose.h interaction_target.h interaction_database.h g1_skeleton.h vec.h quat.h | $(CPP_TEST_DIR)
	$(CXX) $(CPP_TEST_FLAGS) -c tests/cpp/test_interaction_runtime.cpp -o $(CPP_TEST_DIR)/test_interaction_runtime.o
	$(CXX) $(CPP_TEST_FLAGS) $(CPP_TEST_DIR)/test_interaction_runtime.o $(INTERACTION_RUNTIME_SOURCES) -o $@

$(CPP_TEST_DIR)/test_interaction_controller_adapter: tests/cpp/test_interaction_controller_adapter.cpp tests/cpp/interaction_runtime_fixture.h interaction_controller_adapter.cpp interaction_controller_adapter.h interaction_target_rig_ik.cpp interaction_target_rig_ik.h interaction_runtime.h $(INTERACTION_RUNTIME_SOURCES) $(INTERACTION_PLACE_HEADERS) interaction_carry.h interaction_ik.h interaction_rotation_gate.h g1_arm_joint_metadata.h interaction_attachment.h interaction_playback.h interaction_matcher.h interaction_features.h interaction_pose.h interaction_target.h interaction_database.h g1_skeleton.h vec.h quat.h | $(CPP_TEST_DIR)
	$(CXX) $(CPP_TEST_FLAGS) tests/cpp/test_interaction_controller_adapter.cpp interaction_controller_adapter.cpp interaction_target_rig_ik.cpp $(INTERACTION_RUNTIME_SOURCES) -o $@

$(CONTROLLER_SCENE_RELEASE_FAST_MATH_TEST): tests/cpp/test_controller_scene_release_fast_math.cpp interaction_controller_adapter.cpp interaction_controller_adapter.h interaction_target_rig_ik.cpp interaction_target_rig_ik.h interaction_runtime.h $(INTERACTION_RUNTIME_SOURCES) $(INTERACTION_PLACE_HEADERS) interaction_carry.h interaction_ik.h interaction_rotation_gate.h g1_arm_joint_metadata.h interaction_attachment.h interaction_playback.h interaction_matcher.h interaction_features.h interaction_pose.h interaction_target.h interaction_database.h g1_skeleton.h vec.h quat.h | $(CPP_TEST_DIR)
	$(CXX) $(CPP_TEST_FLAGS) $(CONTROLLER_RELEASE_PARITY_FLAGS) tests/cpp/test_controller_scene_release_fast_math.cpp interaction_controller_adapter.cpp interaction_target_rig_ik.cpp $(INTERACTION_RUNTIME_SOURCES) -o $@

$(CPP_TEST_DIR)/test_interaction_target_rig_ik: tests/cpp/test_interaction_target_rig_ik.cpp interaction_target_rig_ik.cpp interaction_target_rig_ik.h interaction_controller_adapter.h interaction_pose.cpp interaction_pose.h interaction_target.h interaction_database.h g1_skeleton.h vec.h quat.h | $(CPP_TEST_DIR)
	$(CXX) $(CPP_TEST_FLAGS) tests/cpp/test_interaction_target_rig_ik.cpp interaction_target_rig_ik.cpp interaction_pose.cpp -o $@

$(CPP_TEST_DIR)/test_stationary_motion_matching: tests/cpp/test_stationary_motion_matching.cpp stationary_motion_matching.h database.h array.h common.h vec.h quat.h locomotion_timing.h | $(CPP_TEST_DIR)
	$(CXX) $(CPP_TEST_FLAGS) -Wno-unused-parameter tests/cpp/test_stationary_motion_matching.cpp -o $@

$(STATIONARY_RELEASE_FAST_MATH_TEST): tests/cpp/test_stationary_motion_matching.cpp stationary_motion_matching.h database.h array.h common.h vec.h quat.h locomotion_timing.h | $(CPP_TEST_DIR)
	$(CXX) $(CPP_TEST_FLAGS) -Wno-unused-parameter -Wno-unused-variable -Wno-unused-result -O3 -DNDEBUG -ffast-math tests/cpp/test_stationary_motion_matching.cpp -o $@

interaction_probe: interaction_probe.cpp interaction_database.h g1_skeleton.h
	$(CXX) $(CPP_TEST_FLAGS) $< -o $@

interaction_query_probe: interaction_query_probe.cpp interaction_features.cpp interaction_features.h interaction_pose.cpp interaction_pose.h interaction_target.h interaction_database.h g1_skeleton.h vec.h quat.h
	$(CXX) $(CPP_TEST_FLAGS) interaction_query_probe.cpp interaction_features.cpp interaction_pose.cpp -o $@

$(SAFE_INTERACTION_QUERY_PROBE): interaction_query_probe.cpp interaction_features.cpp interaction_features.h interaction_pose.cpp interaction_pose.h interaction_target.h interaction_database.h g1_skeleton.h vec.h quat.h | $(TASK12_BUILD_DIR)
	$(CXX) $(CPP_TEST_FLAGS) interaction_query_probe.cpp interaction_features.cpp interaction_pose.cpp -o $@

interaction_runtime_probe: interaction_runtime_probe.cpp interaction_runtime.h $(INTERACTION_RUNTIME_SOURCES) $(INTERACTION_PLACE_HEADERS) interaction_carry.h interaction_ik.h interaction_rotation_gate.h g1_arm_joint_metadata.h interaction_attachment.h interaction_playback.h interaction_matcher.h interaction_features.h interaction_pose.h interaction_target.h interaction_database.h g1_skeleton.h vec.h quat.h
	$(CXX) $(CPP_TEST_FLAGS) interaction_runtime_probe.cpp $(INTERACTION_RUNTIME_SOURCES) -o $@

interaction_place_probe: interaction_place_probe.cpp tests/cpp/pick_entry_oracle_roots.h interaction_controller_adapter.cpp interaction_controller_adapter.h interaction_target_rig_ik.cpp interaction_target_rig_ik.h locomotion_timing.h interaction_runtime.h $(INTERACTION_RUNTIME_SOURCES) $(INTERACTION_PLACE_HEADERS) interaction_carry.h interaction_ik.h interaction_rotation_gate.h g1_arm_joint_metadata.h interaction_attachment.h interaction_playback.h interaction_matcher.h interaction_features.h interaction_pose.h interaction_target.h interaction_database.h g1_skeleton.h vec.h quat.h
	$(CXX) $(CPP_TEST_FLAGS) interaction_place_probe.cpp interaction_controller_adapter.cpp interaction_target_rig_ik.cpp $(INTERACTION_RUNTIME_SOURCES) -o $@

$(RELEASE_INTERACTION_PLACE_PROBE): interaction_place_probe.cpp tests/cpp/pick_entry_oracle_roots.h interaction_controller_adapter.cpp interaction_controller_adapter.h interaction_target_rig_ik.cpp interaction_target_rig_ik.h locomotion_timing.h interaction_runtime.h $(INTERACTION_RUNTIME_SOURCES) $(INTERACTION_PLACE_HEADERS) interaction_carry.h interaction_ik.h interaction_rotation_gate.h g1_arm_joint_metadata.h interaction_attachment.h interaction_playback.h interaction_matcher.h interaction_features.h interaction_pose.h interaction_target.h interaction_database.h g1_skeleton.h vec.h quat.h | $(CPP_TEST_DIR)
	$(CXX) $(CPP_TEST_FLAGS) $(CONTROLLER_RELEASE_PARITY_FLAGS) interaction_place_probe.cpp interaction_controller_adapter.cpp interaction_target_rig_ik.cpp $(INTERACTION_RUNTIME_SOURCES) -o $@

test-python: interaction_probe interaction_query_probe
	python -m unittest discover -s tests/python -t . -v

test-cpp: $(CPP_TEST_BINS)
	@for test_bin in $(CPP_TEST_BINS); do $$test_bin || exit 1; done

test-interaction-target-release-fast-math: $(RELEASE_FAST_MATH_TARGET_TEST)
	$(RELEASE_FAST_MATH_TARGET_TEST)

test-interaction-carry-release-fast-math: $(RELEASE_FAST_MATH_CARRY_TEST)
	$(RELEASE_FAST_MATH_CARRY_TEST)

test-interaction-pick-assist-release-fast-math: $(PICK_ASSIST_RELEASE_FAST_MATH_TEST)
	$(PICK_ASSIST_RELEASE_FAST_MATH_TEST)

test-interaction-arrival-release-fast-math: $(ARRIVAL_RELEASE_FAST_MATH_TEST)
	$(ARRIVAL_RELEASE_FAST_MATH_TEST)

test-interaction-arrival-controller-release-fast-math: $(ARRIVAL_CONTROLLER_RELEASE_FAST_MATH_TEST)
	$(ARRIVAL_CONTROLLER_RELEASE_FAST_MATH_TEST)

test-pick-entry-preview-release-fast-math: $(PICK_ENTRY_PREVIEW_FAST_MATH_TEST)
	$(PICK_ENTRY_PREVIEW_FAST_MATH_TEST) --fast-math-canary

test-interaction-place-selection-fast-math: $(PLACE_SELECTION_FAST_MATH_TEST)
	$(PLACE_SELECTION_FAST_MATH_TEST) --fast-math-canary

test-interaction-place-release-fast-math: $(PLACE_RELEASE_FAST_MATH_TEST)
	$(PLACE_RELEASE_FAST_MATH_TEST)

test-controller-scene-release-fast-math: $(CONTROLLER_SCENE_RELEASE_FAST_MATH_TEST)
	$(CONTROLLER_SCENE_RELEASE_FAST_MATH_TEST)

test-interaction: test-python test-cpp

test-python-interaction-safe: interaction_probe $(SAFE_INTERACTION_QUERY_PROBE)
	MM_INTERACTION_QUERY_PROBE="$(abspath $(SAFE_INTERACTION_QUERY_PROBE))" \
	  G1_XML="$(G1_XML)" \
	  G1_INTERACTION_DIR="" \
	  python -m unittest discover -s tests/python -t . -v

test-interaction-safe: test-python-interaction-safe test-cpp \
  test-interaction-target-release-fast-math \
  test-interaction-carry-release-fast-math \
  test-interaction-pick-assist-release-fast-math \
  test-interaction-arrival-release-fast-math \
  test-interaction-arrival-controller-release-fast-math \
  test-pick-entry-preview-release-fast-math \
  test-interaction-place-selection-fast-math \
  test-interaction-place-release-fast-math \
  test-controller-scene-release-fast-math

gate1-interaction: test-interaction-safe interaction_probe
	python -m resources.build_g1_interaction_database \
	  --source-root "$(GRAIL_PICKUP_ROOT)" \
	  --g1-xml "$(G1_XML)" \
	  --output "$(G1_INTERACTION_DIR)" \
	  --target-fps 25 \
	  --allow-rejections
	python -m resources.validate_g1_interaction_database \
	  --input "$(G1_INTERACTION_DIR)"
	G1_INTERACTION_DIR="$(G1_INTERACTION_DIR)" python -m unittest \
	  tests.python.test_interaction_gate1 -v
	./interaction_probe "$(G1_INTERACTION_DIR)" --json

demo-interaction-pack:
	python -m resources.build_g1_interaction_database \
	  --source-root "$(GRAIL_ROOT)" \
	  --g1-xml "$(G1_XML)" \
	  --output "$(INTERACTION_DEMO_PACK)" \
	  --target-fps 25 \
	  --limit "$(DEMO_INTERACTION_LIMIT)" \
	  --heldout-count 1 \
	  --allow-rejections
	python -m resources.validate_g1_interaction_database \
	  --input "$(INTERACTION_DEMO_PACK)"

gate-live-flat-pick-entry-oracle: $(LIVE_FLAT_PICK_ENTRY_ORACLE_TEST) $(LIVE_FLAT_PICK_ENTRY_ORACLE_RELEASE_TEST)
	@set -eu; \
	  test -f resources/database.bin || { \
	    echo "ERROR resources/database.bin is unavailable" >&2; \
	    exit 1; \
	  }; \
	  test -f "$(INTERACTION_DEMO_PACK)/interaction_database.bin" || { \
	    echo "ERROR $(INTERACTION_DEMO_PACK)/interaction_database.bin is unavailable" >&2; \
	    exit 1; \
	  }; \
	  test -f "$(INTERACTION_DEMO_PACK)/interaction_features.bin" || { \
	    echo "ERROR $(INTERACTION_DEMO_PACK)/interaction_features.bin is unavailable" >&2; \
	    exit 1; \
	  }; \
	  if normal_output="$$(timeout --signal=TERM --kill-after=2s \
	      "$(LIVE_FLAT_PICK_ENTRY_ORACLE_TIMEOUT_SECONDS)s" \
	      "$(LIVE_FLAT_PICK_ENTRY_ORACLE_TEST)" \
	      resources/database.bin "$(INTERACTION_DEMO_PACK)")"; then \
	    :; \
	  else \
	    normal_status=$$?; \
	    echo "ERROR live-flat pickup oracle normal exited $$normal_status" >&2; \
	    exit "$$normal_status"; \
	  fi; \
	  if release_output="$$(timeout --signal=TERM --kill-after=2s \
	      "$(LIVE_FLAT_PICK_ENTRY_ORACLE_TIMEOUT_SECONDS)s" \
	      "$(LIVE_FLAT_PICK_ENTRY_ORACLE_RELEASE_TEST)" \
	      resources/database.bin "$(INTERACTION_DEMO_PACK)")"; then \
	    :; \
	  else \
	    release_status=$$?; \
	    echo "ERROR live-flat pickup oracle release exited $$release_status" >&2; \
	    exit "$$release_status"; \
	  fi; \
	  if test "$$normal_output" != "$$release_output"; then \
	    echo "ERROR live-flat pickup oracle normal/release mismatch" >&2; \
	    echo "normal: $$normal_output" >&2; \
	    echo "release: $$release_output" >&2; \
	    exit 1; \
	  fi; \
	  if test "$$normal_output" != "$(LIVE_FLAT_PICK_ENTRY_ORACLE_GOLDEN)"; then \
	    echo "ERROR live-flat pickup oracle golden summary mismatch" >&2; \
	    echo "expected: $(LIVE_FLAT_PICK_ENTRY_ORACLE_GOLDEN)" >&2; \
	    echo "actual:   $$normal_output" >&2; \
	    exit 1; \
	  fi; \
	  printf '%s\n' "$$normal_output"

gate-live-flat-pick-position-matrix: $(LIVE_FLAT_PICK_ENTRY_ORACLE_TEST) $(LIVE_FLAT_PICK_ENTRY_ORACLE_RELEASE_TEST)
	@set -eu; \
	  if normal_output="$$(timeout --signal=TERM --kill-after=2s \
	      "$(LIVE_FLAT_PICK_ENTRY_ORACLE_TIMEOUT_SECONDS)s" \
	      "$(LIVE_FLAT_PICK_ENTRY_ORACLE_TEST)" \
	      resources/database.bin "$(INTERACTION_DEMO_PACK)" \
	      --position-matrix)"; then \
	    :; \
	  else \
	    normal_status=$$?; \
	    echo "ERROR live-flat pickup position matrix normal exited $$normal_status" >&2; \
	    exit "$$normal_status"; \
	  fi; \
	  if release_output="$$(timeout --signal=TERM --kill-after=2s \
	      "$(LIVE_FLAT_PICK_ENTRY_ORACLE_TIMEOUT_SECONDS)s" \
	      "$(LIVE_FLAT_PICK_ENTRY_ORACLE_RELEASE_TEST)" \
	      resources/database.bin "$(INTERACTION_DEMO_PACK)" \
	      --position-matrix)"; then \
	    :; \
	  else \
	    release_status=$$?; \
	    echo "ERROR live-flat pickup position matrix release exited $$release_status" >&2; \
	    exit "$$release_status"; \
	  fi; \
	  if test "$$normal_output" != "$$release_output"; then \
	    echo "ERROR live-flat pickup position matrix normal/release mismatch" >&2; \
	    echo "normal: $$normal_output" >&2; \
	    echo "release: $$release_output" >&2; \
	    exit 1; \
	  fi; \
	  if test "$$normal_output" != "$(LIVE_FLAT_PICK_POSITION_MATRIX_GOLDEN)"; then \
	    echo "ERROR live-flat pickup position matrix golden summary mismatch" >&2; \
	    echo "expected: $(LIVE_FLAT_PICK_POSITION_MATRIX_GOLDEN)" >&2; \
	    echo "actual:   $$normal_output" >&2; \
	    exit 1; \
	  fi; \
	  printf '%s\n' "$$normal_output"

# Slow/manual diagnostic only; intentionally excluded from default gates.
test-live-flat-pick-entry-oracle-exhaustive: $(LIVE_FLAT_PICK_ENTRY_ORACLE_TEST)
	@echo "WARNING: running slow exhaustive 558-case live-flat oracle" >&2
	$(LIVE_FLAT_PICK_ENTRY_ORACLE_TEST) \
	  resources/database.bin "$(INTERACTION_DEMO_PACK)" --exhaustive

gate-place-headless: test-interaction-safe gate-live-flat-pick-entry-oracle gate-live-flat-pick-position-matrix interaction_place_probe $(RELEASE_INTERACTION_PLACE_PROBE)
	@set -eu; \
	  probe_output="$$(./interaction_place_probe "$(INTERACTION_DEMO_PACK)" --json)"; \
	  printf '%s\n' "$$probe_output"; \
	  INTERACTION_PLACE_PROBE_JSON="$$probe_output" \
	  INTERACTION_PLACE_PROBE_PACK="$(INTERACTION_DEMO_PACK)" \
	    python -m unittest tests.python.test_place_probe -v; \
	  release_probe_output="$$( "$(RELEASE_INTERACTION_PLACE_PROBE)" "$(INTERACTION_DEMO_PACK)" --json )"; \
	  if test "$$probe_output" != "$$release_probe_output"; then \
	    echo "ERROR place probe normal/release matrix mismatch" >&2; \
	    echo "normal:  $$probe_output" >&2; \
	    echo "release: $$release_probe_output" >&2; \
	    exit 1; \
	  fi; \
	  printf '%s\n' "$$release_probe_output"; \
	  INTERACTION_PLACE_PROBE_JSON="$$release_probe_output" \
	  INTERACTION_PLACE_PROBE_PACK="$(INTERACTION_DEMO_PACK)" \
	    python -m unittest tests.python.test_place_probe -v

gate-playable-placement:
	$(MAKE) demo-interaction-pack
	$(MAKE) gate-place-headless
	$(MAKE) bootstrap-raylib
	$(MAKE) controller
	mkdir -p "$(PLACEMENT_EVIDENCE_DIR)"
	@display="$${DISPLAY:-:1}"; \
	  timeout --signal=TERM --kill-after=1s 5s \
	    xdpyinfo -display "$$display" >/dev/null
	@features_before="$$(sha256sum resources/features.bin)" || exit 1; \
	  interaction_database_before="$$(sha256sum "$(INTERACTION_DEMO_PACK)/interaction_database.bin")" || exit 1; \
	  interaction_features_before="$$(sha256sum "$(INTERACTION_DEMO_PACK)/interaction_features.bin")" || exit 1; \
	  display="$${DISPLAY:-:1}"; \
	  DISPLAY="$$display" \
	  MM_INTERACTION_PLACE_AUTODEMO=1 \
	  MM_INTERACTION_PACK="$(INTERACTION_DEMO_PACK)" \
	  MM_FEATURES_OUTPUT="$(PLACEMENT_FEATURES_OUTPUT)" \
	  MM_INTERACTION_LOG="$(PLACEMENT_LOG_PATH)" \
	  MM_INTERACTION_SCREENSHOT="$(PLACEMENT_SCREENSHOT_PATH)" \
	  timeout --signal=TERM --kill-after=5s 60s ./controller; \
	  controller_status=$$?; \
	  features_after="$$(sha256sum resources/features.bin)" || exit 1; \
	  interaction_database_after="$$(sha256sum "$(INTERACTION_DEMO_PACK)/interaction_database.bin")" || exit 1; \
	  interaction_features_after="$$(sha256sum "$(INTERACTION_DEMO_PACK)/interaction_features.bin")" || exit 1; \
	  if test "$$features_before" != "$$features_after"; then \
	    echo "ERROR resources/features.bin changed during placement gate" >&2; \
	    exit 1; \
	  fi; \
	  if test "$$interaction_database_before" != "$$interaction_database_after"; then \
	    echo "ERROR $(INTERACTION_DEMO_PACK)/interaction_database.bin changed during placement gate" >&2; \
	    exit 1; \
	  fi; \
	  if test "$$interaction_features_before" != "$$interaction_features_after"; then \
	    echo "ERROR $(INTERACTION_DEMO_PACK)/interaction_features.bin changed during placement gate" >&2; \
	    exit 1; \
	  fi; \
	  if test "$$controller_status" -ne 0; then \
	    echo "ERROR placement autodemo exited $$controller_status" >&2; \
	    exit "$$controller_status"; \
	  fi
	PLACEMENT_LOG="$(PLACEMENT_LOG_PATH)" \
	PLACEMENT_SCREENSHOT="$(PLACEMENT_SCREENSHOT_PATH)" \
	  python -m unittest \
	    tests.python.test_playable_placement_evidence -v

gate-playable-interaction: test-interaction-safe demo-interaction-pack \
  interaction_probe interaction_runtime_probe
	$(MAKE) bootstrap-raylib
	$(MAKE) controller
	mkdir -p "$(PLAYABLE_EVIDENCE_DIR)"
	@display="$${DISPLAY:-:1}"; \
	  timeout --signal=TERM --kill-after=1s 5s \
	    xdpyinfo -display "$$display" >/dev/null
	./interaction_probe "$(INTERACTION_DEMO_PACK)" --json
	./interaction_runtime_probe "$(INTERACTION_DEMO_PACK)" --json
	@features_before="$$(sha256sum resources/features.bin)" || exit 1; \
	  display="$${DISPLAY:-:1}"; \
	  DISPLAY="$$display" \
	  MM_INTERACTION_AUTODEMO=1 \
	  MM_INTERACTION_PACK="$(INTERACTION_DEMO_PACK)" \
	  MM_FEATURES_OUTPUT="$(PLAYABLE_FEATURES_OUTPUT)" \
	  MM_INTERACTION_LOG="$(PLAYABLE_LOG_PATH)" \
	  MM_INTERACTION_SCREENSHOT="$(PLAYABLE_SCREENSHOT_PATH)" \
	  timeout --signal=TERM --kill-after=5s 45s ./controller; \
	  controller_status=$$?; \
	  features_after="$$(sha256sum resources/features.bin)" || exit 1; \
	  if test "$$features_before" != "$$features_after"; then \
	    echo "ERROR resources/features.bin changed during playable gate" >&2; \
	    exit 1; \
	  fi; \
	  if test "$$controller_status" -ne 0; then \
	    echo "ERROR playable autodemo exited $$controller_status" >&2; \
	    exit "$$controller_status"; \
	  fi
	PLAYABLE_LOG="$(PLAYABLE_LOG_PATH)" \
	PLAYABLE_SCREENSHOT="$(PLAYABLE_SCREENSHOT_PATH)" \
	  python -m unittest \
	    tests.python.test_playable_interaction_evidence -v
