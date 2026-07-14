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
INTERACTION_SOURCES := interaction_pose.cpp interaction_target.cpp \
  interaction_features.cpp interaction_matcher.cpp interaction_playback.cpp \
  interaction_ik.cpp interaction_attachment.cpp interaction_carry.cpp \
  interaction_runtime.cpp interaction_controller_adapter.cpp
SOURCE := controller.cpp $(INTERACTION_SOURCES)
HEADER = $(wildcard *.h)

.PHONY: all bootstrap-raylib controller

all: controller

bootstrap-raylib:
	./scripts/bootstrap_raylib.sh

controller: $(SOURCE) $(HEADER)
	$(CC) $(CONTROLLER_CXXFLAGS) -o $@$(EXT) $(SOURCE) $(CFLAGS) $(LIBS)

clean:
	rm controller$(EXT)

CXX ?= g++
CPP_TEST_FLAGS ?= -std=c++17 -Wall -Wextra -Werror -pedantic -I.
CPP_TEST_DIR := build/tests
TASK12_BUILD_DIR := build/task12
override SAFE_INTERACTION_QUERY_PROBE := build/task12/interaction_query_probe_safe
GRAIL_ROOT ?= /home/ubuntu/datasets/GRAIL/data/pickup_table
G1_XML ?= /home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml
DEMO_INTERACTION_LIMIT ?= 5
INTERACTION_DEMO_PACK ?= resources/g1_interaction
PLAYABLE_EVIDENCE_DIR ?= playable-evidence
PLAYABLE_LOG_PATH ?= $(PLAYABLE_EVIDENCE_DIR)/pickup.jsonl
PLAYABLE_SCREENSHOT_PATH ?= $(PLAYABLE_EVIDENCE_DIR)/pickup.png
PLAYABLE_FEATURES_OUTPUT ?= $(PLAYABLE_EVIDENCE_DIR)/locomotion-features.bin
CPP_TEST_BINS := $(CPP_TEST_DIR)/test_g1_skeleton
CPP_TEST_BINS += $(CPP_TEST_DIR)/test_interaction_database
CPP_TEST_BINS += $(CPP_TEST_DIR)/test_interaction_pose
CPP_TEST_BINS += $(CPP_TEST_DIR)/test_interaction_target
CPP_TEST_BINS += $(CPP_TEST_DIR)/test_interaction_features
CPP_TEST_BINS += $(CPP_TEST_DIR)/test_interaction_matcher
CPP_TEST_BINS += $(CPP_TEST_DIR)/test_interaction_playback
CPP_TEST_BINS += $(CPP_TEST_DIR)/test_interaction_ik
CPP_TEST_BINS += $(CPP_TEST_DIR)/test_interaction_attachment
CPP_TEST_BINS += $(CPP_TEST_DIR)/test_interaction_carry
CPP_TEST_BINS += $(CPP_TEST_DIR)/test_interaction_runtime
CPP_TEST_BINS += $(CPP_TEST_DIR)/test_interaction_controller_adapter

INTERACTION_RUNTIME_SOURCES := interaction_runtime.cpp
INTERACTION_RUNTIME_SOURCES += interaction_carry.cpp interaction_ik.cpp
INTERACTION_RUNTIME_SOURCES += interaction_attachment.cpp interaction_playback.cpp
INTERACTION_RUNTIME_SOURCES += interaction_matcher.cpp interaction_features.cpp
INTERACTION_RUNTIME_SOURCES += interaction_pose.cpp interaction_target.cpp

.PHONY: test-python test-cpp test-interaction
.PHONY: test-python-interaction-safe test-interaction-safe
.PHONY: demo-interaction-pack gate-playable-interaction

$(CPP_TEST_DIR):
	mkdir -p $@

$(TASK12_BUILD_DIR):
	mkdir -p $@

$(CPP_TEST_DIR)/test_g1_skeleton: tests/cpp/test_g1_skeleton.cpp g1_skeleton.h | $(CPP_TEST_DIR)
	$(CXX) $(CPP_TEST_FLAGS) $< -o $@

$(CPP_TEST_DIR)/test_interaction_database: tests/cpp/test_interaction_database.cpp interaction_database.h g1_skeleton.h | $(CPP_TEST_DIR)
	$(CXX) $(CPP_TEST_FLAGS) $< -o $@

$(CPP_TEST_DIR)/test_interaction_pose: tests/cpp/test_interaction_pose.cpp interaction_pose.cpp interaction_pose.h interaction_database.h g1_skeleton.h vec.h quat.h | $(CPP_TEST_DIR)
	$(CXX) $(CPP_TEST_FLAGS) tests/cpp/test_interaction_pose.cpp interaction_pose.cpp -o $@

$(CPP_TEST_DIR)/test_interaction_target: tests/cpp/test_interaction_target.cpp interaction_target.cpp interaction_target.h interaction_pose.h vec.h quat.h | $(CPP_TEST_DIR)
	$(CXX) $(CPP_TEST_FLAGS) tests/cpp/test_interaction_target.cpp interaction_target.cpp -o $@

$(CPP_TEST_DIR)/test_interaction_features: tests/cpp/test_interaction_features.cpp interaction_features.cpp interaction_features.h interaction_pose.cpp interaction_pose.h interaction_target.h interaction_database.h g1_skeleton.h vec.h quat.h | $(CPP_TEST_DIR)
	$(CXX) $(CPP_TEST_FLAGS) tests/cpp/test_interaction_features.cpp interaction_features.cpp interaction_pose.cpp -o $@

$(CPP_TEST_DIR)/test_interaction_matcher: tests/cpp/test_interaction_matcher.cpp tests/cpp/interaction_runtime_fixture.h interaction_matcher.cpp interaction_matcher.h interaction_features.cpp interaction_features.h interaction_pose.cpp interaction_pose.h interaction_target.cpp interaction_target.h interaction_database.h g1_skeleton.h vec.h quat.h | $(CPP_TEST_DIR)
	$(CXX) $(CPP_TEST_FLAGS) tests/cpp/test_interaction_matcher.cpp interaction_matcher.cpp interaction_features.cpp interaction_pose.cpp interaction_target.cpp -o $@

$(CPP_TEST_DIR)/test_interaction_playback: tests/cpp/test_interaction_playback.cpp tests/cpp/interaction_runtime_fixture.h interaction_playback.cpp interaction_playback.h interaction_matcher.cpp interaction_matcher.h interaction_features.cpp interaction_features.h interaction_pose.cpp interaction_pose.h interaction_target.cpp interaction_target.h interaction_database.h g1_skeleton.h vec.h quat.h | $(CPP_TEST_DIR)
	$(CXX) $(CPP_TEST_FLAGS) tests/cpp/test_interaction_playback.cpp interaction_playback.cpp interaction_matcher.cpp interaction_features.cpp interaction_pose.cpp interaction_target.cpp -o $@

$(CPP_TEST_DIR)/test_interaction_ik: tests/cpp/test_interaction_ik.cpp interaction_ik.cpp interaction_ik.h g1_arm_joint_metadata.h interaction_pose.cpp interaction_pose.h interaction_matcher.h interaction_features.h interaction_target.h interaction_database.h g1_skeleton.h vec.h quat.h | $(CPP_TEST_DIR)
	$(CXX) $(CPP_TEST_FLAGS) tests/cpp/test_interaction_ik.cpp interaction_ik.cpp interaction_pose.cpp -o $@

$(CPP_TEST_DIR)/test_interaction_attachment: tests/cpp/test_interaction_attachment.cpp interaction_attachment.cpp interaction_attachment.h interaction_pose.cpp interaction_pose.h interaction_target.cpp interaction_target.h interaction_matcher.h interaction_features.h interaction_database.h g1_skeleton.h vec.h quat.h | $(CPP_TEST_DIR)
	$(CXX) $(CPP_TEST_FLAGS) tests/cpp/test_interaction_attachment.cpp interaction_attachment.cpp interaction_pose.cpp interaction_target.cpp -o $@

$(CPP_TEST_DIR)/test_interaction_carry: tests/cpp/test_interaction_carry.cpp tests/cpp/interaction_runtime_fixture.h interaction_carry.cpp interaction_carry.h interaction_ik.cpp interaction_ik.h g1_arm_joint_metadata.h interaction_matcher.cpp interaction_matcher.h interaction_features.cpp interaction_features.h interaction_pose.cpp interaction_pose.h interaction_target.cpp interaction_target.h interaction_database.h g1_skeleton.h vec.h quat.h | $(CPP_TEST_DIR)
	$(CXX) $(CPP_TEST_FLAGS) -c tests/cpp/test_interaction_carry.cpp -o $(CPP_TEST_DIR)/test_interaction_carry.o
	$(CXX) $(CPP_TEST_FLAGS) $(CPP_TEST_DIR)/test_interaction_carry.o interaction_carry.cpp interaction_ik.cpp interaction_matcher.cpp interaction_features.cpp interaction_pose.cpp interaction_target.cpp -o $@

$(CPP_TEST_DIR)/test_interaction_runtime: tests/cpp/test_interaction_runtime.cpp tests/cpp/interaction_runtime_fixture.h interaction_runtime.h $(INTERACTION_RUNTIME_SOURCES) interaction_carry.h interaction_ik.h g1_arm_joint_metadata.h interaction_attachment.h interaction_playback.h interaction_matcher.h interaction_features.h interaction_pose.h interaction_target.h interaction_database.h g1_skeleton.h vec.h quat.h | $(CPP_TEST_DIR)
	$(CXX) $(CPP_TEST_FLAGS) -c tests/cpp/test_interaction_runtime.cpp -o $(CPP_TEST_DIR)/test_interaction_runtime.o
	$(CXX) $(CPP_TEST_FLAGS) $(CPP_TEST_DIR)/test_interaction_runtime.o $(INTERACTION_RUNTIME_SOURCES) -o $@

$(CPP_TEST_DIR)/test_interaction_controller_adapter: tests/cpp/test_interaction_controller_adapter.cpp tests/cpp/interaction_runtime_fixture.h interaction_controller_adapter.cpp interaction_controller_adapter.h interaction_runtime.h $(INTERACTION_RUNTIME_SOURCES) interaction_carry.h interaction_ik.h g1_arm_joint_metadata.h interaction_attachment.h interaction_playback.h interaction_matcher.h interaction_features.h interaction_pose.h interaction_target.h interaction_database.h g1_skeleton.h vec.h quat.h | $(CPP_TEST_DIR)
	$(CXX) $(CPP_TEST_FLAGS) tests/cpp/test_interaction_controller_adapter.cpp interaction_controller_adapter.cpp $(INTERACTION_RUNTIME_SOURCES) -o $@

interaction_probe: interaction_probe.cpp interaction_database.h g1_skeleton.h
	$(CXX) $(CPP_TEST_FLAGS) $< -o $@

interaction_query_probe: interaction_query_probe.cpp interaction_features.cpp interaction_features.h interaction_pose.cpp interaction_pose.h interaction_target.h interaction_database.h g1_skeleton.h vec.h quat.h
	$(CXX) $(CPP_TEST_FLAGS) interaction_query_probe.cpp interaction_features.cpp interaction_pose.cpp -o $@

$(SAFE_INTERACTION_QUERY_PROBE): interaction_query_probe.cpp interaction_features.cpp interaction_features.h interaction_pose.cpp interaction_pose.h interaction_target.h interaction_database.h g1_skeleton.h vec.h quat.h | $(TASK12_BUILD_DIR)
	$(CXX) $(CPP_TEST_FLAGS) interaction_query_probe.cpp interaction_features.cpp interaction_pose.cpp -o $@

interaction_runtime_probe: interaction_runtime_probe.cpp interaction_runtime.h $(INTERACTION_RUNTIME_SOURCES) interaction_carry.h interaction_ik.h g1_arm_joint_metadata.h interaction_attachment.h interaction_playback.h interaction_matcher.h interaction_features.h interaction_pose.h interaction_target.h interaction_database.h g1_skeleton.h vec.h quat.h
	$(CXX) $(CPP_TEST_FLAGS) interaction_runtime_probe.cpp $(INTERACTION_RUNTIME_SOURCES) -o $@

test-python: interaction_probe interaction_query_probe
	python -m unittest discover -s tests/python -t . -v

test-cpp: $(CPP_TEST_BINS)
	@for test_bin in $(CPP_TEST_BINS); do $$test_bin || exit 1; done

test-interaction: test-python test-cpp

test-python-interaction-safe: interaction_probe $(SAFE_INTERACTION_QUERY_PROBE)
	MM_INTERACTION_QUERY_PROBE="$(abspath $(SAFE_INTERACTION_QUERY_PROBE))" \
	  G1_XML="$(G1_XML)" \
	  python -m unittest discover -s tests/python -t . -v

test-interaction-safe: test-python-interaction-safe test-cpp

demo-interaction-pack:
	python -m resources.build_g1_interaction_database \
	  --source-root "$(GRAIL_ROOT)" \
	  --g1-xml "$(G1_XML)" \
	  --output "$(INTERACTION_DEMO_PACK)" \
	  --target-fps 25 \
	  --limit "$(DEMO_INTERACTION_LIMIT)" \
	  --heldout-count 1
	python -m resources.validate_g1_interaction_database \
	  --input "$(INTERACTION_DEMO_PACK)"

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
