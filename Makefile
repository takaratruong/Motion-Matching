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

SOURCE = controller.cpp
HEADER = $(wildcard *.h)

.PHONY: all bootstrap-raylib controller

all: controller

bootstrap-raylib:
	./scripts/bootstrap_raylib.sh

controller: $(SOURCE) $(HEADER)
	$(CC) -o $@$(EXT) $(SOURCE) $(CFLAGS) $(LIBS) 

clean:
	rm controller$(EXT)

CXX ?= g++
CPP_TEST_FLAGS ?= -std=c++17 -Wall -Wextra -Werror -pedantic -I.
CPP_TEST_DIR := build/tests
CPP_TEST_BINS := $(CPP_TEST_DIR)/test_g1_skeleton
CPP_TEST_BINS += $(CPP_TEST_DIR)/test_interaction_database

.PHONY: test-python test-cpp test-interaction

$(CPP_TEST_DIR):
	mkdir -p $@

$(CPP_TEST_DIR)/test_g1_skeleton: tests/cpp/test_g1_skeleton.cpp g1_skeleton.h | $(CPP_TEST_DIR)
	$(CXX) $(CPP_TEST_FLAGS) $< -o $@

$(CPP_TEST_DIR)/test_interaction_database: tests/cpp/test_interaction_database.cpp interaction_database.h g1_skeleton.h | $(CPP_TEST_DIR)
	$(CXX) $(CPP_TEST_FLAGS) $< -o $@

interaction_probe: interaction_probe.cpp interaction_database.h g1_skeleton.h
	$(CXX) $(CPP_TEST_FLAGS) $< -o $@

test-python: interaction_probe
	python -m unittest discover -s tests/python -t . -v

test-cpp: $(CPP_TEST_BINS)
	@for test_bin in $(CPP_TEST_BINS); do $$test_bin || exit 1; done

test-interaction: test-python test-cpp
