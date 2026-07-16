CXX ?= c++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -Werror -pedantic
CPPFLAGS ?= -I../..
BUILD_DIR ?= ../build
BUILD_COMMIT := $(shell git -C ../.. rev-parse HEAD 2>/dev/null || echo unknown)
JOINT_CONTRACT := $(abspath ../configs/g1_joint_contract.json)

.PHONY: all clean mm_chunk_server

all: mm_chunk_server

mm_chunk_server: $(BUILD_DIR)/mm_chunk_server

$(BUILD_DIR)/mm_chunk_server: mm_chunk_server.cpp mm_chunk_protocol.h \
		mm_chunk_json.h g1_runtime.h g1_joint_projection.h \
		g1_database_validation.h g1_joint_contract_io.h \
		../configs/g1_joint_contract.json
	mkdir -p $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) \
		-DMM_CHUNK_BUILD_COMMIT=\"$(BUILD_COMMIT)\" \
		-DMM_CHUNK_DEFAULT_JOINT_CONTRACT=\"$(JOINT_CONTRACT)\" \
		mm_chunk_server.cpp -o $@

clean:
	rm -f $(BUILD_DIR)/mm_chunk_server
