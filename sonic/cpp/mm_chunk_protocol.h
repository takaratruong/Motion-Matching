#pragma once

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

static constexpr int MM_CHUNK_PROTOCOL_VERSION = 1;
static constexpr int MM_CHUNK_SOURCE_RATE_HZ = 25;
static constexpr int MM_CHUNK_SOURCE_INTERVALS = 10;
static constexpr int MM_CHUNK_JOINT_COUNT = 29;
static constexpr int MM_CHUNK_TERRAIN_SAMPLE_COUNT = 4;

struct mm_chunk_error
{
    std::string code;
    std::string message;
};

static inline bool mm_chunk_fail(
    mm_chunk_error& error,
    const char* code,
    const std::string& message)
{
    error.code = code != nullptr ? code : "internal_error";
    error.message = message;
    return false;
}

static inline void mm_chunk_clear_error(mm_chunk_error& error)
{
    error.code.clear();
    error.message.clear();
}

struct mm_chunk_reset_request
{
    std::string session_id;
    std::string scene_id;
    std::string route_id;
    float terrain_weight = 0.0f;
};

struct mm_chunk_generate_request
{
    std::string session_id;
    std::string candidate_id;
    bool predecessor_is_null = true;
    std::string predecessor_id;
    int source_intervals = 0;
    float requested_velocity_holden[3] = {};
    float desired_heading_holden_wxyz[4] = {};
};

struct mm_chunk_boundary
{
    float joint_position_source[MM_CHUNK_JOINT_COUNT] = {};
    float joint_velocity_source[MM_CHUNK_JOINT_COUNT] = {};
    float physical_pelvis_position_holden[3] = {};
    float physical_pelvis_orientation_holden[4] = {};
    float virtual_root_position_holden[3] = {};
    float virtual_root_orientation_holden[4] = {};
};

static inline bool operator==(
    const mm_chunk_boundary& first,
    const mm_chunk_boundary& second)
{
    for (int index = 0; index < MM_CHUNK_JOINT_COUNT; ++index) {
        if (first.joint_position_source[index] !=
                second.joint_position_source[index] ||
            first.joint_velocity_source[index] !=
                second.joint_velocity_source[index]) {
            return false;
        }
    }
    for (int axis = 0; axis < 3; ++axis) {
        if (first.physical_pelvis_position_holden[axis] !=
                second.physical_pelvis_position_holden[axis] ||
            first.virtual_root_position_holden[axis] !=
                second.virtual_root_position_holden[axis]) {
            return false;
        }
    }
    for (int axis = 0; axis < 4; ++axis) {
        if (first.physical_pelvis_orientation_holden[axis] !=
                second.physical_pelvis_orientation_holden[axis] ||
            first.virtual_root_orientation_holden[axis] !=
                second.virtual_root_orientation_holden[axis]) {
            return false;
        }
    }
    return true;
}

static inline bool operator!=(
    const mm_chunk_boundary& first,
    const mm_chunk_boundary& second)
{
    return !(first == second);
}

struct mm_chunk_step_diagnostic
{
    int selected_database_frame = -1;
    bool searched = false;
    bool transitioned = false;
    float terrain_cost = 0.0f;
    float applied_velocity_holden[3] = {};
    float terrain_values[MM_CHUNK_TERRAIN_SAMPLE_COUNT] = {};
    float terrain_points_holden[MM_CHUNK_TERRAIN_SAMPLE_COUNT][3] = {};
    float support_height = 0.0f;
    float support_target = 0.0f;
};

static inline bool operator==(
    const mm_chunk_step_diagnostic& first,
    const mm_chunk_step_diagnostic& second)
{
    if (first.selected_database_frame != second.selected_database_frame ||
        first.searched != second.searched ||
        first.transitioned != second.transitioned ||
        first.terrain_cost != second.terrain_cost ||
        first.support_height != second.support_height ||
        first.support_target != second.support_target) {
        return false;
    }
    for (int axis = 0; axis < 3; ++axis) {
        if (first.applied_velocity_holden[axis] !=
            second.applied_velocity_holden[axis]) {
            return false;
        }
    }
    for (int sample = 0; sample < MM_CHUNK_TERRAIN_SAMPLE_COUNT; ++sample) {
        if (first.terrain_values[sample] != second.terrain_values[sample]) {
            return false;
        }
        for (int axis = 0; axis < 3; ++axis) {
            if (first.terrain_points_holden[sample][axis] !=
                second.terrain_points_holden[sample][axis]) {
                return false;
            }
        }
    }
    return true;
}

struct mm_chunk_candidate
{
    std::vector<mm_chunk_boundary> boundaries;
    std::vector<mm_chunk_step_diagnostic> steps;
};

static inline bool operator==(
    const mm_chunk_candidate& first,
    const mm_chunk_candidate& second)
{
    return first.boundaries == second.boundaries &&
           first.steps == second.steps;
}

template<typename State>
struct mm_chunk_session
{
    bool reset = false;
    std::string session_id;
    std::string active_candidate_id;
    State active;
    bool candidate_ready = false;
    std::string candidate_id;
    State candidate;
};

template<typename Adapter>
class mm_chunk_protocol
{
public:
    using state_type = typename Adapter::state_type;
    using session_type = mm_chunk_session<state_type>;

    explicit mm_chunk_protocol(Adapter& adapter) : adapter_(adapter) {}

    const session_type& session() const { return session_; }
    bool closed() const { return closed_; }
    bool hello_received() const { return hello_received_; }

    bool hello(mm_chunk_error& error)
    {
        mm_chunk_clear_error(error);
        if (closed_) {
            return mm_chunk_fail(error, "closed", "protocol is closed");
        }
        if (hello_received_) {
            return mm_chunk_fail(
                error,
                "hello_already_received",
                "hello was already received");
        }
        hello_received_ = true;
        return true;
    }

    bool reset(
        const mm_chunk_reset_request& request,
        mm_chunk_boundary& initial_boundary,
        mm_chunk_error& error)
    {
        mm_chunk_clear_error(error);
        if (!ready_for_operation(error)) return false;
        if (session_.candidate_ready) {
            return candidate_outstanding(error);
        }
        if (request.session_id.empty() || request.scene_id.empty() ||
            request.route_id.empty()) {
            return mm_chunk_fail(
                error,
                "invalid_reset",
                "reset identifiers must be nonempty");
        }

        state_type next;
        mm_chunk_boundary next_boundary;
        std::string adapter_error;
        if (!adapter_.reset(next, next_boundary, request, adapter_error)) {
            return mm_chunk_fail(
                error,
                "reset_failed",
                adapter_error.empty() ? "reset adapter failed" : adapter_error);
        }

        adapter_.swap(session_.active, next);
        clear_candidate_state();
        session_.reset = true;
        session_.session_id = request.session_id;
        session_.active_candidate_id.clear();
        initial_boundary = next_boundary;
        return true;
    }

    bool generate(
        const mm_chunk_generate_request& request,
        mm_chunk_candidate& output,
        mm_chunk_error& error)
    {
        mm_chunk_clear_error(error);
        if (!ready_for_operation(error)) return false;
        if (!session_.reset) {
            return mm_chunk_fail(
                error, "reset_required", "reset is required before generate");
        }
        if (session_.candidate_ready) {
            return candidate_outstanding(error);
        }
        if (request.session_id != session_.session_id) {
            return mm_chunk_fail(
                error,
                "session_mismatch",
                "request session does not match active session");
        }
        if (request.candidate_id.empty()) {
            return mm_chunk_fail(
                error, "invalid_candidate", "candidate ID must be nonempty");
        }
        if (request.source_intervals != MM_CHUNK_SOURCE_INTERVALS) {
            return mm_chunk_fail(
                error,
                "invalid_intervals",
                "source_intervals must be exactly 10");
        }
        const bool expected_null = session_.active_candidate_id.empty();
        if (request.predecessor_is_null != expected_null ||
            (!expected_null &&
             request.predecessor_id != session_.active_candidate_id)) {
            return mm_chunk_fail(
                error,
                "predecessor_mismatch",
                "predecessor does not match active candidate");
        }
        if (!expected_null &&
            request.candidate_id == session_.active_candidate_id) {
            return mm_chunk_fail(
                error,
                "invalid_candidate",
                "candidate ID must differ from its predecessor");
        }

        state_type next;
        std::string adapter_error;
        if (!adapter_.clone(next, session_.active, adapter_error)) {
            return mm_chunk_fail(
                error,
                "generation_failed",
                adapter_error.empty() ? "state clone failed" : adapter_error);
        }

        mm_chunk_candidate generated;
        generated.boundaries.reserve(
            static_cast<std::size_t>(MM_CHUNK_SOURCE_INTERVALS + 1));
        generated.steps.reserve(
            static_cast<std::size_t>(MM_CHUNK_SOURCE_INTERVALS));
        mm_chunk_boundary boundary;
        if (!adapter_.observe(boundary, next, adapter_error)) {
            return mm_chunk_fail(
                error,
                "generation_failed",
                adapter_error.empty()
                    ? "boundary observation failed"
                    : adapter_error);
        }
        generated.boundaries.push_back(boundary);
        for (int step = 0; step < MM_CHUNK_SOURCE_INTERVALS; ++step) {
            mm_chunk_step_diagnostic diagnostic;
            if (!adapter_.advance(
                    diagnostic, next, request, step, adapter_error)) {
                return mm_chunk_fail(
                    error,
                    "generation_failed",
                    adapter_error.empty()
                        ? "matcher advance failed"
                        : adapter_error);
            }
            if (!adapter_.observe(boundary, next, adapter_error)) {
                return mm_chunk_fail(
                    error,
                    "generation_failed",
                    adapter_error.empty()
                        ? "boundary observation failed"
                        : adapter_error);
            }
            generated.steps.push_back(diagnostic);
            generated.boundaries.push_back(boundary);
        }

        adapter_.swap(session_.candidate, next);
        session_.candidate_ready = true;
        session_.candidate_id = request.candidate_id;
        output = std::move(generated);
        return true;
    }

    bool commit(
        const std::string& session_id,
        const std::string& candidate_id,
        mm_chunk_error& error)
    {
        mm_chunk_clear_error(error);
        if (!ready_for_operation(error)) return false;
        if (!session_.reset) {
            return mm_chunk_fail(
                error, "reset_required", "reset is required before commit");
        }
        if (session_id != session_.session_id) {
            return mm_chunk_fail(
                error,
                "session_mismatch",
                "request session does not match active session");
        }
        if (!session_.candidate_ready) {
            return mm_chunk_fail(
                error, "no_candidate", "no candidate is outstanding");
        }
        if (session_.candidate_id != candidate_id) {
            return mm_chunk_fail(
                error,
                "candidate_mismatch",
                "commit candidate does not match outstanding candidate");
        }
        adapter_.swap(session_.active, session_.candidate);
        session_.active_candidate_id = session_.candidate_id;
        session_.candidate_id.clear();
        session_.candidate_ready = false;
        release_state(session_.candidate);
        return true;
    }

    bool abort(
        const std::string& session_id,
        const std::string& candidate_id,
        mm_chunk_error& error)
    {
        mm_chunk_clear_error(error);
        if (!ready_for_operation(error)) return false;
        if (!session_.reset) {
            return mm_chunk_fail(
                error, "reset_required", "reset is required before abort");
        }
        if (session_id != session_.session_id) {
            return mm_chunk_fail(
                error,
                "session_mismatch",
                "request session does not match active session");
        }
        if (!session_.candidate_ready) {
            return mm_chunk_fail(
                error, "no_candidate", "no candidate is outstanding");
        }
        if (session_.candidate_id != candidate_id) {
            return mm_chunk_fail(
                error,
                "candidate_mismatch",
                "abort candidate does not match outstanding candidate");
        }
        session_.candidate_id.clear();
        session_.candidate_ready = false;
        release_state(session_.candidate);
        return true;
    }

    bool close(mm_chunk_error& error)
    {
        mm_chunk_clear_error(error);
        if (!ready_for_operation(error)) return false;
        if (session_.candidate_ready) {
            return candidate_outstanding(error);
        }
        closed_ = true;
        return true;
    }

private:
    bool ready_for_operation(mm_chunk_error& error) const
    {
        if (closed_) {
            return mm_chunk_fail(error, "closed", "protocol is closed");
        }
        if (!hello_received_) {
            return mm_chunk_fail(
                error,
                "hello_required",
                "hello must be the first request");
        }
        return true;
    }

    bool candidate_outstanding(mm_chunk_error& error) const
    {
        return mm_chunk_fail(
            error,
            "candidate_outstanding",
            "candidate " + session_.candidate_id +
                " must be committed or aborted");
    }

    void release_state(state_type& state)
    {
        state_type discarded;
        adapter_.swap(state, discarded);
    }

    void clear_candidate_state()
    {
        if (session_.candidate_ready) release_state(session_.candidate);
        session_.candidate_ready = false;
        session_.candidate_id.clear();
    }

    Adapter& adapter_;
    session_type session_;
    bool hello_received_ = false;
    bool closed_ = false;
};
