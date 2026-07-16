#include "sonic/cpp/mm_chunk_protocol.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

static void check(bool condition, const char* expression, int line)
{
    if (!condition) {
        std::fprintf(
            stderr,
            "MM chunk protocol check failed at line %d: %s\n",
            line,
            expression);
        std::exit(1);
    }
}

#define CHECK(expression) check((expression), #expression, __LINE__)

struct fake_state
{
    int frame = 0;
    std::vector<int> history;
};

struct fake_reset_context
{
    std::string scene_id;
};

static std::uint64_t fake_hash(const fake_state& state)
{
    std::uint64_t hash = UINT64_C(1469598103934665603);
    const auto add = [&](std::uint64_t value, std::uint64_t& output) {
        for (int byte = 0; byte < 8; ++byte) {
            output ^= (value >> (byte * 8)) & UINT64_C(0xff);
            output *= UINT64_C(1099511628211);
        }
    };
    add(static_cast<std::uint64_t>(state.frame), hash);
    add(static_cast<std::uint64_t>(state.history.size()), hash);
    for (int value : state.history) {
        add(static_cast<std::uint64_t>(value), hash);
    }
    return hash;
}

struct fake_adapter
{
    using state_type = fake_state;
    using reset_context_type = fake_reset_context;

    int clone_count = 0;
    int swap_count = 0;
    int fail_step = -1;
    int observe_count = 0;
    int fail_observe_call = -1;
    bool fail_clone = false;
    bool fail_reset = false;
    int reset_publish_count = 0;
    std::string published_scene_id;

    bool prepare_reset(
        state_type& state,
        mm_chunk_boundary& boundary,
        reset_context_type& context,
        const mm_chunk_reset_request& request,
        std::string& error)
    {
        state.frame = 100 + static_cast<int>(request.scene_id.size());
        state.history.assign(1, state.frame);
        context.scene_id = request.scene_id;
        if (fail_reset) {
            error = "injected reset failure";
            return false;
        }
        return observe(boundary, state, error);
    }

    void publish_reset(reset_context_type& context)
    {
        ++reset_publish_count;
        published_scene_id.swap(context.scene_id);
    }

    bool reset(
        state_type& state,
        mm_chunk_boundary& boundary,
        const mm_chunk_reset_request& request,
        std::string& error)
    {
        reset_context_type context;
        if (!prepare_reset(state, boundary, context, request, error)) {
            return false;
        }
        publish_reset(context);
        return true;
    }

    bool clone(
        state_type& destination,
        const state_type& source,
        std::string& error)
    {
        ++clone_count;
        destination = source;
        if (fail_clone) {
            destination.frame += 7000;
            destination.history.push_back(destination.frame);
            error = "injected clone failure after destination mutation";
            return false;
        }
        return true;
    }

    void swap(state_type& first, state_type& second)
    {
        ++swap_count;
        using std::swap;
        swap(first, second);
    }

    bool observe(
        mm_chunk_boundary& boundary,
        const state_type& state,
        std::string& error)
    {
        ++observe_count;
        if (observe_count == fail_observe_call) {
            error = "injected observation failure at configured call";
            return false;
        }
        boundary = mm_chunk_boundary();
        for (int joint = 0; joint < MM_CHUNK_JOINT_COUNT; ++joint) {
            boundary.joint_position_source[joint] =
                static_cast<float>(state.frame * 100 + joint);
            boundary.joint_velocity_source[joint] =
                static_cast<float>(state.frame - joint);
        }
        boundary.physical_pelvis_position_holden[0] =
            static_cast<float>(state.frame);
        boundary.physical_pelvis_orientation_holden[0] = 1.0f;
        boundary.virtual_root_position_holden[2] =
            static_cast<float>(state.frame);
        boundary.virtual_root_orientation_holden[0] = 1.0f;
        return true;
    }

    bool advance(
        mm_chunk_step_diagnostic& diagnostic,
        state_type& state,
        const mm_chunk_generate_request& request,
        int step,
        std::string& error)
    {
        if (step == fail_step) {
            state.frame += 9000;
            state.history.push_back(state.frame);
            error = "injected step failure";
            return false;
        }
        ++state.frame;
        state.history.push_back(state.frame);
        diagnostic = mm_chunk_step_diagnostic();
        diagnostic.selected_database_frame = state.frame;
        diagnostic.searched = (step % 2) == 0;
        diagnostic.transitioned = step == 5;
        diagnostic.terrain_cost =
            request.requested_velocity_holden[2] + static_cast<float>(step);
        diagnostic.applied_velocity_holden[0] =
            request.requested_velocity_holden[0];
        diagnostic.applied_velocity_holden[1] =
            request.requested_velocity_holden[1];
        diagnostic.applied_velocity_holden[2] =
            request.requested_velocity_holden[2];
        diagnostic.support_height = static_cast<float>(state.frame);
        diagnostic.support_target = static_cast<float>(state.frame + 1);
        for (int sample = 0; sample < MM_CHUNK_TERRAIN_SAMPLE_COUNT;
             ++sample) {
            diagnostic.terrain_values[sample] =
                static_cast<float>(state.frame + sample);
            for (int axis = 0; axis < 3; ++axis) {
                diagnostic.terrain_points_holden[sample][axis] =
                    static_cast<float>(state.frame + sample + axis);
            }
        }
        return true;
    }
};

using fake_engine = mm_chunk_protocol<fake_adapter>;

static mm_chunk_reset_request reset_request(const std::string& session = "s1")
{
    mm_chunk_reset_request request;
    request.session_id = session;
    request.scene_id = "fixture-scene";
    request.route_id = "fixture-route";
    request.terrain_weight = 4.0f;
    return request;
}

static mm_chunk_generate_request generate_request(
    const std::string& candidate,
    const bool predecessor_is_null,
    const std::string& predecessor = std::string())
{
    mm_chunk_generate_request request;
    request.session_id = "s1";
    request.candidate_id = candidate;
    request.predecessor_is_null = predecessor_is_null;
    request.predecessor_id = predecessor;
    request.source_intervals = MM_CHUNK_SOURCE_INTERVALS;
    request.requested_velocity_holden[2] = 0.5f;
    request.desired_heading_holden_wxyz[0] = 1.0f;
    return request;
}

static void require_error(
    bool result,
    const mm_chunk_error& error,
    const char* code)
{
    CHECK(!result);
    CHECK(error.code == code);
    CHECK(!error.message.empty());
}

enum matrix_state
{
    matrix_pre_hello,
    matrix_hello_no_reset,
    matrix_reset_no_candidate,
    matrix_outstanding_candidate,
    matrix_committed,
    matrix_aborted,
    matrix_closed,
    matrix_state_count,
};

enum matrix_operation
{
    matrix_hello,
    matrix_reset,
    matrix_generate,
    matrix_commit,
    matrix_abort,
    matrix_close,
    matrix_operation_count,
};

static void setup_matrix_state(
    fake_engine& protocol,
    matrix_state state,
    mm_chunk_error& error)
{
    if (state == matrix_pre_hello) return;
    CHECK(protocol.hello(error));
    if (state == matrix_hello_no_reset) return;
    if (state == matrix_closed) {
        CHECK(protocol.close(error));
        return;
    }
    mm_chunk_boundary initial;
    CHECK(protocol.reset(reset_request(), initial, error));
    if (state == matrix_reset_no_candidate) return;
    mm_chunk_candidate candidate;
    CHECK(protocol.generate(
        generate_request("c000000", true), candidate, error));
    if (state == matrix_outstanding_candidate) return;
    if (state == matrix_committed) {
        CHECK(protocol.commit("s1", "c000000", error));
    } else {
        CHECK(state == matrix_aborted);
        CHECK(protocol.abort("s1", "c000000", error));
    }
}

static bool invoke_matrix_operation(
    fake_engine& protocol,
    matrix_state state,
    matrix_operation operation,
    mm_chunk_error& error)
{
    mm_chunk_boundary boundary;
    mm_chunk_candidate candidate;
    switch (operation) {
    case matrix_hello:
        return protocol.hello(error);
    case matrix_reset:
        return protocol.reset(reset_request("matrix-reset"), boundary, error);
    case matrix_generate:
        if (state == matrix_committed) {
            return protocol.generate(
                generate_request("c000001", false, "c000000"),
                candidate,
                error);
        }
        return protocol.generate(
            generate_request("c000000", true), candidate, error);
    case matrix_commit:
        return protocol.commit("s1", "c000000", error);
    case matrix_abort:
        return protocol.abort("s1", "c000000", error);
    default:
        return protocol.close(error);
    }
}

static void test_complete_protocol_state_matrix()
{
    static const char* const expected[matrix_state_count]
                                     [matrix_operation_count] = {
        {nullptr, "hello_required", "hello_required", "hello_required",
         "hello_required", "hello_required"},
        {"hello_already_received", nullptr, "reset_required",
         "reset_required", "reset_required", nullptr},
        {"hello_already_received", nullptr, nullptr, "no_candidate",
         "no_candidate", nullptr},
        {"hello_already_received", "candidate_outstanding",
         "candidate_outstanding", nullptr, nullptr,
         "candidate_outstanding"},
        {"hello_already_received", nullptr, nullptr, "no_candidate",
         "no_candidate", nullptr},
        {"hello_already_received", nullptr, nullptr, "no_candidate",
         "no_candidate", nullptr},
        {"closed", "closed", "closed", "closed", "closed", "closed"},
    };

    for (int state_index = 0; state_index < matrix_state_count;
         ++state_index) {
        for (int operation_index = 0;
             operation_index < matrix_operation_count;
             ++operation_index) {
            fake_adapter adapter;
            fake_engine protocol(adapter);
            mm_chunk_error error;
            const matrix_state state =
                static_cast<matrix_state>(state_index);
            setup_matrix_state(protocol, state, error);
            const bool result = invoke_matrix_operation(
                protocol,
                state,
                static_cast<matrix_operation>(operation_index),
                error);
            const char* code = expected[state_index][operation_index];
            if (code == nullptr) {
                CHECK(result);
                CHECK(error.code.empty());
            } else {
                require_error(result, error, code);
            }
        }
    }
}

static void test_preparation_is_inert_until_explicit_publication()
{
    fake_adapter adapter;
    fake_engine protocol(adapter);
    mm_chunk_error error;
    CHECK(protocol.hello(error));

    fake_engine::reset_preparation reset_prepared;
    CHECK(protocol.prepare_reset(
        reset_request(), reset_prepared, error));
    CHECK(reset_prepared.ready);
    CHECK(!protocol.session().reset);
    CHECK(protocol.session().session_id.empty());
    CHECK(adapter.reset_publish_count == 0);
    CHECK(adapter.published_scene_id.empty());
    CHECK(reset_prepared.boundary.physical_pelvis_position_holden[0] ==
          static_cast<float>(reset_prepared.state.frame));

    CHECK(protocol.publish_reset(reset_prepared, error));
    CHECK(!reset_prepared.ready);
    CHECK(protocol.session().reset);
    CHECK(protocol.session().session_id == "s1");
    CHECK(adapter.reset_publish_count == 1);
    CHECK(adapter.published_scene_id == "fixture-scene");
    require_error(
        protocol.publish_reset(reset_prepared, error),
        error,
        "invalid_preparation");

    const std::uint64_t active = fake_hash(protocol.session().active);
    const std::uint64_t candidate = fake_hash(protocol.session().candidate);
    fake_engine::generate_preparation generate_prepared;
    CHECK(protocol.prepare_generate(
        generate_request("c000000", true), generate_prepared, error));
    CHECK(generate_prepared.ready);
    CHECK(generate_prepared.candidate.boundaries.size() == 11u);
    CHECK(generate_prepared.candidate.steps.size() == 10u);
    CHECK(fake_hash(protocol.session().active) == active);
    CHECK(fake_hash(protocol.session().candidate) == candidate);
    CHECK(!protocol.session().candidate_ready);

    CHECK(protocol.publish_generate(generate_prepared, error));
    CHECK(!generate_prepared.ready);
    CHECK(fake_hash(protocol.session().active) == active);
    CHECK(protocol.session().candidate_ready);
    CHECK(protocol.session().candidate_id == "c000000");
    require_error(
        protocol.publish_generate(generate_prepared, error),
        error,
        "invalid_preparation");
}

static void test_legal_sequence_is_transactional_and_continuous()
{
    fake_adapter adapter;
    fake_engine protocol(adapter);
    mm_chunk_error error;
    mm_chunk_boundary initial;

    require_error(
        protocol.reset(reset_request(), initial, error),
        error,
        "hello_required");
    CHECK(protocol.hello(error));
    require_error(protocol.hello(error), error, "hello_already_received");
    CHECK(protocol.reset(reset_request(), initial, error));
    CHECK(protocol.session().reset);
    CHECK(protocol.session().session_id == "s1");
    CHECK(!protocol.session().candidate_ready);
    CHECK(initial.physical_pelvis_position_holden[0] ==
          static_cast<float>(protocol.session().active.frame));

    const std::uint64_t active_before = fake_hash(protocol.session().active);
    mm_chunk_candidate chunk;
    CHECK(protocol.generate(
        generate_request("c000000", true), chunk, error));
    CHECK(chunk.boundaries.size() ==
          static_cast<std::size_t>(MM_CHUNK_SOURCE_INTERVALS + 1));
    CHECK(chunk.steps.size() ==
          static_cast<std::size_t>(MM_CHUNK_SOURCE_INTERVALS));
    CHECK(chunk.steps.front().applied_velocity_holden[2] == 0.5f);
    CHECK(chunk.boundaries.front().physical_pelvis_position_holden[0] ==
          initial.physical_pelvis_position_holden[0]);
    CHECK(chunk.boundaries.back().physical_pelvis_position_holden[0] ==
          initial.physical_pelvis_position_holden[0] +
              static_cast<float>(MM_CHUNK_SOURCE_INTERVALS));
    CHECK(fake_hash(protocol.session().active) == active_before);
    CHECK(protocol.session().candidate_ready);
    CHECK(protocol.session().candidate_id == "c000000");
    CHECK(adapter.clone_count == 1);

    mm_chunk_candidate ignored;
    require_error(
        protocol.generate(
            generate_request("c000001", true), ignored, error),
        error,
        "candidate_outstanding");
    require_error(
        protocol.reset(reset_request("s2"), initial, error),
        error,
        "candidate_outstanding");
    require_error(
        protocol.commit("wrong-session", "c000000", error),
        error,
        "session_mismatch");
    require_error(
        protocol.commit("s1", "wrong-candidate", error),
        error,
        "candidate_mismatch");
    CHECK(fake_hash(protocol.session().active) == active_before);

    CHECK(protocol.abort("s1", "c000000", error));
    CHECK(!protocol.session().candidate_ready);
    CHECK(fake_hash(protocol.session().active) == active_before);

    mm_chunk_candidate regenerated;
    CHECK(protocol.generate(
        generate_request("c000000", true), regenerated, error));
    CHECK(chunk == regenerated);
    CHECK(fake_hash(protocol.session().active) == active_before);
    const std::uint64_t candidate_hash =
        fake_hash(protocol.session().candidate);
    CHECK(protocol.commit("s1", "c000000", error));
    CHECK(protocol.session().active_candidate_id == "c000000");
    CHECK(fake_hash(protocol.session().active) == candidate_hash);
    CHECK(fake_hash(protocol.session().active) != active_before);
    CHECK(!protocol.session().candidate_ready);
    std::printf(
        "transaction evidence: active_before=%016llx candidate=%016llx "
        "committed=%016llx boundaries=%zu steps=%zu "
        "abort_regenerate_equal=true\n",
        static_cast<unsigned long long>(active_before),
        static_cast<unsigned long long>(candidate_hash),
        static_cast<unsigned long long>(fake_hash(protocol.session().active)),
        chunk.boundaries.size(),
        chunk.steps.size());

    require_error(
        protocol.generate(
            generate_request("c000001", true), ignored, error),
        error,
        "predecessor_mismatch");
    require_error(
        protocol.generate(
            generate_request("c000001", false, "other"), ignored, error),
        error,
        "predecessor_mismatch");
    CHECK(protocol.generate(
        generate_request("c000001", false, "c000000"), ignored, error));
    CHECK(ignored.boundaries.front() == chunk.boundaries.back());
    require_error(protocol.close(error), error, "candidate_outstanding");
    CHECK(protocol.abort("s1", "c000001", error));
    CHECK(protocol.close(error));
    CHECK(protocol.closed());
    require_error(
        protocol.reset(reset_request(), initial, error), error, "closed");
}

static void test_failures_never_publish_or_mutate_active_state()
{
    fake_adapter adapter;
    fake_engine protocol(adapter);
    mm_chunk_error error;
    mm_chunk_boundary initial;
    CHECK(protocol.hello(error));
    CHECK(protocol.reset(reset_request(), initial, error));
    const std::uint64_t active = fake_hash(protocol.session().active);
    const std::uint64_t candidate = fake_hash(protocol.session().candidate);

    adapter.fail_clone = true;
    mm_chunk_candidate output;
    output.boundaries.push_back(mm_chunk_boundary());
    output.boundaries[0].joint_position_source[0] = 12345.0f;
    const mm_chunk_candidate sentinel = output;
    require_error(
        protocol.generate(
            generate_request("failed-clone", true), output, error),
        error,
        "generation_failed");
    CHECK(output == sentinel);
    CHECK(fake_hash(protocol.session().active) == active);
    CHECK(fake_hash(protocol.session().candidate) == candidate);
    CHECK(!protocol.session().candidate_ready);

    adapter.fail_clone = false;
    adapter.fail_step = 4;
    require_error(
        protocol.generate(
            generate_request("failed", true), output, error),
        error,
        "generation_failed");
    CHECK(output == sentinel);
    CHECK(fake_hash(protocol.session().active) == active);
    CHECK(fake_hash(protocol.session().candidate) == candidate);
    CHECK(!protocol.session().candidate_ready);

    adapter.fail_step = -1;
    adapter.fail_observe_call = adapter.observe_count + 7;
    require_error(
        protocol.generate(
            generate_request("failed-late-observe", true), output, error),
        error,
        "generation_failed");
    CHECK(output == sentinel);
    CHECK(fake_hash(protocol.session().active) == active);
    CHECK(fake_hash(protocol.session().candidate) == candidate);
    CHECK(!protocol.session().candidate_ready);

    adapter.fail_observe_call = -1;
    adapter.fail_reset = true;
    mm_chunk_boundary reset_sentinel;
    reset_sentinel.joint_position_source[0] = 54321.0f;
    initial = reset_sentinel;
    require_error(
        protocol.reset(reset_request("replacement"), initial, error),
        error,
        "reset_failed");
    CHECK(initial == reset_sentinel);
    CHECK(protocol.session().session_id == "s1");
    CHECK(fake_hash(protocol.session().active) == active);
    CHECK(fake_hash(protocol.session().candidate) == candidate);
}

static void test_every_illegal_identity_and_interval_is_rejected()
{
    fake_adapter adapter;
    fake_engine protocol(adapter);
    mm_chunk_error error;
    mm_chunk_boundary initial;
    mm_chunk_candidate output;
    CHECK(protocol.hello(error));
    CHECK(protocol.reset(reset_request(), initial, error));

    mm_chunk_generate_request request = generate_request("c0", true);
    request.session_id = "other";
    require_error(
        protocol.generate(request, output, error), error, "session_mismatch");
    request = generate_request("c0", true);
    request.source_intervals = 9;
    require_error(
        protocol.generate(request, output, error), error, "invalid_intervals");
    request = generate_request("", true);
    require_error(
        protocol.generate(request, output, error), error, "invalid_candidate");
    require_error(
        protocol.commit("s1", "c0", error), error, "no_candidate");
    require_error(
        protocol.abort("s1", "c0", error), error, "no_candidate");

    CHECK(protocol.generate(generate_request("c0", true), output, error));
    require_error(
        protocol.abort("other", "c0", error), error, "session_mismatch");
    require_error(
        protocol.abort("s1", "other", error), error, "candidate_mismatch");
    CHECK(protocol.abort("s1", "c0", error));
}

int main()
{
    test_complete_protocol_state_matrix();
    test_preparation_is_inert_until_explicit_publication();
    test_legal_sequence_is_transactional_and_continuous();
    test_failures_never_publish_or_mutate_active_state();
    test_every_illegal_identity_and_interval_is_rejected();
    std::puts("MM chunk protocol tests passed");
    return 0;
}
