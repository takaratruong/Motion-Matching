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

    int clone_count = 0;
    int swap_count = 0;
    int fail_step = -1;
    bool fail_reset = false;
    bool fail_observe = false;

    bool reset(
        state_type& state,
        mm_chunk_boundary& boundary,
        const mm_chunk_reset_request& request,
        std::string& error)
    {
        state.frame = 100 + static_cast<int>(request.scene_id.size());
        state.history.assign(1, state.frame);
        if (fail_reset) {
            error = "injected reset failure";
            return false;
        }
        return observe(boundary, state, error);
    }

    bool clone(
        state_type& destination,
        const state_type& source,
        std::string&)
    {
        ++clone_count;
        destination = source;
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
        if (fail_observe) {
            error = "injected observation failure";
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

    adapter.fail_step = 4;
    mm_chunk_candidate output;
    output.boundaries.push_back(mm_chunk_boundary());
    output.boundaries[0].joint_position_source[0] = 12345.0f;
    const mm_chunk_candidate sentinel = output;
    require_error(
        protocol.generate(
            generate_request("failed", true), output, error),
        error,
        "generation_failed");
    CHECK(output == sentinel);
    CHECK(fake_hash(protocol.session().active) == active);
    CHECK(!protocol.session().candidate_ready);

    adapter.fail_step = -1;
    adapter.fail_observe = true;
    require_error(
        protocol.generate(
            generate_request("failed-observe", true), output, error),
        error,
        "generation_failed");
    CHECK(output == sentinel);
    CHECK(fake_hash(protocol.session().active) == active);
    CHECK(!protocol.session().candidate_ready);

    adapter.fail_observe = false;
    adapter.fail_reset = true;
    require_error(
        protocol.reset(reset_request("replacement"), initial, error),
        error,
        "reset_failed");
    CHECK(protocol.session().session_id == "s1");
    CHECK(fake_hash(protocol.session().active) == active);
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
}

int main()
{
    test_legal_sequence_is_transactional_and_continuous();
    test_failures_never_publish_or_mutate_active_state();
    test_every_illegal_identity_and_interval_is_rejected();
    std::puts("MM chunk protocol tests passed");
    return 0;
}
