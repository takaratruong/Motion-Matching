#include "interaction_funnel_worker.h"

#include <array>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

interaction::FunnelProposalRequest request() {
    interaction::FunnelProposalRequest value{};
    value.request_id = 7U;
    value.batch_seed = 2026071901U;
    for (size_t i = 0; i < value.checkpoint_sha256.size(); ++i) {
        value.checkpoint_sha256[i] = static_cast<uint8_t>(i);
    }
    for (size_t i = 0; i < value.condition.size(); ++i) {
        value.condition[i] = static_cast<float>(i) * 0.125F;
    }
    return value;
}

void test_request_bytes_match_python_layout() {
    const std::vector<uint8_t> bytes =
        interaction::serialize_funnel_proposal_request(request());
    assert(bytes.size() == 156U);
    const std::array<uint8_t, 8> magic{
        'G', '1', 'F', 'R', 'E', 'Q', '0', '2'};
    for (size_t i = 0; i < magic.size(); ++i) assert(bytes[i] == magic[i]);
    assert(bytes[8] == 2U && bytes[9] == 0U);
    assert(bytes[12] == 7U);  // request_id, little endian
    assert(bytes[28] == 0U && bytes[59] == 31U);  // checkpoint digest
}

void test_invalid_request_is_rejected() {
    interaction::FunnelProposalRequest invalid = request();
    invalid.request_id = 0U;
    bool threw = false;
    try {
        (void)interaction::serialize_funnel_proposal_request(invalid);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
}

void test_nonzero_worker_exit_fails_without_artifact() {
    const std::filesystem::path directory =
        std::filesystem::temp_directory_path() /
        "g1-funnel-worker-nonzero-test";
    std::filesystem::remove_all(directory);
    interaction::AsyncPythonFunnelProvider provider(
        "/bin/false", "ignored-worker.py", "ignored-checkpoint.pt", directory);
    assert(provider.begin(request()));
    assert(!provider.begin(request()));
    interaction::FunnelProposalPoll result{};
    for (int attempt = 0; attempt < 1000; ++attempt) {
        result = provider.poll();
        if (result.state != interaction::FunnelProposalPollState::Pending) break;
        std::this_thread::yield();
    }
    assert(result.state == interaction::FunnelProposalPollState::Failed);
    assert(!result.artifact.has_value());
    assert(!result.error.empty());
    std::filesystem::remove_all(directory);
}

}  // namespace

int main() {
    test_request_bytes_match_python_layout();
    test_invalid_request_is_rejected();
    test_nonzero_worker_exit_fails_without_artifact();
    return 0;
}
