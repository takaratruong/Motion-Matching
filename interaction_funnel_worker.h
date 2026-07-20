#pragma once

#include "interaction_funnel_artifact.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace interaction {

struct FunnelProposalRequest {
    uint64_t request_id = 0U;
    uint64_t batch_seed = 0U;
    std::array<uint8_t, 32> checkpoint_sha256{};
    std::array<float, kFunnelConditionDim> condition{};
};

std::vector<uint8_t> serialize_funnel_proposal_request(
    const FunnelProposalRequest& request);

enum class FunnelProposalPollState {
    Pending,
    Ready,
    Failed,
};

struct FunnelProposalPoll {
    FunnelProposalPollState state = FunnelProposalPollState::Pending;
    std::optional<InteractionFunnelArtifact> artifact{};
    std::string error{};
};

class FunnelProposalProvider {
public:
    virtual ~FunnelProposalProvider() = default;
    virtual bool begin(const FunnelProposalRequest& request) = 0;
    virtual FunnelProposalPoll poll() = 0;
    virtual void cancel() = 0;
};

class AsyncPythonFunnelProvider final : public FunnelProposalProvider {
public:
    AsyncPythonFunnelProvider(
        std::filesystem::path python,
        std::filesystem::path worker,
        std::filesystem::path checkpoint,
        std::filesystem::path work_directory);
    ~AsyncPythonFunnelProvider() override;

    bool begin(const FunnelProposalRequest& request) override;
    FunnelProposalPoll poll() override;
    void cancel() override;

private:
    FunnelProposalPoll fail(std::string error);
    void remove_attempt_files();

    std::filesystem::path python_;
    std::filesystem::path worker_;
    std::filesystem::path checkpoint_;
    std::filesystem::path work_directory_;
    std::filesystem::path request_path_;
    std::filesystem::path response_path_;
    std::optional<FunnelProposalRequest> request_{};
    int child_pid_ = -1;
};

}  // namespace interaction
