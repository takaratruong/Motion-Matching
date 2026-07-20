#include "interaction_funnel_worker.h"

#include <cmath>
#include <cstring>
#include <fstream>
#include <iterator>
#include <spawn.h>
#include <stdexcept>
#include <system_error>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

extern char** environ;

namespace interaction {
namespace {

constexpr std::array<uint8_t, 8> kRequestMagic{
    'G', '1', 'F', 'R', 'E', 'Q', '0', '2'};
constexpr uint32_t kRequestSchemaVersion = 2U;

void append_u32(std::vector<uint8_t>& bytes, uint32_t value) {
    for (int index = 0; index < 4; ++index) {
        bytes.push_back(static_cast<uint8_t>(
            (value >> (8 * index)) & 0xffU));
    }
}

void append_u64(std::vector<uint8_t>& bytes, uint64_t value) {
    for (int index = 0; index < 8; ++index) {
        bytes.push_back(static_cast<uint8_t>(
            (value >> (8 * index)) & 0xffU));
    }
}

void append_f32(std::vector<uint8_t>& bytes, float value) {
    uint32_t raw = 0U;
    static_assert(sizeof(raw) == sizeof(value));
    std::memcpy(&raw, &value, sizeof(raw));
    append_u32(bytes, raw);
}

void write_atomic(
    const std::filesystem::path& path,
    const std::vector<uint8_t>& bytes) {
    const std::filesystem::path temporary = path.string() + ".tmp";
    {
        std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
        if (!stream) throw std::runtime_error("cannot create funnel request");
        stream.write(
            reinterpret_cast<const char*>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()));
        if (!stream) throw std::runtime_error("cannot write funnel request");
    }
    std::filesystem::rename(temporary, path);
}

std::vector<uint8_t> read_bytes(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot open funnel response");
    return std::vector<uint8_t>(
        std::istreambuf_iterator<char>(stream),
        std::istreambuf_iterator<char>());
}

bool same_request(
    const FunnelProposalRequest& request,
    const InteractionFunnelArtifact& artifact) {
    return request.request_id == artifact.request_id() &&
        request.batch_seed == artifact.batch_seed() &&
        request.checkpoint_sha256 == artifact.checkpoint_sha256() &&
        std::memcmp(
            request.condition.data(), artifact.condition().data(),
            sizeof(float) * request.condition.size()) == 0;
}

}  // namespace

std::vector<uint8_t> serialize_funnel_proposal_request(
    const FunnelProposalRequest& request) {
    if (request.request_id == 0U) {
        throw std::invalid_argument("funnel request id must be nonzero");
    }
    for (float value : request.condition) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("funnel request condition must be finite");
        }
    }
    std::vector<uint8_t> bytes;
    bytes.reserve(156U);
    bytes.insert(bytes.end(), kRequestMagic.begin(), kRequestMagic.end());
    append_u32(bytes, kRequestSchemaVersion);
    append_u64(bytes, request.request_id);
    append_u64(bytes, request.batch_seed);
    bytes.insert(
        bytes.end(), request.checkpoint_sha256.begin(),
        request.checkpoint_sha256.end());
    for (float value : request.condition) append_f32(bytes, value);
    return bytes;
}

AsyncPythonFunnelProvider::AsyncPythonFunnelProvider(
    std::filesystem::path python,
    std::filesystem::path worker,
    std::filesystem::path checkpoint,
    std::filesystem::path work_directory)
    : python_(std::move(python)),
      worker_(std::move(worker)),
      checkpoint_(std::move(checkpoint)),
      work_directory_(std::move(work_directory)) {}

AsyncPythonFunnelProvider::~AsyncPythonFunnelProvider() {
    cancel();
}

bool AsyncPythonFunnelProvider::begin(
    const FunnelProposalRequest& request) {
    if (child_pid_ >= 0 || request_.has_value()) return false;
    try {
        const std::vector<uint8_t> bytes =
            serialize_funnel_proposal_request(request);
        std::filesystem::create_directories(work_directory_);
        const std::string stem = "request-" + std::to_string(request.request_id);
        request_path_ = work_directory_ / (stem + ".req");
        response_path_ = work_directory_ / (stem + ".funnel");
        remove_attempt_files();
        write_atomic(request_path_, bytes);

        std::vector<std::string> arguments{
            python_.string(), worker_.string(),
            "--request", request_path_.string(),
            "--checkpoint", checkpoint_.string(),
            "--output", response_path_.string(),
        };
        std::vector<char*> argv;
        argv.reserve(arguments.size() + 1U);
        for (std::string& argument : arguments) argv.push_back(argument.data());
        argv.push_back(nullptr);
        pid_t pid = -1;
        const int result = posix_spawnp(
            &pid, python_.c_str(), nullptr, nullptr, argv.data(), environ);
        if (result != 0) {
            remove_attempt_files();
            return false;
        }
        child_pid_ = static_cast<int>(pid);
        request_ = request;
        return true;
    } catch (const std::exception&) {
        remove_attempt_files();
        return false;
    }
}

FunnelProposalPoll AsyncPythonFunnelProvider::poll() {
    if (child_pid_ < 0 || !request_.has_value()) {
        return {FunnelProposalPollState::Failed, std::nullopt,
                "no active funnel proposal request"};
    }
    int status = 0;
    const pid_t result = waitpid(
        static_cast<pid_t>(child_pid_), &status, WNOHANG);
    if (result == 0) return {};
    if (result < 0) return fail("failed to poll funnel proposal worker");
    child_pid_ = -1;
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        return fail("funnel proposal worker exited unsuccessfully");
    }
    try {
        InteractionFunnelArtifact artifact =
            InteractionFunnelArtifact::load(read_bytes(response_path_));
        if (!same_request(*request_, artifact)) {
            return fail("funnel proposal response identity mismatch");
        }
        request_.reset();
        remove_attempt_files();
        return {FunnelProposalPollState::Ready, std::move(artifact), {}};
    } catch (const std::exception& error) {
        return fail(error.what());
    }
}

void AsyncPythonFunnelProvider::cancel() {
    if (child_pid_ >= 0) {
        (void)kill(static_cast<pid_t>(child_pid_), SIGTERM);
        int status = 0;
        (void)waitpid(static_cast<pid_t>(child_pid_), &status, 0);
        child_pid_ = -1;
    }
    request_.reset();
    remove_attempt_files();
}

FunnelProposalPoll AsyncPythonFunnelProvider::fail(std::string error) {
    child_pid_ = -1;
    request_.reset();
    remove_attempt_files();
    return {FunnelProposalPollState::Failed, std::nullopt, std::move(error)};
}

void AsyncPythonFunnelProvider::remove_attempt_files() {
    std::error_code ignored;
    if (!request_path_.empty()) std::filesystem::remove(request_path_, ignored);
    if (!request_path_.empty()) {
        std::filesystem::remove(request_path_.string() + ".tmp", ignored);
    }
    if (!response_path_.empty()) std::filesystem::remove(response_path_, ignored);
    if (!response_path_.empty()) {
        std::filesystem::remove(response_path_.string() + ".tmp", ignored);
    }
}

}  // namespace interaction
