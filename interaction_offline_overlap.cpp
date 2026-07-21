#include "interaction_offline_overlap.h"

#include "g1_skeleton.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace interaction::offline_overlap {
namespace {

constexpr std::array<uint8_t, 8> kMagic = {
    'G', '1', 'O', 'V', 'L', 'P', '0', '1'};
constexpr uint32_t kSchemaVersion = 1U;
constexpr float kQuaternionUnitTolerance = 1.0e-3F;

class Reader {
public:
    explicit Reader(std::vector<uint8_t> bytes) : bytes_(std::move(bytes)) {}

    uint32_t u32(const char* label) {
        require(4U, label);
        const uint32_t value =
            static_cast<uint32_t>(bytes_[offset_]) |
            (static_cast<uint32_t>(bytes_[offset_ + 1U]) << 8U) |
            (static_cast<uint32_t>(bytes_[offset_ + 2U]) << 16U) |
            (static_cast<uint32_t>(bytes_[offset_ + 3U]) << 24U);
        offset_ += 4U;
        return value;
    }

    float f32(const char* label) {
        static_assert(sizeof(float) == sizeof(uint32_t),
                      "offline overlap requires IEEE-754 binary32 floats");
        const uint32_t bits = u32(label);
        float value = 0.0F;
        std::memcpy(&value, &bits, sizeof(value));
        if (!std::isfinite(value)) {
            throw std::runtime_error(std::string("offline overlap has nonfinite ") + label);
        }
        return value;
    }

    std::string bytes(size_t count, const char* label) {
        require(count, label);
        const char* start = reinterpret_cast<const char*>(bytes_.data() + offset_);
        offset_ += count;
        return std::string(start, count);
    }

    uint8_t u8(const char* label) {
        require(1U, label);
        return bytes_[offset_++];
    }

    bool consumed() const { return offset_ == bytes_.size(); }

private:
    void require(size_t count, const char* label) const {
        if (count > bytes_.size() - offset_) {
            throw std::runtime_error(std::string("truncated offline overlap ") + label);
        }
    }

    std::vector<uint8_t> bytes_;
    size_t offset_ = 0U;
};

void expect_u32(Reader& reader, uint32_t expected, const char* label) {
    if (reader.u32(label) != expected) {
        throw std::runtime_error(std::string("offline overlap has wrong ") + label);
    }
}

void expect_string(
    Reader& reader,
    std::string_view expected,
    const char* label) {
    const uint32_t size = reader.u32(label);
    if (size != expected.size() || reader.bytes(size, label) != expected) {
        throw std::runtime_error(std::string("offline overlap has wrong ") + label);
    }
}

void validate_quaternion(quat value) {
    const float length = std::sqrt(
        value.w * value.w + value.x * value.x +
        value.y * value.y + value.z * value.z);
    if (!std::isfinite(length) ||
        std::fabs(length - 1.0F) > kQuaternionUnitTolerance) {
        throw std::runtime_error("offline overlap has non-unit quaternion");
    }
}

vec3 read_vec3(Reader& reader, const char* label) {
    const float x = reader.f32(label);
    const float y = reader.f32(label);
    const float z = reader.f32(label);
    return vec3(x, y, z);
}

quat read_quaternion(Reader& reader) {
    const float w = reader.f32("rotation");
    const float x = reader.f32("rotation");
    const float y = reader.f32("rotation");
    const float z = reader.f32("rotation");
    return quat(w, x, y, z);
}

}  // namespace

Clip Clip::load(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream.is_open()) {
        throw std::runtime_error("could not open offline overlap file: " + path.string());
    }
    const std::vector<uint8_t> payload{
        std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
    if (stream.bad()) {
        throw std::runtime_error("could not read offline overlap file: " + path.string());
    }
    Reader reader(payload);
    for (const uint8_t expected : kMagic) {
        if (reader.u8("magic") != expected) {
            throw std::runtime_error("offline overlap has wrong magic");
        }
    }
    expect_u32(reader, kSchemaVersion, "schema version");
    expect_u32(reader, static_cast<uint32_t>(kFrameRateHz), "frame rate");
    expect_u32(reader, static_cast<uint32_t>(kFrameCount), "frame count");
    expect_u32(reader, static_cast<uint32_t>(g1_skeleton::BoneCount), "bone count");
    expect_u32(reader, static_cast<uint32_t>(kContactCount), "contact count");
    expect_string(reader, g1_skeleton::kSkeletonSignature, "skeleton signature");
    for (const std::string_view name : g1_skeleton::kBoneNames) {
        expect_string(reader, name, "skeleton bone order");
    }

    Clip clip{};
    for (Pose& pose : clip.frames) {
        for (vec3& value : pose.positions) {
            value = read_vec3(reader, "position");
        }
    }
    for (Pose& pose : clip.frames) {
        for (vec3& value : pose.velocities) {
            value = read_vec3(reader, "velocity");
        }
    }
    for (Pose& pose : clip.frames) {
        for (vec3& value : pose.angular_velocities) {
            value = read_vec3(reader, "angular velocity");
        }
    }
    for (Pose& pose : clip.frames) {
        for (quat& value : pose.rotations) {
            value = read_quaternion(reader);
            validate_quaternion(value);
        }
    }
    for (Pose& pose : clip.frames) {
        for (uint8_t& contact : pose.foot_contacts) {
            contact = reader.u8("contact");
            if (contact > 1U) {
                throw std::runtime_error("offline overlap has non-binary foot contact");
            }
        }
    }
    if (!reader.consumed()) {
        throw std::runtime_error("trailing bytes in offline overlap file");
    }
    return clip;
}

Player::Player(Clip clip, bool loop) : clip_(std::move(clip)), loop_(loop) {}

const Pose& Player::pose() const {
    return clip_.frames[frame_index_];
}

size_t Player::frame_index() const {
    return frame_index_;
}

bool Player::finished() const {
    return finished_;
}

void Player::advance_25hz() {
    if (finished_) return;
    if (frame_index_ + 1U < kFrameCount) {
        ++frame_index_;
        return;
    }
    if (loop_) {
        frame_index_ = 0U;
        return;
    }
    finished_ = true;
}

void Player::restart() {
    frame_index_ = 0U;
    finished_ = false;
}

}  // namespace interaction::offline_overlap
