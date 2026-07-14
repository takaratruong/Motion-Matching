#include "interaction_database.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

class Sha256 {
public:
    void update(const uint8_t* data, size_t size) {
        total_bytes_ += static_cast<uint64_t>(size);
        for (size_t index = 0; index < size; ++index) {
            buffer_[buffer_size_++] = data[index];
            if (buffer_size_ == buffer_.size()) {
                transform(buffer_.data());
                buffer_size_ = 0;
            }
        }
    }

    std::array<uint8_t, 32> final() {
        const uint64_t bit_length = total_bytes_ * 8U;
        buffer_[buffer_size_++] = 0x80U;
        if (buffer_size_ > 56U) {
            std::fill(buffer_.begin() + static_cast<std::ptrdiff_t>(buffer_size_),
                      buffer_.end(), 0U);
            transform(buffer_.data());
            buffer_size_ = 0;
        }
        std::fill(buffer_.begin() + static_cast<std::ptrdiff_t>(buffer_size_),
                  buffer_.begin() + 56, 0U);
        for (size_t index = 0; index < 8U; ++index) {
            buffer_[63U - index] =
                static_cast<uint8_t>(bit_length >> (index * 8U));
        }
        transform(buffer_.data());

        std::array<uint8_t, 32> digest{};
        for (size_t index = 0; index < state_.size(); ++index) {
            digest[index * 4U] = static_cast<uint8_t>(state_[index] >> 24U);
            digest[index * 4U + 1U] =
                static_cast<uint8_t>(state_[index] >> 16U);
            digest[index * 4U + 2U] =
                static_cast<uint8_t>(state_[index] >> 8U);
            digest[index * 4U + 3U] = static_cast<uint8_t>(state_[index]);
        }
        return digest;
    }

private:
    static constexpr uint32_t rotate_right(uint32_t value, uint32_t count) {
        return (value >> count) | (value << (32U - count));
    }

    void transform(const uint8_t* block) {
        static constexpr std::array<uint32_t, 64> constants = {
            0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
            0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
            0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
            0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
            0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
            0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
            0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
            0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
            0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
            0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
            0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
            0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
            0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
            0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
            0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
            0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U,
        };

        std::array<uint32_t, 64> words{};
        for (size_t index = 0; index < 16U; ++index) {
            const size_t offset = index * 4U;
            words[index] =
                (static_cast<uint32_t>(block[offset]) << 24U) |
                (static_cast<uint32_t>(block[offset + 1U]) << 16U) |
                (static_cast<uint32_t>(block[offset + 2U]) << 8U) |
                static_cast<uint32_t>(block[offset + 3U]);
        }
        for (size_t index = 16U; index < words.size(); ++index) {
            const uint32_t first =
                rotate_right(words[index - 15U], 7U) ^
                rotate_right(words[index - 15U], 18U) ^
                (words[index - 15U] >> 3U);
            const uint32_t second =
                rotate_right(words[index - 2U], 17U) ^
                rotate_right(words[index - 2U], 19U) ^
                (words[index - 2U] >> 10U);
            words[index] = words[index - 16U] + first +
                           words[index - 7U] + second;
        }

        uint32_t a = state_[0];
        uint32_t b = state_[1];
        uint32_t c = state_[2];
        uint32_t d = state_[3];
        uint32_t e = state_[4];
        uint32_t f = state_[5];
        uint32_t g = state_[6];
        uint32_t h = state_[7];

        for (size_t index = 0; index < words.size(); ++index) {
            const uint32_t upper_sigma = rotate_right(e, 6U) ^
                                         rotate_right(e, 11U) ^
                                         rotate_right(e, 25U);
            const uint32_t choose = (e & f) ^ ((~e) & g);
            const uint32_t first =
                h + upper_sigma + choose + constants[index] + words[index];
            const uint32_t lower_sigma = rotate_right(a, 2U) ^
                                         rotate_right(a, 13U) ^
                                         rotate_right(a, 22U);
            const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t second = lower_sigma + majority;

            h = g;
            g = f;
            f = e;
            e = d + first;
            d = c;
            c = b;
            b = a;
            a = first + second;
        }

        state_[0] += a;
        state_[1] += b;
        state_[2] += c;
        state_[3] += d;
        state_[4] += e;
        state_[5] += f;
        state_[6] += g;
        state_[7] += h;
    }

    std::array<uint32_t, 8> state_ = {
        0x6a09e667U,
        0xbb67ae85U,
        0x3c6ef372U,
        0xa54ff53aU,
        0x510e527fU,
        0x9b05688cU,
        0x1f83d9abU,
        0x5be0cd19U,
    };
    std::array<uint8_t, 64> buffer_{};
    size_t buffer_size_ = 0;
    uint64_t total_bytes_ = 0;
};

std::string hexadecimal(const std::array<uint8_t, 32>& digest) {
    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (uint8_t byte : digest) {
        output << std::setw(2) << static_cast<unsigned int>(byte);
    }
    return output.str();
}

std::string sha256_file(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot open " + path.string());
    }

    Sha256 hash;
    std::array<char, 16 * 1024> buffer{};
    while (input) {
        input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const std::streamsize count = input.gcount();
        if (count > 0) {
            hash.update(
                reinterpret_cast<const uint8_t*>(buffer.data()),
                static_cast<size_t>(count));
        }
    }
    if (input.bad()) {
        throw std::runtime_error("could not hash " + path.string());
    }
    return hexadecimal(hash.final());
}

double maximum_quaternion_norm_error(const std::vector<float>& values) {
    double maximum = 0.0;
    for (size_t index = 0; index < values.size(); index += 4U) {
        double squared_norm = 0.0;
        for (size_t component = 0; component < 4U; ++component) {
            const double value = static_cast<double>(values[index + component]);
            squared_norm += value * value;
        }
        maximum = std::max(maximum, std::abs(std::sqrt(squared_norm) - 1.0));
    }
    return maximum;
}

double maximum_quaternion_norm_error(const interaction::Database& database) {
    return std::max(
        {maximum_quaternion_norm_error(database.rotations),
         maximum_quaternion_norm_error(database.object_rotations),
         maximum_quaternion_norm_error(database.table_rotations),
         maximum_quaternion_norm_error(database.grasp_rotations_object)});
}

void print_probe_json(
    const interaction::Database& database,
    const interaction::Features& features,
    std::string_view database_sha256,
    std::string_view feature_sha256) {
    std::array<uint64_t, 5> phase_counts{};
    for (uint8_t phase : database.phases) {
        ++phase_counts[phase];
    }
    const double quaternion_error =
        maximum_quaternion_norm_error(database);

    std::cout << std::setprecision(9)
              << "{\"bone_count\":" << database.bone_count
              << ",\"clip_count\":" << database.clip_count
              << ",\"database_sha256\":\"" << database_sha256 << '"'
              << ",\"feature_count\":" << features.feature_count
              << ",\"feature_sha256\":\"" << feature_sha256 << '"'
              << ",\"first_source_frame\":" << database.source_frames.front()
              << ",\"frame_count\":" << database.frame_count
              << ",\"last_source_frame\":" << database.source_frames.back()
              << ",\"max_quaternion_norm_error\":";
    if (quaternion_error == 0.0) {
        std::cout << "0.0";
    } else {
        std::cout << quaternion_error;
    }
    std::cout << ",\"phase_counts\":[" << phase_counts[0] << ','
              << phase_counts[1] << ',' << phase_counts[2] << ','
              << phase_counts[3] << ',' << phase_counts[4] << "]}\n";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3 || std::string_view(argv[2]) != "--json") {
        std::cerr << "usage: interaction_probe <pack> --json\n";
        return 2;
    }

    try {
        const std::filesystem::path pack = argv[1];
        const std::filesystem::path database_path =
            pack / "interaction_database.bin";
        const std::filesystem::path feature_path =
            pack / "interaction_features.bin";
        const interaction::Database database =
            interaction::load_database(database_path);
        const interaction::Features features =
            interaction::load_features(feature_path);
        if (database.frame_count != features.frame_count) {
            throw interaction::FormatError(
                "feature/database frame count mismatch");
        }

        const std::string database_sha256 = sha256_file(database_path);
        const std::string feature_sha256 = sha256_file(feature_path);
        static_cast<void>(sha256_file(pack / "manifest.json"));
        static_cast<void>(sha256_file(pack / "evaluation_split.json"));
        static_cast<void>(sha256_file(pack / "validation_report.json"));
        print_probe_json(
            database, features, database_sha256, feature_sha256);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "interaction_probe: " << error.what() << '\n';
        return 1;
    }
}
