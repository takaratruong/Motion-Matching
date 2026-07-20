#pragma once

#include <array>
#include <cstdint>
#include <filesystem>

namespace interaction {

std::array<uint8_t, 32> sha256_file_bytes(
    const std::filesystem::path& path);

}  // namespace interaction
