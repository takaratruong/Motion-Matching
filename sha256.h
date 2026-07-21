#pragma once

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

struct sha256_state
{
    uint8_t block[64] = {};
    uint32_t length = 0;
    uint64_t bit_length = 0;
    uint32_t hash[8] = {
        0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u,
    };
};

static inline uint32_t sha256_rotr(uint32_t value, uint32_t bits)
{
    return (value >> bits) | (value << (32u - bits));
}

static inline void sha256_transform(sha256_state& state)
{
    static const uint32_t constants[64] = {
        0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
        0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
        0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
        0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
        0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
        0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
        0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
        0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u,
    };
    uint32_t words[64] = {};
    for (int i = 0; i < 16; ++i)
        words[i] = (static_cast<uint32_t>(state.block[i * 4]) << 24) |
                   (static_cast<uint32_t>(state.block[i * 4 + 1]) << 16) |
                   (static_cast<uint32_t>(state.block[i * 4 + 2]) << 8) |
                    static_cast<uint32_t>(state.block[i * 4 + 3]);
    for (int i = 16; i < 64; ++i) {
        const uint32_t s0 = sha256_rotr(words[i - 15], 7) ^
                            sha256_rotr(words[i - 15], 18) ^
                            (words[i - 15] >> 3);
        const uint32_t s1 = sha256_rotr(words[i - 2], 17) ^
                            sha256_rotr(words[i - 2], 19) ^
                            (words[i - 2] >> 10);
        words[i] = words[i - 16] + s0 + words[i - 7] + s1;
    }
    uint32_t a=state.hash[0], b=state.hash[1], c=state.hash[2], d=state.hash[3];
    uint32_t e=state.hash[4], f=state.hash[5], g=state.hash[6], h=state.hash[7];
    for (int i = 0; i < 64; ++i) {
        const uint32_t s1 = sha256_rotr(e, 6) ^ sha256_rotr(e, 11) ^ sha256_rotr(e, 25);
        const uint32_t choose = (e & f) ^ ((~e) & g);
        const uint32_t first = h + s1 + choose + constants[i] + words[i];
        const uint32_t s0 = sha256_rotr(a, 2) ^ sha256_rotr(a, 13) ^ sha256_rotr(a, 22);
        const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
        const uint32_t second = s0 + majority;
        h=g; g=f; f=e; e=d+first; d=c; c=b; b=a; a=first+second;
    }
    state.hash[0]+=a; state.hash[1]+=b; state.hash[2]+=c; state.hash[3]+=d;
    state.hash[4]+=e; state.hash[5]+=f; state.hash[6]+=g; state.hash[7]+=h;
}

static inline void sha256_update(
    sha256_state& state, const uint8_t* data, const size_t size)
{
    for (size_t i = 0; i < size; ++i) {
        state.block[state.length++] = data[i];
        if (state.length == 64) {
            sha256_transform(state);
            state.bit_length += 512;
            state.length = 0;
        }
    }
}

static inline std::string sha256_finish(sha256_state& state)
{
    uint32_t i = state.length;
    state.block[i++] = 0x80u;
    if (i > 56) {
        while (i < 64) state.block[i++] = 0;
        sha256_transform(state);
        std::memset(state.block, 0, 56);
    } else while (i < 56) state.block[i++] = 0;
    state.bit_length += static_cast<uint64_t>(state.length) * 8u;
    for (int byte = 0; byte < 8; ++byte)
        state.block[63 - byte] = static_cast<uint8_t>(state.bit_length >> (byte * 8));
    sha256_transform(state);
    static const char hex[] = "0123456789abcdef";
    std::string out(64, '0');
    for (int word = 0; word < 8; ++word)
        for (int byte = 0; byte < 4; ++byte) {
            const uint8_t value = static_cast<uint8_t>(
                state.hash[word] >> (24 - byte * 8));
            out[(word * 4 + byte) * 2] = hex[value >> 4];
            out[(word * 4 + byte) * 2 + 1] = hex[value & 15];
        }
    return out;
}

static inline bool sha256_file_hex(
    std::string& out, const char* path, char* error, const int error_capacity)
{
    if (path == NULL || path[0] == '\0') {
        if (error && error_capacity > 0)
            std::snprintf(error, static_cast<size_t>(error_capacity),
                "%s: invalid SHA-256 path", path ? path : "<null>");
        return false;
    }
    FILE* file = std::fopen(path, "rb");
    if (file == NULL) {
        if (error && error_capacity > 0)
            std::snprintf(error, static_cast<size_t>(error_capacity),
                "%s: cannot open for SHA-256 (%s)", path, std::strerror(errno));
        return false;
    }
    sha256_state state;
    uint8_t buffer[64 * 1024];
    while (true) {
        const size_t count = std::fread(buffer, 1, sizeof(buffer), file);
        sha256_update(state, buffer, count);
        if (count != sizeof(buffer)) break;
    }
    bool failed = std::ferror(file) != 0;
    if (std::fclose(file) != 0) failed = true;
    if (failed) {
        if (error && error_capacity > 0)
            std::snprintf(error, static_cast<size_t>(error_capacity),
                "%s: failed while hashing", path);
        return false;
    }
    std::string digest = sha256_finish(state);
    out.swap(digest);
    return true;
}
