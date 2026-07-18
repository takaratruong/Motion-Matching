#include "database.h"
#include "interaction_controller_adapter.h"
#include "interaction_database.h"
#include "interaction_runtime.h"
#include "locomotion_timing.h"
#include "stationary_motion_matching.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

namespace {

constexpr size_t kMaximumJsonDepth = 128U;
constexpr size_t kMaximumJsonItems = 1000000U;
constexpr uintmax_t kMaximumManifestBytes = 64U * 1024U * 1024U;
constexpr uintmax_t kMaximumReportBytes = 8U * 1024U * 1024U;
constexpr size_t kExpectedClipCount = 2045U;
constexpr int64_t kExpectedFrameCount = 511250;
constexpr size_t kExpectedRetainedCandidateCount = 19U;
constexpr size_t kExpectedStationarySnapshotCount = 186U;
constexpr size_t kMaximumPreviewWorkers = 16U;
constexpr int64_t kRightHand = 1;
constexpr const char* kExpectedDatasetId =
    "nvidia/PhysicalAI-Robotics-Locomanipulation-GRAIL";
constexpr const char* kExpectedTargetSequenceId =
    "pickup_table__beer_10__001";
constexpr const char* kExpectedManifestSha256 =
    "c9024d12b05c59a492f9f06a6fe2614cb1e814ece6e35e9e96e5be7210d800e4";
constexpr const char* kExpectedReportSha256 =
    "6e93f87e455513c887f4deeb2900980f0c13ca8f1f560ffa89c3ce1de715f7e1";

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error(message);
}

void require(bool condition, const std::string& message) {
    if (!condition) fail(message);
}

class Sha256 {
public:
    void update(const uint8_t* data, size_t size) {
        total_bytes_ += static_cast<uint64_t>(size);
        for (size_t index = 0U; index < size; ++index) {
            buffer_[buffer_size_++] = data[index];
            if (buffer_size_ == buffer_.size()) {
                transform(buffer_.data());
                buffer_size_ = 0U;
            }
        }
    }

    std::array<uint8_t, 32> final() {
        const uint64_t bit_length = total_bytes_ * 8U;
        buffer_[buffer_size_++] = 0x80U;
        if (buffer_size_ > 56U) {
            std::fill(
                buffer_.begin() + static_cast<std::ptrdiff_t>(buffer_size_),
                buffer_.end(),
                0U);
            transform(buffer_.data());
            buffer_size_ = 0U;
        }
        std::fill(
            buffer_.begin() + static_cast<std::ptrdiff_t>(buffer_size_),
            buffer_.begin() + 56,
            0U);
        for (size_t index = 0U; index < 8U; ++index) {
            buffer_[63U - index] =
                static_cast<uint8_t>(bit_length >> (index * 8U));
        }
        transform(buffer_.data());

        std::array<uint8_t, 32> digest{};
        for (size_t index = 0U; index < state_.size(); ++index) {
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
        for (size_t index = 0U; index < 16U; ++index) {
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

        for (size_t index = 0U; index < words.size(); ++index) {
            const uint32_t upper_sigma = rotate_right(e, 6U) ^
                rotate_right(e, 11U) ^ rotate_right(e, 25U);
            const uint32_t choose = (e & f) ^ ((~e) & g);
            const uint32_t first =
                h + upper_sigma + choose + constants[index] + words[index];
            const uint32_t lower_sigma = rotate_right(a, 2U) ^
                rotate_right(a, 13U) ^ rotate_right(a, 22U);
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
    size_t buffer_size_ = 0U;
    uint64_t total_bytes_ = 0U;
};

std::string sha256_bytes(const std::string& bytes) {
    Sha256 hash;
    if (!bytes.empty()) {
        hash.update(
            reinterpret_cast<const uint8_t*>(bytes.data()), bytes.size());
    }
    const std::array<uint8_t, 32> digest = hash.final();
    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (uint8_t byte : digest) {
        output << std::setw(2) << static_cast<unsigned int>(byte);
    }
    return output.str();
}

struct JsonValue {
    enum class Kind { Null, Boolean, Number, String, Array, Object };

    Kind kind = Kind::Null;
    bool boolean = false;
    double number = 0.0;
    std::string text;
    std::vector<JsonValue> array;
    std::map<std::string, JsonValue> object;

    static JsonValue make_null() { return JsonValue{}; }

    static JsonValue make_boolean(bool value) {
        JsonValue result;
        result.kind = Kind::Boolean;
        result.boolean = value;
        return result;
    }

    static JsonValue make_number(double value, std::string token) {
        JsonValue result;
        result.kind = Kind::Number;
        result.number = value;
        result.text = std::move(token);
        return result;
    }

    static JsonValue make_string(std::string value) {
        JsonValue result;
        result.kind = Kind::String;
        result.text = std::move(value);
        return result;
    }

    static JsonValue make_array(std::vector<JsonValue> value) {
        JsonValue result;
        result.kind = Kind::Array;
        result.array = std::move(value);
        return result;
    }

    static JsonValue make_object(std::map<std::string, JsonValue> value) {
        JsonValue result;
        result.kind = Kind::Object;
        result.object = std::move(value);
        return result;
    }
};

class JsonParser {
public:
    JsonParser(const std::string& source, std::string label)
        : source_(source), label_(std::move(label)) {}

    JsonValue parse_document() {
        skip_whitespace();
        require(position_ < source_.size(), label_ + " is empty");
        JsonValue value = parse_value(0U);
        skip_whitespace();
        if (position_ != source_.size()) {
            syntax_error("trailing bytes after the JSON value");
        }
        return value;
    }

private:
    [[noreturn]] void syntax_error(const std::string& message) const {
        fail(
            label_ + " JSON syntax error at byte " +
            std::to_string(position_) + ": " + message);
    }

    void skip_whitespace() {
        while (position_ < source_.size()) {
            const char character = source_[position_];
            if (character != ' ' && character != '\t' &&
                character != '\n' && character != '\r') {
                break;
            }
            ++position_;
        }
    }

    bool consume(char expected) {
        if (position_ < source_.size() && source_[position_] == expected) {
            ++position_;
            return true;
        }
        return false;
    }

    void expect(char expected, const std::string& message) {
        if (!consume(expected)) syntax_error(message);
    }

    JsonValue parse_value(size_t depth) {
        if (depth > kMaximumJsonDepth) {
            syntax_error("nesting exceeds the safety limit");
        }
        if (position_ >= source_.size()) {
            syntax_error("unexpected end of input");
        }
        switch (source_[position_]) {
        case 'n': return parse_literal("null", JsonValue::make_null());
        case 't': return parse_literal("true", JsonValue::make_boolean(true));
        case 'f': return parse_literal("false", JsonValue::make_boolean(false));
        case '"': return JsonValue::make_string(parse_string());
        case '[': return parse_array(depth);
        case '{': return parse_object(depth);
        default:
            if (source_[position_] == '-' ||
                (source_[position_] >= '0' && source_[position_] <= '9')) {
                return parse_number();
            }
            syntax_error("expected a JSON value");
        }
    }

    JsonValue parse_literal(const char* literal, JsonValue value) {
        const size_t start = position_;
        for (size_t index = 0U; literal[index] != '\0'; ++index) {
            if (position_ >= source_.size() ||
                source_[position_] != literal[index]) {
                position_ = start;
                syntax_error(std::string("invalid literal; expected ") + literal);
            }
            ++position_;
        }
        return value;
    }

    static int hexadecimal_digit(unsigned char character) {
        if (character >= '0' && character <= '9') return character - '0';
        if (character >= 'a' && character <= 'f') return 10 + character - 'a';
        if (character >= 'A' && character <= 'F') return 10 + character - 'A';
        return -1;
    }

    uint32_t parse_hex_quad() {
        if (source_.size() - position_ < 4U) {
            syntax_error("incomplete Unicode escape");
        }
        uint32_t result = 0U;
        for (size_t index = 0U; index < 4U; ++index) {
            const int digit = hexadecimal_digit(
                static_cast<unsigned char>(source_[position_++]));
            if (digit < 0) syntax_error("invalid Unicode escape");
            result = result * 16U + static_cast<uint32_t>(digit);
        }
        return result;
    }

    static void append_utf8(std::string& output, uint32_t codepoint) {
        if (codepoint <= 0x7fU) {
            output.push_back(static_cast<char>(codepoint));
        } else if (codepoint <= 0x7ffU) {
            output.push_back(static_cast<char>(0xc0U | (codepoint >> 6U)));
            output.push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
        } else if (codepoint <= 0xffffU) {
            output.push_back(static_cast<char>(0xe0U | (codepoint >> 12U)));
            output.push_back(
                static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3fU)));
            output.push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
        } else {
            output.push_back(static_cast<char>(0xf0U | (codepoint >> 18U)));
            output.push_back(
                static_cast<char>(0x80U | ((codepoint >> 12U) & 0x3fU)));
            output.push_back(
                static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3fU)));
            output.push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
        }
    }

    void append_raw_utf8(std::string& output) {
        const size_t start = position_;
        const unsigned char lead =
            static_cast<unsigned char>(source_[position_]);
        size_t length = 0U;
        if (lead >= 0xc2U && lead <= 0xdfU) {
            length = 2U;
        } else if (lead >= 0xe0U && lead <= 0xefU) {
            length = 3U;
        } else if (lead >= 0xf0U && lead <= 0xf4U) {
            length = 4U;
        } else {
            syntax_error("invalid UTF-8 leading byte in string");
        }
        if (source_.size() - position_ < length) {
            syntax_error("truncated UTF-8 sequence in string");
        }
        for (size_t index = 1U; index < length; ++index) {
            const unsigned char continuation =
                static_cast<unsigned char>(source_[position_ + index]);
            if ((continuation & 0xc0U) != 0x80U) {
                syntax_error("invalid UTF-8 continuation byte in string");
            }
        }
        const unsigned char second =
            static_cast<unsigned char>(source_[position_ + 1U]);
        if ((lead == 0xe0U && second < 0xa0U) ||
            (lead == 0xedU && second > 0x9fU) ||
            (lead == 0xf0U && second < 0x90U) ||
            (lead == 0xf4U && second > 0x8fU)) {
            syntax_error("non-scalar or overlong UTF-8 sequence in string");
        }
        position_ += length;
        output.append(source_, start, length);
    }

    void append_escaped_unicode(std::string& output) {
        uint32_t codepoint = parse_hex_quad();
        if (codepoint >= 0xd800U && codepoint <= 0xdbffU) {
            if (source_.size() - position_ < 2U ||
                source_[position_] != '\\' ||
                source_[position_ + 1U] != 'u') {
                syntax_error("high surrogate is not followed by a low surrogate");
            }
            position_ += 2U;
            const uint32_t low = parse_hex_quad();
            if (low < 0xdc00U || low > 0xdfffU) {
                syntax_error("high surrogate is followed by an invalid low surrogate");
            }
            codepoint = 0x10000U +
                ((codepoint - 0xd800U) << 10U) + (low - 0xdc00U);
        } else if (codepoint >= 0xdc00U && codepoint <= 0xdfffU) {
            syntax_error("unpaired low surrogate in string");
        }
        append_utf8(output, codepoint);
    }

    std::string parse_string() {
        expect('"', "expected opening quote");
        std::string output;
        while (position_ < source_.size()) {
            const unsigned char character =
                static_cast<unsigned char>(source_[position_++]);
            if (character == '"') return output;
            if (character < 0x20U) {
                syntax_error("unescaped control byte in string");
            }
            if (character == '\\') {
                if (position_ >= source_.size()) {
                    syntax_error("truncated string escape");
                }
                const char escape = source_[position_++];
                switch (escape) {
                case '"': output.push_back('"'); break;
                case '\\': output.push_back('\\'); break;
                case '/': output.push_back('/'); break;
                case 'b': output.push_back('\b'); break;
                case 'f': output.push_back('\f'); break;
                case 'n': output.push_back('\n'); break;
                case 'r': output.push_back('\r'); break;
                case 't': output.push_back('\t'); break;
                case 'u': append_escaped_unicode(output); break;
                default: syntax_error("invalid string escape");
                }
            } else if (character < 0x80U) {
                output.push_back(static_cast<char>(character));
            } else {
                --position_;
                append_raw_utf8(output);
            }
        }
        syntax_error("unterminated string");
    }

    JsonValue parse_number() {
        const size_t start = position_;
        consume('-');
        if (position_ >= source_.size()) {
            syntax_error("number ends after minus sign");
        }
        if (consume('0')) {
            if (position_ < source_.size() &&
                source_[position_] >= '0' && source_[position_] <= '9') {
                syntax_error("leading zero in number");
            }
        } else {
            if (source_[position_] < '1' || source_[position_] > '9') {
                syntax_error("invalid integer part");
            }
            while (position_ < source_.size() &&
                   source_[position_] >= '0' && source_[position_] <= '9') {
                ++position_;
            }
        }
        if (consume('.')) {
            const size_t fraction_start = position_;
            while (position_ < source_.size() &&
                   source_[position_] >= '0' && source_[position_] <= '9') {
                ++position_;
            }
            if (position_ == fraction_start) {
                syntax_error("fraction requires at least one digit");
            }
        }
        if (position_ < source_.size() &&
            (source_[position_] == 'e' || source_[position_] == 'E')) {
            ++position_;
            if (position_ < source_.size() &&
                (source_[position_] == '+' || source_[position_] == '-')) {
                ++position_;
            }
            const size_t exponent_start = position_;
            while (position_ < source_.size() &&
                   source_[position_] >= '0' && source_[position_] <= '9') {
                ++position_;
            }
            if (position_ == exponent_start) {
                syntax_error("exponent requires at least one digit");
            }
        }
        const std::string token = source_.substr(start, position_ - start);
        char* conversion_end = nullptr;
        errno = 0;
        const double value = std::strtod(token.c_str(), &conversion_end);
        if (conversion_end != token.c_str() + token.size() ||
            errno == ERANGE || !std::isfinite(value)) {
            syntax_error("number is not a finite representable double");
        }
        return JsonValue::make_number(value, token);
    }

    JsonValue parse_array(size_t depth) {
        expect('[', "expected array");
        skip_whitespace();
        std::vector<JsonValue> values;
        if (consume(']')) return JsonValue::make_array(std::move(values));
        for (;;) {
            if (values.size() >= kMaximumJsonItems) {
                syntax_error("array exceeds the item safety limit");
            }
            values.push_back(parse_value(depth + 1U));
            skip_whitespace();
            if (consume(']')) break;
            expect(',', "expected comma or closing bracket");
            skip_whitespace();
        }
        return JsonValue::make_array(std::move(values));
    }

    JsonValue parse_object(size_t depth) {
        expect('{', "expected object");
        skip_whitespace();
        std::map<std::string, JsonValue> values;
        if (consume('}')) return JsonValue::make_object(std::move(values));
        for (;;) {
            if (values.size() >= kMaximumJsonItems) {
                syntax_error("object exceeds the member safety limit");
            }
            if (position_ >= source_.size() || source_[position_] != '"') {
                syntax_error("object key must be a string");
            }
            std::string key = parse_string();
            skip_whitespace();
            expect(':', "expected colon after object key");
            skip_whitespace();
            JsonValue value = parse_value(depth + 1U);
            const auto inserted = values.emplace(std::move(key), std::move(value));
            if (!inserted.second) {
                syntax_error("duplicate object key " + inserted.first->first);
            }
            skip_whitespace();
            if (consume('}')) break;
            expect(',', "expected comma or closing brace");
            skip_whitespace();
        }
        return JsonValue::make_object(std::move(values));
    }

    const std::string& source_;
    std::string label_;
    size_t position_ = 0U;
};

std::string read_text_file(
    const std::filesystem::path& path,
    uintmax_t maximum_bytes,
    const std::string& label) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) fail("could not open " + label + ": " + path.string());
    const std::streampos end = input.tellg();
    if (end < std::streampos(0)) {
        fail("could not determine " + label + " size: " + path.string());
    }
    const uintmax_t byte_count = static_cast<uintmax_t>(end);
    if (byte_count > maximum_bytes) {
        fail(label + " exceeds the input size safety limit");
    }
    input.seekg(0, std::ios::beg);
    if (!input) fail("could not seek " + label + ": " + path.string());
    std::string contents(static_cast<size_t>(byte_count), '\0');
    if (!contents.empty()) {
        input.read(contents.data(), static_cast<std::streamsize>(contents.size()));
        if (input.gcount() != static_cast<std::streamsize>(contents.size())) {
            fail("short read while loading " + label + ": " + path.string());
        }
    }
    char unexpected = '\0';
    if (input.read(&unexpected, 1)) {
        fail(label + " changed while it was being read");
    }
    if (!input.eof()) {
        fail("I/O failure while loading " + label + ": " + path.string());
    }
    return contents;
}

void require_readable_nonempty_file(
    const std::filesystem::path& path,
    const std::string& label) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) fail("could not open " + label + ": " + path.string());
    const std::streampos end = input.tellg();
    if (end <= std::streampos(0)) {
        fail(label + " must be a nonempty seekable file: " + path.string());
    }
}

const JsonValue& require_kind(
    const JsonValue& value,
    JsonValue::Kind expected,
    const std::string& label) {
    if (value.kind != expected) fail(label + " has the wrong JSON type");
    return value;
}

std::reference_wrapper<const JsonValue> object_member(
    const JsonValue& object,
    const char* key,
    const std::string& label) {
    require_kind(object, JsonValue::Kind::Object, label);
    const auto found = object.object.find(key);
    if (found == object.object.end()) {
        fail(label + " is missing required member " + key);
    }
    return std::cref(found->second);
}

void require_exact_keys(
    const JsonValue& object,
    std::initializer_list<const char*> expected,
    const std::string& label) {
    require_kind(object, JsonValue::Kind::Object, label);
    std::set<std::string> expected_keys;
    for (const char* key : expected) expected_keys.emplace(key);
    std::set<std::string> actual_keys;
    for (const auto& item : object.object) actual_keys.emplace(item.first);
    if (actual_keys != expected_keys) {
        std::string details;
        for (const std::string& key : expected_keys) {
            if (actual_keys.count(key) == 0U) details += " missing=" + key;
        }
        for (const std::string& key : actual_keys) {
            if (expected_keys.count(key) == 0U) details += " unexpected=" + key;
        }
        fail(label + " members differ from the frozen schema:" + details);
    }
}

const std::string& json_string(
    const JsonValue& value,
    const std::string& label) {
    require_kind(value, JsonValue::Kind::String, label);
    if (value.text.empty()) fail(label + " must be a nonempty string");
    return value.text;
}

bool json_boolean(const JsonValue& value, const std::string& label) {
    require_kind(value, JsonValue::Kind::Boolean, label);
    return value.boolean;
}

double json_number(const JsonValue& value, const std::string& label) {
    require_kind(value, JsonValue::Kind::Number, label);
    if (!std::isfinite(value.number)) fail(label + " must be finite");
    return value.number;
}

int64_t json_integer(const JsonValue& value, const std::string& label) {
    require_kind(value, JsonValue::Kind::Number, label);
    if (value.text.find_first_of(".eE") != std::string::npos) {
        fail(label + " must use integer JSON syntax");
    }
    int64_t result = 0;
    const char* begin = value.text.data();
    const char* end = begin + value.text.size();
    const auto conversion = std::from_chars(begin, end, result, 10);
    if (conversion.ec != std::errc() || conversion.ptr != end) {
        fail(label + " is outside the signed 64-bit integer range");
    }
    return result;
}

int64_t nonnegative_integer(
    const JsonValue& value,
    const std::string& label) {
    const int64_t result = json_integer(value, label);
    if (result < 0) fail(label + " must be nonnegative");
    return result;
}

std::reference_wrapper<const std::vector<JsonValue>> json_array(
    const JsonValue& value,
    const std::string& label) {
    require_kind(value, JsonValue::Kind::Array, label);
    return std::cref(value.array);
}

template <size_t Size>
std::array<double, Size> finite_number_array(
    const JsonValue& value,
    const std::string& label) {
    const std::vector<JsonValue>& source = json_array(value, label).get();
    if (source.size() != Size) {
        fail(label + " must contain exactly " + std::to_string(Size) + " numbers");
    }
    std::array<double, Size> result{};
    for (size_t index = 0U; index < Size; ++index) {
        result[index] = json_number(
            source[index], label + "[" + std::to_string(index) + "]");
    }
    return result;
}

JsonValue parse_json_file(
    const std::filesystem::path& path,
    uintmax_t maximum_bytes,
    const std::string& label,
    const std::string& expected_sha256) {
    const std::string source = read_text_file(path, maximum_bytes, label);
    const std::string actual_sha256 = sha256_bytes(source);
    require(
        actual_sha256 == expected_sha256,
        label + " SHA-256 differs from the reviewed byte authority: expected " +
            expected_sha256 + ", got " + actual_sha256);
    return JsonParser(source, label).parse_document();
}

struct ManifestClip {
    std::string sequence_id;
    std::string object_id;
    int64_t active_hand = -1;
    int64_t range_start = -1;
    int64_t range_stop = -1;
};

struct Manifest {
    std::string dataset_id;
    int64_t schema_version = -1;
    std::vector<ManifestClip> clips;
    std::map<std::string, size_t> sequence_ordinals;
};

struct ReportTarget {
    std::string sequence_id;
    std::string object_id;
    int64_t active_hand = -1;
    int64_t range_start = -1;
    int64_t range_stop = -1;
    int64_t entry_local_frame = -1;
    int64_t object_alignment_local_frame = -1;
    int64_t contact_local_frame = -1;
    int64_t lift_local_frame = -1;
    int64_t hold_local_frame = -1;
    int64_t stop_local_frame = -1;
    std::array<double, 3> object_alignment_position{};
    std::array<double, 4> object_alignment_rotation{};
    std::array<double, 3> table_position{};
    std::array<double, 4> table_rotation{};
    std::array<double, 3> table_size{};
    std::array<double, 3> object_dimensions{};
    std::array<double, 3> grasp_position_object{};
    std::array<double, 4> grasp_rotation_object{};
    std::array<double, 3> approach_direction_object{};
};

struct StableKey {
    std::string dataset_id;
    int64_t schema_version = -1;
    std::string sequence_id;
    int64_t entry_local_frame = -1;
    int64_t active_hand = -1;

    auto ordered_tuple() const {
        return std::tie(
            dataset_id,
            schema_version,
            sequence_id,
            entry_local_frame,
            active_hand);
    }
};

struct RetainedCandidate {
    StableKey stable_key;
    std::string sequence_id;
    std::string object_id;
    int64_t active_hand = -1;
    int64_t source_entry_frame = -1;
    int64_t entry_local_frame = -1;
    int64_t object_alignment_local_frame = -1;
    int64_t contact_local_frame = -1;
    int64_t lift_local_frame = -1;
    int64_t hold_local_frame = -1;
    int64_t stop_local_frame = -1;
    double root_x_object_report_m = 0.0;
    double root_z_object_report_m = 0.0;
    double root_yaw_object_report_radians = 0.0;
    float root_x_object_m = 0.0F;
    float root_z_object_m = 0.0F;
    float root_yaw_object_radians = 0.0F;
    double contact_hand_position_error_m = 0.0;
    double contact_hand_orientation_error_radians = 0.0;
    double entry_root_planar_speed_mps = 0.0;
    size_t source_clip_ordinal = 0U;
};

struct SlotReport {
    std::string dataset_id;
    int64_t schema_version = -1;
    ReportTarget target;
    std::vector<RetainedCandidate> retained_candidates;
    int64_t deduplicated_candidate_count = -1;
    std::map<std::string, int64_t> rejected_counts_by_gate;
};

Manifest parse_manifest(const JsonValue& root) {
    require_kind(root, JsonValue::Kind::Object, "full-pack manifest");
    Manifest manifest;
    manifest.dataset_id = json_string(
        object_member(root, "dataset_id", "full-pack manifest"),
        "full-pack manifest.dataset_id");
    manifest.schema_version = json_integer(
        object_member(root, "schema_version", "full-pack manifest"),
        "full-pack manifest.schema_version");
    require(
        manifest.dataset_id == kExpectedDatasetId,
        "full-pack manifest dataset_id differs from the Task4 authority");
    require(
        manifest.schema_version == 1,
        "full-pack manifest schema_version must be exactly 1");
    require(
        object_member(root, "diagnostic_limit", "full-pack manifest").get().kind ==
            JsonValue::Kind::Null,
        "full-pack manifest diagnostic_limit must be null");
    require(
        json_number(
            object_member(root, "target_fps", "full-pack manifest"),
            "full-pack manifest.target_fps") == 25.0,
        "full-pack manifest target_fps must be exactly 25");

    const auto clips_member = object_member(
        root, "clips", "full-pack manifest");
    const JsonValue& clips_value = clips_member.get();
    const std::string clips_label = "full-pack manifest.clips";
    const auto clips_array = json_array(clips_value, clips_label);
    const std::vector<JsonValue>& clips = clips_array.get();
    require(
        clips.size() == kExpectedClipCount,
        "full-pack manifest must contain exactly 2045 clips");
    manifest.clips.reserve(clips.size());
    int64_t expected_start = 0;
    std::set<std::pair<int64_t, int64_t>> seen_ranges;
    for (size_t ordinal = 0U; ordinal < clips.size(); ++ordinal) {
        const std::string label =
            "full-pack manifest.clips[" + std::to_string(ordinal) + "]";
        const JsonValue& clip_value = clips[ordinal];
        require_exact_keys(
            clip_value,
            {"active_hand", "object_id", "range_start", "range_stop",
             "sequence_id"},
            label);
        ManifestClip clip;
        clip.sequence_id = json_string(
            object_member(clip_value, "sequence_id", label),
            label + ".sequence_id");
        clip.object_id = json_string(
            object_member(clip_value, "object_id", label),
            label + ".object_id");
        clip.active_hand = json_integer(
            object_member(clip_value, "active_hand", label),
            label + ".active_hand");
        clip.range_start = nonnegative_integer(
            object_member(clip_value, "range_start", label),
            label + ".range_start");
        clip.range_stop = nonnegative_integer(
            object_member(clip_value, "range_stop", label),
            label + ".range_stop");
        require(
            clip.active_hand == 0 || clip.active_hand == 1,
            label + ".active_hand must be integer 0 or 1");
        require(
            clip.range_start == expected_start,
            label + " must preserve contiguous manifest range order");
        require(
            clip.range_start < clip.range_stop,
            label + " range must be nonempty");
        require(
            clip.range_stop <= std::numeric_limits<int32_t>::max(),
            label + " range exceeds runtime frame index capacity");
        require(
            seen_ranges.emplace(clip.range_start, clip.range_stop).second,
            label + " duplicates a manifest range");
        const auto inserted = manifest.sequence_ordinals.emplace(
            clip.sequence_id, ordinal);
        require(inserted.second, label + " duplicates sequence_id");
        expected_start = clip.range_stop;
        manifest.clips.push_back(std::move(clip));
    }
    require(
        expected_start == kExpectedFrameCount,
        "full-pack manifest ranges must cover exactly 511250 frames");
    return manifest;
}

StableKey parse_stable_key(const JsonValue& value, const std::string& label) {
    const std::vector<JsonValue>& fields = json_array(value, label).get();
    require(fields.size() == 5U, label + " must contain exactly five fields");
    StableKey key;
    key.dataset_id = json_string(fields[0], label + "[0]");
    key.schema_version = json_integer(fields[1], label + "[1]");
    key.sequence_id = json_string(fields[2], label + "[2]");
    key.entry_local_frame = nonnegative_integer(fields[3], label + "[3]");
    key.active_hand = json_integer(fields[4], label + "[4]");
    return key;
}

void require_event_order(
    int64_t entry,
    int64_t alignment,
    int64_t contact,
    int64_t lift,
    int64_t hold,
    int64_t stop,
    const std::string& label) {
    require(
        0 <= entry && entry < alignment && alignment < contact &&
            contact < lift && lift < hold && hold < stop,
        label + " local event frames are not strictly ordered");
    require(
        alignment == contact - 1,
        label + " object alignment must be exactly Contact-1");
}

ReportTarget parse_report_target(const JsonValue& value) {
    const std::string label = "Task4 report.target";
    require_exact_keys(
        value,
        {"active_hand", "approach_direction_object", "contact_local_frame",
         "entry_local_frame", "grasp_position_object",
         "grasp_rotation_object", "hold_local_frame", "lift_local_frame",
         "object_alignment_local_frame", "object_alignment_position",
         "object_alignment_rotation", "object_dimensions", "object_id",
         "range_start", "range_stop", "sequence_id", "stop_local_frame",
         "table_position", "table_rotation", "table_size"},
        label);
    ReportTarget target;
    target.sequence_id = json_string(
        object_member(value, "sequence_id", label), label + ".sequence_id");
    target.object_id = json_string(
        object_member(value, "object_id", label), label + ".object_id");
    target.active_hand = json_integer(
        object_member(value, "active_hand", label), label + ".active_hand");
    target.range_start = nonnegative_integer(
        object_member(value, "range_start", label), label + ".range_start");
    target.range_stop = nonnegative_integer(
        object_member(value, "range_stop", label), label + ".range_stop");
    target.entry_local_frame = nonnegative_integer(
        object_member(value, "entry_local_frame", label),
        label + ".entry_local_frame");
    target.object_alignment_local_frame = nonnegative_integer(
        object_member(value, "object_alignment_local_frame", label),
        label + ".object_alignment_local_frame");
    target.contact_local_frame = nonnegative_integer(
        object_member(value, "contact_local_frame", label),
        label + ".contact_local_frame");
    target.lift_local_frame = nonnegative_integer(
        object_member(value, "lift_local_frame", label),
        label + ".lift_local_frame");
    target.hold_local_frame = nonnegative_integer(
        object_member(value, "hold_local_frame", label),
        label + ".hold_local_frame");
    target.stop_local_frame = nonnegative_integer(
        object_member(value, "stop_local_frame", label),
        label + ".stop_local_frame");
    target.object_alignment_position = finite_number_array<3>(
        object_member(value, "object_alignment_position", label),
        label + ".object_alignment_position");
    target.object_alignment_rotation = finite_number_array<4>(
        object_member(value, "object_alignment_rotation", label),
        label + ".object_alignment_rotation");
    target.table_position = finite_number_array<3>(
        object_member(value, "table_position", label),
        label + ".table_position");
    target.table_rotation = finite_number_array<4>(
        object_member(value, "table_rotation", label),
        label + ".table_rotation");
    target.table_size = finite_number_array<3>(
        object_member(value, "table_size", label), label + ".table_size");
    target.object_dimensions = finite_number_array<3>(
        object_member(value, "object_dimensions", label),
        label + ".object_dimensions");
    target.grasp_position_object = finite_number_array<3>(
        object_member(value, "grasp_position_object", label),
        label + ".grasp_position_object");
    target.grasp_rotation_object = finite_number_array<4>(
        object_member(value, "grasp_rotation_object", label),
        label + ".grasp_rotation_object");
    target.approach_direction_object = finite_number_array<3>(
        object_member(value, "approach_direction_object", label),
        label + ".approach_direction_object");
    return target;
}

RetainedCandidate parse_retained_candidate(
    const JsonValue& value,
    size_t ordinal) {
    const std::string label =
        "Task4 report.retained_candidates[" + std::to_string(ordinal) + "]";
    require_exact_keys(
        value,
        {"active_hand", "contact_hand_orientation_error_radians",
         "contact_hand_position_error_m", "contact_local_frame",
         "entry_local_frame", "entry_root_planar_speed_mps",
         "hand_object_clear", "hand_table_clear", "hold_local_frame",
         "lift_local_frame", "object_alignment_local_frame", "object_id",
         "root_table_clear", "root_x_object_m", "root_yaw_object_radians",
         "root_z_object_m", "sequence_id", "source_entry_frame",
         "stable_key", "static_path_feasible", "stop_local_frame"},
        label);
    RetainedCandidate candidate;
    candidate.stable_key = parse_stable_key(
        object_member(value, "stable_key", label), label + ".stable_key");
    candidate.sequence_id = json_string(
        object_member(value, "sequence_id", label), label + ".sequence_id");
    candidate.object_id = json_string(
        object_member(value, "object_id", label), label + ".object_id");
    candidate.active_hand = json_integer(
        object_member(value, "active_hand", label), label + ".active_hand");
    candidate.source_entry_frame = nonnegative_integer(
        object_member(value, "source_entry_frame", label),
        label + ".source_entry_frame");
    candidate.entry_local_frame = nonnegative_integer(
        object_member(value, "entry_local_frame", label),
        label + ".entry_local_frame");
    candidate.object_alignment_local_frame = nonnegative_integer(
        object_member(value, "object_alignment_local_frame", label),
        label + ".object_alignment_local_frame");
    candidate.contact_local_frame = nonnegative_integer(
        object_member(value, "contact_local_frame", label),
        label + ".contact_local_frame");
    candidate.lift_local_frame = nonnegative_integer(
        object_member(value, "lift_local_frame", label),
        label + ".lift_local_frame");
    candidate.hold_local_frame = nonnegative_integer(
        object_member(value, "hold_local_frame", label),
        label + ".hold_local_frame");
    candidate.stop_local_frame = nonnegative_integer(
        object_member(value, "stop_local_frame", label),
        label + ".stop_local_frame");
    candidate.root_x_object_report_m = json_number(
        object_member(value, "root_x_object_m", label),
        label + ".root_x_object_m");
    candidate.root_z_object_report_m = json_number(
        object_member(value, "root_z_object_m", label),
        label + ".root_z_object_m");
    candidate.root_yaw_object_report_radians = json_number(
        object_member(value, "root_yaw_object_radians", label),
        label + ".root_yaw_object_radians");
    candidate.root_x_object_m =
        static_cast<float>(candidate.root_x_object_report_m);
    candidate.root_z_object_m =
        static_cast<float>(candidate.root_z_object_report_m);
    candidate.root_yaw_object_radians =
        static_cast<float>(candidate.root_yaw_object_report_radians);
    require(
        std::isfinite(candidate.root_x_object_m) &&
            std::isfinite(candidate.root_z_object_m) &&
            std::isfinite(candidate.root_yaw_object_radians),
        label + " root values are not finite after float32 authorship");
    require(
        std::abs(candidate.root_yaw_object_report_radians) <=
            3.14159265358979323846,
        label + ".root_yaw_object_radians is outside [-pi, pi]");
    candidate.contact_hand_position_error_m = json_number(
        object_member(value, "contact_hand_position_error_m", label),
        label + ".contact_hand_position_error_m");
    candidate.contact_hand_orientation_error_radians = json_number(
        object_member(value, "contact_hand_orientation_error_radians", label),
        label + ".contact_hand_orientation_error_radians");
    candidate.entry_root_planar_speed_mps = json_number(
        object_member(value, "entry_root_planar_speed_mps", label),
        label + ".entry_root_planar_speed_mps");
    require(
        candidate.contact_hand_position_error_m >= 0.0 &&
            candidate.contact_hand_orientation_error_radians >= 0.0 &&
            candidate.entry_root_planar_speed_mps >= 0.0,
        label + " error/speed diagnostics must be nonnegative");
    require(
        json_boolean(
            object_member(value, "root_table_clear", label),
            label + ".root_table_clear"),
        label + " must be root-table clear");
    require(
        json_boolean(
            object_member(value, "hand_table_clear", label),
            label + ".hand_table_clear"),
        label + " must be hand-table clear");
    require(
        json_boolean(
            object_member(value, "hand_object_clear", label),
            label + ".hand_object_clear"),
        label + " must be hand-object clear");
    require(
        json_boolean(
            object_member(value, "static_path_feasible", label),
            label + ".static_path_feasible"),
        label + " must be statically path-feasible");
    return candidate;
}

SlotReport parse_slot_report(const JsonValue& root) {
    const std::string label = "Task4 report";
    require_exact_keys(
        root,
        {"dataset_id", "deduplicated_candidate_count",
         "rejected_counts_by_gate", "retained_candidates", "schema_version",
         "target"},
        label);
    SlotReport report;
    report.dataset_id = json_string(
        object_member(root, "dataset_id", label), label + ".dataset_id");
    report.schema_version = json_integer(
        object_member(root, "schema_version", label),
        label + ".schema_version");
    report.deduplicated_candidate_count = nonnegative_integer(
        object_member(root, "deduplicated_candidate_count", label),
        label + ".deduplicated_candidate_count");
    report.target = parse_report_target(object_member(root, "target", label));

    const JsonValue& rejected = object_member(
        root, "rejected_counts_by_gate", label);
    require_exact_keys(
        rejected,
        {"contact_orientation", "contact_position", "hand_mismatch",
         "hand_object", "hand_table", "root_table"},
        label + ".rejected_counts_by_gate");
    for (const auto& item : rejected.object) {
        report.rejected_counts_by_gate[item.first] = nonnegative_integer(
            item.second,
            label + ".rejected_counts_by_gate." + item.first);
    }

    const auto candidates_member = object_member(
        root, "retained_candidates", label);
    const JsonValue& candidates_value = candidates_member.get();
    const std::string candidates_label = label + ".retained_candidates";
    const auto candidates_array = json_array(
        candidates_value, candidates_label);
    const std::vector<JsonValue>& candidates = candidates_array.get();
    require(
        candidates.size() == kExpectedRetainedCandidateCount,
        "Task4 report must contain exactly 19 retained candidates");
    report.retained_candidates.reserve(candidates.size());
    for (size_t ordinal = 0U; ordinal < candidates.size(); ++ordinal) {
        report.retained_candidates.push_back(
            parse_retained_candidate(candidates[ordinal], ordinal));
    }
    return report;
}

const ManifestClip& manifest_clip_for_sequence(
    const Manifest& manifest,
    const std::string& sequence_id,
    size_t* ordinal,
    const std::string& label) {
    const auto found = manifest.sequence_ordinals.find(sequence_id);
    if (found == manifest.sequence_ordinals.end()) {
        fail(label + " sequence_id is absent from the full-pack manifest");
    }
    if (ordinal != nullptr) *ordinal = found->second;
    return manifest.clips.at(found->second);
}

void validate_target_against_manifest(
    const ReportTarget& target,
    const Manifest& manifest) {
    const std::string label = "Task4 report.target";
    require(
        target.sequence_id == kExpectedTargetSequenceId,
        label + " must be the frozen beer target sequence");
    require(target.active_hand == kRightHand, label + " must use the right hand");
    const ManifestClip& clip = manifest_clip_for_sequence(
        manifest, target.sequence_id, nullptr, label);
    require(target.object_id == clip.object_id, label + " object_id differs from manifest");
    require(target.active_hand == clip.active_hand, label + " active_hand differs from manifest");
    require(
        target.range_start == clip.range_start &&
            target.range_stop == clip.range_stop,
        label + " global range differs from manifest");
    const int64_t clip_length = clip.range_stop - clip.range_start;
    require(
        target.stop_local_frame == clip_length,
        label + " stop_local_frame differs from manifest clip length");
    require_event_order(
        target.entry_local_frame,
        target.object_alignment_local_frame,
        target.contact_local_frame,
        target.lift_local_frame,
        target.hold_local_frame,
        target.stop_local_frame,
        label);
    require(
        target.entry_local_frame == 101 && target.contact_local_frame == 126,
        label + " entry/contact provenance differs from the frozen target");
    for (double dimension : target.table_size) {
        require(dimension > 0.0, label + ".table_size must be positive");
    }
    for (double dimension : target.object_dimensions) {
        require(dimension > 0.0, label + ".object_dimensions must be positive");
    }
}

void validate_candidates_against_manifest(
    SlotReport& report,
    const Manifest& manifest) {
    std::set<std::tuple<std::string, int64_t, int64_t>> provenance;
    StableKey previous_key;
    bool have_previous = false;
    for (size_t ordinal = 0U;
         ordinal < report.retained_candidates.size();
         ++ordinal) {
        RetainedCandidate& candidate = report.retained_candidates[ordinal];
        const std::string label =
            "Task4 report.retained_candidates[" +
            std::to_string(ordinal) + "]";
        require(
            candidate.active_hand == kRightHand,
            label + " must use the right hand");
        require(
            candidate.stable_key.dataset_id == report.dataset_id &&
                candidate.stable_key.schema_version == report.schema_version &&
                candidate.stable_key.sequence_id == candidate.sequence_id &&
                candidate.stable_key.entry_local_frame ==
                    candidate.entry_local_frame &&
                candidate.stable_key.active_hand == candidate.active_hand,
            label + ".stable_key differs from candidate provenance");
        if (have_previous) {
            require(
                previous_key.ordered_tuple() < candidate.stable_key.ordered_tuple(),
                label + " violates strict stable-key report order");
        }
        previous_key = candidate.stable_key;
        have_previous = true;
        require(
            provenance.emplace(
                candidate.sequence_id,
                candidate.entry_local_frame,
                candidate.contact_local_frame).second,
            label + " duplicates retained candidate provenance");

        const ManifestClip& clip = manifest_clip_for_sequence(
            manifest,
            candidate.sequence_id,
            &candidate.source_clip_ordinal,
            label);
        require(
            candidate.object_id == clip.object_id,
            label + " object_id differs from manifest");
        require(
            candidate.active_hand == clip.active_hand,
            label + " active_hand differs from manifest");
        const int64_t clip_length = clip.range_stop - clip.range_start;
        require(
            candidate.stop_local_frame == clip_length,
            label + " stop_local_frame differs from manifest clip length");
        require(
            candidate.source_entry_frame < clip_length,
            label + ".source_entry_frame is outside the local clip range");
        require_event_order(
            candidate.entry_local_frame,
            candidate.object_alignment_local_frame,
            candidate.contact_local_frame,
            candidate.lift_local_frame,
            candidate.hold_local_frame,
            candidate.stop_local_frame,
            label);
    }
}

void validate_report_against_manifest(
    SlotReport& report,
    const Manifest& manifest) {
    require(
        report.dataset_id == kExpectedDatasetId &&
            report.dataset_id == manifest.dataset_id,
        "Task4 report dataset_id differs from the full-pack authority");
    require(
        report.schema_version == 1 &&
            report.schema_version == manifest.schema_version,
        "Task4 report schema_version differs from the full-pack authority");
    require(
        report.deduplicated_candidate_count == 82,
        "Task4 report deduplicated candidate count differs from the reviewed report");
    validate_target_against_manifest(report.target, manifest);
    validate_candidates_against_manifest(report, manifest);
}

void validate_loaded_pack_against_manifest(
    const interaction::Database& database,
    const interaction::Features& features,
    const Manifest& manifest) {
    require(
        database.frame_count == static_cast<uint32_t>(kExpectedFrameCount),
        "loaded interaction database frame count differs from manifest authority");
    require(
        database.clip_count == manifest.clips.size() &&
            database.range_starts.size() == manifest.clips.size() &&
            database.range_stops.size() == manifest.clips.size() &&
            database.active_hands.size() == manifest.clips.size(),
        "loaded interaction database clip arrays differ from manifest count");
    for (size_t ordinal = 0U; ordinal < manifest.clips.size(); ++ordinal) {
        const ManifestClip& clip = manifest.clips[ordinal];
        require(
            database.range_starts[ordinal] == clip.range_start &&
                database.range_stops[ordinal] == clip.range_stop,
            "loaded interaction database range differs from manifest clip " +
                std::to_string(ordinal));
        require(
            database.active_hands[ordinal] == clip.active_hand,
            "loaded interaction database active hand differs from manifest clip " +
                std::to_string(ordinal));
    }
    require(
        features.frame_count == database.frame_count,
        "loaded interaction features frame count differs from database");
    require(
        features.feature_count == 71U && features.dimension == 71U &&
            features.group_count == 5U,
        "loaded interaction feature dimensions differ from controller authority");
    require(
        features.offsets.size() == features.feature_count &&
            features.scales.size() == features.feature_count &&
            features.values.size() ==
                static_cast<size_t>(features.frame_count) *
                    static_cast<size_t>(features.feature_count),
        "loaded interaction feature arrays have inconsistent counts");
}

vec3 report_vec3(const std::array<double, 3>& value) {
    return vec3(
        static_cast<float>(value[0]),
        static_cast<float>(value[1]),
        static_cast<float>(value[2]));
}

quat report_quat(const std::array<double, 4>& value) {
    return quat(
        static_cast<float>(value[0]),
        static_cast<float>(value[1]),
        static_cast<float>(value[2]),
        static_cast<float>(value[3]));
}

bool exact_vec3(vec3 left, vec3 right) {
    return left.x == right.x && left.y == right.y && left.z == right.z;
}

bool exact_quat(quat left, quat right) {
    return left.w == right.w && left.x == right.x &&
        left.y == right.y && left.z == right.z;
}

interaction::InteractionTarget make_report_demo_target(
    const ReportTarget& source) {
    require(
        exact_quat(
            report_quat(source.table_rotation),
            quat(1.0F, 0.0F, 0.0F, 0.0F)),
        "Task4 report target table rotation must be identity for demo mapping");
    interaction::InteractionTarget target;
    target.handle = {1U, 1U};
    target.object_profile_id = 1U;
    target.table_world = {
        vec3(
            static_cast<float>(source.table_position[0]),
            static_cast<float>(source.table_position[1]),
            3.0F),
        report_quat(source.table_rotation),
    };
    target.table_size = report_vec3(source.table_size);
    const double table_z_translation = 3.0 - source.table_position[2];
    target.object_world = {
        vec3(
            static_cast<float>(source.object_alignment_position[0]),
            static_cast<float>(source.object_alignment_position[1]),
            static_cast<float>(
                source.object_alignment_position[2] + table_z_translation)),
        report_quat(source.object_alignment_rotation),
    };
    target.object_dimensions = report_vec3(source.object_dimensions);
    target.object_bounds = {
        vec3(), 0.5F * target.object_dimensions};
    target.state = interaction::ObjectState::Free;
    target.owner_request = 0U;

    interaction::GraspAffordance affordance;
    affordance.id = 1U;
    affordance.hand = interaction::Hand::Right;
    affordance.hand_in_object = {
        report_vec3(source.grasp_position_object),
        report_quat(source.grasp_rotation_object),
    };
    affordance.approach_direction_object =
        report_vec3(source.approach_direction_object);
    affordance.clearance_radius = 0.04F;
    require(
        affordance.interaction_slots.empty(),
        "preview target must not bake uncertified interaction slots");
    target.affordances.push_back(std::move(affordance));
    return target;
}

void validate_report_demo_target(
    const interaction::InteractionTarget& target) {
    require(
        target.handle == (interaction::TargetHandle{1U, 1U}) &&
            target.object_profile_id == 1U &&
            target.state == interaction::ObjectState::Free &&
            target.owner_request == 0U,
        "report-derived demo target identity/state differs");
    require(
        exact_vec3(
            target.table_world.position,
            vec3(0.0F, 0.360757500F, 3.0F)) &&
            exact_quat(
                target.table_world.rotation,
                quat(1.0F, 0.0F, 0.0F, 0.0F)) &&
            exact_vec3(
                target.table_size,
                vec3(2.0F, 0.0399999991F, 0.600000024F)),
        "report-derived demo table differs from frozen beer constants");
    require(
        exact_vec3(
            target.object_world.position,
            vec3(0.00394439697F, 0.503655553F, 2.77000808716F)) &&
            exact_quat(
                target.object_world.rotation,
                quat(
                    -0.0669774629F,
                    0.670088462F,
                    0.0799996098F,
                    -0.734911910F)) &&
            exact_vec3(
                target.object_dimensions,
                vec3(0.0645366386F, 0.0645366609F, 0.240097240F)) &&
            exact_vec3(target.object_bounds.center_object, vec3()) &&
            exact_vec3(
                target.object_bounds.half_extents_object,
                0.5F * target.object_dimensions),
        "report-derived demo object differs from frozen beer constants");
    require(
        target.affordances.size() == 1U,
        "report-derived demo target must have exactly one affordance");
    const interaction::GraspAffordance& affordance =
        target.affordances.front();
    require(
        affordance.id == 1U &&
            affordance.hand == interaction::Hand::Right &&
            affordance.interaction_slots.empty() &&
            affordance.clearance_radius == 0.04F &&
            exact_vec3(
                affordance.hand_in_object.position,
                vec3(0.0930671170F, -0.119263843F, 0.0375832170F)) &&
            exact_quat(
                affordance.hand_in_object.rotation,
                quat(
                    0.308746904F,
                    0.0609171167F,
                    -0.155041456F,
                    0.936443567F)) &&
            exact_vec3(
                affordance.approach_direction_object,
                vec3(-0.997760296F, 0.0F, 0.0668911785F)),
        "report-derived demo grasp differs from frozen beer constants");
}

struct RegisteredTarget {
    interaction::TargetRegistry registry;
    interaction::TargetHandle handle{};
    size_t registration_count = 0U;
};

RegisteredTarget register_report_demo_target(const ReportTarget& source) {
    const interaction::InteractionTarget authored =
        make_report_demo_target(source);
    validate_report_demo_target(authored);
    RegisteredTarget result;
    result.handle = result.registry.upsert(authored);
    ++result.registration_count;
    require(
        result.registration_count == 1U && result.handle == authored.handle,
        "preview target must be registered exactly once with handle {1,1}");
    const interaction::InteractionTarget* found =
        result.registry.find(result.handle);
    require(
        found != nullptr &&
            result.registry.find_by_id(result.handle.id) == found &&
            interaction::same_interaction_target_snapshot(*found, authored),
        "registered preview target lookup differs from authored snapshot");
    return result;
}

int containing_flat_clip_stop(const database& flat_database, int frame) {
    for (int range = 0; range < flat_database.nranges(); ++range) {
        if (frame >= flat_database.range_starts(range) &&
            frame < flat_database.range_stops(range)) {
            return flat_database.range_stops(range);
        }
    }
    return -1;
}

void validate_flat_database_shape(const database& flat_database) {
    require(flat_database.nframes() > 0, "flat database has no frames");
    require(
        flat_database.nbones() ==
            static_cast<int>(interaction::kFlatControllerBoneCount),
        "flat database bone count does not match the 23-bone controller");
    require(
        flat_database.bone_velocities.rows == flat_database.nframes() &&
            flat_database.bone_velocities.cols == flat_database.nbones() &&
            flat_database.bone_rotations.rows == flat_database.nframes() &&
            flat_database.bone_rotations.cols == flat_database.nbones() &&
            flat_database.bone_angular_velocities.rows ==
                flat_database.nframes() &&
            flat_database.bone_angular_velocities.cols ==
                flat_database.nbones() &&
            flat_database.contact_states.rows == flat_database.nframes() &&
            flat_database.contact_states.cols >= 2,
        "flat database pose arrays have inconsistent shapes");
    require(
        flat_database.bone_parents.size ==
            static_cast<int>(interaction::kFlatControllerBoneCount),
        "flat database parent count does not match the 23-bone controller");
    for (size_t bone = 0U;
         bone < interaction::kFlatControllerBoneCount;
         ++bone) {
        require(
            flat_database.bone_parents(static_cast<int>(bone)) ==
                interaction::kFlatControllerParents[bone],
            "flat database parent tree does not match the controller");
    }
}

interaction::Pose make_interaction_reference(
    const interaction::Database& interaction_database) {
    require(
        interaction_database.clip_count > 0U &&
            !interaction_database.range_starts.empty(),
        "interaction pack has no reference frame");
    interaction::Pose reference = interaction::pose_at_frame(
        interaction_database, interaction_database.range_starts.at(0U));
    reference.velocities.fill(vec3());
    reference.angular_velocities.fill(vec3());
    reference.hand_dof = interaction::kFlatControllerRestHandDof;
    reference.hand_dof_velocities =
        interaction::kFlatControllerRestHandDofVelocities;
    reference.foot_contacts = {};
    return reference;
}

interaction::FlatControllerPose flat_pose_at(
    const database& flat_database,
    int frame) {
    require(
        frame >= 0 && frame < flat_database.nframes(),
        "flat pose frame is out of range");
    interaction::FlatControllerPose pose{};
    for (size_t bone = 0U;
         bone < interaction::kFlatControllerBoneCount;
         ++bone) {
        const int index = static_cast<int>(bone);
        pose.positions[bone] = flat_database.bone_positions(frame, index);
        pose.velocities[bone] = flat_database.bone_velocities(frame, index);
        pose.rotations[bone] = flat_database.bone_rotations(frame, index);
        pose.angular_velocities[bone] =
            flat_database.bone_angular_velocities(frame, index);
    }
    pose.foot_contacts[0] =
        flat_database.contact_states(frame, 0) ? 1U : 0U;
    pose.foot_contacts[1] =
        flat_database.contact_states(frame, 1) ? 1U : 0U;
    return pose;
}

struct FixedControllerPoseBridge {
    interaction::FlatControllerPose flat_reference{};
    interaction::Pose interaction_reference{};
};

FixedControllerPoseBridge make_fixed_controller_pose_bridge(
    const database& flat_database,
    const interaction::Database& interaction_database) {
    require(
        flat_database.nranges() > 0,
        "flat database has no fixed controller reference frame");
    FixedControllerPoseBridge bridge{};
    bridge.flat_reference = flat_pose_at(
        flat_database, flat_database.range_starts(0));
    bridge.flat_reference.velocities.fill(vec3());
    bridge.flat_reference.angular_velocities.fill(vec3());
    bridge.flat_reference.foot_contacts = {};

    bridge.interaction_reference =
        make_interaction_reference(interaction_database);
    constexpr size_t root = g1_skeleton::Simulation;
    bridge.interaction_reference.positions[root] =
        bridge.flat_reference.positions[0];
    bridge.interaction_reference.velocities[root] =
        bridge.flat_reference.velocities[0];
    bridge.interaction_reference.rotations[root] =
        bridge.flat_reference.rotations[0];
    bridge.interaction_reference.angular_velocities[root] =
        bridge.flat_reference.angular_velocities[0];
    return bridge;
}

interaction::LocomotionSnapshot make_stationary_snapshot(
    const database& flat_database,
    int frame,
    const FixedControllerPoseBridge& bridge) {
    const int clip_stop = containing_flat_clip_stop(flat_database, frame);
    require(clip_stop > frame, "stationary frame is outside a flat clip");
    interaction::LocomotionSnapshot snapshot{};
    snapshot.pose = interaction::expand_flat_controller_pose(
        flat_pose_at(flat_database, frame),
        bridge.interaction_reference,
        bridge.flat_reference);
    for (size_t index = 0U;
         index < locomotion_timing::kTrajectoryFrameOffsets.size();
         ++index) {
        const int future =
            frame + locomotion_timing::kTrajectoryFrameOffsets[index];
        require(
            future < clip_stop,
            "stationary snapshot trajectory crosses a flat clip boundary");
        snapshot.future_root_positions[index] =
            flat_database.bone_positions(future, 0);
        snapshot.future_root_rotations[index] =
            flat_database.bone_rotations(future, 0);
    }
    return snapshot;
}

void require_fixed_reference_calibration(
    const database& flat_database,
    const FixedControllerPoseBridge& bridge) {
    const int reference_frame = flat_database.range_starts(0);
    const interaction::FlatControllerPose current =
        flat_pose_at(flat_database, reference_frame);
    const interaction::Pose expected =
        interaction::expand_flat_controller_pose(
            current,
            bridge.interaction_reference,
            bridge.flat_reference);
    const interaction::Pose legacy =
        interaction::expand_flat_controller_pose(
            current, bridge.interaction_reference);
    const interaction::LocomotionSnapshot actual =
        make_stationary_snapshot(
            flat_database, reference_frame, bridge);
    const interaction::WorldPose expected_world =
        interaction::world_pose(expected);
    const interaction::WorldPose legacy_world =
        interaction::world_pose(legacy);
    const interaction::WorldPose actual_world =
        interaction::world_pose(actual.pose);

    float legacy_maximum_error_m = 0.0F;
    float actual_maximum_error_m = 0.0F;
    for (const interaction::FlatControllerAnchor anchor :
         interaction::kFlatControllerAnchors) {
        legacy_maximum_error_m = std::max(
            legacy_maximum_error_m,
            length(
                legacy_world.positions[anchor.g1_bone] -
                expected_world.positions[anchor.g1_bone]));
        actual_maximum_error_m = std::max(
            actual_maximum_error_m,
            length(
                actual_world.positions[anchor.g1_bone] -
                expected_world.positions[anchor.g1_bone]));
    }
    require(
        legacy_maximum_error_m > 0.20F,
        "fixed-reference witness no longer distinguishes the legacy bridge");
    require(
        actual_maximum_error_m <= 1.0e-4F,
        "stationary snapshot differs from fixed-reference bridge calibration");
}

void validate_stationary_frames(const std::vector<int>& frames) {
    require(
        frames.size() == kExpectedStationarySnapshotCount,
        "stationary candidate count changed: expected=186 actual=" +
            std::to_string(frames.size()));
    require(
        frames.front() == 0 && frames.at(92U) == 92 &&
            frames.at(93U) == 118 && frames.back() == 210,
        "stationary candidate spans changed");
    for (size_t index = 0U; index < 93U; ++index) {
        require(
            frames[index] == static_cast<int>(index) &&
                frames[93U + index] == 118 + static_cast<int>(index),
            "stationary candidate span is not contiguous");
    }
}

struct FrozenStationarySnapshot {
    int frame = -1;
    interaction::LocomotionSnapshot snapshot{};
    uint64_t fingerprint = 0U;
};

std::vector<FrozenStationarySnapshot> make_frozen_stationary_snapshots(
    const database& flat_database,
    const FixedControllerPoseBridge& bridge) {
    const std::vector<int> frames =
        stationary_motion_matching::derive_candidates(flat_database);
    validate_stationary_frames(frames);
    std::vector<FrozenStationarySnapshot> frozen;
    frozen.reserve(frames.size());
    for (int frame : frames) {
        FrozenStationarySnapshot record;
        record.frame = frame;
        record.snapshot = make_stationary_snapshot(
            flat_database, frame, bridge);
        record.fingerprint =
            interaction::runtime_detail::locomotion_snapshot_fingerprint(
                record.snapshot);
        require(
            record.fingerprint != 0U,
            "stationary snapshot fingerprint must be nonzero");
        frozen.push_back(std::move(record));
    }
    require(
        frozen.size() == kExpectedStationarySnapshotCount,
        "frozen stationary snapshot count changed");
    return frozen;
}

interaction::PickEntryRoot map_candidate_root_to_world(
    const RetainedCandidate& candidate,
    const interaction::InteractionTarget& target) {
    const vec3 projected = quat_mul_vec3(
        target.object_world.rotation, vec3(0.0F, 0.0F, 1.0F));
    const double planar_length_squared =
        static_cast<double>(projected.x) * projected.x +
        static_cast<double>(projected.z) * projected.z;
    require(
        std::isfinite(planar_length_squared) &&
            planar_length_squared > 0.0,
        "preview target has no finite planar object facing");
    const double inverse_length = 1.0 / std::sqrt(planar_length_squared);
    const vec3 object_forward(
        static_cast<float>(projected.x * inverse_length),
        0.0F,
        static_cast<float>(projected.z * inverse_length));
    const vec3 object_right(
        object_forward.z, 0.0F, -object_forward.x);
    const float world_x = target.object_world.position.x +
        candidate.root_x_object_m * object_right.x +
        candidate.root_z_object_m * object_forward.x;
    const float world_z = target.object_world.position.z +
        candidate.root_x_object_m * object_right.z +
        candidate.root_z_object_m * object_forward.z;
    const float object_yaw = std::atan2(
        object_forward.x, object_forward.z);
    const float unwrapped_yaw =
        object_yaw + candidate.root_yaw_object_radians;
    const float world_yaw = std::atan2(
        std::sin(unwrapped_yaw), std::cos(unwrapped_yaw));
    require(
        std::isfinite(world_x) && std::isfinite(world_z) &&
            std::isfinite(world_yaw),
        "candidate object-planar root does not map to finite world geometry");
    return {world_x, world_z, world_yaw};
}

struct PreviewCaseResult {
    interaction::PickEntryPreview preview{};
};

struct ReadyCostEvidence {
    int stationary_flat_frame = -1;
    float total_cost = 0.0F;
};

struct MatcherProvenance {
    size_t clip_ordinal = 0U;
    int32_t entry_global_frame = -1;
    int32_t contact_global_frame = -1;
    int32_t entry_local_frame = -1;
    int32_t contact_local_frame = -1;
    std::string sequence_id;
};

struct CandidatePreviewSummary {
    size_t candidate_ordinal = 0U;
    interaction::PickEntryRoot prospective_root{};
    bool path_feasible = false;
    bool match_ready = false;
    interaction::Reason path_reason = interaction::Reason::None;
    interaction::Reason match_reason = interaction::Reason::None;
    int32_t feasible_entry_frame = -1;
    int32_t preview_contact_frame = -1;
    std::optional<MatcherProvenance> matcher_provenance{};
    std::vector<ReadyCostEvidence> runtime_ready_cost_evidence;
    std::optional<int> selected_stationary_flat_frame{};
    uint64_t snapshot_fingerprint = 0U;
    std::optional<float> total_cost{};
    size_t preview_count = 0U;
};

struct PreviewEvaluation {
    std::vector<CandidatePreviewSummary> candidates;
    size_t case_count = 0U;
    size_t ready_candidate_count = 0U;
    size_t worker_count = 0U;
};

bool same_pick_entry_root(
    interaction::PickEntryRoot left,
    interaction::PickEntryRoot right) {
    return left.world_x == right.world_x &&
        left.world_z == right.world_z &&
        left.world_yaw_radians == right.world_yaw_radians;
}

size_t containing_manifest_clip(
    const Manifest& manifest,
    int32_t entry_global,
    int32_t contact_global,
    const std::string& label) {
    size_t found_ordinal = manifest.clips.size();
    size_t match_count = 0U;
    for (size_t ordinal = 0U; ordinal < manifest.clips.size(); ++ordinal) {
        const ManifestClip& clip = manifest.clips[ordinal];
        if (entry_global >= clip.range_start &&
            entry_global < contact_global &&
            contact_global < clip.range_stop) {
            found_ordinal = ordinal;
            ++match_count;
        }
    }
    require(
        match_count == 1U,
        label + " global entry/contact do not join exactly one manifest clip");
    return found_ordinal;
}

MatcherProvenance matcher_provenance_for(
    const interaction::MatchCandidate& candidate,
    const Manifest& manifest,
    const std::string& label) {
    require(
        candidate.clip >= 0 &&
            static_cast<size_t>(candidate.clip) < manifest.clips.size(),
        label + " selected matcher clip is outside the manifest");
    const size_t clip_ordinal = static_cast<size_t>(candidate.clip);
    const size_t joined_ordinal = containing_manifest_clip(
        manifest,
        candidate.entry_frame,
        candidate.contact_frame,
        label);
    require(
        joined_ordinal == clip_ordinal,
        label + " selected matcher clip ordinal differs from global frames");
    const ManifestClip& clip = manifest.clips[clip_ordinal];
    MatcherProvenance provenance;
    provenance.clip_ordinal = clip_ordinal;
    provenance.entry_global_frame = candidate.entry_frame;
    provenance.contact_global_frame = candidate.contact_frame;
    provenance.entry_local_frame = static_cast<int32_t>(
        candidate.entry_frame - clip.range_start);
    provenance.contact_local_frame = static_cast<int32_t>(
        candidate.contact_frame - clip.range_start);
    provenance.sequence_id = clip.sequence_id;
    require(
        provenance.entry_local_frame >= 0 &&
            provenance.entry_local_frame < provenance.contact_local_frame &&
            provenance.contact_local_frame <
                clip.range_stop - clip.range_start,
        label + " selected matcher local event frames are invalid");
    return provenance;
}

void validate_preview_case(
    const interaction::PickEntryPreview& preview,
    interaction::PickEntryRoot expected_root,
    const Manifest& manifest,
    const std::string& label) {
    require(
        same_pick_entry_root(preview.prospective_root, expected_root),
        label + " runtime did not preserve the prospective root");
    require(
        (preview.path_reason == interaction::Reason::None) ==
            preview.path_feasible,
        label + " path reason/feasibility invariant failed");
    require(
        (preview.match_reason == interaction::Reason::None) ==
            preview.match_ready,
        label + " match reason/readiness invariant failed");
    require(
        !preview.match_ready || preview.path_feasible,
        label + " match-ready preview is not path-feasible");
    if (preview.path_feasible) {
        static_cast<void>(containing_manifest_clip(
            manifest,
            preview.feasible_entry_frame,
            preview.contact_frame,
            label + " path evidence"));
    } else {
        require(
            preview.feasible_entry_frame == -1 &&
                preview.contact_frame == -1,
            label + " infeasible preview carries event frames");
    }
    if (preview.match_ready) {
        require(
            std::isfinite(preview.total_cost) && preview.total_cost >= 0.0F &&
                preview.match_candidate.total_cost == preview.total_cost,
            label + " ready preview has invalid selected total cost");
        static_cast<void>(matcher_provenance_for(
            preview.match_candidate, manifest, label + " matcher"));
    }
}

CandidatePreviewSummary reduce_candidate_previews(
    size_t candidate_ordinal,
    interaction::PickEntryRoot prospective_root,
    const std::vector<PreviewCaseResult>& case_results,
    const std::vector<FrozenStationarySnapshot>& stationary_snapshots,
    const Manifest& manifest) {
    CandidatePreviewSummary summary;
    summary.candidate_ordinal = candidate_ordinal;
    summary.prospective_root = prospective_root;
    summary.preview_count = stationary_snapshots.size();
    const size_t first_case =
        candidate_ordinal * stationary_snapshots.size();
    std::optional<size_t> best_ready_index;
    std::optional<size_t> first_path_feasible_index;
    for (size_t stationary_index = 0U;
         stationary_index < stationary_snapshots.size();
         ++stationary_index) {
        const interaction::PickEntryPreview& preview =
            case_results[first_case + stationary_index].preview;
        const std::string label =
            "candidate " + std::to_string(candidate_ordinal) +
            " stationary frame " +
            std::to_string(stationary_snapshots[stationary_index].frame);
        validate_preview_case(preview, prospective_root, manifest, label);
        if (preview.path_feasible && !first_path_feasible_index.has_value()) {
            first_path_feasible_index = stationary_index;
        }
        if (!preview.match_ready) continue;
        summary.runtime_ready_cost_evidence.push_back(
            {stationary_snapshots[stationary_index].frame,
             preview.total_cost});
        if (!best_ready_index.has_value()) {
            best_ready_index = stationary_index;
            continue;
        }
        const interaction::PickEntryPreview& current_best =
            case_results[first_case + *best_ready_index].preview;
        const int frame = stationary_snapshots[stationary_index].frame;
        const int best_frame =
            stationary_snapshots[*best_ready_index].frame;
        if (preview.total_cost < current_best.total_cost ||
            (preview.total_cost == current_best.total_cost &&
             frame < best_frame)) {
            best_ready_index = stationary_index;
        }
    }

    const size_t representative_index = best_ready_index.has_value()
        ? *best_ready_index
        : first_path_feasible_index.value_or(0U);
    const interaction::PickEntryPreview& representative =
        case_results[first_case + representative_index].preview;
    summary.snapshot_fingerprint =
        stationary_snapshots[representative_index].fingerprint;
    summary.path_feasible = representative.path_feasible;
    summary.match_ready = representative.match_ready;
    summary.path_reason = representative.path_reason;
    summary.match_reason = representative.match_reason;
    if (best_ready_index.has_value()) {
        summary.path_feasible = true;
        summary.match_ready = true;
        summary.path_reason = interaction::Reason::None;
        summary.match_reason = interaction::Reason::None;
        summary.feasible_entry_frame =
            representative.match_candidate.entry_frame;
        summary.preview_contact_frame =
            representative.match_candidate.contact_frame;
        summary.matcher_provenance = matcher_provenance_for(
            representative.match_candidate,
            manifest,
            "candidate " + std::to_string(candidate_ordinal));
        summary.selected_stationary_flat_frame =
            stationary_snapshots[representative_index].frame;
        summary.total_cost = representative.total_cost;
        require(
            summary.feasible_entry_frame ==
                    summary.matcher_provenance->entry_global_frame &&
                summary.preview_contact_frame ==
                    summary.matcher_provenance->contact_global_frame,
            "ready candidate frames differ from selected matcher provenance");
    } else {
        summary.feasible_entry_frame = representative.feasible_entry_frame;
        summary.preview_contact_frame = representative.contact_frame;
        require(
            summary.runtime_ready_cost_evidence.empty() &&
                !summary.matcher_provenance.has_value() &&
                !summary.selected_stationary_flat_frame.has_value() &&
                !summary.total_cost.has_value(),
            "non-ready candidate retained ready-only evidence");
    }
    require(
        summary.snapshot_fingerprint != 0U &&
            summary.preview_count == kExpectedStationarySnapshotCount,
        "candidate preview summary lost stationary evidence authority");
    return summary;
}

PreviewEvaluation evaluate_unrestricted_previews(
    const interaction::Database& interaction_database,
    const interaction::Features& interaction_features,
    interaction::TargetRegistry& immutable_registry,
    interaction::TargetHandle target_handle,
    uint32_t affordance_id,
    const std::vector<RetainedCandidate>& candidates,
    const std::vector<FrozenStationarySnapshot>& stationary_snapshots,
    const Manifest& manifest) {
    const interaction::InteractionTarget* target =
        immutable_registry.find(target_handle);
    require(
        target != nullptr && target->affordances.size() == 1U &&
            target->affordances.front().id == affordance_id,
        "unrestricted preview target lookup failed");
    std::vector<interaction::PickEntryRoot> prospective_roots;
    prospective_roots.reserve(candidates.size());
    for (const RetainedCandidate& candidate : candidates) {
        prospective_roots.push_back(
            map_candidate_root_to_world(candidate, *target));
    }

    const size_t case_count =
        candidates.size() * stationary_snapshots.size();
    require(case_count > 0U, "unrestricted preview case matrix is empty");
    std::vector<PreviewCaseResult> case_results(case_count);
    std::vector<std::exception_ptr> case_errors(case_count);
    const unsigned int hardware = std::thread::hardware_concurrency();
    const size_t available_workers =
        hardware == 0U ? 4U : static_cast<size_t>(hardware);
    const size_t worker_count = std::min(
        {kMaximumPreviewWorkers, available_workers, case_count});
    require(worker_count > 0U, "unrestricted preview has no workers");
    std::atomic<size_t> next_case{0U};
    std::vector<std::exception_ptr> worker_errors(worker_count);
    std::vector<std::thread> workers;
    workers.reserve(worker_count);
    try {
        for (size_t worker_index = 0U;
             worker_index < worker_count;
             ++worker_index) {
            workers.emplace_back([&, worker_index]() {
                try {
                    interaction::InteractionRuntime runtime(
                        interaction_database,
                        interaction_features,
                        immutable_registry,
                        interaction::RuntimeConfig{});
                    for (;;) {
                        const size_t case_index = next_case.fetch_add(
                            1U, std::memory_order_relaxed);
                        if (case_index >= case_count) return;
                        const size_t candidate_index =
                            case_index / stationary_snapshots.size();
                        const size_t stationary_index =
                            case_index % stationary_snapshots.size();
                        try {
                            case_results[case_index].preview =
                                runtime.preview_pick(
                                    stationary_snapshots[stationary_index].snapshot,
                                    prospective_roots[candidate_index],
                                    target_handle,
                                    affordance_id);
                        } catch (...) {
                            case_errors[case_index] =
                                std::current_exception();
                        }
                    }
                } catch (...) {
                    worker_errors[worker_index] = std::current_exception();
                }
            });
        }
    } catch (...) {
        for (std::thread& worker : workers) {
            if (worker.joinable()) worker.join();
        }
        throw;
    }
    for (std::thread& worker : workers) worker.join();
    for (size_t worker_index = 0U;
         worker_index < worker_errors.size();
         ++worker_index) {
        if (!worker_errors[worker_index]) continue;
        try {
            std::rethrow_exception(worker_errors[worker_index]);
        } catch (const std::exception& error) {
            fail(
                "preview worker " + std::to_string(worker_index) +
                " failed: " + error.what());
        } catch (...) {
            fail(
                "preview worker " + std::to_string(worker_index) +
                " failed with a non-standard exception");
        }
    }
    for (size_t case_index = 0U;
         case_index < case_errors.size();
         ++case_index) {
        if (!case_errors[case_index]) continue;
        try {
            std::rethrow_exception(case_errors[case_index]);
        } catch (const std::exception& error) {
            fail(
                "preview case " + std::to_string(case_index) +
                " failed: " + error.what());
        } catch (...) {
            fail(
                "preview case " + std::to_string(case_index) +
                " failed with a non-standard exception");
        }
    }

    PreviewEvaluation evaluation;
    evaluation.case_count = case_count;
    evaluation.worker_count = worker_count;
    evaluation.candidates.reserve(candidates.size());
    for (size_t candidate_index = 0U;
         candidate_index < candidates.size();
         ++candidate_index) {
        CandidatePreviewSummary summary = reduce_candidate_previews(
            candidate_index,
            prospective_roots[candidate_index],
            case_results,
            stationary_snapshots,
            manifest);
        if (summary.path_feasible && summary.match_ready) {
            ++evaluation.ready_candidate_count;
        }
        evaluation.candidates.push_back(std::move(summary));
    }
    return evaluation;
}

const char* reason_name(interaction::Reason reason) {
    using interaction::Reason;
    switch (reason) {
    case Reason::None: return "None";
    case Reason::PackUnavailable: return "PackUnavailable";
    case Reason::TargetUnavailable: return "TargetUnavailable";
    case Reason::TargetChanged: return "TargetChanged";
    case Reason::OutOfRange: return "OutOfRange";
    case Reason::NoCandidate: return "NoCandidate";
    case Reason::PoorMatch: return "PoorMatch";
    case Reason::BlockedPath: return "BlockedPath";
    case Reason::CorrectionLimit: return "CorrectionLimit";
    case Reason::Cancelled: return "Cancelled";
    case Reason::ContactPosition: return "ContactPosition";
    case Reason::ContactOrientation: return "ContactOrientation";
    case Reason::JointLimit: return "JointLimit";
    case Reason::LostContact: return "LostContact";
    case Reason::ClipEnded: return "ClipEnded";
    case Reason::Reset: return "Reset";
    case Reason::SurfaceUnavailable: return "SurfaceUnavailable";
    case Reason::SurfaceChanged: return "SurfaceChanged";
    case Reason::PlacementOutOfBounds: return "PlacementOutOfBounds";
    case Reason::ReleasePosition: return "ReleasePosition";
    case Reason::ReleaseOrientation: return "ReleaseOrientation";
    }
    fail("runtime preview produced an unknown reason");
}

void append_json_string(std::string& output, std::string_view value) {
    static constexpr char hexadecimal[] = "0123456789abcdef";
    output.push_back('"');
    for (unsigned char character : value) {
        switch (character) {
        case '"': output += "\\\""; break;
        case '\\': output += "\\\\"; break;
        case '\b': output += "\\b"; break;
        case '\f': output += "\\f"; break;
        case '\n': output += "\\n"; break;
        case '\r': output += "\\r"; break;
        case '\t': output += "\\t"; break;
        default:
            if (character < 0x20U) {
                output += "\\u00";
                output.push_back(hexadecimal[(character >> 4U) & 0x0fU]);
                output.push_back(hexadecimal[character & 0x0fU]);
            } else {
                require(
                    character < 0x80U,
                    "canonical evidence strings must be ASCII");
                output.push_back(static_cast<char>(character));
            }
        }
    }
    output.push_back('"');
}

class CanonicalJsonObject {
public:
    explicit CanonicalJsonObject(std::string& output) : output_(output) {
        output_.push_back('{');
    }

    void key(std::string_view value) {
        require(!finished_, "canonical JSON object is already closed");
        const std::string current(value);
        require(
            previous_key_.empty() || previous_key_ < current,
            "canonical JSON object keys are not strictly alphabetical");
        if (!first_) output_.push_back(',');
        append_json_string(output_, value);
        output_.push_back(':');
        previous_key_ = current;
        first_ = false;
    }

    void finish() {
        require(!finished_, "canonical JSON object was closed twice");
        output_.push_back('}');
        finished_ = true;
    }

private:
    std::string& output_;
    std::string previous_key_;
    bool first_ = true;
    bool finished_ = false;
};

std::string significant_number(double value, int precision) {
    require(std::isfinite(value), "JSON evidence number must be finite");
    if (value == 0.0) return "0.0";
    std::ostringstream output;
    output.imbue(std::locale::classic());
    output << std::setprecision(precision) << std::defaultfloat << value;
    const std::string result = output.str();
    require(!result.empty(), "could not format JSON evidence number");
    return result;
}

std::string float32_hex(float value) {
    require(std::isfinite(value), "float32 evidence must be finite");
    uint32_t bits = 0U;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    std::ostringstream output;
    output.imbue(std::locale::classic());
    output << "0x" << std::hex << std::setfill('0')
           << std::setw(8) << bits;
    return output.str();
}

void append_matcher_provenance(
    std::string& output,
    const MatcherProvenance& provenance) {
    CanonicalJsonObject object(output);
    object.key("clip_ordinal");
    output += std::to_string(provenance.clip_ordinal);
    object.key("contact_global_frame");
    output += std::to_string(provenance.contact_global_frame);
    object.key("contact_local_frame");
    output += std::to_string(provenance.contact_local_frame);
    object.key("entry_global_frame");
    output += std::to_string(provenance.entry_global_frame);
    object.key("entry_local_frame");
    output += std::to_string(provenance.entry_local_frame);
    object.key("sequence_id");
    append_json_string(output, provenance.sequence_id);
    object.finish();
}

void append_ready_cost_evidence(
    std::string& output,
    const std::vector<ReadyCostEvidence>& evidence) {
    output.push_back('[');
    int previous_frame = -1;
    for (size_t index = 0U; index < evidence.size(); ++index) {
        if (index != 0U) output.push_back(',');
        require(
            evidence[index].stationary_flat_frame > previous_frame,
            "ready cost evidence is not in stationary-frame order");
        previous_frame = evidence[index].stationary_flat_frame;
        CanonicalJsonObject object(output);
        object.key("stationary_flat_frame");
        output += std::to_string(evidence[index].stationary_flat_frame);
        object.key("total_cost");
        output += significant_number(
            evidence[index].total_cost,
            std::numeric_limits<float>::max_digits10);
        object.finish();
    }
    output.push_back(']');
}

std::string serialize_candidate_preview(
    const RetainedCandidate& source,
    const CandidatePreviewSummary& summary) {
    require(
        summary.candidate_ordinal < kExpectedRetainedCandidateCount &&
            summary.preview_count == kExpectedStationarySnapshotCount,
        "candidate summary identity/count differs before serialization");
    std::string output;
    CanonicalJsonObject object(output);
    object.key("candidate_ordinal");
    output += std::to_string(summary.candidate_ordinal);
    object.key("contact_local_frame");
    output += std::to_string(source.contact_local_frame);
    object.key("entry_local_frame");
    output += std::to_string(source.entry_local_frame);
    object.key("feasible_entry_frame");
    output += std::to_string(summary.feasible_entry_frame);
    object.key("match_ready");
    output += summary.match_ready ? "true" : "false";
    object.key("match_reason");
    append_json_string(output, reason_name(summary.match_reason));
    object.key("matcher_provenance");
    if (summary.matcher_provenance.has_value()) {
        append_matcher_provenance(output, *summary.matcher_provenance);
    } else {
        output += "null";
    }
    object.key("path_feasible");
    output += summary.path_feasible ? "true" : "false";
    object.key("path_reason");
    append_json_string(output, reason_name(summary.path_reason));
    object.key("preview_authority");
    {
        CanonicalJsonObject authority(output);
        authority.key("candidate_subset_authority");
        output += "false";
        authority.key("mode");
        append_json_string(output, "global_unrestricted");
        authority.key("runtime_clip_allowlist_applied");
        output += "false";
        authority.finish();
    }
    object.key("preview_contact_frame");
    output += std::to_string(summary.preview_contact_frame);
    object.key("preview_count");
    output += std::to_string(summary.preview_count);
    object.key("prospective_root_x_object_f32_hex");
    append_json_string(output, float32_hex(source.root_x_object_m));
    object.key("prospective_root_x_object_m");
    output += significant_number(
        source.root_x_object_report_m, 9);
    object.key("prospective_root_yaw_object_f32_hex");
    append_json_string(
        output, float32_hex(source.root_yaw_object_radians));
    object.key("prospective_root_yaw_object_radians");
    output += significant_number(
        source.root_yaw_object_report_radians, 9);
    object.key("prospective_root_z_object_f32_hex");
    append_json_string(output, float32_hex(source.root_z_object_m));
    object.key("prospective_root_z_object_m");
    output += significant_number(
        source.root_z_object_report_m, 9);
    object.key("record_type");
    append_json_string(output, "candidate_preview");
    object.key("runtime_ready_cost_evidence");
    append_ready_cost_evidence(
        output, summary.runtime_ready_cost_evidence);
    object.key("selected_stationary_flat_frame");
    if (summary.selected_stationary_flat_frame.has_value()) {
        output += std::to_string(*summary.selected_stationary_flat_frame);
    } else {
        output += "null";
    }
    object.key("sequence_id");
    append_json_string(output, source.sequence_id);
    object.key("snapshot_fingerprint");
    output += std::to_string(summary.snapshot_fingerprint);
    object.key("source_clip_ordinal");
    output += std::to_string(source.source_clip_ordinal);
    object.key("total_cost");
    if (summary.total_cost.has_value()) {
        output += significant_number(
            *summary.total_cost,
            std::numeric_limits<float>::max_digits10);
    } else {
        output += "null";
    }
    object.finish();
    return output;
}

void append_target_scalar(
    CanonicalJsonObject& scalars,
    std::string& output,
    std::string_view name,
    float value) {
    scalars.key(name);
    append_json_string(output, float32_hex(value));
}

void append_target_record(
    std::string& output,
    const ReportTarget& source,
    const interaction::InteractionTarget& target) {
    require(
        target.affordances.size() == 1U,
        "target scalar serialization requires one affordance");
    const interaction::GraspAffordance& affordance =
        target.affordances.front();
    CanonicalJsonObject object(output);
    object.key("contact_local_frame");
    output += std::to_string(source.contact_local_frame);
    object.key("entry_local_frame");
    output += std::to_string(source.entry_local_frame);
    object.key("scalars_f32_hex");
    {
        CanonicalJsonObject scalars(output);
        append_target_scalar(
            scalars, output, "approach_direction_object_x",
            affordance.approach_direction_object.x);
        append_target_scalar(
            scalars, output, "approach_direction_object_y",
            affordance.approach_direction_object.y);
        append_target_scalar(
            scalars, output, "approach_direction_object_z",
            affordance.approach_direction_object.z);
        append_target_scalar(
            scalars, output, "clearance_radius_m",
            affordance.clearance_radius);
        append_target_scalar(
            scalars, output, "grasp_position_object_x",
            affordance.hand_in_object.position.x);
        append_target_scalar(
            scalars, output, "grasp_position_object_y",
            affordance.hand_in_object.position.y);
        append_target_scalar(
            scalars, output, "grasp_position_object_z",
            affordance.hand_in_object.position.z);
        append_target_scalar(
            scalars, output, "grasp_rotation_object_w",
            affordance.hand_in_object.rotation.w);
        append_target_scalar(
            scalars, output, "grasp_rotation_object_x",
            affordance.hand_in_object.rotation.x);
        append_target_scalar(
            scalars, output, "grasp_rotation_object_y",
            affordance.hand_in_object.rotation.y);
        append_target_scalar(
            scalars, output, "grasp_rotation_object_z",
            affordance.hand_in_object.rotation.z);
        append_target_scalar(
            scalars, output, "object_dimensions_x",
            target.object_dimensions.x);
        append_target_scalar(
            scalars, output, "object_dimensions_y",
            target.object_dimensions.y);
        append_target_scalar(
            scalars, output, "object_dimensions_z",
            target.object_dimensions.z);
        append_target_scalar(
            scalars, output, "object_position_x",
            target.object_world.position.x);
        append_target_scalar(
            scalars, output, "object_position_y",
            target.object_world.position.y);
        append_target_scalar(
            scalars, output, "object_position_z",
            target.object_world.position.z);
        append_target_scalar(
            scalars, output, "object_rotation_w",
            target.object_world.rotation.w);
        append_target_scalar(
            scalars, output, "object_rotation_x",
            target.object_world.rotation.x);
        append_target_scalar(
            scalars, output, "object_rotation_y",
            target.object_world.rotation.y);
        append_target_scalar(
            scalars, output, "object_rotation_z",
            target.object_world.rotation.z);
        append_target_scalar(
            scalars, output, "table_position_x",
            target.table_world.position.x);
        append_target_scalar(
            scalars, output, "table_position_y",
            target.table_world.position.y);
        append_target_scalar(
            scalars, output, "table_position_z",
            target.table_world.position.z);
        append_target_scalar(
            scalars, output, "table_rotation_w",
            target.table_world.rotation.w);
        append_target_scalar(
            scalars, output, "table_rotation_x",
            target.table_world.rotation.x);
        append_target_scalar(
            scalars, output, "table_rotation_y",
            target.table_world.rotation.y);
        append_target_scalar(
            scalars, output, "table_rotation_z",
            target.table_world.rotation.z);
        append_target_scalar(
            scalars, output, "table_size_x", target.table_size.x);
        append_target_scalar(
            scalars, output, "table_size_y", target.table_size.y);
        append_target_scalar(
            scalars, output, "table_size_z", target.table_size.z);
        scalars.finish();
    }
    object.key("sequence_id");
    append_json_string(output, source.sequence_id);
    object.finish();
}

void append_selected_slot(
    std::string& output,
    const RetainedCandidate& source,
    uint32_t slot_id) {
    CanonicalJsonObject slot(output);
    slot.key("contact_local_frame");
    output += std::to_string(source.contact_local_frame);
    slot.key("entry_local_frame");
    output += std::to_string(source.entry_local_frame);
    slot.key("prospective_root_x_object_f32_hex");
    append_json_string(output, float32_hex(source.root_x_object_m));
    slot.key("prospective_root_yaw_object_f32_hex");
    append_json_string(
        output, float32_hex(source.root_yaw_object_radians));
    slot.key("prospective_root_z_object_f32_hex");
    append_json_string(output, float32_hex(source.root_z_object_m));
    slot.key("sequence_id");
    append_json_string(output, source.sequence_id);
    slot.key("slot_id");
    output += std::to_string(slot_id);
    slot.finish();
}

std::string serialize_selected_slots(
    const std::vector<size_t>& selected_candidate_indices,
    const std::vector<RetainedCandidate>& candidates,
    const ReportTarget& source_target,
    const interaction::InteractionTarget& target,
    size_t registration_count) {
    require(
        selected_candidate_indices.size() == 3U,
        "selected_slots requires exactly three runtime-ready candidates");
    std::string output;
    CanonicalJsonObject object(output);
    object.key("record_type");
    append_json_string(output, "selected_slots");
    object.key("slots");
    output.push_back('[');
    size_t previous_index = 0U;
    for (size_t selected_index = 0U;
         selected_index < selected_candidate_indices.size();
         ++selected_index) {
        if (selected_index != 0U) output.push_back(',');
        const size_t candidate_index =
            selected_candidate_indices[selected_index];
        require(
            candidate_index < candidates.size() &&
                (selected_index == 0U || candidate_index > previous_index),
            "selected candidate indices do not preserve stable report order");
        previous_index = candidate_index;
        append_selected_slot(
            output,
            candidates[candidate_index],
            static_cast<uint32_t>(selected_index + 1U));
    }
    output.push_back(']');
    object.key("target");
    append_target_record(output, source_target, target);
    object.key("target_registry_registration_count");
    output += std::to_string(registration_count);
    object.finish();
    return output;
}

void validate_buffered_json_lines(const std::vector<std::string>& lines) {
    require(
        lines.size() == kExpectedRetainedCandidateCount + 1U,
        "buffered preview JSONL record count differs");
    for (size_t index = 0U; index < lines.size(); ++index) {
        require(
            !lines[index].empty() &&
                lines[index].find('\n') == std::string::npos &&
                lines[index].find('\r') == std::string::npos,
            "buffered preview record is empty or multiline");
        const std::string label =
            "buffered preview record " + std::to_string(index);
        const JsonValue record = JsonParser(lines[index], label).parse_document();
        const auto record_type_member = object_member(
            record, "record_type", label);
        const JsonValue& record_type_value = record_type_member.get();
        const std::string record_type_label = label + ".record_type";
        const std::string& record_type = json_string(
            record_type_value, record_type_label);
        const char* expected = index + 1U == lines.size()
            ? "selected_slots"
            : "candidate_preview";
        require(
            record_type == expected,
            label + " has the wrong record_type");
    }
}

std::string serialize_complete_preview_evidence(
    const PreviewEvaluation& evaluation,
    const SlotReport& report,
    const interaction::InteractionTarget& target,
    size_t registration_count) {
    require(
        evaluation.candidates.size() == report.retained_candidates.size() &&
            evaluation.case_count ==
                report.retained_candidates.size() *
                    kExpectedStationarySnapshotCount,
        "preview evaluation dimensions differ before serialization");
    std::vector<std::string> lines;
    lines.reserve(evaluation.candidates.size() + 1U);
    std::vector<size_t> selected_candidate_indices;
    selected_candidate_indices.reserve(3U);
    for (size_t candidate_index = 0U;
         candidate_index < evaluation.candidates.size();
         ++candidate_index) {
        const CandidatePreviewSummary& summary =
            evaluation.candidates[candidate_index];
        require(
            summary.candidate_ordinal == candidate_index,
            "candidate summary order differs from Task4 stable order");
        lines.push_back(serialize_candidate_preview(
            report.retained_candidates[candidate_index], summary));
        if (summary.path_feasible && summary.match_ready &&
            selected_candidate_indices.size() < 3U) {
            selected_candidate_indices.push_back(candidate_index);
        }
    }
    require(
        selected_candidate_indices.size() == 3U,
        "fewer than three candidates are runtime-ready; refusing partial stdout");
    lines.push_back(serialize_selected_slots(
        selected_candidate_indices,
        report.retained_candidates,
        report.target,
        target,
        registration_count));
    validate_buffered_json_lines(lines);

    std::string output;
    size_t byte_count = lines.size();
    for (const std::string& line : lines) byte_count += line.size();
    output.reserve(byte_count);
    for (const std::string& line : lines) {
        output += line;
        output.push_back('\n');
    }
    return output;
}

int run(int argc, char** argv) {
    if (argc != 4) {
        std::cerr
            << "usage: interaction_smart_pickup_preview_probe "
            << "<flat-database.bin> <full-pack-directory> <Task4-report.json>\n";
        return 2;
    }
    const std::filesystem::path flat_database_path(argv[1]);
    const std::filesystem::path full_pack_path(argv[2]);
    const std::filesystem::path report_path(argv[3]);
    require_readable_nonempty_file(flat_database_path, "flat database");
    require_readable_nonempty_file(
        full_pack_path / "interaction_database.bin",
        "full-pack interaction database");
    require_readable_nonempty_file(
        full_pack_path / "interaction_features.bin",
        "full-pack interaction features");

    const JsonValue manifest_json = parse_json_file(
        full_pack_path / "manifest.json",
        kMaximumManifestBytes,
        "full-pack manifest",
        kExpectedManifestSha256);
    const JsonValue report_json = parse_json_file(
        report_path,
        kMaximumReportBytes,
        "Task4 report",
        kExpectedReportSha256);
    Manifest manifest = parse_manifest(manifest_json);
    SlotReport report = parse_slot_report(report_json);
    validate_report_against_manifest(report, manifest);

    database flat_database{};
    const std::string flat_database_filename = flat_database_path.string();
    database_load(flat_database, flat_database_filename.c_str());
    validate_flat_database_shape(flat_database);
    const interaction::Database interaction_database =
        interaction::load_database(
            full_pack_path / "interaction_database.bin");
    const interaction::Features interaction_features =
        interaction::load_features(
            full_pack_path / "interaction_features.bin");
    interaction::validate_controller_interaction_pack(
        interaction_database, interaction_features);
    validate_loaded_pack_against_manifest(
        interaction_database, interaction_features, manifest);
    RegisteredTarget registered_target =
        register_report_demo_target(report.target);
    const FixedControllerPoseBridge bridge =
        make_fixed_controller_pose_bridge(
            flat_database, interaction_database);
    require_fixed_reference_calibration(flat_database, bridge);
    const std::vector<FrozenStationarySnapshot> stationary_snapshots =
        make_frozen_stationary_snapshots(flat_database, bridge);
    const PreviewEvaluation preview_evaluation =
        evaluate_unrestricted_previews(
            interaction_database,
            interaction_features,
            registered_target.registry,
            registered_target.handle,
            1U,
            report.retained_candidates,
            stationary_snapshots,
            manifest);
    const interaction::InteractionTarget* registered_target_snapshot =
        registered_target.registry.find(registered_target.handle);
    require(
        registered_target_snapshot != nullptr,
        "registered target disappeared before evidence serialization");
    const std::string buffered_evidence =
        serialize_complete_preview_evidence(
            preview_evaluation,
            report,
            *registered_target_snapshot,
            registered_target.registration_count);
    require(
        buffered_evidence.size() <=
            static_cast<size_t>(
                std::numeric_limits<std::streamsize>::max()),
        "buffered preview evidence exceeds stream capacity");

    std::cerr
        << "interaction_smart_pickup_preview_probe: loaded flat_frames="
        << flat_database.nframes()
        << " flat_bones=" << flat_database.nbones()
        << " interaction_frames=" << interaction_database.frame_count
        << " interaction_clips=" << interaction_database.clip_count
        << " interaction_features=" << interaction_features.feature_count
        << " retained_candidates=" << report.retained_candidates.size()
        << " registry_count=" << registered_target.registration_count
        << " stationary_count=" << stationary_snapshots.size()
        << " stationary_first=" << stationary_snapshots.front().frame
        << '/' << stationary_snapshots.front().fingerprint
        << " stationary_last=" << stationary_snapshots.back().frame
        << '/' << stationary_snapshots.back().fingerprint
        << " preview_cases=" << preview_evaluation.case_count
        << " preview_workers=" << preview_evaluation.worker_count
        << " ready_candidates=" << preview_evaluation.ready_candidate_count
        << " evidence_records="
        << preview_evaluation.candidates.size() + 1U
        << '\n';
    std::cout.write(
        buffered_evidence.data(),
        static_cast<std::streamsize>(buffered_evidence.size()));
    if (!std::cout) fail("could not write complete preview evidence");
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "interaction_smart_pickup_preview_probe: "
                  << error.what() << '\n';
        return 1;
    }
}
