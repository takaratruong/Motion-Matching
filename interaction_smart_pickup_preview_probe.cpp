#include <algorithm>
#include <array>
#include <cerrno>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
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

    std::cerr
        << "interaction_smart_pickup_preview_probe: validated "
        << manifest.clips.size() << " manifest clips and "
        << report.retained_candidates.size()
        << " retained candidates; runtime preview is not yet implemented\n";
    return 1;
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
