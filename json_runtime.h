#pragma once

#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

enum json_kind { json_null, json_boolean, json_number, json_string,
                 json_array, json_object };

struct json_value
{
    json_kind kind = json_null;
    bool boolean_value = false;
    double number_value = 0.0;
    std::string string_value;
    std::vector<json_value> array_value;
    std::vector<std::pair<std::string, json_value> > object_value;
    size_t source_offset = 0;
};

static inline const json_value* json_member(
    const json_value& object, const char* key)
{
    if (object.kind != json_object || key == NULL) return NULL;
    for (size_t i = 0; i < object.object_value.size(); ++i)
        if (object.object_value[i].first == key)
            return &object.object_value[i].second;
    return NULL;
}

struct json_parser
{
    const char* path;
    const std::string& text;
    size_t cursor = 0;
    int depth = 0;
    std::string reason;

    json_parser(const char* input_path, const std::string& input)
        : path(input_path), text(input) {}

    bool fail(const char* message)
    {
        if (reason.empty()) reason = message;
        return false;
    }

    void whitespace()
    {
        while (cursor < text.size() &&
               (text[cursor] == ' ' || text[cursor] == '\t' ||
                text[cursor] == '\n' || text[cursor] == '\r')) ++cursor;
    }

    static int hex(const char value)
    {
        if (value >= '0' && value <= '9') return value - '0';
        if (value >= 'a' && value <= 'f') return value - 'a' + 10;
        if (value >= 'A' && value <= 'F') return value - 'A' + 10;
        return -1;
    }

    bool code_unit(uint32_t& value)
    {
        if (text.size() - cursor < 4) return fail("truncated unicode escape");
        value = 0;
        for (int i = 0; i < 4; ++i) {
            const int digit = hex(text[cursor++]);
            if (digit < 0) return fail("invalid unicode escape");
            value = value * 16u + static_cast<uint32_t>(digit);
        }
        return true;
    }

    static void append_utf8(std::string& out, const uint32_t value)
    {
        if (value <= 0x7fu) out.push_back(static_cast<char>(value));
        else if (value <= 0x7ffu) {
            out.push_back(static_cast<char>(0xc0u | (value >> 6)));
            out.push_back(static_cast<char>(0x80u | (value & 0x3fu)));
        } else if (value <= 0xffffu) {
            out.push_back(static_cast<char>(0xe0u | (value >> 12)));
            out.push_back(static_cast<char>(0x80u | ((value >> 6) & 0x3fu)));
            out.push_back(static_cast<char>(0x80u | (value & 0x3fu)));
        } else {
            out.push_back(static_cast<char>(0xf0u | (value >> 18)));
            out.push_back(static_cast<char>(0x80u | ((value >> 12) & 0x3fu)));
            out.push_back(static_cast<char>(0x80u | ((value >> 6) & 0x3fu)));
            out.push_back(static_cast<char>(0x80u | (value & 0x3fu)));
        }
    }

    bool raw_utf8(std::string& out)
    {
        const unsigned char first = static_cast<unsigned char>(text[cursor++]);
        int tails = 0;
        uint32_t value = 0, minimum = 0;
        if (first >= 0xc2 && first <= 0xdf) {
            tails = 1; value = first & 0x1fu; minimum = 0x80u;
        } else if (first >= 0xe0 && first <= 0xef) {
            tails = 2; value = first & 0x0fu; minimum = 0x800u;
        } else if (first >= 0xf0 && first <= 0xf4) {
            tails = 3; value = first & 0x07u; minimum = 0x10000u;
        } else return fail("invalid UTF-8 lead byte");
        if (text.size() - cursor < static_cast<size_t>(tails))
            return fail("truncated UTF-8");
        for (int i = 0; i < tails; ++i) {
            const unsigned char next = static_cast<unsigned char>(text[cursor++]);
            if ((next & 0xc0u) != 0x80u) return fail("invalid UTF-8 tail byte");
            value = (value << 6) | (next & 0x3fu);
        }
        if (value < minimum || value > 0x10ffffu ||
            (value >= 0xd800u && value <= 0xdfffu))
            return fail("invalid UTF-8 codepoint");
        append_utf8(out, value);
        return true;
    }

    bool string(std::string& out)
    {
        whitespace();
        if (cursor >= text.size() || text[cursor++] != '"')
            return fail("expected string");
        out.clear();
        while (cursor < text.size()) {
            const unsigned char value = static_cast<unsigned char>(text[cursor++]);
            if (value == '"') return true;
            if (value < 0x20u) return fail("control byte in string");
            if (value >= 0x80u) { --cursor; if (!raw_utf8(out)) return false; continue; }
            if (value != '\\') { out.push_back(static_cast<char>(value)); continue; }
            if (cursor >= text.size()) return fail("truncated escape");
            const char escape = text[cursor++];
            if (escape == '"' || escape == '\\' || escape == '/')
                out.push_back(escape);
            else if (escape == 'b') out.push_back('\b');
            else if (escape == 'f') out.push_back('\f');
            else if (escape == 'n') out.push_back('\n');
            else if (escape == 'r') out.push_back('\r');
            else if (escape == 't') out.push_back('\t');
            else if (escape == 'u') {
                uint32_t first = 0;
                if (!code_unit(first)) return false;
                uint32_t codepoint = first;
                if (first >= 0xd800u && first <= 0xdbffu) {
                    if (text.size() - cursor < 6 || text[cursor] != '\\' ||
                        text[cursor + 1] != 'u') return fail("missing low surrogate");
                    cursor += 2;
                    uint32_t second = 0;
                    if (!code_unit(second) || second < 0xdc00u || second > 0xdfffu)
                        return fail("invalid low surrogate");
                    codepoint = 0x10000u + ((first - 0xd800u) << 10) +
                                (second - 0xdc00u);
                } else if (first >= 0xdc00u && first <= 0xdfffu) {
                    return fail("unpaired low surrogate");
                }
                append_utf8(out, codepoint);
            } else return fail("invalid string escape");
        }
        return fail("unterminated string");
    }

    bool number(json_value& out)
    {
        const size_t begin = cursor;
        if (cursor < text.size() && text[cursor] == '-') ++cursor;
        if (cursor >= text.size()) return fail("truncated number");
        if (text[cursor] == '0') ++cursor;
        else if (text[cursor] >= '1' && text[cursor] <= '9')
            while (cursor < text.size() && text[cursor] >= '0' &&
                   text[cursor] <= '9') ++cursor;
        else return fail("invalid number integer");
        if (cursor < text.size() && text[cursor] == '.') {
            ++cursor;
            const size_t digits = cursor;
            while (cursor < text.size() && text[cursor] >= '0' &&
                   text[cursor] <= '9') ++cursor;
            if (cursor == digits) return fail("invalid number fraction");
        }
        if (cursor < text.size() && (text[cursor] == 'e' || text[cursor] == 'E')) {
            ++cursor;
            if (cursor < text.size() && (text[cursor] == '+' || text[cursor] == '-'))
                ++cursor;
            const size_t digits = cursor;
            while (cursor < text.size() && text[cursor] >= '0' &&
                   text[cursor] <= '9') ++cursor;
            if (cursor == digits) return fail("invalid number exponent");
        }
        const std::string token = text.substr(begin, cursor - begin);
        errno = 0;
        char* end = NULL;
        const double parsed = std::strtod(token.c_str(), &end);
        if (errno == ERANGE || end == NULL || *end != '\0' || !std::isfinite(parsed))
            return fail("non-finite or out-of-range number");
        out.kind = json_number;
        out.number_value = parsed;
        return true;
    }

    bool literal(const char* word)
    {
        const size_t size = std::strlen(word);
        if (text.size() - cursor < size || text.compare(cursor, size, word) != 0)
            return fail("invalid literal");
        cursor += size;
        return true;
    }

    bool value(json_value& out)
    {
        whitespace();
        if (cursor >= text.size()) return fail("expected value");
        out.source_offset = cursor;
        if (text[cursor] == '"') {
            out.kind = json_string;
            return string(out.string_value);
        }
        if (text[cursor] == '-' || (text[cursor] >= '0' && text[cursor] <= '9'))
            return number(out);
        if (text[cursor] == 'n') { out.kind = json_null; return literal("null"); }
        if (text[cursor] == 't') {
            out.kind = json_boolean; out.boolean_value = true; return literal("true");
        }
        if (text[cursor] == 'f') {
            out.kind = json_boolean; out.boolean_value = false; return literal("false");
        }
        if (depth >= 64) return fail("JSON nesting exceeds 64");
        if (text[cursor] == '[') {
            out.kind = json_array; ++cursor; ++depth; whitespace();
            if (cursor < text.size() && text[cursor] == ']') {
                ++cursor; --depth; return true;
            }
            while (true) {
                out.array_value.push_back(json_value());
                if (!value(out.array_value.back())) return false;
                whitespace();
                if (cursor < text.size() && text[cursor] == ']') {
                    ++cursor; --depth; return true;
                }
                if (cursor >= text.size() || text[cursor++] != ',')
                    return fail("expected array comma or close");
            }
        }
        if (text[cursor] == '{') {
            out.kind = json_object; ++cursor; ++depth; whitespace();
            if (cursor < text.size() && text[cursor] == '}') {
                ++cursor; --depth; return true;
            }
            while (true) {
                std::string key;
                if (!string(key)) return false;
                for (size_t i = 0; i < out.object_value.size(); ++i)
                    if (out.object_value[i].first == key)
                        return fail("duplicate object key");
                whitespace();
                if (cursor >= text.size() || text[cursor++] != ':')
                    return fail("expected object colon");
                out.object_value.push_back(std::make_pair(key, json_value()));
                if (!value(out.object_value.back().second)) return false;
                whitespace();
                if (cursor < text.size() && text[cursor] == '}') {
                    ++cursor; --depth; return true;
                }
                if (cursor >= text.size() || text[cursor++] != ',')
                    return fail("expected object comma or close");
            }
        }
        return fail("invalid value token");
    }
};

static inline bool json_runtime_error(
    char* error, const int capacity, const char* path,
    const size_t offset, const char* reason)
{
    if (error != NULL && capacity > 0)
        std::snprintf(error, static_cast<size_t>(capacity),
            "%s: JSON error at byte %zu: %s", path, offset, reason);
    return false;
}

static const size_t JSON_DOCUMENT_DEFAULT_MAXIMUM_BYTES =
    16u * 1024u * 1024u;
static const size_t JSON_MOTION_MANIFEST_MAXIMUM_BYTES =
    256u * 1024u * 1024u;

static inline bool json_document_load_with_limit(
    json_value& out, const char* path, const size_t maximum_size,
    char* error, const int error_capacity)
{
    if (path == NULL || path[0] == '\0')
        return json_runtime_error(error, error_capacity,
            path != NULL ? path : "<null>", 0, "invalid path");
    if (maximum_size == 0 ||
        maximum_size > JSON_MOTION_MANIFEST_MAXIMUM_BYTES)
        return json_runtime_error(error, error_capacity, path, 0,
            "invalid maximum document size");
    FILE* file = std::fopen(path, "rb");
    if (file == NULL)
        return json_runtime_error(error, error_capacity, path, 0, "cannot open");
    if (std::fseek(file, 0, SEEK_END) != 0) {
        std::fclose(file);
        return json_runtime_error(error, error_capacity, path, 0, "cannot size");
    }
    const long end = std::ftell(file);
    if (end < 0 ||
        static_cast<uintmax_t>(end) > static_cast<uintmax_t>(SIZE_MAX) ||
        static_cast<uintmax_t>(end) >
            static_cast<uintmax_t>(maximum_size) ||
        std::fseek(file, 0, SEEK_SET) != 0) {
        std::fclose(file);
        const char* reason = end < 0 ? "cannot size" :
            (maximum_size == JSON_DOCUMENT_DEFAULT_MAXIMUM_BYTES ?
                "document exceeds 16 MiB" :
                "document exceeds explicit maximum size");
        return json_runtime_error(error, error_capacity, path, 0,
            reason);
    }
    std::string text(static_cast<size_t>(end), '\0');
    bool read_failed = !text.empty() &&
        std::fread(&text[0], 1, text.size(), file) != text.size();
    if (std::fclose(file) != 0) read_failed = true;
    if (read_failed)
        return json_runtime_error(error, error_capacity, path, 0, "cannot read");

    json_value loaded;
    json_parser parser(path, text);
    if (!parser.value(loaded))
        return json_runtime_error(error, error_capacity, path,
            parser.cursor, parser.reason.c_str());
    parser.whitespace();
    if (parser.cursor != text.size())
        return json_runtime_error(error, error_capacity, path,
            parser.cursor, "trailing data");
    out = std::move(loaded);
    return true;
}

static inline bool json_document_load(
    json_value& out, const char* path, char* error, const int error_capacity)
{
    return json_document_load_with_limit(
        out, path, JSON_DOCUMENT_DEFAULT_MAXIMUM_BYTES,
        error, error_capacity);
}
