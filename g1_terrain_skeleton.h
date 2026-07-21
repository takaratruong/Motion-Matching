#pragma once

#include "g1_kinematic_contract.h"
#include "database.h"

#include <errno.h>
#include <limits.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static inline bool g1_error(char* output, int capacity, const char* format, ...)
{
    if (output != NULL && capacity > 0) {
        va_list arguments;
        va_start(arguments, format);
        vsnprintf(output, static_cast<size_t>(capacity), format, arguments);
        va_end(arguments);
    }
    return false;
}

static inline bool g1_skeleton_validate(
    const database& db, char* error, int error_capacity)
{
    static const int parents[G1_BoneCount] = {
        -1, 0, 1, 2, 3, 4, 5, 6, 1, 8, 9, 10, 11, 12, 1, 14,
        15, 16, 17, 18, 19, 20, 21, 22, 16, 24, 25, 26, 27, 28, 29
    };

    if (db.bone_positions.cols != G1_BoneCount) {
        return g1_error(
            error,
            error_capacity,
            "G1 bone count mismatch: expected %d position columns, got %d",
            G1_BoneCount,
            db.bone_positions.cols);
    }
    if (db.bone_parents.size != G1_BoneCount ||
        db.bone_parents.data == NULL) {
        return g1_error(
            error,
            error_capacity,
            "G1 parent array size mismatch: expected %d, got %d",
            G1_BoneCount,
            db.bone_parents.size);
    }
    for (int bone = 0; bone < G1_BoneCount; ++bone) {
        if (db.bone_parents.data[bone] != parents[bone]) {
            return g1_error(
                error,
                error_capacity,
                "G1 parent mismatch at bone %d: expected %d, got %d",
                bone,
                parents[bone],
                db.bone_parents.data[bone]);
        }
    }
    return true;
}

// This focused parser validates the complete JSON document while retaining
// only string spans needed to identify root.skeleton.signature.
struct g1_json_parser
{
    struct string_span
    {
        size_t begin;
        size_t end;
    };

    enum scope
    {
        root_scope,
        skeleton_scope,
        other_scope
    };

    const char* text;
    size_t size;
    size_t position;
    int depth;
    int skeleton_count;
    int signature_count;
    int matching_signature_count;

    g1_json_parser(const char* input, size_t input_size)
        : text(input),
          size(input_size),
          position(0),
          depth(0),
          skeleton_count(0),
          signature_count(0),
          matching_signature_count(0)
    {
    }

    void whitespace()
    {
        while (position < size &&
               (text[position] == ' ' || text[position] == '\t' ||
                text[position] == '\n' || text[position] == '\r')) {
            ++position;
        }
    }

    bool take(char expected)
    {
        whitespace();
        if (position >= size || text[position] != expected) {
            return false;
        }
        ++position;
        return true;
    }

    static int hex(unsigned char value)
    {
        if (value >= '0' && value <= '9') return value - '0';
        if (value >= 'a' && value <= 'f') return value - 'a' + 10;
        if (value >= 'A' && value <= 'F') return value - 'A' + 10;
        return -1;
    }

    bool utf8_tail(unsigned char first)
    {
        int count = 0;
        uint32_t codepoint = 0;
        uint32_t minimum = 0;
        if (first >= 0xc2 && first <= 0xdf) {
            count = 1; codepoint = first & 0x1f; minimum = 0x80;
        } else if (first >= 0xe0 && first <= 0xef) {
            count = 2; codepoint = first & 0x0f; minimum = 0x800;
        } else if (first >= 0xf0 && first <= 0xf4) {
            count = 3; codepoint = first & 0x07; minimum = 0x10000;
        } else {
            return false;
        }
        if (size - position < static_cast<size_t>(count)) return false;
        for (int i = 0; i < count; ++i) {
            const unsigned char next =
                static_cast<unsigned char>(text[position++]);
            if ((next & 0xc0) != 0x80) return false;
            codepoint = (codepoint << 6) | (next & 0x3f);
        }
        return codepoint >= minimum && codepoint <= 0x10ffff &&
               !(codepoint >= 0xd800 && codepoint <= 0xdfff);
    }

    bool string(string_span& span)
    {
        whitespace();
        if (position >= size || text[position++] != '"') return false;
        span.begin = position;
        while (position < size) {
            const unsigned char value =
                static_cast<unsigned char>(text[position++]);
            if (value == '"') {
                span.end = position - 1;
                return true;
            }
            if (value < 0x20) return false;
            if (value == '\\') {
                if (position >= size) return false;
                const char escape = text[position++];
                if (strchr("\"\\/bfnrt", escape) != NULL) continue;
                if (escape != 'u' || size - position < 4) return false;
                for (int i = 0; i < 4; ++i) {
                    if (hex(static_cast<unsigned char>(text[position++])) < 0)
                        return false;
                }
            } else if (value >= 0x80 && !utf8_tail(value)) {
                return false;
            }
        }
        return false;
    }

    bool string_equals(const string_span& span, const char* expected) const
    {
        size_t cursor = span.begin;
        size_t expected_position = 0;
        while (cursor < span.end) {
            uint32_t value = static_cast<unsigned char>(text[cursor++]);
            if (value == '\\') {
                const char escape = text[cursor++];
                if (escape == 'u') {
                    value = 0;
                    for (int i = 0; i < 4; ++i) {
                        value = value * 16 + static_cast<uint32_t>(
                            hex(static_cast<unsigned char>(text[cursor++])));
                    }
                } else {
                    switch (escape) {
                    case '"': value = '"'; break;
                    case '\\': value = '\\'; break;
                    case '/': value = '/'; break;
                    case 'b': value = '\b'; break;
                    case 'f': value = '\f'; break;
                    case 'n': value = '\n'; break;
                    case 'r': value = '\r'; break;
                    default: value = '\t'; break;
                    }
                }
            }
            if (value > 0x7f || expected[expected_position] == '\0' ||
                static_cast<unsigned char>(expected[expected_position]) != value) {
                return false;
            }
            ++expected_position;
        }
        return expected[expected_position] == '\0';
    }

    bool literal(const char* value)
    {
        const size_t length = strlen(value);
        if (size - position < length ||
            memcmp(text + position, value, length) != 0) return false;
        position += length;
        return true;
    }

    bool number()
    {
        if (position < size && text[position] == '-') ++position;
        if (position >= size) return false;
        if (text[position] == '0') {
            ++position;
        } else if (text[position] >= '1' && text[position] <= '9') {
            while (position < size && text[position] >= '0' &&
                   text[position] <= '9') ++position;
        } else {
            return false;
        }
        if (position < size && text[position] == '.') {
            ++position;
            const size_t start = position;
            while (position < size && text[position] >= '0' &&
                   text[position] <= '9') ++position;
            if (position == start) return false;
        }
        if (position < size &&
            (text[position] == 'e' || text[position] == 'E')) {
            ++position;
            if (position < size &&
                (text[position] == '+' || text[position] == '-')) ++position;
            const size_t start = position;
            while (position < size && text[position] >= '0' &&
                   text[position] <= '9') ++position;
            if (position == start) return false;
        }
        return true;
    }

    bool array()
    {
        if (depth >= 128 || !take('[')) return false;
        ++depth;
        whitespace();
        if (position < size && text[position] == ']') {
            ++position; --depth; return true;
        }
        while (value(other_scope)) {
            whitespace();
            if (position < size && text[position] == ']') {
                ++position; --depth; return true;
            }
            if (!take(',')) return false;
        }
        return false;
    }

    bool object(scope current_scope)
    {
        if (depth >= 128 || !take('{')) return false;
        ++depth;
        whitespace();
        if (position < size && text[position] == '}') {
            ++position; --depth; return true;
        }
        while (true) {
            string_span key = {};
            if (!string(key) || !take(':')) return false;
            const bool is_skeleton = string_equals(key, "skeleton");
            const bool is_signature = string_equals(key, "signature");

            if (current_scope == root_scope && is_skeleton) {
                ++skeleton_count;
                whitespace();
                if (position < size && text[position] == '{') {
                    if (!object(skeleton_scope)) return false;
                } else if (!value(other_scope)) {
                    return false;
                }
            } else if (current_scope == skeleton_scope && is_signature) {
                ++signature_count;
                whitespace();
                if (position < size && text[position] == '"') {
                    string_span signature = {};
                    if (!string(signature)) return false;
                    if (string_equals(signature, G1_SkeletonSignature))
                        ++matching_signature_count;
                } else if (!value(other_scope)) {
                    return false;
                }
            } else if (!value(other_scope)) {
                return false;
            }

            whitespace();
            if (position < size && text[position] == '}') {
                ++position; --depth; return true;
            }
            if (!take(',')) return false;
        }
    }

    bool value(scope object_scope)
    {
        whitespace();
        if (position >= size) return false;
        if (text[position] == '{') return object(object_scope);
        if (text[position] == '[') return array();
        if (text[position] == '"') {
            string_span ignored = {};
            return string(ignored);
        }
        if (text[position] == 't') return literal("true");
        if (text[position] == 'f') return literal("false");
        if (text[position] == 'n') return literal("null");
        return number();
    }

    bool document()
    {
        whitespace();
        if (position >= size || text[position] != '{' || !object(root_scope))
            return false;
        whitespace();
        return position == size;
    }
};

static inline bool g1_manifest_validate(
    const char* path, char* error, int error_capacity)
{
    const char* display_path = path != NULL ? path : "<null>";
    if (path == NULL || path[0] == '\0') {
        return g1_error(
            error, error_capacity, "%s: invalid manifest path", display_path);
    }

    FILE* file = fopen(path, "rb");
    if (file == NULL) {
        return g1_error(
            error,
            error_capacity,
            "%s: cannot open manifest (%s)",
            path,
            strerror(errno));
    }
    if (fseek(file, 0, SEEK_END) != 0) {
        fclose(file);
        return g1_error(
            error, error_capacity, "%s: cannot size manifest", path);
    }
    const long end = ftell(file);
    if (end < 0 || fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        return g1_error(
            error, error_capacity, "%s: cannot size manifest", path);
    }
    const size_t file_size = static_cast<size_t>(end);
    if (file_size > static_cast<size_t>(INT_MAX - 1)) {
        fclose(file);
        return g1_error(
            error,
            error_capacity,
            "%s: manifest is too large (%zu bytes; maximum %d)",
            path,
            file_size,
            INT_MAX - 1);
    }

    char* text = static_cast<char*>(malloc(file_size + 1));
    if (text == NULL) {
        fclose(file);
        return g1_error(
            error,
            error_capacity,
            "%s: cannot allocate %zu bytes for manifest",
            path,
            file_size + 1);
    }
    const size_t bytes_read = fread(text, 1, file_size, file);
    const int extra = bytes_read == file_size ? fgetc(file) : EOF;
    const bool read_error = ferror(file) != 0;
    const bool close_error = fclose(file) != 0;
    if (bytes_read != file_size || extra != EOF || read_error || close_error) {
        free(text);
        return g1_error(
            error, error_capacity, "%s: cannot read complete manifest", path);
    }
    text[file_size] = '\0';

    g1_json_parser parser(text, file_size);
    const bool json_valid = parser.document();
    free(text);
    if (!json_valid) {
        return g1_error(
            error, error_capacity, "%s: malformed JSON manifest", path);
    }
    if (parser.skeleton_count == 0) {
        return g1_error(
            error, error_capacity, "%s: G1 skeleton object is missing", path);
    }
    if (parser.skeleton_count != 1) {
        return g1_error(
            error,
            error_capacity,
            "%s: ambiguous G1 skeleton objects (%d)",
            path,
            parser.skeleton_count);
    }
    if (parser.signature_count == 0) {
        return g1_error(
            error,
            error_capacity,
            "%s: G1 skeleton signature field is missing",
            path);
    }
    if (parser.signature_count != 1) {
        return g1_error(
            error,
            error_capacity,
            "%s: ambiguous G1 skeleton signature fields (%d)",
            path,
            parser.signature_count);
    }
    if (parser.matching_signature_count != 1) {
        return g1_error(
            error,
            error_capacity,
            "%s: G1 skeleton signature mismatch",
            path);
    }
    return true;
}
