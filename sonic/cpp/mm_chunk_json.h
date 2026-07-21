#pragma once

#include "json_runtime.h"
#include "sonic/cpp/mm_chunk_protocol.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <string>

enum mm_chunk_operation
{
    mm_chunk_op_invalid,
    mm_chunk_op_hello,
    mm_chunk_op_reset,
    mm_chunk_op_generate,
    mm_chunk_op_commit,
    mm_chunk_op_abort,
    mm_chunk_op_close,
};

struct mm_chunk_request
{
    int version = 0;
    mm_chunk_operation operation = mm_chunk_op_invalid;
    std::string operation_text = "<invalid>";
    std::string request_id;
    mm_chunk_reset_request reset;
    mm_chunk_generate_request generate;
    std::string session_id;
    std::string candidate_id;
};

static inline bool mm_chunk_json_exact_keys(
    const json_value& object,
    std::initializer_list<const char*> expected,
    mm_chunk_error& error)
{
    if (object.kind != json_object ||
        object.object_value.size() != expected.size()) {
        return mm_chunk_fail(
            error, "invalid_request", "request has an incorrect key set");
    }
    for (const char* key : expected) {
        if (json_member(object, key) == nullptr) {
            return mm_chunk_fail(
                error, "invalid_request", "request has an incorrect key set");
        }
    }
    return true;
}

static inline bool mm_chunk_json_has_no_duplicate_members(
    const json_value& value,
    std::string& duplicate)
{
    if (value.kind == json_array) {
        for (const json_value& child : value.array_value) {
            if (!mm_chunk_json_has_no_duplicate_members(child, duplicate)) {
                return false;
            }
        }
        return true;
    }
    if (value.kind != json_object) return true;
    for (std::size_t index = 0; index < value.object_value.size(); ++index) {
        for (std::size_t prior = 0; prior < index; ++prior) {
            if (value.object_value[index].first ==
                value.object_value[prior].first) {
                duplicate = value.object_value[index].first;
                return false;
            }
        }
        if (!mm_chunk_json_has_no_duplicate_members(
                value.object_value[index].second, duplicate)) {
            return false;
        }
    }
    return true;
}

static inline bool mm_chunk_json_identifier_is_valid(
    const std::string& value)
{
    return !value.empty();
}

static inline bool mm_chunk_json_string_member(
    std::string& output,
    const json_value& object,
    const char* key,
    mm_chunk_error& error)
{
    const json_value* value = json_member(object, key);
    if (value == nullptr || value->kind != json_string ||
        !mm_chunk_json_identifier_is_valid(value->string_value)) {
        return mm_chunk_fail(
            error,
            "invalid_request",
            std::string(key) + " must be a nonempty valid string");
    }
    output = value->string_value;
    return true;
}

static inline bool mm_chunk_json_integer_value(
    int& output,
    const json_value& value,
    const char* label,
    mm_chunk_error& error)
{
    if (value.kind != json_number || !std::isfinite(value.number_value) ||
        std::floor(value.number_value) != value.number_value ||
        value.number_value <
            static_cast<double>(std::numeric_limits<int>::min()) ||
        value.number_value >
            static_cast<double>(std::numeric_limits<int>::max())) {
        return mm_chunk_fail(
            error,
            "invalid_request",
            std::string(label) + " must be an integral integer");
    }
    output = static_cast<int>(value.number_value);
    return true;
}

static inline bool mm_chunk_json_integer_member(
    int& output,
    const json_value& object,
    const char* key,
    mm_chunk_error& error)
{
    const json_value* value = json_member(object, key);
    if (value == nullptr) {
        return mm_chunk_fail(
            error,
            "invalid_request",
            std::string(key) + " is required");
    }
    return mm_chunk_json_integer_value(output, *value, key, error);
}

static inline bool mm_chunk_json_binary32_value(
    float& output,
    const json_value& value,
    const char* label,
    mm_chunk_error& error)
{
    if (value.kind != json_number || !std::isfinite(value.number_value) ||
        value.number_value <
            -static_cast<double>(std::numeric_limits<float>::max()) ||
        value.number_value >
            static_cast<double>(std::numeric_limits<float>::max())) {
        return mm_chunk_fail(
            error,
            "invalid_request",
            std::string(label) + " must be a finite binary32 number");
    }
    const float encoded = static_cast<float>(value.number_value);
    if (!std::isfinite(encoded) ||
        static_cast<double>(encoded) != value.number_value) {
        return mm_chunk_fail(
            error,
            "invalid_request",
            std::string(label) +
                " must be an exact finite binary32 number");
    }
    output = encoded == 0.0f ? 0.0f : encoded;
    return true;
}

static inline bool mm_chunk_json_binary32_member(
    float& output,
    const json_value& object,
    const char* key,
    mm_chunk_error& error)
{
    const json_value* value = json_member(object, key);
    if (value == nullptr) {
        return mm_chunk_fail(
            error,
            "invalid_request",
            std::string(key) + " is required");
    }
    return mm_chunk_json_binary32_value(output, *value, key, error);
}

template<std::size_t Size>
static inline bool mm_chunk_json_binary32_array_member(
    float (&output)[Size],
    const json_value& object,
    const char* key,
    mm_chunk_error& error)
{
    const json_value* value = json_member(object, key);
    if (value == nullptr || value->kind != json_array ||
        value->array_value.size() != Size) {
        return mm_chunk_fail(
            error,
            "invalid_request",
            std::string(key) + " has an invalid array shape");
    }
    for (std::size_t index = 0; index < Size; ++index) {
        if (!mm_chunk_json_binary32_value(
                output[index], value->array_value[index], key, error)) {
            return false;
        }
    }
    return true;
}

static inline mm_chunk_operation mm_chunk_operation_from_string(
    const std::string& operation)
{
    if (operation == "hello") return mm_chunk_op_hello;
    if (operation == "reset") return mm_chunk_op_reset;
    if (operation == "generate") return mm_chunk_op_generate;
    if (operation == "commit") return mm_chunk_op_commit;
    if (operation == "abort") return mm_chunk_op_abort;
    if (operation == "close") return mm_chunk_op_close;
    return mm_chunk_op_invalid;
}

static inline bool mm_chunk_json_parse_request(
    mm_chunk_request& output,
    const std::string& line,
    mm_chunk_error& error)
{
    output = mm_chunk_request();
    mm_chunk_clear_error(error);
    json_value document;
    json_parser parser("<stdin>", line);
    if (!parser.value(document)) {
        return mm_chunk_fail(
            error,
            "invalid_json",
            "JSON error at byte " + std::to_string(parser.cursor) +
                ": " + parser.reason);
    }
    parser.whitespace();
    if (parser.cursor != line.size()) {
        return mm_chunk_fail(
            error,
            "invalid_json",
            "JSON error at byte " + std::to_string(parser.cursor) +
                ": trailing data");
    }
    if (document.kind != json_object) {
        return mm_chunk_fail(
            error, "invalid_request", "request must be one JSON object");
    }
    std::string duplicate;
    if (!mm_chunk_json_has_no_duplicate_members(document, duplicate)) {
        return mm_chunk_fail(
            error,
            "invalid_json",
            "duplicate object key: " + duplicate);
    }

    const json_value* version = json_member(document, "v");
    if (version == nullptr ||
        !mm_chunk_json_integer_value(
            output.version, *version, "v", error)) {
        return false;
    }
    if (output.version != MM_CHUNK_PROTOCOL_VERSION) {
        return mm_chunk_fail(
            error,
            "unsupported_version",
            "only protocol version 1 is supported");
    }
    if (!mm_chunk_json_string_member(
            output.operation_text, document, "op", error) ||
        !mm_chunk_json_string_member(
            output.request_id, document, "request_id", error)) {
        return false;
    }
    output.operation = mm_chunk_operation_from_string(output.operation_text);
    if (output.operation == mm_chunk_op_invalid) {
        return mm_chunk_fail(
            error,
            "unknown_operation",
            "unknown operation: " + output.operation_text);
    }

    switch (output.operation) {
    case mm_chunk_op_hello:
    case mm_chunk_op_close:
        return mm_chunk_json_exact_keys(
            document, {"v", "op", "request_id"}, error);
    case mm_chunk_op_reset: {
        // Accept the historical seven-key request (movement_model defaults to
        // raw) or that set plus an explicit movement_model string.
        const bool has_movement_model =
            json_member(document, "movement_model") != nullptr;
        const bool keys_ok = has_movement_model
            ? mm_chunk_json_exact_keys(
                  document,
                  {"v", "op", "request_id", "session_id", "scene_id",
                   "route_id", "terrain_weight", "movement_model"},
                  error)
            : mm_chunk_json_exact_keys(
                  document,
                  {"v", "op", "request_id", "session_id", "scene_id",
                   "route_id", "terrain_weight"},
                  error);
        if (!keys_ok ||
            !mm_chunk_json_string_member(
                output.reset.session_id, document, "session_id", error) ||
            !mm_chunk_json_string_member(
                output.reset.scene_id, document, "scene_id", error) ||
            !mm_chunk_json_string_member(
                output.reset.route_id, document, "route_id", error) ||
            !mm_chunk_json_binary32_member(
                output.reset.terrain_weight,
                document,
                "terrain_weight",
                error)) {
            return false;
        }
        if (has_movement_model &&
            !mm_chunk_json_string_member(
                output.reset.movement_model,
                document,
                "movement_model",
                error)) {
            return false;
        }
        return true;
    }
    case mm_chunk_op_generate: {
        if (!mm_chunk_json_exact_keys(
                document,
                {"v", "op", "request_id", "session_id", "candidate_id",
                 "predecessor_id", "source_intervals",
                 "requested_velocity_holden",
                 "desired_heading_holden_wxyz"},
                error) ||
            !mm_chunk_json_string_member(
                output.generate.session_id, document, "session_id", error) ||
            !mm_chunk_json_string_member(
                output.generate.candidate_id,
                document,
                "candidate_id",
                error) ||
            !mm_chunk_json_integer_member(
                output.generate.source_intervals,
                document,
                "source_intervals",
                error) ||
            !mm_chunk_json_binary32_array_member(
                output.generate.requested_velocity_holden,
                document,
                "requested_velocity_holden",
                error) ||
            !mm_chunk_json_binary32_array_member(
                output.generate.desired_heading_holden_wxyz,
                document,
                "desired_heading_holden_wxyz",
                error)) {
            return false;
        }
        const json_value* predecessor = json_member(document, "predecessor_id");
        if (predecessor == nullptr) {
            return mm_chunk_fail(
                error, "invalid_request", "predecessor_id is required");
        }
        if (predecessor->kind == json_null) {
            output.generate.predecessor_is_null = true;
            output.generate.predecessor_id.clear();
        } else if (predecessor->kind == json_string &&
                   mm_chunk_json_identifier_is_valid(
                       predecessor->string_value)) {
            output.generate.predecessor_is_null = false;
            output.generate.predecessor_id = predecessor->string_value;
        } else {
            return mm_chunk_fail(
                error,
                "invalid_request",
                "predecessor_id must be null or a nonempty valid string");
        }
        double norm_squared = 0.0;
        for (float value : output.generate.desired_heading_holden_wxyz) {
            norm_squared += static_cast<double>(value) *
                            static_cast<double>(value);
        }
        if (!std::isfinite(norm_squared) ||
            std::fabs(norm_squared - 1.0) > 2.0e-4) {
            return mm_chunk_fail(
                error,
                "invalid_request",
                "desired_heading_holden_wxyz must be a unit quaternion");
        }
        return true;
    }
    case mm_chunk_op_commit:
    case mm_chunk_op_abort:
        return mm_chunk_json_exact_keys(
                   document,
                   {"v", "op", "request_id", "session_id", "candidate_id"},
                   error) &&
               mm_chunk_json_string_member(
                   output.session_id, document, "session_id", error) &&
               mm_chunk_json_string_member(
                   output.candidate_id, document, "candidate_id", error);
    default:
        break;
    }
    return mm_chunk_fail(error, "internal_error", "unhandled operation");
}

class mm_chunk_json_writer
{
public:
    void raw(const char* value)
    {
        if (value == nullptr) {
            valid_ = false;
            return;
        }
        text_ += value;
    }

    void raw(const std::string& value) { text_ += value; }
    void character(char value) { text_.push_back(value); }

    void string(const std::string& value)
    {
        text_.push_back('"');
        static const char hex[] = "0123456789abcdef";
        for (unsigned char character : value) {
            switch (character) {
            case '"': text_ += "\\\""; break;
            case '\\': text_ += "\\\\"; break;
            case '\b': text_ += "\\b"; break;
            case '\f': text_ += "\\f"; break;
            case '\n': text_ += "\\n"; break;
            case '\r': text_ += "\\r"; break;
            case '\t': text_ += "\\t"; break;
            default:
                if (character < 0x20u) {
                    text_ += "\\u00";
                    text_.push_back(hex[(character >> 4) & 0x0fu]);
                    text_.push_back(hex[character & 0x0fu]);
                } else {
                    text_.push_back(static_cast<char>(character));
                }
                break;
            }
        }
        text_.push_back('"');
    }

    void boolean(bool value) { text_ += value ? "true" : "false"; }

    void integer(int value)
    {
        char encoded[32] = {};
        const int count = std::snprintf(
            encoded, sizeof(encoded), "%d", value);
        if (count <= 0 || count >= static_cast<int>(sizeof(encoded))) {
            valid_ = false;
            return;
        }
        text_.append(encoded, static_cast<std::size_t>(count));
    }

    void number(float value)
    {
        if (!std::isfinite(value)) {
            valid_ = false;
            return;
        }
        char encoded[64] = {};
        const int count = std::snprintf(
            encoded,
            sizeof(encoded),
            "%.9g",
            static_cast<double>(value == 0.0f ? 0.0f : value));
        if (count <= 0 || count >= static_cast<int>(sizeof(encoded))) {
            valid_ = false;
            return;
        }
        text_.append(encoded, static_cast<std::size_t>(count));
    }

    void number(double value)
    {
        if (!std::isfinite(value)) {
            valid_ = false;
            return;
        }
        char encoded[64] = {};
        const int count = std::snprintf(
            encoded,
            sizeof(encoded),
            "%.17g",
            value == 0.0 ? 0.0 : value);
        if (count <= 0 || count >= static_cast<int>(sizeof(encoded))) {
            valid_ = false;
            return;
        }
        text_.append(encoded, static_cast<std::size_t>(count));
    }

    bool valid() const { return valid_; }
    const std::string& text() const { return text_; }

private:
    std::string text_;
    bool valid_ = true;
};
