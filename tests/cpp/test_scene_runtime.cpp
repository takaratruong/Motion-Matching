#include "json_runtime.h"
#include "sha256.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static const char* const json_fixture_path = "/tmp/test_scene_json.json";

static void check(bool value, const char* message)
{
    if (!value) {
        std::fprintf(stderr, "scene runtime test failed: %s\n", message);
        std::exit(1);
    }
}

static void write_text(const char* path, const std::string& text)
{
    FILE* file = std::fopen(path, "wb");
    check(file != NULL, "open fixture");
    check(std::fwrite(text.data(), 1, text.size(), file) == text.size(),
          "write fixture");
    check(std::fclose(file) == 0, "close fixture");
}

static bool load_text(
    json_value& out, const std::string& text, char error[512])
{
    write_text(json_fixture_path, text);
    error[0] = '\0';
    return json_document_load(out, json_fixture_path, error, 512);
}

static json_value sentinel_value()
{
    json_value value;
    value.kind = json_string;
    value.boolean_value = true;
    value.number_value = 42.5;
    value.string_value = "sentinel";
    value.array_value.push_back(json_value());
    value.object_value.push_back(
        std::make_pair(std::string("preserved"), json_value()));
    value.source_offset = 123;
    return value;
}

static void check_sentinel(const json_value& value, const char* message)
{
    check(value.kind == json_string, message);
    check(value.boolean_value, message);
    check(value.number_value == 42.5, message);
    check(value.string_value == "sentinel", message);
    check(value.array_value.size() == 1, message);
    check(value.object_value.size() == 1, message);
    check(value.object_value[0].first == "preserved", message);
    check(value.source_offset == 123, message);
}

static void expect_json_failure(const std::string& text, const char* message)
{
    char error[512] = {};
    json_value value = sentinel_value();
    check(!load_text(value, text, error), message);
    check(std::strstr(error, json_fixture_path) != NULL,
          "JSON failure contains fixture path");
    check_sentinel(value, "JSON failure is transactional");
}

static void test_json_value_kinds_members_and_offsets()
{
    char error[512] = {};
    json_value value;
    const std::string text =
        " \n{\"none\":null,\"yes\":true,\"no\":false,"
        "\"number\":-1.25e2,\"string\":\"value\","
        "\"array\":[0],\"object\":{}}";
    check(load_text(value, text, error), error);
    check(value.kind == json_object, "root object kind");
    check(value.source_offset == text.find('{'), "root source offset");

    const json_value* none = json_member(value, "none");
    const json_value* yes = json_member(value, "yes");
    const json_value* no = json_member(value, "no");
    const json_value* number = json_member(value, "number");
    const json_value* string = json_member(value, "string");
    const json_value* array = json_member(value, "array");
    const json_value* object = json_member(value, "object");
    check(none != NULL && none->kind == json_null, "null kind");
    check(yes != NULL && yes->kind == json_boolean && yes->boolean_value,
          "true boolean kind");
    check(no != NULL && no->kind == json_boolean && !no->boolean_value,
          "false boolean kind");
    check(number != NULL && number->kind == json_number &&
          number->number_value == -125.0, "number kind and decode");
    check(string != NULL && string->kind == json_string &&
          string->string_value == "value", "string kind and decode");
    check(array != NULL && array->kind == json_array &&
          array->array_value.size() == 1, "array kind");
    check(object != NULL && object->kind == json_object &&
          object->object_value.empty(), "object kind");

    check(none->source_offset == text.find("null"), "null source offset");
    check(array->source_offset == text.find('['), "array source offset");
    check(array->array_value[0].source_offset == text.find("[0") + 1,
          "array member source offset");
    check(json_member(value, "missing") == NULL, "missing member lookup");
    check(json_member(value, NULL) == NULL, "null member key");
    check(json_member(*string, "anything") == NULL,
          "member lookup requires object");
}

static void test_json_full_document_and_unique_keys()
{
    const std::string failures[] = {
        "",
        " \t\r\n",
        "[] trailing",
        "[1,]",
        "{\"x\":1,}",
        "{\"x\" 1}",
        "{\"x\":1",
        "{\"x\":tru}",
        "{\"x\":1,\"x\":2}",
        "{\"x\":1,\"\\u0078\":2}",
    };
    for (size_t i = 0; i < sizeof(failures) / sizeof(failures[0]); ++i)
        expect_json_failure(failures[i], "strict full-document rejection");
}

static void test_json_number_grammar_and_finite_range()
{
    char error[512] = {};
    json_value value;
    check(load_text(value,
        "[0,-0,17,-1.25,6.022e23,1E-3]", error), error);
    check(value.kind == json_array && value.array_value.size() == 6,
          "valid JSON number forms");
    for (size_t i = 0; i < value.array_value.size(); ++i)
        check(value.array_value[i].kind == json_number,
              "valid number has number kind");
    check(value.array_value[3].number_value == -1.25,
          "fraction number decode");
    check(value.array_value[4].number_value > 6.0e23,
          "exponent number decode");

    const std::string failures[] = {
        "{\"x\":01}",
        "{\"x\":-01}",
        "{\"x\":+1}",
        "{\"x\":.1}",
        "{\"x\":1.}",
        "{\"x\":1e}",
        "{\"x\":1e+}",
        "{\"x\":-}",
        "{\"x\":NaN}",
        "{\"x\":Infinity}",
        "{\"x\":1e999}",
        "{\"x\":1e-9999}",
    };
    for (size_t i = 0; i < sizeof(failures) / sizeof(failures[0]); ++i)
        expect_json_failure(failures[i], "invalid or non-finite number rejection");
}

static std::string json_string_containing(const std::string& bytes)
{
    return std::string("{\"x\":\"") + bytes + "\"}";
}

static void test_json_utf8_escapes_and_surrogates()
{
    char error[512] = {};
    json_value value;
    const std::string valid =
        "{\"escapes\":\"\\\"\\\\\\/\\b\\f\\n\\r\\t\","
        "\"bmp\":\"\\u2603\",\"pair\":\"\\uD83D\\uDE00\","
        "\"raw\":\"\xe2\x98\x83\",\"nul\":\"\\u0000\"}";
    check(load_text(value, valid, error), error);
    check(json_member(value, "escapes")->string_value ==
          std::string("\"\\/\b\f\n\r\t"), "simple escape decode");
    check(json_member(value, "bmp")->string_value == "\xe2\x98\x83",
          "BMP unicode escape decode");
    check(json_member(value, "pair")->string_value == "\xf0\x9f\x98\x80",
          "surrogate pair decode");
    check(json_member(value, "raw")->string_value == "\xe2\x98\x83",
          "valid raw UTF-8 preservation");
    check(json_member(value, "nul")->string_value.size() == 1 &&
          json_member(value, "nul")->string_value[0] == '\0',
          "escaped null decode");

    std::vector<std::string> failures;
    failures.push_back("{\"x\":\"\\q\"}");
    failures.push_back("{\"x\":\"\\u12G4\"}");
    failures.push_back("{\"x\":\"\\u123\"}");
    failures.push_back("{\"x\":\"\\uD800\"}");
    failures.push_back("{\"x\":\"\\uD800\\u0041\"}");
    failures.push_back("{\"x\":\"\\uDC00\"}");
    failures.push_back("{\"x\":\"unterminated}");
    failures.push_back(json_string_containing(std::string(1, '\0')));
    failures.push_back(json_string_containing(std::string(1, '\x01')));
    failures.push_back(json_string_containing(std::string(1, '\x80')));
    failures.push_back(json_string_containing(std::string("\xc0\xaf", 2)));
    failures.push_back(json_string_containing(std::string("\xe0\x80\xaf", 3)));
    failures.push_back(json_string_containing(std::string("\xe2\x28\xa1", 3)));
    failures.push_back(json_string_containing(std::string("\xed\xa0\x80", 3)));
    failures.push_back(json_string_containing(std::string("\xf4\x90\x80\x80", 4)));
    failures.push_back(std::string("{\"x\":\"") + std::string(1, '\xe2'));
    for (size_t i = 0; i < failures.size(); ++i)
        expect_json_failure(failures[i], "invalid string encoding rejection");
}

static std::string nested_array_document(const int depth)
{
    std::string text(static_cast<size_t>(depth), '[');
    text += "null";
    text.append(static_cast<size_t>(depth), ']');
    return text;
}

static void test_json_depth_limit_is_exactly_64()
{
    char error[512] = {};
    json_value value;
    check(load_text(value, nested_array_document(64), error), error);
    check(value.kind == json_array, "depth 64 accepted");
    expect_json_failure(nested_array_document(65), "depth 65 rejected");
}

static void test_json_size_limit_is_exactly_16_mib()
{
    static const size_t maximum = 16u * 1024u * 1024u;
    char error[512] = {};
    json_value value;
    std::string exact = "null";
    exact.append(maximum - exact.size(), ' ');
    check(exact.size() == maximum, "exact JSON ceiling fixture size");
    check(load_text(value, exact, error), error);
    check(value.kind == json_null, "exact 16 MiB JSON accepted");

    exact.push_back(' ');
    json_value prior = sentinel_value();
    check(!load_text(prior, exact, error), "JSON above 16 MiB rejected");
    check(std::strstr(error, json_fixture_path) != NULL,
          "oversize JSON path diagnostic");
    check(std::strstr(error, "exceeds 16 MiB") != NULL,
          "oversize JSON limit diagnostic");
    check_sentinel(prior, "oversize JSON failure is transactional");
}

static void test_json_file_errors_are_transactional()
{
    const char* const missing = "/tmp/test_scene_json_missing.json";
    std::remove(missing);
    char error[512] = {};
    json_value value = sentinel_value();
    check(!json_document_load(value, missing, error, 512),
          "missing JSON file rejected");
    check(std::strstr(error, missing) != NULL, "missing JSON path diagnostic");
    check_sentinel(value, "missing JSON failure is transactional");

    check(!json_document_load(value, NULL, error, 512),
          "null JSON path rejected");
    check(std::strstr(error, "<null>") != NULL, "null JSON path diagnostic");
    check_sentinel(value, "null path JSON failure is transactional");
    check(!json_document_load(value, "", NULL, 0),
          "empty JSON path rejected without error buffer");
    check_sentinel(value, "empty path JSON failure is transactional");
}

static void check_sha_vector(
    const char* path, const std::string& contents, const char* expected)
{
    write_text(path, contents);
    char error[512] = {};
    std::string digest = "sentinel";
    check(sha256_file_hex(digest, path, error, 512), error);
    check(digest == expected, "SHA-256 known vector");
    check(digest.size() == 64, "SHA-256 digest length");
    for (size_t i = 0; i < digest.size(); ++i)
        check((digest[i] >= '0' && digest[i] <= '9') ||
              (digest[i] >= 'a' && digest[i] <= 'f'),
              "SHA-256 lowercase hexadecimal output");
}

static void test_sha256_known_vectors_and_file_errors()
{
    check_sha_vector("/tmp/test_sha_empty", "",
        "e3b0c44298fc1c149afbf4c8996fb924"
        "27ae41e4649b934ca495991b7852b855");
    check_sha_vector("/tmp/test_sha_abc", "abc",
        "ba7816bf8f01cfea414140de5dae2223"
        "b00361a396177a9cb410ff61f20015ad");
    check_sha_vector("/tmp/test_sha_multiblock",
        "abcdbcdecdefdefgefghfghighijhijk"
        "ijkljklmklmnlmnomnopnopq",
        "248d6a61d20638b8e5c026930c3e6039"
        "a33ce45964ff2167f6ecedd419db06c1");
    check_sha_vector("/tmp/test_sha_stream", std::string(1000000, 'a'),
        "cdc76e5c9914fb9281a1c7e284d73e67"
        "f1809a48a497200e046d39ccc7112cd0");

    const char* const missing = "/tmp/test_sha_missing_task2";
    std::remove(missing);
    char error[512] = {};
    std::string digest = "sentinel";
    check(!sha256_file_hex(digest, missing, error, 512),
          "missing SHA file rejected");
    check(std::strstr(error, missing) != NULL, "missing SHA path diagnostic");
    check(digest == "sentinel", "missing SHA failure is transactional");

    check(!sha256_file_hex(digest, NULL, error, 512),
          "null SHA path rejected");
    check(std::strstr(error, "<null>") != NULL, "null SHA path diagnostic");
    check(digest == "sentinel", "null SHA failure is transactional");
    check(!sha256_file_hex(digest, "", NULL, 0),
          "empty SHA path rejected without error buffer");
    check(digest == "sentinel", "empty SHA failure is transactional");
}

int main()
{
    test_json_value_kinds_members_and_offsets();
    test_json_full_document_and_unique_keys();
    test_json_number_grammar_and_finite_range();
    test_json_utf8_escapes_and_surrogates();
    test_json_depth_limit_is_exactly_64();
    test_json_size_limit_is_exactly_16_mib();
    test_json_file_errors_are_transactional();
    test_sha256_known_vectors_and_file_errors();
    return 0;
}
