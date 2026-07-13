#include "motion_match_log.h"

#include <stdio.h>
#include <string.h>
#include <unistd.h>

static void check(bool condition, const char* expression, int line)
{
    if (!condition)
    {
        fprintf(stderr, "CHECK failed at line %d: %s\n", line, expression);
        exit(1);
    }
}

#define CHECK(expression) check((expression), #expression, __LINE__)

static void replace_with_dev_full(motion_match_log& log)
{
    FILE* replacement = freopen("/dev/full", "w", log.file);
    CHECK(replacement != NULL);
    log.file = replacement;
}

static motion_match_log_row make_row(const char* query_bits_hex = "")
{
    motion_match_log_row row = {};
    row.query_bits_hex = query_bits_hex;
    return row;
}

static float float_from_bits(const uint32_t bits)
{
    float value = 0.0f;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static void check_query_bits_reject_nonfinite()
{
    array1d<float> query(31);
    query.zero();
    char output[31 * 8 + 1] = {};

    query(0) = float_from_bits(UINT32_C(0x7f7fffff));
    CHECK(motion_match_query_is_finite_31d(query));
    CHECK(motion_match_query_bits_hex(output, sizeof(output), query));
    CHECK(strncmp(output, "7f7fffff", 8) == 0);

    query(0) = float_from_bits(UINT32_C(0x7f800000));
    CHECK(!motion_match_query_is_finite_31d(query));
    CHECK(!motion_match_query_bits_hex(output, sizeof(output), query));

    query(0) = float_from_bits(UINT32_C(0x7fc00000));
    CHECK(!motion_match_query_is_finite_31d(query));
    CHECK(!motion_match_query_bits_hex(output, sizeof(output), query));
}

int main(int argc, char** argv)
{
    CHECK(argc == 1 || argc == 2);
    char error[512] = {};

    check_query_bits_reject_nonfinite();

    {
        motion_match_log log;
        CHECK(!log.open("/dev/full", error, sizeof(error)));
        CHECK(strstr(error, "/dev/full") != NULL);
        CHECK(strstr(error, "initialize") != NULL);
        CHECK(log.file == NULL);
    }

    char write_path[] = "/tmp/test_motion_match_log_write_XXXXXX";
    int write_fd = mkstemp(write_path);
    CHECK(write_fd >= 0);
    CHECK(close(write_fd) == 0);

    {
        motion_match_log log;
        CHECK(log.open(write_path, error, sizeof(error)));
        replace_with_dev_full(log);
        motion_match_log_row row = make_row();
        CHECK(!log.write(row, error, sizeof(error)));
        CHECK(strstr(error, "write") != NULL);
        (void)log.close(error, sizeof(error));
        CHECK(log.file == NULL);
    }

    CHECK(remove(write_path) == 0);

    char close_path[] = "/tmp/test_motion_match_log_close_XXXXXX";
    int close_fd = mkstemp(close_path);
    CHECK(close_fd >= 0);
    CHECK(close(close_fd) == 0);

    {
        motion_match_log log;
        CHECK(log.open(close_path, error, sizeof(error)));
        replace_with_dev_full(log);
        CHECK(fputs("buffered close failure", log.file) >= 0);
        CHECK(!log.close(error, sizeof(error)));
        CHECK(strstr(error, "close") != NULL);
        CHECK(log.file == NULL);
    }

    CHECK(remove(close_path) == 0);

    char generated_success_path[] =
        "/tmp/test_motion_match_log_success_XXXXXX";
    const bool retain_success = argc == 2;
    const char* success_path = retain_success
        ? argv[1] : generated_success_path;
    if (!retain_success)
    {
        int success_fd = mkstemp(generated_success_path);
        CHECK(success_fd >= 0);
        CHECK(close(success_fd) == 0);
    }

    {
        motion_match_log log;
        CHECK(log.open(success_path, error, sizeof(error)));
        array1d<float> query(31);
        query.zero();
        char query_bits_hex[31 * 8 + 1] = {};
        CHECK(motion_match_query_bits_hex(
            query_bits_hex, sizeof(query_bits_hex), query));
        motion_match_log_row row = make_row(query_bits_hex);
        CHECK(log.write(row, error, sizeof(error)));
        CHECK(log.close(error, sizeof(error)));
        CHECK(log.file == NULL);
    }

    if (!retain_success)
    {
        CHECK(remove(success_path) == 0);
    }

    printf("VALID motion-log finite-bits and io failures surfaced\n");
    return 0;
}
