#pragma once

#include <cerrno>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <string>

struct cleanup_report
{
    int exit_code = 0;
    int motion_pack_load_count = 0;
    int model_load_count = 0;
    int model_unload_count = 0;
    bool log_closed = false;
    bool window_closed = false;
};

static inline bool cleanup_error(
    char* error, int capacity, const char* message)
{
    if (error != NULL && capacity > 0) {
        std::snprintf(
            error, static_cast<std::size_t>(capacity), "%s",
            message != NULL ? message : "cleanup report error");
    }
    return false;
}

static inline bool cleanup_report_write(
    const char* path,
    const cleanup_report& report,
    char* error,
    int capacity)
{
    if (path == NULL) return true;
    if (path[0] == '\0' ||
        report.motion_pack_load_count != 1 ||
        report.model_load_count < 0 ||
        report.model_unload_count < 0 ||
        report.model_load_count != report.model_unload_count ||
        !report.log_closed || !report.window_closed) {
        return cleanup_error(
            error, capacity,
            "cleanup report has invalid path or incomplete cleanup");
    }

    char payload[256] = {};
    const int payload_size = std::snprintf(
        payload, sizeof(payload),
        "{\"exit_code\":%d,\"live_model_count\":0,"
        "\"log_closed\":true,\"model_load_count\":%d,"
        "\"model_unload_count\":%d,"
        "\"motion_pack_load_count\":1,"
        "\"window_closed\":true}\n",
        report.exit_code,
        report.model_load_count,
        report.model_unload_count);
    if (payload_size < 0 ||
        static_cast<std::size_t>(payload_size) >= sizeof(payload)) {
        return cleanup_error(
            error, capacity, "cleanup report formatting failed");
    }

    const std::string temporary = std::string(path) + ".tmp";
    errno = 0;
    FILE* file = std::fopen(temporary.c_str(), "wb");
    if (file == NULL) {
        const int saved_errno = errno != 0 ? errno : EIO;
        const std::string message =
            temporary + ": cannot open cleanup report (" +
            std::strerror(saved_errno) + ")";
        return cleanup_error(error, capacity, message.c_str());
    }

    int saved_errno = 0;
    errno = 0;
    const std::size_t written = std::fwrite(
        payload, 1, static_cast<std::size_t>(payload_size), file);
    if (written != static_cast<std::size_t>(payload_size)) {
        saved_errno = errno != 0 ? errno : EIO;
    }

    errno = 0;
    if (std::fflush(file) != 0 && saved_errno == 0) {
        saved_errno = errno != 0 ? errno : EIO;
    }

    errno = 0;
    if (std::fclose(file) != 0 && saved_errno == 0) {
        saved_errno = errno != 0 ? errno : EIO;
    }

    if (saved_errno == 0) {
        errno = 0;
        if (std::rename(temporary.c_str(), path) != 0) {
            saved_errno = errno != 0 ? errno : EIO;
        }
    }

    if (saved_errno != 0) {
        std::remove(temporary.c_str());
        const std::string message =
            std::string(path) + ": cannot finish cleanup report (" +
            std::strerror(saved_errno) + ")";
        return cleanup_error(error, capacity, message.c_str());
    }
    return true;
}
