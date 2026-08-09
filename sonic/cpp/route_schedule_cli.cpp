#include "../../route_runtime.h"

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

static const float g1_dt = 1.0f / 60.0f;

static bool fail(const std::string& message)
{
    std::cerr << "route_schedule_cli: " << message << '\n';
    return false;
}

static int open_without_symlinks(const char* path)
{
    if (path == NULL || path[0] == '\0') return -1;
    const std::string value(path);
    std::vector<std::string> parts;
    size_t cursor = 0;
    while (cursor <= value.size()) {
        const size_t slash = value.find('/', cursor);
        const size_t end = slash == std::string::npos ? value.size() : slash;
        const std::string part = value.substr(cursor, end - cursor);
        if (!part.empty() && part != ".") {
            if (part == "..") return -1;
            parts.push_back(part);
        }
        if (slash == std::string::npos) break;
        cursor = slash + 1;
    }
    if (parts.empty()) return -1;
    const int directory_flags = O_RDONLY | O_CLOEXEC | O_DIRECTORY | O_NOFOLLOW;
    int directory = open(value[0] == '/' ? "/" : ".", directory_flags);
    if (directory < 0) return -1;
    for (size_t index = 0; index + 1 < parts.size(); ++index) {
        const int child = openat(directory, parts[index].c_str(), directory_flags);
        close(directory);
        if (child < 0) return -1;
        directory = child;
    }
    const int descriptor = openat(
        directory, parts.back().c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    close(directory);
    return descriptor;
}

static bool load_json(json_value& document, const char* path)
{
    const int descriptor = open_without_symlinks(path);
    struct stat metadata = {};
    if (descriptor < 0 || fstat(descriptor, &metadata) != 0 ||
        !S_ISREG(metadata.st_mode) || metadata.st_nlink != 1) {
        if (descriptor >= 0) close(descriptor);
        return fail("scene JSON must be a regular single-link file");
    }
    static const size_t maximum = 16u * 1024u * 1024u;
    std::string text;
    char buffer[65536];
    bool read_failed = false;
    for (;;) {
        const ssize_t count = read(descriptor, buffer, sizeof(buffer));
        if (count > 0) {
            if (text.size() > maximum - static_cast<size_t>(count)) {
                read_failed = true;
                break;
            }
            text.append(buffer, static_cast<size_t>(count));
            continue;
        }
        if (count == 0) break;
        if (errno == EINTR) continue;
        read_failed = true;
        break;
    }
    if (close(descriptor) != 0) read_failed = true;
    if (read_failed)
        return fail("cannot read scene JSON or document exceeds 16 MiB");
    json_value parsed;
    json_parser parser(path, text);
    if (!parser.value(parsed))
        return fail(std::string("invalid scene JSON: ") + parser.reason);
    parser.whitespace();
    if (parser.cursor != text.size()) return fail("scene JSON has trailing data");
    document = std::move(parsed);
    return true;
}

static bool parse_route(
    std::string& scene_id,
    scene_route& route,
    const json_value& document,
    const char* route_id)
{
    char error[512] = {};
    const json_value* id = json_member(document, "id");
    const json_value* routes = json_member(document, "routes");
    if (id == NULL || routes == NULL ||
        !scene_string(scene_id, *id, "scene id", error, sizeof(error)) ||
        !scene_id_is_safe(scene_id) || routes->kind != json_array)
        return fail(error[0] != '\0' ? error : "invalid scene identity/routes");

    int matches = 0;
    scene_route candidate;
    for (size_t index = 0; index < routes->array_value.size(); ++index) {
        const json_value& value = routes->array_value[index];
        const json_value* route_name = json_member(value, "id");
        std::string parsed_id;
        if (route_name == NULL ||
            !scene_string(parsed_id, *route_name, "route id", error, sizeof(error)))
            return fail(error);
        if (parsed_id != route_id) continue;
        ++matches;
        if (!scene_exact_keys(
                value,
                {"id", "waypoints_xz", "expected_outcome",
                 "walkability_class", "landing_hold_seconds"},
                "route", error, sizeof(error)))
            return fail(error);
        const json_value* outcome = json_member(value, "expected_outcome");
        const json_value* classification = json_member(value, "walkability_class");
        const json_value* hold = json_member(value, "landing_hold_seconds");
        const json_value* waypoints = json_member(value, "waypoints_xz");
        if (outcome == NULL || classification == NULL || hold == NULL ||
            waypoints == NULL ||
            !scene_string(candidate.expected_outcome, *outcome,
                          "route outcome", error, sizeof(error)) ||
            !scene_number_int(candidate.walkability_class, *classification,
                              "route class", error, sizeof(error)) ||
            !scene_number_binary32(candidate.landing_hold_seconds, *hold,
                                   "route hold", error, sizeof(error)) ||
            candidate.expected_outcome != "traverse" ||
            candidate.walkability_class != 1 ||
            waypoints->kind != json_array || waypoints->array_value.size() < 2)
            return fail(error[0] != '\0' ? error : "invalid registered route");
        candidate.id = parsed_id;
        for (size_t point = 0; point < waypoints->array_value.size(); ++point) {
            float values[2] = {};
            if (!scene_binary32_array(
                    values, 2, waypoints->array_value[point],
                    "route waypoint", error, sizeof(error)))
                return fail(error);
            candidate.waypoints_xz.push_back(
                std::make_pair(values[0], values[1]));
        }
    }
    if (matches != 1) return fail("route ID is missing or duplicated");
    if (!deterministic_route_inputs_valid(candidate, g1_dt, 0.50f))
        return fail("route fails deterministic command validation");
    route = std::move(candidate);
    return true;
}

static uint32_t bits(const float value)
{
    uint32_t output = 0;
    std::memcpy(&output, &value, sizeof(output));
    return output;
}

int main(int argc, char** argv)
{
    if (argc != 3) {
        std::cerr << "usage: route_schedule_cli SCENE_JSON ROUTE_ID\n";
        return 2;
    }
    json_value document;
    if (!load_json(document, argv[1])) return 2;
    std::string scene_id;
    scene_route route;
    if (!parse_route(scene_id, route, document, argv[2])) return 2;
    const int motion_frames = deterministic_route_motion_frames(
        route, g1_dt, 0.50f);
    if (motion_frames <= 0) {
        std::cerr << "route_schedule_cli: invalid route motion frame count\n";
        return 2;
    }

    std::cout
        << "{\"schema\":\"mm-sonic-route-schedule/v1\","
        << "\"scene_id\":\"" << scene_id << "\","
        << "\"route_id\":\"" << route.id << "\","
        << "\"source_rate_hz\":60,\"speed_mps\":0.5,"
        << "\"motion_frames\":" << motion_frames << ",\"frames\":[";
    char error[512] = {};
    for (int frame = 0; frame < motion_frames; ++frame) {
        deterministic_route_sample sample;
        if (!deterministic_route_command(
                sample, route, frame, g1_dt, 0.50f,
                error, static_cast<int>(sizeof(error)))) {
            std::cerr << "route_schedule_cli: " << error << '\n';
            return 2;
        }
        if (frame != 0) std::cout << ',';
        std::cout
            << "{\"frame_index\":" << frame
            << ",\"waypoint_index\":" << sample.waypoint
            << ",\"complete\":" << (sample.complete ? "true" : "false")
            << ",\"velocity_holden_bits\":["
            << bits(sample.command.x) << ','
            << bits(sample.command.y) << ','
            << bits(sample.command.z) << "]}";
    }
    std::cout << "]}\n";
    return 0;
}
