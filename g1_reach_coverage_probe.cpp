#include "reach_coverage.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr float kPi = 3.14159265358979323846F;
constexpr float kZeroPositionTolerance = 0.001F;
constexpr float kZeroApproachTolerance = 0.008726646F;

struct Count {
    size_t evaluations = 0U;
    size_t accepted = 0U;
};

struct Report {
    size_t zero_tested = 0U;
    size_t zero_failed = 0U;
    float zero_max_position = 0.0F;
    float zero_max_approach = 0.0F;
    Count position_perturbations{};
    Count orientation_perturbations{};
    std::array<size_t, reach::kRejectionCount> rejections{};
    std::map<std::string, Count> hand;
    std::map<std::string, Count> source;
    std::map<std::string, Count> height_band;
    std::map<std::string, Count> direction_band;
    std::map<std::string, Count> augmentation;
    Count union_counts{};
};

std::string escape_json(const std::string& value) {
    std::ostringstream output;
    for (const char character : value) {
        switch (character) {
            case '\\': output << "\\\\"; break;
            case '"': output << "\\\""; break;
            case '\n': output << "\\n"; break;
            case '\r': output << "\\r"; break;
            case '\t': output << "\\t"; break;
            default: output << character; break;
        }
    }
    return output.str();
}

std::string height_band(vec3 endpoint) {
    if (endpoint.y < 0.50F) return "ground";
    if (endpoint.y < 1.20F) return "middle";
    return "overhead";
}

std::string direction_band(vec3 endpoint) {
    if (std::abs(endpoint.x) >= std::abs(endpoint.z)) {
        return endpoint.x >= 0.0F ? "front" : "back";
    }
    return endpoint.z >= 0.0F ? "left" : "right";
}

void increment(Count& count, reach::Rejection rejection) {
    ++count.evaluations;
    if (rejection == reach::Rejection::None) ++count.accepted;
}

void record(
    Report& report,
    const reach::Pack& pack,
    size_t clip,
    const reach::Evaluation& evaluation,
    bool position_perturbation) {
    if (position_perturbation) {
        increment(report.position_perturbations, evaluation.rejection);
    } else {
        increment(report.orientation_perturbations, evaluation.rejection);
    }
    ++report.rejections.at(static_cast<size_t>(evaluation.rejection));
    increment(report.union_counts, evaluation.rejection);
    const reach::Hand hand = static_cast<reach::Hand>(
        pack.database.active_hands.at(clip));
    increment(report.hand[hand == reach::Hand::Left ? "left" : "right"],
        evaluation.rejection);
    increment(report.source[pack.database.source_names.at(
        pack.database.source_indices.at(clip))], evaluation.rejection);
    const interaction::Transform endpoint = reach::endpoint_transform(
        pack.database, clip);
    increment(report.height_band[height_band(endpoint.position)],
        evaluation.rejection);
    increment(report.direction_band[direction_band(endpoint.position)],
        evaluation.rejection);
    const reach::Augmentation augmentation = static_cast<reach::Augmentation>(
        pack.database.augmentations.at(clip));
    increment(report.augmentation[
        augmentation == reach::Augmentation::Captured
            ? "captured" : "mirrored"], evaluation.rejection);
}

reach::Query own_query(const reach::Pack& pack, size_t clip) {
    reach::Query query{};
    query.hand = static_cast<reach::Hand>(
        pack.database.active_hands.at(clip));
    query.target = reach::endpoint_transform(pack.database, clip);
    query.approach_world = reach::approach_direction(pack.database, clip);
    return query;
}

Report run_clip(const reach::Pack& pack, size_t clip) {
    Report report{};
    const reach::Evaluation zero = reach::shape_candidate(
        pack, reach::Candidate{clip}, own_query(pack, clip));
    ++report.zero_tested;
    report.zero_max_position = zero.position_error_m;
    report.zero_max_approach = zero.approach_error_radians;
    if (zero.rejection != reach::Rejection::None ||
        zero.position_error_m > kZeroPositionTolerance ||
        zero.approach_error_radians > kZeroApproachTolerance) {
        ++report.zero_failed;
    }

    constexpr std::array<float, 5U> position_offsets = {
        0.05F, 0.10F, 0.20F, 0.30F, 0.45F};
    constexpr std::array<vec3, 3U> axes = {
        vec3(1, 0, 0), vec3(0, 1, 0), vec3(0, 0, 1)};
    for (const vec3 axis : axes) {
        for (const float offset : position_offsets) {
            for (const float sign : {-1.0F, 1.0F}) {
                reach::Query query = own_query(pack, clip);
                query.target.position = query.target.position +
                    sign * offset * axis;
                const reach::Evaluation evaluation = reach::shape_candidate(
                    pack, reach::Candidate{clip}, query);
                record(report, pack, clip, evaluation, true);
            }
        }
    }

    constexpr std::array<float, 4U> orientation_angles = {
        0.261799388F, 0.523598776F, 1.047197551F, 1.570796327F};
    for (const vec3 axis : axes) {
        for (const float angle : orientation_angles) {
            for (const float sign : {-1.0F, 1.0F}) {
                reach::Query query = own_query(pack, clip);
                query.target.rotation = quat_mul(
                    query.target.rotation,
                    quat_from_angle_axis(sign * angle, axis));
                const reach::Evaluation evaluation = reach::shape_candidate(
                    pack, reach::Candidate{clip}, query);
                record(report, pack, clip, evaluation, false);
            }
        }
    }
    reach::Query twist = own_query(pack, clip);
    twist.target.rotation = quat_mul(
        twist.target.rotation,
        quat_from_angle_axis(kPi, vec3(1, 0, 0)));
    record(report, pack, clip,
        reach::shape_candidate(pack, reach::Candidate{clip}, twist), false);
    return report;
}

void merge_count(Count& destination, const Count& source) {
    destination.evaluations += source.evaluations;
    destination.accepted += source.accepted;
}

void merge_groups(
    std::map<std::string, Count>& destination,
    const std::map<std::string, Count>& source) {
    for (const auto& [name, count] : source) {
        merge_count(destination[name], count);
    }
}

void merge_report(Report& destination, const Report& source) {
    destination.zero_tested += source.zero_tested;
    destination.zero_failed += source.zero_failed;
    destination.zero_max_position = std::max(
        destination.zero_max_position, source.zero_max_position);
    destination.zero_max_approach = std::max(
        destination.zero_max_approach, source.zero_max_approach);
    merge_count(
        destination.position_perturbations, source.position_perturbations);
    merge_count(
        destination.orientation_perturbations,
        source.orientation_perturbations);
    for (size_t reason = 0U; reason < reach::kRejectionCount; ++reason) {
        destination.rejections[reason] += source.rejections[reason];
    }
    merge_groups(destination.hand, source.hand);
    merge_groups(destination.source, source.source);
    merge_groups(destination.height_band, source.height_band);
    merge_groups(destination.direction_band, source.direction_band);
    merge_groups(destination.augmentation, source.augmentation);
    merge_count(destination.union_counts, source.union_counts);
}

Report run_probe(const reach::Pack& pack) {
    const size_t worker_count = std::max<size_t>(1U, std::min<size_t>(
        8U,
        std::min<size_t>(
            pack.database.clip_count,
            std::max(1U, std::thread::hardware_concurrency()))));
    std::atomic<size_t> next_clip{0U};
    std::vector<Report> partial(worker_count);
    std::vector<std::thread> workers;
    workers.reserve(worker_count);
    for (size_t worker = 0U; worker < worker_count; ++worker) {
        workers.emplace_back([&, worker]() {
            while (true) {
                const size_t clip = next_clip.fetch_add(1U);
                if (clip >= pack.database.clip_count) break;
                merge_report(partial[worker], run_clip(pack, clip));
            }
        });
    }
    for (std::thread& worker : workers) worker.join();
    Report report{};
    for (const Report& value : partial) merge_report(report, value);
    return report;
}

void write_count(std::ostream& output, const Count& count) {
    output << "{\"accepted\":" << count.accepted
           << ",\"evaluations\":" << count.evaluations << '}';
}

void write_groups(
    std::ostream& output,
    const std::map<std::string, Count>& groups) {
    output << '{';
    bool first = true;
    for (const auto& [name, count] : groups) {
        if (!first) output << ',';
        first = false;
        output << '"' << escape_json(name) << "\":";
        write_count(output, count);
    }
    output << '}';
}

std::string to_json(const Report& report) {
    std::ostringstream output;
    output << std::fixed << std::setprecision(7);
    output << "{\"augmentation\":";
    write_groups(output, report.augmentation);
    output << ",\"direction_band\":";
    write_groups(output, report.direction_band);
    output << ",\"hand\":";
    write_groups(output, report.hand);
    output << ",\"height_band\":";
    write_groups(output, report.height_band);
    output << ",\"orientation_perturbations\":";
    write_count(output, report.orientation_perturbations);
    output << ",\"position_perturbations\":";
    write_count(output, report.position_perturbations);
    output << ",\"rejections\":{";
    for (size_t reason = 0U; reason < report.rejections.size(); ++reason) {
        if (reason != 0U) output << ',';
        output << '"' << reach::rejection_name(
            static_cast<reach::Rejection>(reason)) << "\":"
               << report.rejections[reason];
    }
    output << "},\"source\":";
    write_groups(output, report.source);
    output << ",\"union\":";
    write_count(output, report.union_counts);
    output << ",\"zero_retarget\":{\"failed\":" << report.zero_failed
           << ",\"max_approach_radians\":" << report.zero_max_approach
           << ",\"max_position_m\":" << report.zero_max_position
           << ",\"tested\":" << report.zero_tested << "}}\n";
    return output.str();
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2 && argc != 4) {
            throw std::invalid_argument(
                "usage: g1_reach_coverage_probe PACK [--json OUTPUT]");
        }
        std::filesystem::path json_path;
        if (argc == 4) {
            if (std::string(argv[2]) != "--json") {
                throw std::invalid_argument("expected --json OUTPUT");
            }
            json_path = argv[3];
        }
        const reach::Pack pack = reach::load_pack(argv[1]);
        const Report report = run_probe(pack);
        const std::string json = to_json(report);
        std::cout << json;
        if (!json_path.empty()) {
            if (!json_path.parent_path().empty()) {
                std::filesystem::create_directories(json_path.parent_path());
            }
            std::ofstream output(json_path);
            if (!output) throw std::runtime_error("cannot open JSON output");
            output << json;
            if (!output) throw std::runtime_error("cannot write JSON output");
        }
        return report.zero_failed == 0U ? EXIT_SUCCESS : EXIT_FAILURE;
    } catch (const std::exception& error) {
        std::cerr << "g1 reach coverage probe: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
