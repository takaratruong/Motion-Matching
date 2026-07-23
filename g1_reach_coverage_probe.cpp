#include "reach_coverage_metrics.h"
#include "reach_search.h"
#include "reach_placement.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr float kPi = 3.14159265358979323846F;
constexpr float kWristContactOffset = 0.04F;
constexpr float kTableTop = 0.65F;
constexpr float kTableThickness = 0.06F;
constexpr float kTableCenterY = 0.62F;
constexpr vec3 kTableDimensions(1.20F, kTableThickness, 0.75F);
constexpr float kAcceptedPositionM = 0.001F;
constexpr size_t kExpectedInstances = 4608U;
constexpr size_t kRootAzimuthSectors = 12U;
constexpr size_t kMinimumOpenAccepted = 94U;
constexpr size_t kMinimumOpenAcceptedPerHand = 47U;
constexpr size_t kMinimumOpenAzimuthSectors = 5U;

struct Fixture {
    std::string name;
    reach::ExhaustiveQuery query{};
    interaction::OrientedBox object{};
    interaction::EnvironmentGeometry environment{};
};

struct FixtureReport {
    bool complete = false;
    size_t raw_instances = 0U;
    size_t processed_instances = 0U;
    size_t accepted = 0U;
    std::array<size_t, 2U> hands{};
    size_t root_azimuth_sectors = 0U;
    float maximum_accepted_position_m = 0.0F;
    float maximum_accepted_approach_radians = 0.0F;
    float maximum_accepted_orientation_radians = 0.0F;
    std::array<size_t, reach::kRejectionCount> rejections{};
    size_t object_collision_observed = 0U;
    size_t environment_collision_observed = 0U;
    double elapsed_seconds = 0.0;
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

Fixture make_fixture(
    std::string name,
    vec3 object_position,
    const interaction::EnvironmentGeometry& environment) {
    constexpr vec3 dimensions(0.10F, 0.10F, 0.10F);
    const interaction::Transform object{object_position, quat()};
    const vec3 target_position = object_position + vec3(
        0.5F * dimensions.x + kWristContactOffset, 0.0F, 0.0F);
    return {
        std::move(name),
        {
            {
                target_position,
                quat_from_angle_axis(kPi, vec3(0.0F, 1.0F, 0.0F)),
            },
            vec3(-1.0F, 0.0F, 0.0F),
        },
        {object, dimensions},
        environment,
    };
}

std::vector<Fixture> shared_grasps() {
    const interaction::Transform table_world{
        vec3(0.0F, kTableCenterY, 0.0F), quat()};
    const interaction::EnvironmentGeometry coverage =
        interaction::make_coverage_environment(
            table_world, kTableDimensions);
    const interaction::EnvironmentGeometry open{};
    const auto supported_center = [](const interaction::OrientedBox& support) {
        return support.world.position +
            vec3(0.0F, 0.5F * support.dimensions.y + 0.05F, 0.0F);
    };
    return {
        make_fixture("open_space", vec3(0.0F, 0.90F, 0.0F), open),
        make_fixture("table", supported_center(coverage.boxes.at(0U)), coverage),
        make_fixture("shelf", supported_center(coverage.boxes.at(5U)), coverage),
        make_fixture(
            "below_table", vec3(0.0F, kTableTop - 0.32F, 0.0F), coverage),
        make_fixture(
            "lower_table", supported_center(coverage.boxes.at(8U)), coverage),
    };
}

FixtureReport evaluate_fixture(
    const reach::Pack& pack,
    const Fixture& fixture,
    const reach::SearchConfig& config) {
    const reach::SearchResult result = reach::search_all(
        pack,
        fixture.query,
        fixture.object,
        fixture.environment,
        config);
    FixtureReport report{};
    report.complete = result.complete;
    report.raw_instances = result.total;
    report.processed_instances = result.processed;
    report.accepted = result.accepted.size();
    report.elapsed_seconds =
        std::chrono::duration<double>(result.elapsed).count();
    std::vector<reach::Candidate> accepted_candidates;
    accepted_candidates.reserve(result.accepted.size());

    for (const reach::CompactEvaluation& compact : result.evaluations) {
        const reach::Evaluation& evaluation = compact.evaluation;
        ++report.rejections.at(
            static_cast<size_t>(evaluation.rejection));
        if (evaluation.object_collision_observed) {
            ++report.object_collision_observed;
        }
        if (evaluation.environment_collision_observed) {
            ++report.environment_collision_observed;
        }
    }
    for (const size_t index : result.accepted) {
        const reach::CompactEvaluation& compact = result.evaluations[index];
        const reach::Evaluation& evaluation = compact.evaluation;
        accepted_candidates.push_back(evaluation.candidate);
        ++report.hands.at(
            pack.database.active_hands.at(evaluation.candidate.clip));
        report.maximum_accepted_position_m = std::max(
            report.maximum_accepted_position_m,
            evaluation.position_error_m);
        report.maximum_accepted_approach_radians = std::max(
            report.maximum_accepted_approach_radians,
            evaluation.approach_error_radians);
        report.maximum_accepted_orientation_radians = std::max(
            report.maximum_accepted_orientation_radians,
            evaluation.orientation_error_radians);
    }
    report.root_azimuth_sectors =
        reach::count_placed_root_azimuth_sectors(
            pack,
            accepted_candidates,
            fixture.query.target.position,
            kRootAzimuthSectors);
    return report;
}

void write_fixture_report(
    std::ostream& output,
    const FixtureReport& report) {
    output << "{\"accepted\":" << report.accepted
           << ",\"complete\":" << (report.complete ? "true" : "false")
           << ",\"coverage_demonstrated\":"
           << (report.accepted > 0U ? "true" : "false")
           << ",\"elapsed_seconds\":" << report.elapsed_seconds
           << ",\"hands\":{\"left\":" << report.hands[0]
           << ",\"right\":" << report.hands[1] << '}';
    const auto accepted_metric = [&](const char* name, float value) {
        output << ",\"" << name << "\":";
        if (report.accepted == 0U) {
            output << "null";
        } else {
            output << value;
        }
    };
    accepted_metric(
        "maximum_accepted_approach_radians",
        report.maximum_accepted_approach_radians);
    accepted_metric(
        "maximum_accepted_orientation_radians",
        report.maximum_accepted_orientation_radians);
    accepted_metric(
        "maximum_accepted_position_m",
        report.maximum_accepted_position_m);
    output << ",\"observed_collisions\":{\"environment\":"
           << report.environment_collision_observed
           << ",\"object\":" << report.object_collision_observed << '}'
           << ",\"processed_instances\":" << report.processed_instances
           << ",\"raw_instances\":" << report.raw_instances
           << ",\"rejections\":{";
    for (size_t reason = 0U; reason < report.rejections.size(); ++reason) {
        if (reason != 0U) output << ',';
        output << '"' << reach::rejection_name(
            static_cast<reach::Rejection>(reason)) << "\":"
               << report.rejections[reason];
    }
    output << "},\"root_azimuth_sectors\":"
           << report.root_azimuth_sectors << '}';
}

std::string to_json(
    const std::vector<Fixture>& fixtures,
    const std::vector<FixtureReport>& reports,
    bool search_integrity_passed) {
    std::ostringstream output;
    output << std::fixed << std::setprecision(7);
    output << "{\"search_integrity_passed\":"
           << (search_integrity_passed ? "true" : "false")
           << ",\"expected_instances\":" << kExpectedInstances
           << ",\"shared_grasps\":{";
    for (size_t index = 0U; index < fixtures.size(); ++index) {
        if (index != 0U) output << ',';
        output << '"' << escape_json(fixtures[index].name) << "\":";
        write_fixture_report(output, reports[index]);
    }
    output << "}}\n";
    return output.str();
}

bool valid_report(
    const std::vector<Fixture>& fixtures,
    const std::vector<FixtureReport>& reports) {
    bool valid = true;
    for (size_t index = 0U; index < reports.size(); ++index) {
        const FixtureReport& report = reports[index];
        valid = valid && report.complete &&
                report.raw_instances == kExpectedInstances &&
                report.processed_instances == kExpectedInstances &&
                (report.accepted == 0U ||
                 report.maximum_accepted_position_m <= kAcceptedPositionM) &&
                report.elapsed_seconds <= 30.0;
        if (fixtures[index].name == "open_space") {
            valid = valid &&
                    report.accepted >= kMinimumOpenAccepted &&
                    report.hands[0] >= kMinimumOpenAcceptedPerHand &&
                    report.hands[1] >= kMinimumOpenAcceptedPerHand &&
                    report.root_azimuth_sectors >= kMinimumOpenAzimuthSectors;
        }
    }
    return valid;
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
        const std::vector<Fixture> fixtures = shared_grasps();
        reach::SearchConfig config{};
        config.worker_count = std::max<size_t>(1U, std::min<size_t>(
            8U, std::max(1U, std::thread::hardware_concurrency())));
        config.deadline = std::chrono::seconds(30);
        std::vector<FixtureReport> reports;
        reports.reserve(fixtures.size());
        for (const Fixture& fixture : fixtures) {
            reports.push_back(evaluate_fixture(pack, fixture, config));
        }
        const bool search_integrity_passed = valid_report(fixtures, reports);
        const std::string json = to_json(
            fixtures, reports, search_integrity_passed);
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
        return search_integrity_passed ? EXIT_SUCCESS : EXIT_FAILURE;
    } catch (const std::exception& error) {
        std::cerr << "g1 reach coverage probe: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
