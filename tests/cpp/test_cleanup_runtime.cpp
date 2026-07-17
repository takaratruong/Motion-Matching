#include "cleanup_runtime.h"

#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>

#include <sys/stat.h>
#include <unistd.h>

static void check(bool value, const char* message)
{
    if (!value) {
        std::fprintf(stderr, "cleanup runtime test failed: %s\n", message);
        std::exit(1);
    }
}

static std::string fixture_path(const char* suffix)
{
    char path[256] = {};
    const int written = std::snprintf(
        path, sizeof(path), "/tmp/test_g1_cleanup_%ld_%s",
        static_cast<long>(getpid()), suffix);
    check(written > 0 && static_cast<std::size_t>(written) < sizeof(path),
          "fixture path fits");
    return std::string(path);
}

static void remove_file_if_present(const std::string& path)
{
    if (unlink(path.c_str()) != 0) {
        check(errno == ENOENT, "remove stale fixture file");
    }
}

static void remove_directory_if_present(const std::string& path)
{
    if (rmdir(path.c_str()) != 0) {
        check(errno == ENOENT, "remove stale fixture directory");
    }
}

static bool path_exists(const std::string& path)
{
    struct stat status = {};
    if (lstat(path.c_str(), &status) == 0) return true;
    check(errno == ENOENT, "fixture existence check");
    return false;
}

static bool path_is_directory(const std::string& path)
{
    struct stat status = {};
    check(lstat(path.c_str(), &status) == 0, "stat fixture directory");
    return S_ISDIR(status.st_mode);
}

static void write_text(const std::string& path, const char* text)
{
    FILE* file = std::fopen(path.c_str(), "wb");
    check(file != NULL, "open fixture for writing");
    const std::size_t length = std::strlen(text);
    check(std::fwrite(text, 1, length, file) == length,
          "write fixture text");
    check(std::fclose(file) == 0, "close fixture text");
}

static std::string read_text(const std::string& path)
{
    FILE* file = std::fopen(path.c_str(), "rb");
    check(file != NULL, "open fixture for reading");
    std::string text;
    char buffer[256] = {};
    while (true) {
        const std::size_t count =
            std::fread(buffer, 1, sizeof(buffer), file);
        text.append(buffer, count);
        if (count != sizeof(buffer)) break;
    }
    check(std::ferror(file) == 0, "read fixture text");
    check(std::fclose(file) == 0, "close fixture after reading");
    return text;
}

static cleanup_report valid_report()
{
    cleanup_report report;
    report.exit_code = -7;
    report.motion_pack_load_count = 1;
    report.model_load_count = 28;
    report.model_unload_count = 28;
    report.log_closed = true;
    report.window_closed = true;
    return report;
}

static void check_report_equal(
    const cleanup_report& actual,
    const cleanup_report& expected,
    const char* message)
{
    check(actual.exit_code == expected.exit_code &&
              actual.motion_pack_load_count ==
                  expected.motion_pack_load_count &&
              actual.model_load_count == expected.model_load_count &&
              actual.model_unload_count == expected.model_unload_count &&
              actual.log_closed == expected.log_closed &&
              actual.window_closed == expected.window_closed,
          message);
}

static void check_contains(
    const char* text, const std::string& fragment, const char* message)
{
    check(text != NULL && std::strstr(text, fragment.c_str()) != NULL,
          message);
}

static void test_null_path_is_noop_and_preserves_inputs()
{
    cleanup_report report;
    const cleanup_report prior = report;
    char error[32] = "unchanged";
    check(cleanup_report_write(NULL, report, error, sizeof(error)),
          "NULL cleanup path is disabled");
    check(std::strcmp(error, "unchanged") == 0,
          "NULL cleanup path preserves error text");
    check_report_equal(report, prior, "NULL cleanup path preserves report");
}

static void test_exact_json_replaces_final_atomically()
{
    const std::string path = fixture_path("exact.json");
    const std::string temporary = path + ".tmp";
    remove_file_if_present(temporary);
    remove_file_if_present(path);
    write_text(path, "old-final\n");
    write_text(temporary, "stale-temporary\n");

    cleanup_report report = valid_report();
    const cleanup_report prior = report;
    char error[256] = "success-sentinel";
    check(cleanup_report_write(path.c_str(), report, error, sizeof(error)),
          error);
    check(read_text(path) ==
              "{\"exit_code\":-7,\"live_model_count\":0,"
              "\"log_closed\":true,\"model_load_count\":28,"
              "\"model_unload_count\":28,"
              "\"motion_pack_load_count\":1,"
              "\"window_closed\":true}\n",
          "cleanup report has exact deterministic JSON");
    check(!path_exists(temporary),
          "successful cleanup report leaves no temporary file");
    check(std::strcmp(error, "success-sentinel") == 0,
          "successful cleanup report preserves error text");
    check_report_equal(report, prior, "successful write preserves report");

    report.exit_code = 2;
    report.model_load_count = 0;
    report.model_unload_count = 0;
    const cleanup_report replacement_prior = report;
    check(cleanup_report_write(path.c_str(), report, error, sizeof(error)),
          error);
    check(read_text(path) ==
              "{\"exit_code\":2,\"live_model_count\":0,"
              "\"log_closed\":true,\"model_load_count\":0,"
              "\"model_unload_count\":0,"
              "\"motion_pack_load_count\":1,"
              "\"window_closed\":true}\n",
          "second cleanup report atomically replaces final output");
    check(!path_exists(temporary),
          "replacement leaves no temporary file");
    check_report_equal(
        report, replacement_prior, "replacement preserves report");
    remove_file_if_present(path);
}

static void test_invalid_inputs_are_transactional()
{
    static const char* invalid_message =
        "cleanup report has invalid path or incomplete cleanup";
    cleanup_report report = valid_report();
    const cleanup_report empty_prior = report;
    char error[256] = {};
    check(!cleanup_report_write("", report, error, sizeof(error)),
          "empty cleanup path is rejected");
    check(std::strcmp(error, invalid_message) == 0,
          "empty cleanup path has exact error");
    check_report_equal(report, empty_prior, "empty path preserves report");

    const std::string path = fixture_path("invalid.json");
    const std::string temporary = path + ".tmp";
    remove_file_if_present(temporary);
    remove_file_if_present(path);
    write_text(path, "prior-final\n");
    write_text(temporary, "prior-temporary\n");

    const auto expect_invalid = [&](const cleanup_report& candidate,
                                    const char* message) {
        const cleanup_report prior = candidate;
        char local_error[256] = {};
        check(!cleanup_report_write(
                  path.c_str(), candidate, local_error,
                  static_cast<int>(sizeof(local_error))),
              message);
        check(std::strcmp(local_error, invalid_message) == 0,
              "invalid report has exact error");
        check(read_text(path) == "prior-final\n",
              "invalid report preserves final file");
        check(read_text(temporary) == "prior-temporary\n",
              "invalid report preserves preexisting temporary file");
        check_report_equal(candidate, prior, "invalid write preserves report");
    };

    report = valid_report();
    report.motion_pack_load_count = 0;
    expect_invalid(report, "zero motion-pack loads rejected");
    report = valid_report();
    report.motion_pack_load_count = 2;
    expect_invalid(report, "multiple motion-pack loads rejected");
    report = valid_report();
    report.model_load_count = -1;
    expect_invalid(report, "negative model loads rejected");
    report = valid_report();
    report.model_unload_count = -1;
    expect_invalid(report, "negative model unloads rejected");
    report = valid_report();
    report.model_load_count = 2;
    report.model_unload_count = 1;
    expect_invalid(report, "live model rejected");
    report = valid_report();
    report.model_load_count = 0;
    report.model_unload_count = INT_MAX;
    expect_invalid(report, "excess model unloads rejected");
    report = valid_report();
    report.log_closed = false;
    expect_invalid(report, "open log rejected");
    report = valid_report();
    report.window_closed = false;
    expect_invalid(report, "open window rejected");

    remove_file_if_present(temporary);
    remove_file_if_present(path);
}

static void test_error_truncation_is_exact_and_bounded()
{
    cleanup_report report = valid_report();
    const cleanup_report prior = report;

    char guarded[10] = {'L', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'R'};
    check(!cleanup_report_write("", report, guarded + 1, 8),
          "guarded error call fails");
    check(guarded[0] == 'L' && guarded[9] == 'R',
          "truncated error preserves surrounding canaries");
    check(std::strcmp(guarded + 1, "cleanup") == 0,
          "truncated error has exact prefix and terminator");

    char one[3] = {'L', 'x', 'R'};
    check(!cleanup_report_write("", report, one + 1, 1),
          "one-byte error call fails");
    check(one[0] == 'L' && one[1] == '\0' && one[2] == 'R',
          "one-byte error writes only a terminator");

    char zero[8] = "zero";
    check(!cleanup_report_write("", report, zero, 0),
          "zero-capacity error call fails");
    check(std::strcmp(zero, "zero") == 0,
          "zero-capacity error buffer is untouched");

    char negative[16] = "negative";
    check(!cleanup_report_write("", report, negative, -4),
          "negative-capacity error call fails");
    check(std::strcmp(negative, "negative") == 0,
          "negative-capacity error buffer is untouched");
    check(!cleanup_report_write("", report, NULL, 32),
          "NULL error buffer is safe");
    check_report_equal(report, prior, "error paths preserve report");
}

static void test_open_failure_preserves_existing_state()
{
    const std::string path = fixture_path("open_failure.json");
    const std::string temporary = path + ".tmp";
    remove_file_if_present(path);
    remove_file_if_present(temporary);
    remove_directory_if_present(temporary);
    write_text(path, "prior-final\n");
    check(mkdir(temporary.c_str(), 0700) == 0,
          "create temporary-path directory");

    const cleanup_report report = valid_report();
    const cleanup_report prior = report;
    char error[256] = {};
    check(!cleanup_report_write(path.c_str(), report, error, sizeof(error)),
          "temporary open failure is reported");
    check_contains(error, temporary, "open failure names temporary path");
    check_contains(error, "cannot open cleanup report",
                   "open failure names operation");
    check(read_text(path) == "prior-final\n",
          "open failure preserves final output");
    check(path_is_directory(temporary),
          "open failure preserves unowned temporary directory");
    check_report_equal(report, prior, "open failure preserves report");

    remove_directory_if_present(temporary);
    remove_file_if_present(path);
}

static void test_write_failure_removes_temporary_and_preserves_final()
{
    check(access("/dev/full", F_OK) == 0, "/dev/full is available");
    const std::string path = fixture_path("write_failure.json");
    const std::string temporary = path + ".tmp";
    remove_file_if_present(temporary);
    remove_file_if_present(path);
    write_text(path, "prior-final\n");
    check(symlink("/dev/full", temporary.c_str()) == 0,
          "create deterministic write-failure symlink");

    const cleanup_report report = valid_report();
    const cleanup_report prior = report;
    char error[256] = {};
    check(!cleanup_report_write(path.c_str(), report, error, sizeof(error)),
          "write failure is reported");
    check_contains(error, path, "write failure names final path");
    check_contains(error, "cannot finish cleanup report",
                   "write failure names operation");
    check(read_text(path) == "prior-final\n",
          "write failure preserves final output");
    check(!path_exists(temporary),
          "write failure removes owned temporary symlink");
    check_report_equal(report, prior, "write failure preserves report");

    remove_file_if_present(path);
}

static void test_rename_failure_removes_temporary_and_preserves_destination()
{
    const std::string path = fixture_path("rename_failure.json");
    const std::string temporary = path + ".tmp";
    remove_file_if_present(temporary);
    remove_file_if_present(path);
    remove_directory_if_present(path);
    check(mkdir(path.c_str(), 0700) == 0,
          "create rename-failure destination directory");

    const cleanup_report report = valid_report();
    const cleanup_report prior = report;
    char error[256] = {};
    check(!cleanup_report_write(path.c_str(), report, error, sizeof(error)),
          "rename failure is reported");
    check_contains(error, path, "rename failure names final path");
    check_contains(error, "cannot finish cleanup report",
                   "rename failure names operation");
    check(path_is_directory(path),
          "rename failure preserves destination directory");
    check(!path_exists(temporary),
          "rename failure removes temporary file");
    check_report_equal(report, prior, "rename failure preserves report");

    remove_directory_if_present(path);
}

static std::size_t count_occurrences(
    const std::string& text, const char* needle)
{
    std::size_t count = 0;
    std::size_t position = 0;
    while ((position = text.find(needle, position)) != std::string::npos) {
        ++count;
        position += std::strlen(needle);
    }
    return count;
}

static std::size_t find_required(
    const std::string& text,
    const char* needle,
    std::size_t start,
    const char* message)
{
    const std::size_t position = text.find(needle, start);
    check(position != std::string::npos, message);
    return position;
}

static void test_controller_has_one_post_window_cleanup_path(
    const char* controller_path)
{
    std::ifstream input(controller_path);
    check(input.good(), "controller source opens");
    const std::string source(
        (std::istreambuf_iterator<char>(input)),
        std::istreambuf_iterator<char>());
    check(source.find("#include \"cleanup_runtime.h\"") !=
              std::string::npos,
          "controller includes cleanup report interface");

    const std::size_t window = find_required(
        source, "SetTargetFPS(25);", 0,
        "controller enters the post-window-success lifetime");
    const std::string post_window = source.substr(window);
    check(count_occurrences(post_window, "CloseWindow();") == 1,
          "post-window controller has exactly one window close");
    check(count_occurrences(
              post_window, "model_unloader(terrain_model);") == 1,
          "post-window controller unloads the active model once at the tail");
    check(post_window.find("UnloadModel(terrain_model)") ==
              std::string::npos,
          "post-window controller never bypasses the counted model unloader");
    check(post_window.find("return 2;") == std::string::npos,
          "post-window failures do not return before cleanup");
    check(post_window.find("_Exit(") == std::string::npos &&
              post_window.find("std::exit(") == std::string::npos,
          "post-window controller has no process-terminating shortcut");

    const std::size_t close_log = find_required(
        post_window, "deterministic_log.close(", 0,
        "normal tail closes deterministic log");
    const std::size_t unload_model = find_required(
        post_window, "model_unloader(terrain_model);", close_log,
        "normal tail unloads terrain model after log close");
    const std::size_t close_window = find_required(
        post_window, "CloseWindow();", unload_model,
        "normal tail closes window after model unload");
    const std::size_t write_cleanup = find_required(
        post_window, "cleanup_report_write(", close_window,
        "normal tail writes cleanup report after window close");
    static const char main_tail_text[] =
        "    normal_cleanup();\n"
        "    return controller_exit_code;\n"
        "}";
    const std::size_t main_tail = find_required(
        post_window, main_tail_text, write_cleanup,
        "normal cleanup and return form the actual controller-main tail");
    const std::size_t final_return = find_required(
        post_window, "return controller_exit_code;", main_tail,
        "normal tail returns only after cleanup reporting");
    const std::size_t main_end = main_tail + sizeof(main_tail_text) - 1U;
    check(close_log < unload_model && unload_model < close_window &&
              close_window < write_cleanup && write_cleanup < final_return,
          "normal cleanup stage order is fixed");
    const std::string post_window_main = post_window.substr(0, main_end);
    check(count_occurrences(
              post_window_main,
              "        normal_cleanup();\n"
              "        return controller_exit_code;") == 2 &&
              count_occurrences(
                  post_window_main, "normal_cleanup();") == 3 &&
              count_occurrences(
                  post_window_main,
                  "return controller_exit_code;") == 3,
          "two early failures and the normal tail each use authenticated cleanup returns");
    check(post_window_main.find("return ", final_return + 1) ==
              std::string::npos,
          "normal cleanup return is the final controller return");

    const std::size_t counter = find_required(
        post_window, "int model_load_count = 0;", 0,
        "model load counter starts before initial model allocation");
    const std::size_t loader_owner = find_required(
        post_window,
        "G1ModelLoader initial_model_loader{&model_load_count};",
        counter,
        "initial model loader exclusively owns the load counter");
    static const char initial_load_text[] =
        "const scene_model_load_result initial_model = initial_model_loader(\n"
        "        terrain_model,\n"
        "        active_scene.mesh_path.c_str(),\n"
        "        artifact_error,\n"
        "        static_cast<int>(sizeof(artifact_error)));";
    const std::size_t initial_load = find_required(
        post_window,
        initial_load_text,
        loader_owner,
        "initial terrain model uses the counted loader owner");
    static const char initial_failure_text[] =
        "if (!initial_model.ready || !initial_model.allocated) {\n"
        "        ::controlled_runtime_error(artifact_error);\n"
        "        controller_exit_code = 2;\n"
        "        normal_cleanup();\n"
        "        return controller_exit_code;\n"
        "    }";
    const std::size_t initial_failure = find_required(
        post_window,
        initial_failure_text,
        initial_load,
        "initial model readiness/allocation gate uses authenticated cleanup");
    const std::size_t loader_definition = find_required(
        source, "struct G1ModelLoader", 0,
        "controller defines the counted model loader");
    const std::size_t model_load = find_required(
        source, "output = ::LoadModel(path);", loader_definition,
        "counted loader performs the unique model load");
    const std::size_t count_load = find_required(
        source, "if (allocated) ++*load_count;", model_load,
        "counted loader increments only for allocated ownership");
    const std::size_t loader_definition_end = find_required(
        source, "struct G1ModelUnloader", count_load,
        "counted loader ends before the unloader owner");
    check(counter < loader_owner && loader_owner < initial_load &&
              initial_load < initial_failure && initial_failure < main_tail &&
              model_load < count_load && count_load < loader_definition_end &&
              count_occurrences(
                  source, "if (allocated) ++*load_count;") == 1,
          "initial model load is counted transactionally by one loader operator");

    for (const char* field : {
             "cleanup.exit_code = controller_exit_code;",
             "cleanup.motion_pack_load_count = motion_pack_load_count;",
             "cleanup.model_load_count = model_load_count;",
             "cleanup.model_unload_count = model_unload_count;",
             "cleanup.log_closed = log_closed;",
             "cleanup.window_closed = window_closed;"}) {
        check(post_window.find(field, close_window) != std::string::npos,
              "cleanup report consumes final controller state");
    }
    static const char cleanup_path_owner_text[] =
        "const char* cleanup_path = test_config.cleanup_log_path.empty()\n"
        "        ? nullptr\n"
        "        : test_config.cleanup_log_path.c_str();";
    const std::size_t cleanup_path_owner = find_required(
        post_window,
        cleanup_path_owner_text,
        0,
        "cleanup report path freezes from checked argument configuration");
    const std::size_t cleanup_path_use = find_required(
        post_window,
        "cleanup_report_write(\n"
        "                cleanup_path,",
        cleanup_path_owner,
        "cleanup report write uses the frozen path owner");
    check(cleanup_path_owner < close_log &&
              close_window < cleanup_path_use &&
              cleanup_path_use < final_return &&
              count_occurrences(post_window_main, "cleanup_path") == 2,
          "frozen cleanup path is declared once and consumed only by the report write");

    const std::size_t cleanup_lambda = find_required(
        post_window, "auto normal_cleanup =", 0,
        "controller defines one shared normal cleanup operation");
    static const char update_loop_text[] =
        "while (!::WindowShouldClose() && !controller_exit_requested) {";
    const std::size_t update_loop = find_required(
        post_window,
        update_loop_text,
        cleanup_lambda,
        "shared cleanup is defined before the ordinary update loop");
    check(cleanup_lambda < update_loop && update_loop < main_tail &&
              count_occurrences(
                  post_window_main, update_loop_text) == 1 &&
              count_occurrences(
                  post_window_main, "normal_cleanup();") == 3,
          "one ordinary update loop precedes the shared tail cleanup owned by two startup failures and the final return");
    check(post_window.find("if (cleanup_complete) return;", cleanup_lambda) !=
              std::string::npos &&
              post_window.find("cleanup_complete = true;", cleanup_lambda) !=
                  std::string::npos,
          "shared cleanup is idempotent");
}

int main(int argc, char** argv)
{
    test_null_path_is_noop_and_preserves_inputs();
    test_exact_json_replaces_final_atomically();
    test_invalid_inputs_are_transactional();
    test_error_truncation_is_exact_and_bounded();
    test_open_failure_preserves_existing_state();
    test_write_failure_removes_temporary_and_preserves_final();
    test_rename_failure_removes_temporary_and_preserves_destination();
    const char* controller_path = "controller.cpp";
    if (argc == 3 && std::strcmp(argv[1], "--controller") == 0) {
        controller_path = argv[2];
    } else {
        check(argc == 1,
              "usage: test_cleanup_runtime [--controller controller.cpp]");
    }
    test_controller_has_one_post_window_cleanup_path(controller_path);
    return 0;
}
