#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <condition_variable>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <cerrno>
#include <execinfo.h>
#include <fcntl.h>
#include <limits.h>
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <dxrt/dxrt_api.h>
#include <opencv2/opencv.hpp>

#ifdef USE_X11
#include <X11/Xlib.h>
#endif

#ifndef DEPTH_ROOT_DIR
#define DEPTH_ROOT_DIR "."
#endif

namespace
{

constexpr int kDefaultCameraIndex = 0;
constexpr int kDefaultCameraWidth = 640;
constexpr int kDefaultCameraHeight = 480;
constexpr int kDefaultCameraFps = 30;
constexpr int kMaxAsyncQueueSize = 4;
constexpr double kFpsWindowSec = 5.0;
constexpr double kFpsUpdateIntervalSec = 1.0;
const char *kWindowName = "Depth Anything v2 Demo";

enum CrashPhase
{
    kPhaseStartup = 0,
    kPhaseParseArgs,
    kPhaseCheckModel,
    kPhaseOpenCamera,
    kPhaseCreateEngine,
    kPhaseReadModelInputs,
    kPhaseRegisterCallback,
    kPhaseCaptureThread,
    kPhaseInferenceThread,
    kPhaseRunAsync,
    kPhaseInferenceCallback,
    kPhaseDisplayLoop,
    kPhaseShutdown
};

thread_local volatile std::sig_atomic_t g_crash_phase = kPhaseStartup;

const char *crash_phase_name(std::sig_atomic_t phase)
{
    switch (phase)
    {
    case kPhaseStartup:
        return "startup";
    case kPhaseParseArgs:
        return "argument parsing";
    case kPhaseCheckModel:
        return "model validation";
    case kPhaseOpenCamera:
        return "camera initialization";
    case kPhaseCreateEngine:
        return "DXRT InferenceEngine creation";
    case kPhaseReadModelInputs:
        return "DXRT model input inspection";
    case kPhaseRegisterCallback:
        return "DXRT callback registration";
    case kPhaseCaptureThread:
        return "camera capture thread";
    case kPhaseInferenceThread:
        return "inference thread";
    case kPhaseRunAsync:
        return "DXRT RunAsync";
    case kPhaseInferenceCallback:
        return "DXRT inference callback";
    case kPhaseDisplayLoop:
        return "display loop";
    case kPhaseShutdown:
        return "shutdown";
    default:
        return "unknown";
    }
}

void set_crash_phase(CrashPhase phase)
{
    g_crash_phase = phase;
}

class CrashPhaseScope
{
public:
    explicit CrashPhaseScope(CrashPhase phase)
        : previous_(g_crash_phase)
    {
        set_crash_phase(phase);
    }

    ~CrashPhaseScope()
    {
        g_crash_phase = previous_;
    }

private:
    std::sig_atomic_t previous_;
};

void write_signal_text(const char *text)
{
    if (text == nullptr)
    {
        return;
    }

    std::size_t len = 0;
    while (text[len] != '\0')
    {
        ++len;
    }
    const ssize_t written = ::write(STDERR_FILENO, text, len);
    (void)written;
}

const char *signal_name(int signal)
{
    switch (signal)
    {
    case SIGSEGV:
        return "SIGSEGV";
    case SIGABRT:
        return "SIGABRT";
    case SIGBUS:
        return "SIGBUS";
    case SIGILL:
        return "SIGILL";
    case SIGFPE:
        return "SIGFPE";
    default:
        return "signal";
    }
}

void crash_signal_handler(int signal)
{
    write_signal_text("\nFatal native crash: ");
    write_signal_text(signal_name(signal));
    write_signal_text("\nCurrent phase: ");
    write_signal_text(crash_phase_name(g_crash_phase));
    write_signal_text("\nBacktrace:\n");

    void *frames[64];
    const int frame_count = ::backtrace(frames, 64);
    if (frame_count > 0)
    {
        ::backtrace_symbols_fd(frames, frame_count, STDERR_FILENO);
    }

    ::_exit(128 + signal);
}

void install_crash_handlers()
{
    struct sigaction action {};
    action.sa_handler = crash_signal_handler;
    sigemptyset(&action.sa_mask);
    action.sa_flags = SA_RESETHAND | SA_NODEFER;

    sigaction(SIGSEGV, &action, nullptr);
    sigaction(SIGABRT, &action, nullptr);
    sigaction(SIGBUS, &action, nullptr);
    sigaction(SIGILL, &action, nullptr);
    sigaction(SIGFPE, &action, nullptr);

    void *frames[1];
    (void)::backtrace(frames, 1);
}

bool file_exists(const std::string &path)
{
    std::ifstream in(path.c_str(), std::ios::binary);
    return static_cast<bool>(in);
}

long long file_size_bytes(const std::string &path)
{
    struct stat st {};
    if (::stat(path.c_str(), &st) != 0)
    {
        return -1;
    }
    return static_cast<long long>(st.st_size);
}

std::string current_working_directory()
{
    char cwd[PATH_MAX];
    if (::getcwd(cwd, sizeof(cwd)) == nullptr)
    {
        return "<unknown>";
    }
    return cwd;
}

std::string resolved_path(const std::string &path)
{
    char resolved[PATH_MAX];
    if (::realpath(path.c_str(), resolved) == nullptr)
    {
        return std::string();
    }
    return resolved;
}

std::string camera_device_path(int camera_index)
{
    return "/dev/video" + std::to_string(camera_index);
}

bool get_v4l2_control(int camera_index, uint32_t control_id, const char *name, int *value)
{
    if (value == nullptr)
    {
        return false;
    }

    const std::string path = camera_device_path(camera_index);
    const int fd = ::open(path.c_str(), O_RDWR | O_NONBLOCK);
    if (fd < 0)
    {
        std::cerr << "Warning: could not open " << path << " to read " << name
                  << ": " << std::strerror(errno) << std::endl;
        return false;
    }

    struct v4l2_control control {};
    control.id = control_id;
    const bool ok = (::ioctl(fd, VIDIOC_G_CTRL, &control) == 0);
    if (!ok)
    {
        std::cerr << "Warning: could not read camera control " << name
                  << " from " << path << ": " << std::strerror(errno) << std::endl;
    }
    else
    {
        *value = control.value;
    }

    ::close(fd);
    return ok;
}

bool set_v4l2_control(int camera_index, uint32_t control_id, const char *name, int value)
{
    const std::string path = camera_device_path(camera_index);
    const int fd = ::open(path.c_str(), O_RDWR | O_NONBLOCK);
    if (fd < 0)
    {
        std::cerr << "Warning: could not open " << path << " to set " << name
                  << ": " << std::strerror(errno) << std::endl;
        return false;
    }

    struct v4l2_control control {};
    control.id = control_id;
    control.value = value;
    const bool ok = (::ioctl(fd, VIDIOC_S_CTRL, &control) == 0);
    if (!ok)
    {
        std::cerr << "Warning: could not set camera control " << name
                  << "=" << value << " on " << path << ": " << std::strerror(errno) << std::endl;
    }
    else
    {
        std::cout << "Camera control: " << name << "=" << value << " on " << path << std::endl;
    }

    ::close(fd);
    return ok;
}

void configure_camera_controls(int camera_index, bool disable_dynamic_framerate)
{
    if (!disable_dynamic_framerate)
    {
        std::cout << "Camera dynamic framerate control: leaving device default enabled/unchanged" << std::endl;
        return;
    }

    set_v4l2_control(camera_index,
                     V4L2_CID_EXPOSURE_AUTO_PRIORITY,
                     "exposure_dynamic_framerate",
                     0);

    int value = 0;
    if (get_v4l2_control(camera_index,
                         V4L2_CID_EXPOSURE_AUTO_PRIORITY,
                         "exposure_dynamic_framerate",
                         &value))
    {
        std::cout << "Camera control actual: exposure_dynamic_framerate=" << value << std::endl;
    }
}

std::string default_model_path()
{
    const std::string root = DEPTH_ROOT_DIR;
    const std::vector<std::string> candidates = {
        root + "/models/depth_anything_v2.dxnn",
        root + "/depth_anything_v2_vits_294x518.dxnn",
        root + "/depth_anything_v2_vits_294x518_sim_aggsv.dxnn",
        root + "/depth_anything_v2_vits_224x224.dxnn",
        root + "/depth_anything_v2_vitb.dxnn",
    };
    for (const std::string &candidate : candidates)
    {
        if (file_exists(candidate))
        {
            return candidate;
        }
    }
    return candidates.front();
}

struct Options
{
    std::string model_path = default_model_path();
    int camera_index = kDefaultCameraIndex;
    int width = kDefaultCameraWidth;
    int height = kDefaultCameraHeight;
    int fps = kDefaultCameraFps;
    int queue_size = kMaxAsyncQueueSize;
    int camera_buffer_size = 0;
    std::string camera_backend = "any";
    std::string camera_fourcc;
    bool side_by_side = false;
    bool grayscale = false;
    bool camera_only = false;
    bool disable_dynamic_framerate = false;
    cv::Scalar margin_bgr = cv::Scalar(0, 0, 0);
};

void print_startup_diagnostics(const Options &options)
{
    std::cout << "Working directory: " << current_working_directory() << std::endl;
    std::cout << "Model path: " << options.model_path << std::endl;

    const std::string absolute_model_path = resolved_path(options.model_path);
    if (!absolute_model_path.empty())
    {
        std::cout << "Resolved model path: " << absolute_model_path << std::endl;
    }

    const long long model_size = file_size_bytes(options.model_path);
    if (model_size >= 0)
    {
        std::cout << "Model file size: " << model_size << " bytes" << std::endl;
    }
}

struct CameraPacket
{
    cv::Mat frame;
    uint64_t seq = 0;
};

struct DepthResult
{
    uint64_t frame_id = 0;
    cv::Mat original_bgr;
    cv::Mat depth;
    double latency_ms = 0.0;
};

std::mutex g_camera_mutex;
std::condition_variable g_camera_cv;
CameraPacket g_latest_camera;
std::deque<std::chrono::steady_clock::time_point> g_camera_timestamps;

std::atomic<uint64_t> g_camera_frames(0);
std::atomic<uint64_t> g_submitted_frames(0);
std::atomic<uint64_t> g_skipped_frames(0);
std::atomic<uint64_t> g_completed_frames(0);
std::atomic<uint64_t> g_displayed_frames(0);

struct TimingCounter
{
    std::atomic<uint64_t> count{0};
    std::atomic<uint64_t> total_us{0};
};

struct TimingSnapshot
{
    uint64_t count = 0;
    uint64_t total_us = 0;
};

struct PipelineTimingStats
{
    TimingCounter camera_read;
    TimingCounter camera_grab;
    TimingCounter camera_retrieve;
    TimingCounter camera_interval;
    TimingCounter camera_loop_gap;
    TimingCounter camera_store;
    TimingCounter camera_lock_wait;
    TimingCounter inference_wait;
    TimingCounter inference_copy;
    TimingCounter submit_total;
    TimingCounter preprocess;
    TimingCounter run_async;
    TimingCounter callback_total;
    TimingCounter callback_tensor;
    TimingCounter display_total;
    TimingCounter display_postprocess;
    TimingCounter display_show;
    TimingCounter wait_key;
};

struct PipelineTimingSnapshot
{
    TimingSnapshot camera_read;
    TimingSnapshot camera_grab;
    TimingSnapshot camera_retrieve;
    TimingSnapshot camera_interval;
    TimingSnapshot camera_loop_gap;
    TimingSnapshot camera_store;
    TimingSnapshot camera_lock_wait;
    TimingSnapshot inference_wait;
    TimingSnapshot inference_copy;
    TimingSnapshot submit_total;
    TimingSnapshot preprocess;
    TimingSnapshot run_async;
    TimingSnapshot callback_total;
    TimingSnapshot callback_tensor;
    TimingSnapshot display_total;
    TimingSnapshot display_postprocess;
    TimingSnapshot display_show;
    TimingSnapshot wait_key;
};

PipelineTimingStats g_timing;

void record_timing(TimingCounter &counter,
                   const std::chrono::steady_clock::time_point &start,
                   const std::chrono::steady_clock::time_point &end = std::chrono::steady_clock::now())
{
    const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    if (elapsed < 0)
    {
        return;
    }

    counter.count.fetch_add(1, std::memory_order_relaxed);
    counter.total_us.fetch_add(static_cast<uint64_t>(elapsed), std::memory_order_relaxed);
}

void record_timing_us(TimingCounter &counter, uint64_t elapsed_us)
{
    counter.count.fetch_add(1, std::memory_order_relaxed);
    counter.total_us.fetch_add(elapsed_us, std::memory_order_relaxed);
}

double average_ms_since_last(const TimingCounter &counter, TimingSnapshot *snapshot)
{
    const uint64_t count = counter.count.load(std::memory_order_relaxed);
    const uint64_t total_us = counter.total_us.load(std::memory_order_relaxed);
    const uint64_t delta_count = count - snapshot->count;
    const uint64_t delta_total_us = total_us - snapshot->total_us;

    snapshot->count = count;
    snapshot->total_us = total_us;

    if (delta_count == 0)
    {
        return 0.0;
    }
    return static_cast<double>(delta_total_us) / static_cast<double>(delta_count) / 1000.0;
}

double rate_since_last(uint64_t current, uint64_t *last, double elapsed_sec)
{
    const uint64_t delta = current - *last;
    *last = current;
    if (elapsed_sec <= 0.0)
    {
        return 0.0;
    }
    return static_cast<double>(delta) / elapsed_sec;
}

class ScopedTimer
{
public:
    explicit ScopedTimer(TimingCounter &counter)
        : counter_(counter), start_(std::chrono::steady_clock::now())
    {
    }

    ~ScopedTimer()
    {
        record_timing(counter_, start_);
    }

private:
    TimingCounter &counter_;
    std::chrono::steady_clock::time_point start_;
};

std::string basename_of(const std::string &path)
{
    const std::size_t slash = path.find_last_of("/\\");
    return slash == std::string::npos ? path : path.substr(slash + 1);
}

void ensure_parent_dirs(const std::string &full_path)
{
    const std::size_t slash = full_path.find_last_of('/');
    if (slash == std::string::npos)
    {
        return;
    }

    const std::string dir = full_path.substr(0, slash);
    std::string partial;
    for (char ch : dir)
    {
        partial.push_back(ch);
        if (ch == '/' && partial.size() > 1)
        {
            if (::mkdir(partial.c_str(), 0755) != 0 && errno != EEXIST)
            {
                return;
            }
        }
    }
    if (!partial.empty())
    {
        (void)::mkdir(partial.c_str(), 0755);
    }
}

void notify_launcher_ready()
{
    const char *path = std::getenv("DX_LAUNCHER_READY_FILE");
    if (path == nullptr || *path == '\0')
    {
        return;
    }

    const std::string full_path(path);
    ensure_parent_dirs(full_path);
    std::ofstream out(full_path.c_str(), std::ios::trunc);
    if (out)
    {
        out << "ready\n";
    }
}

bool parse_int_value(const std::string &value, int *out)
{
    if (out == nullptr)
    {
        return false;
    }
    try
    {
        std::size_t used = 0;
        const int parsed = std::stoi(value, &used);
        if (used != value.size())
        {
            return false;
        }
        *out = parsed;
        return true;
    }
    catch (...)
    {
        return false;
    }
}

bool parse_bg_color_rgb(const std::string &value, cv::Scalar *margin_bgr)
{
    std::vector<int> parts;
    std::size_t start = 0;
    while (start <= value.size())
    {
        const std::size_t comma = value.find(',', start);
        const std::string token = value.substr(
            start, comma == std::string::npos ? std::string::npos : comma - start);
        int parsed = 0;
        if (!parse_int_value(token, &parsed))
        {
            return false;
        }
        parts.push_back(parsed);
        if (comma == std::string::npos)
        {
            break;
        }
        start = comma + 1;
    }

    if (parts.size() != 3)
    {
        return false;
    }
    for (int v : parts)
    {
        if (v < 0 || v > 255)
        {
            return false;
        }
    }
    *margin_bgr = cv::Scalar(parts[2], parts[1], parts[0]);
    return true;
}

void print_usage(const char *argv0)
{
    std::cout
        << "Usage: " << argv0 << " [OPTIONS]\n"
        << "  -m, --model <PATH>          Path to .dxnn model\n"
        << "  -s, --side                  Show original and depth map side by side\n"
        << "  -g, --grayscale             Use grayscale depth map\n"
        << "      --camera-index <N>      Camera index (default: 0)\n"
        << "      --width <N>             Camera width (default: 640)\n"
        << "      --height <N>            Camera height (default: 480)\n"
        << "      --fps <N>               Camera FPS request (default: 30)\n"
        << "      --queue-size <N>        Async in-flight queue size, 1..4 (default: 4)\n"
        << "      --backend any|v4l2      OpenCV camera backend (default: any)\n"
        << "      --camera-buffer-size <N>\n"
        << "                              Set OpenCV camera buffer size when N > 0\n"
        << "      --camera-fourcc CODE    Request camera pixel format, e.g. MJPG or YUYV\n"
        << "      --camera-only           Benchmark camera capture without NPU/display\n"
        << "      --disable-dynamic-framerate\n"
        << "                              Set camera exposure dynamic framerate control to 0\n"
        << "      --bg-color R,G,B        Fullscreen letterbox margin color (default: 0,0,0)\n"
        << "  -h, --help                  Show this help\n";
}

bool parse_args(int argc, char **argv, Options *options)
{
    for (int i = 1; i < argc;)
    {
        const std::string arg(argv[i++]);
        auto require_value = [&](const char *name) -> const char * {
            if (i >= argc)
            {
                std::cerr << "Error: missing value for " << name << std::endl;
                return nullptr;
            }
            return argv[i++];
        };

        if (arg == "-m" || arg == "--model")
        {
            const char *value = require_value(arg.c_str());
            if (value == nullptr)
            {
                return false;
            }
            options->model_path = value;
        }
        else if (arg == "-s" || arg == "--side")
        {
            options->side_by_side = true;
        }
        else if (arg == "-g" || arg == "--grayscale")
        {
            options->grayscale = true;
        }
        else if (arg == "--camera-index")
        {
            const char *value = require_value(arg.c_str());
            if (value == nullptr || !parse_int_value(value, &options->camera_index) || options->camera_index < 0)
            {
                std::cerr << "Error: --camera-index expects a non-negative integer" << std::endl;
                return false;
            }
        }
        else if (arg == "--width")
        {
            const char *value = require_value(arg.c_str());
            if (value == nullptr || !parse_int_value(value, &options->width) || options->width <= 0)
            {
                std::cerr << "Error: --width expects a positive integer" << std::endl;
                return false;
            }
        }
        else if (arg == "--height")
        {
            const char *value = require_value(arg.c_str());
            if (value == nullptr || !parse_int_value(value, &options->height) || options->height <= 0)
            {
                std::cerr << "Error: --height expects a positive integer" << std::endl;
                return false;
            }
        }
        else if (arg == "--fps")
        {
            const char *value = require_value(arg.c_str());
            if (value == nullptr || !parse_int_value(value, &options->fps) || options->fps <= 0)
            {
                std::cerr << "Error: --fps expects a positive integer" << std::endl;
                return false;
            }
        }
        else if (arg == "--queue-size")
        {
            const char *value = require_value(arg.c_str());
            if (value == nullptr || !parse_int_value(value, &options->queue_size) ||
                options->queue_size < 1 || options->queue_size > kMaxAsyncQueueSize)
            {
                std::cerr << "Error: --queue-size expects an integer from 1 to 4" << std::endl;
                return false;
            }
        }
        else if (arg == "--backend")
        {
            const char *value = require_value(arg.c_str());
            if (value == nullptr)
            {
                return false;
            }
            options->camera_backend = value;
            if (options->camera_backend != "any" && options->camera_backend != "v4l2")
            {
                std::cerr << "Error: --backend expects any or v4l2" << std::endl;
                return false;
            }
        }
        else if (arg == "--camera-buffer-size")
        {
            const char *value = require_value(arg.c_str());
            if (value == nullptr || !parse_int_value(value, &options->camera_buffer_size) ||
                options->camera_buffer_size < 0)
            {
                std::cerr << "Error: --camera-buffer-size expects a non-negative integer" << std::endl;
                return false;
            }
        }
        else if (arg == "--camera-fourcc")
        {
            const char *value = require_value(arg.c_str());
            if (value == nullptr)
            {
                return false;
            }
            options->camera_fourcc = value;
            if (options->camera_fourcc.size() != 4)
            {
                std::cerr << "Error: --camera-fourcc expects a 4-character code" << std::endl;
                return false;
            }
        }
        else if (arg == "--camera-only")
        {
            options->camera_only = true;
        }
        else if (arg == "--allow-dynamic-framerate")
        {
            options->disable_dynamic_framerate = false;
        }
        else if (arg == "--disable-dynamic-framerate")
        {
            options->disable_dynamic_framerate = true;
        }
        else if (arg == "--bg-color")
        {
            const char *value = require_value(arg.c_str());
            if (value == nullptr || !parse_bg_color_rgb(value, &options->margin_bgr))
            {
                std::cerr << "Error: --bg-color expects R,G,B values in 0-255" << std::endl;
                return false;
            }
        }
        else if (arg == "-h" || arg == "--help")
        {
            print_usage(argv[0]);
            std::exit(0);
        }
        else
        {
            std::cerr << "Error: unknown argument " << arg << std::endl;
            return false;
        }
    }
    return true;
}

int fourcc_from_string(const std::string &code)
{
    if (code.size() != 4)
    {
        return 0;
    }
    return cv::VideoWriter::fourcc(code[0], code[1], code[2], code[3]);
}

int camera_backend_api(const std::string &backend)
{
    if (backend == "v4l2")
    {
        return cv::CAP_V4L2;
    }
    return cv::CAP_ANY;
}

std::string fourcc_to_string(double value)
{
    const int fourcc = static_cast<int>(value);
    std::string code(4, ' ');
    code[0] = static_cast<char>(fourcc & 0xff);
    code[1] = static_cast<char>((fourcc >> 8) & 0xff);
    code[2] = static_cast<char>((fourcc >> 16) & 0xff);
    code[3] = static_cast<char>((fourcc >> 24) & 0xff);
    for (char &ch : code)
    {
        if (ch < 32 || ch > 126)
        {
            ch = '?';
        }
    }
    return code;
}

std::pair<int, int> screen_size()
{
#ifdef USE_X11
    Display *display = XOpenDisplay(nullptr);
    if (display != nullptr)
    {
        const int screen = DefaultScreen(display);
        const int w = DisplayWidth(display, screen);
        const int h = DisplayHeight(display, screen);
        XCloseDisplay(display);
        if (w > 0 && h > 0)
        {
            return std::make_pair(w, h);
        }
    }
#endif
    return std::make_pair(1920, 1080);
}

cv::Mat letterbox_to_screen(const cv::Mat &img, int screen_w, int screen_h, const cv::Scalar &bg_bgr)
{
    cv::Mat canvas(screen_h, screen_w, CV_8UC3, bg_bgr);
    if (img.empty())
    {
        return canvas;
    }

    const double scale = std::min(static_cast<double>(screen_w) / img.cols,
                                  static_cast<double>(screen_h) / img.rows);
    const int new_w = std::max(1, static_cast<int>(std::round(img.cols * scale)));
    const int new_h = std::max(1, static_cast<int>(std::round(img.rows * scale)));

    cv::Mat resized;
    cv::resize(img, resized, cv::Size(new_w, new_h), 0.0, 0.0, cv::INTER_LINEAR);
    const int x0 = (screen_w - new_w) / 2;
    const int y0 = (screen_h - new_h) / 2;
    resized.copyTo(canvas(cv::Rect(x0, y0, new_w, new_h)));
    return canvas;
}

void draw_fps_overlay(cv::Mat &bgr, double fps)
{
    char text[64];
    std::snprintf(text, sizeof(text), "%.1f FPS", fps);
    const int font = cv::FONT_HERSHEY_DUPLEX;
    const double scale = 1.08;
    const int thick = 3;
    const int pad_x = 24;
    const int pad_y = 17;
    const int margin = 15;

    int baseline = 0;
    const cv::Size text_size = cv::getTextSize(text, font, scale, thick, &baseline);
    const int box_w = text_size.width + pad_x * 2;
    const int box_h = text_size.height + baseline + pad_y * 2;
    const int x0 = std::min(margin, std::max(0, bgr.cols - box_w - margin));
    const int y0 = std::min(margin, std::max(0, bgr.rows - box_h - margin));
    const int x1 = std::min(x0 + box_w, bgr.cols);
    const int y1 = std::min(y0 + box_h, bgr.rows);
    if (x1 <= x0 + 4 || y1 <= y0 + 4)
    {
        return;
    }

    cv::Mat roi = bgr(cv::Rect(x0, y0, x1 - x0, y1 - y0));
    cv::Mat panel(roi.size(), roi.type(), cv::Scalar(38, 40, 44));
    cv::addWeighted(roi, 0.52, panel, 0.48, 0.0, roi);

    const int stripe_w = std::min(6, x1 - x0);
    if (stripe_w > 0)
    {
        cv::Mat stripe = bgr(cv::Rect(x0, y0, stripe_w, y1 - y0));
        cv::Mat accent(stripe.size(), stripe.type(), cv::Scalar(92, 168, 255));
        cv::addWeighted(stripe, 0.35, accent, 0.65, 0.0, stripe);
    }

    cv::rectangle(bgr, cv::Rect(x0, y0, x1 - x0, y1 - y0), cv::Scalar(96, 102, 110), 1, cv::LINE_AA);
    const int tx = x0 + pad_x;
    const int ty = y0 + pad_y + text_size.height;
    const cv::Point offsets[] = {
        {-1, -1}, {-1, 1}, {1, -1}, {1, 1}, {0, -1}, {0, 1}, {-1, 0}, {1, 0}};
    for (const cv::Point &offset : offsets)
    {
        cv::putText(bgr, text, cv::Point(tx, ty) + offset, font, scale, cv::Scalar(0, 0, 0), thick + 1, cv::LINE_AA);
    }
    cv::putText(bgr, text, cv::Point(tx, ty), font, scale, cv::Scalar(236, 240, 245), thick, cv::LINE_AA);
}

void draw_model_name_overlay(cv::Mat &bgr, const std::string &model_path)
{
    const std::string text = "Model: " + basename_of(model_path);
    const int font = cv::FONT_HERSHEY_DUPLEX;
    const double scale = 0.93;
    const int thick = 2;
    const int pad_x = 21;
    const int pad_y = 14;
    const int margin = 12;

    int baseline = 0;
    const cv::Size text_size = cv::getTextSize(text, font, scale, thick, &baseline);
    const int box_w = text_size.width + pad_x * 2;
    const int box_h = text_size.height + baseline + pad_y * 2;
    const int x0 = std::max(margin, (bgr.cols - box_w) / 2);
    const int y0 = margin;
    const int x1 = std::min(x0 + box_w, bgr.cols);
    const int y1 = std::min(y0 + box_h, bgr.rows);
    if (x1 <= x0 + 4 || y1 <= y0 + 4)
    {
        return;
    }

    cv::Mat roi = bgr(cv::Rect(x0, y0, x1 - x0, y1 - y0));
    cv::Mat panel(roi.size(), roi.type(), cv::Scalar(32, 34, 38));
    cv::addWeighted(roi, 0.45, panel, 0.55, 0.0, roi);
    cv::rectangle(bgr, cv::Rect(x0, y0, x1 - x0, y1 - y0), cv::Scalar(80, 86, 94), 1, cv::LINE_AA);

    const int tx = x0 + pad_x;
    const int ty = y0 + pad_y + text_size.height;
    const cv::Point offsets[] = {{-1, -1}, {-1, 1}, {1, -1}, {1, 1}, {0, -1}, {0, 1}};
    for (const cv::Point &offset : offsets)
    {
        cv::putText(bgr, text, cv::Point(tx, ty) + offset, font, scale, cv::Scalar(0, 0, 0), thick + 1, cv::LINE_AA);
    }
    cv::putText(bgr, text, cv::Point(tx, ty), font, scale, cv::Scalar(220, 224, 230), thick, cv::LINE_AA);
}

void draw_frame_id_overlay(cv::Mat &bgr, uint64_t frame_id, double latency_ms)
{
    std::ostringstream oss;
    oss << "Frame ID: " << frame_id << "  Latency: " << std::fixed << std::setprecision(1) << latency_ms << " ms";
    const std::string text = oss.str();
    const int font = cv::FONT_HERSHEY_DUPLEX;
    const double scale = 0.72;
    const int thick = 2;
    const int pad_x = 16;
    const int pad_y = 11;
    const int margin = 15;

    int baseline = 0;
    const cv::Size text_size = cv::getTextSize(text, font, scale, thick, &baseline);
    const int box_w = text_size.width + pad_x * 2;
    const int box_h = text_size.height + baseline + pad_y * 2;
    const int x0 = std::min(margin, std::max(0, bgr.cols - box_w - margin));
    const int y0 = std::max(margin, bgr.rows - box_h - margin);
    const int x1 = std::min(x0 + box_w, bgr.cols);
    const int y1 = std::min(y0 + box_h, bgr.rows);
    if (x1 <= x0 + 4 || y1 <= y0 + 4)
    {
        return;
    }

    cv::Mat roi = bgr(cv::Rect(x0, y0, x1 - x0, y1 - y0));
    cv::Mat panel(roi.size(), roi.type(), cv::Scalar(28, 30, 34));
    cv::addWeighted(roi, 0.50, panel, 0.50, 0.0, roi);
    cv::rectangle(bgr, cv::Rect(x0, y0, x1 - x0, y1 - y0), cv::Scalar(75, 82, 92), 1, cv::LINE_AA);

    const int tx = x0 + pad_x;
    const int ty = y0 + pad_y + text_size.height;
    for (const cv::Point &offset : std::vector<cv::Point>{{-1, -1}, {-1, 1}, {1, -1}, {1, 1}})
    {
        cv::putText(bgr, text, cv::Point(tx, ty) + offset, font, scale, cv::Scalar(0, 0, 0), thick + 1, cv::LINE_AA);
    }
    cv::putText(bgr, text, cv::Point(tx, ty), font, scale, cv::Scalar(222, 227, 234), thick, cv::LINE_AA);
}

int64_t shape_dim(const std::vector<int64_t> &shape, std::size_t index, int64_t fallback)
{
    if (index >= shape.size() || shape[index] <= 0)
    {
        return fallback;
    }
    return shape[index];
}

std::size_t shape_element_count(const std::vector<int64_t> &shape)
{
    std::size_t count = 1;
    for (int64_t dim : shape)
    {
        if (dim > 0)
        {
            count *= static_cast<std::size_t>(dim);
        }
    }
    return count;
}

class AsyncDepthAnything
{
public:
    AsyncDepthAnything(const std::string &model_path, int queue_size)
        : queue_size_(queue_size)
    {
        std::cout << "Initializing DX Engine (Async Mode) with Depth Anything..." << std::endl;

        {
            CrashPhaseScope phase(kPhaseCreateEngine);
            std::cout << "Creating DXRT InferenceEngine..." << std::endl;
            try
            {
                dxrt::InferenceOption option;
                engine_.reset(new dxrt::InferenceEngine(model_path, option));
            }
            catch (const std::exception &e)
            {
                throw std::runtime_error(std::string("failed to create DXRT InferenceEngine: ") + e.what());
            }
            catch (...)
            {
                throw std::runtime_error("failed to create DXRT InferenceEngine: unknown exception");
            }

            if (!engine_)
            {
                throw std::runtime_error("failed to create DXRT InferenceEngine: engine is null");
            }
            std::cout << "DXRT InferenceEngine created." << std::endl;
        }

        dxrt::Tensors inputs;
        {
            CrashPhaseScope phase(kPhaseReadModelInputs);
            try
            {
                inputs = engine_->GetInputs();
            }
            catch (const std::exception &e)
            {
                throw std::runtime_error(std::string("failed to inspect model input tensors: ") + e.what());
            }
            catch (...)
            {
                throw std::runtime_error("failed to inspect model input tensors: unknown exception");
            }
        }
        if (inputs.empty())
        {
            throw std::runtime_error("model has no input tensor");
        }

        input_shape_ = inputs.front().shape();
        input_type_ = inputs.front().type();
        if (input_type_ != dxrt::DataType::FLOAT)
        {
            throw std::runtime_error("only FLOAT input models are supported by this demo");
        }

        analyze_input_layout();
        input_element_count_ = shape_element_count(input_shape_);
        if (input_element_count_ < static_cast<std::size_t>(net_w_ * net_h_ * 3))
        {
            throw std::runtime_error("model input tensor is smaller than expected RGB image size");
        }

        {
            std::lock_guard<std::mutex> lock(slots_mutex_);
            slots_.resize(static_cast<std::size_t>(queue_size_));
            for (int i = 0; i < queue_size_; ++i)
            {
                slots_[static_cast<std::size_t>(i)].input.assign(input_element_count_, 0.0f);
                free_slots_.push(i);
            }
        }

        {
            CrashPhaseScope phase(kPhaseRegisterCallback);
            engine_->RegisterCallback([this](dxrt::TensorPtrs &outputs, void *user_arg) -> int {
                return on_complete(outputs, user_arg);
            });
        }

        std::cout << "Model Input Size: " << net_w_ << "x" << net_h_ << std::endl;
        std::cout << "Async queue size: " << queue_size_ << " (max " << kMaxAsyncQueueSize << ")" << std::endl;
    }

    ~AsyncDepthAnything()
    {
        wait_all();
    }

    bool try_submit(const cv::Mat &frame_bgr)
    {
        if (!engine_)
        {
            throw std::runtime_error("DXRT engine is not initialized");
        }

        int slot = -1;
        {
            std::lock_guard<std::mutex> lock(slots_mutex_);
            if (free_slots_.empty())
            {
                return false;
            }
            slot = free_slots_.front();
            free_slots_.pop();
            ++in_flight_;
        }

        try
        {
            ScopedTimer submit_timer(g_timing.submit_total);

            {
                ScopedTimer preprocess_timer(g_timing.preprocess);
                preprocess(frame_bgr, slots_[static_cast<std::size_t>(slot)].input);
            }

            std::unique_ptr<JobContext> job(new JobContext);
            job->owner = this;
            job->slot_index = slot;
            job->frame_id = next_submit_id_++;
            job->original_bgr = frame_bgr.clone();
            job->submit_ts = std::chrono::steady_clock::now();

            void *input_ptr = slots_[static_cast<std::size_t>(slot)].input.data();
            JobContext *raw_job = job.release();
            int job_id = -1;
            try
            {
                CrashPhaseScope phase(kPhaseRunAsync);
                ScopedTimer run_async_timer(g_timing.run_async);
                job_id = engine_->RunAsync(input_ptr, static_cast<void *>(raw_job));
            }
            catch (...)
            {
                delete raw_job;
                throw;
            }

            {
                std::lock_guard<std::mutex> lock(submit_mutex_);
                last_job_id_ = job_id;
            }
            g_submitted_frames.fetch_add(1, std::memory_order_relaxed);
            return true;
        }
        catch (...)
        {
            release_slot(slot);
            throw;
        }
    }

    bool pop_next_result(DepthResult *result)
    {
        if (result == nullptr)
        {
            return false;
        }

        std::lock_guard<std::mutex> lock(results_mutex_);
        auto it = pending_results_.find(next_display_id_);
        if (it == pending_results_.end())
        {
            return false;
        }

        *result = std::move(it->second);
        pending_results_.erase(it);
        ++next_display_id_;
        g_displayed_frames.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    double completion_fps()
    {
        const auto now = std::chrono::steady_clock::now();
        const auto cutoff = now - std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                                      std::chrono::duration<double>(kFpsWindowSec));
        std::lock_guard<std::mutex> lock(stats_mutex_);
        while (!completion_timestamps_.empty() && completion_timestamps_.front() < cutoff)
        {
            completion_timestamps_.pop_front();
        }
        return static_cast<double>(completion_timestamps_.size()) / kFpsWindowSec;
    }

    int in_flight() const
    {
        std::lock_guard<std::mutex> lock(slots_mutex_);
        return in_flight_;
    }

    uint64_t next_display_id() const
    {
        std::lock_guard<std::mutex> lock(results_mutex_);
        return next_display_id_;
    }

    void wait_all()
    {
        int job_id = -1;
        {
            std::lock_guard<std::mutex> lock(submit_mutex_);
            job_id = last_job_id_;
            last_job_id_ = -1;
        }
        if (job_id >= 0 && engine_)
        {
            engine_->Wait(job_id);
        }

        std::unique_lock<std::mutex> lock(slots_mutex_);
        slots_cv_.wait(lock, [this]() { return in_flight_ == 0; });
    }

private:
    struct Slot
    {
        std::vector<float> input;
    };

    struct JobContext
    {
        AsyncDepthAnything *owner = nullptr;
        int slot_index = -1;
        uint64_t frame_id = 0;
        cv::Mat original_bgr;
        std::chrono::steady_clock::time_point submit_ts;
    };

    int on_complete(dxrt::TensorPtrs &outputs, void *user_arg)
    {
        CrashPhaseScope phase(kPhaseInferenceCallback);
        ScopedTimer callback_timer(g_timing.callback_total);
        std::unique_ptr<JobContext> job(static_cast<JobContext *>(user_arg));
        if (!job)
        {
            return 0;
        }

        DepthResult result;
        result.frame_id = job->frame_id;
        result.original_bgr = job->original_bgr;
        result.latency_ms = std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - job->submit_ts)
                                .count();
        if (!outputs.empty() && outputs.front())
        {
            ScopedTimer tensor_timer(g_timing.callback_tensor);
            result.depth = tensor_to_depth_mat(outputs.front().get());
        }

        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            completion_timestamps_.push_back(std::chrono::steady_clock::now());
        }
        g_completed_frames.fetch_add(1, std::memory_order_relaxed);

        push_result(std::move(result));
        release_slot(job->slot_index);
        return 0;
    }

    void push_result(DepthResult &&result)
    {
        std::lock_guard<std::mutex> lock(results_mutex_);
        if (result.frame_id < next_display_id_)
        {
            return;
        }

        pending_results_[result.frame_id] = std::move(result);

        while (pending_results_.size() > static_cast<std::size_t>(queue_size_))
        {
            auto it = pending_results_.begin();
            if (it->first >= next_display_id_)
            {
                next_display_id_ = it->first + 1;
            }
            pending_results_.erase(it);
        }
    }

    void release_slot(int slot)
    {
        if (slot < 0)
        {
            return;
        }

        {
            std::lock_guard<std::mutex> lock(slots_mutex_);
            free_slots_.push(slot);
            --in_flight_;
        }
        slots_cv_.notify_all();
    }

    static cv::Mat tensor_to_depth_mat(dxrt::Tensor *tensor)
    {
        if (tensor == nullptr || tensor->data() == nullptr)
        {
            return cv::Mat();
        }

        const std::vector<int64_t> &shape = tensor->shape();
        if (shape.empty())
        {
            return cv::Mat();
        }

        int64_t h = 0;
        int64_t w = 0;
        if (shape.size() >= 4 && shape.back() == 1)
        {
            h = shape[shape.size() - 3];
            w = shape[shape.size() - 2];
        }
        else if (shape.size() >= 2)
        {
            h = shape[shape.size() - 2];
            w = shape[shape.size() - 1];
        }
        if (h <= 0 || w <= 0)
        {
            return cv::Mat();
        }

        const int64_t count = h * w;
        cv::Mat depth(static_cast<int>(h), static_cast<int>(w), CV_32F);
        float *dst = depth.ptr<float>();

        if (tensor->type() == dxrt::DataType::FLOAT)
        {
            const float *src = static_cast<const float *>(tensor->data());
            std::copy(src, src + count, dst);
        }
        else if (tensor->type() == dxrt::DataType::UINT16)
        {
            const uint16_t *src = static_cast<const uint16_t *>(tensor->data());
            for (int64_t i = 0; i < count; ++i)
            {
                dst[i] = static_cast<float>(src[i]);
            }
        }
        else if (tensor->type() == dxrt::DataType::UINT8)
        {
            const uint8_t *src = static_cast<const uint8_t *>(tensor->data());
            for (int64_t i = 0; i < count; ++i)
            {
                dst[i] = static_cast<float>(src[i]);
            }
        }
        else
        {
            return cv::Mat();
        }

        return depth;
    }

    void analyze_input_layout()
    {
        if (input_shape_.size() != 4)
        {
            throw std::runtime_error("expected a 4D input tensor");
        }

        if (shape_dim(input_shape_, 1, 0) == 3)
        {
            nchw_ = true;
            net_h_ = static_cast<int>(shape_dim(input_shape_, 2, 0));
            net_w_ = static_cast<int>(shape_dim(input_shape_, 3, 0));
        }
        else if (shape_dim(input_shape_, 3, 0) == 3)
        {
            nchw_ = false;
            net_h_ = static_cast<int>(shape_dim(input_shape_, 1, 0));
            net_w_ = static_cast<int>(shape_dim(input_shape_, 2, 0));
        }
        else
        {
            throw std::runtime_error("could not detect NCHW/NHWC input layout");
        }
        if (net_w_ <= 0 || net_h_ <= 0)
        {
            throw std::runtime_error("invalid model input size");
        }
    }

    void preprocess(const cv::Mat &frame_bgr, std::vector<float> &input) const
    {
        cv::Mat rgb;
        cv::cvtColor(frame_bgr, rgb, cv::COLOR_BGR2RGB);

        cv::Mat resized;
        cv::resize(rgb, resized, cv::Size(net_w_, net_h_), 0.0, 0.0, cv::INTER_LINEAR);

        cv::Mat f32;
        resized.convertTo(f32, CV_32FC3, 1.0 / 255.0);

        if (input.size() != input_element_count_)
        {
            input.assign(input_element_count_, 0.0f);
        }

        const float mean[3] = {0.485f, 0.456f, 0.406f};
        const float stddev[3] = {0.229f, 0.224f, 0.225f};

        if (nchw_)
        {
            const int plane = net_w_ * net_h_;
            for (int y = 0; y < net_h_; ++y)
            {
                const cv::Vec3f *row = f32.ptr<cv::Vec3f>(y);
                for (int x = 0; x < net_w_; ++x)
                {
                    for (int c = 0; c < 3; ++c)
                    {
                        input[static_cast<std::size_t>(c * plane + y * net_w_ + x)] =
                            (row[x][c] - mean[c]) / stddev[c];
                    }
                }
            }
        }
        else
        {
            std::size_t idx = 0;
            for (int y = 0; y < net_h_; ++y)
            {
                const cv::Vec3f *row = f32.ptr<cv::Vec3f>(y);
                for (int x = 0; x < net_w_; ++x)
                {
                    for (int c = 0; c < 3; ++c)
                    {
                        input[idx++] = (row[x][c] - mean[c]) / stddev[c];
                    }
                }
            }
        }
    }

    std::unique_ptr<dxrt::InferenceEngine> engine_;
    int queue_size_ = kMaxAsyncQueueSize;
    std::vector<Slot> slots_;
    std::queue<int> free_slots_;
    mutable std::mutex slots_mutex_;
    std::condition_variable slots_cv_;
    int in_flight_ = 0;

    std::mutex submit_mutex_;
    int last_job_id_ = -1;
    uint64_t next_submit_id_ = 0;

    mutable std::mutex results_mutex_;
    std::map<uint64_t, DepthResult> pending_results_;
    uint64_t next_display_id_ = 0;

    std::mutex stats_mutex_;
    std::deque<std::chrono::steady_clock::time_point> completion_timestamps_;

    std::vector<int64_t> input_shape_;
    dxrt::DataType input_type_ = dxrt::DataType::NONE_TYPE;
    std::size_t input_element_count_ = 0;
    int net_w_ = 0;
    int net_h_ = 0;
    bool nchw_ = true;
};

cv::Mat create_depth_map(const cv::Mat &depth, bool grayscale)
{
    if (depth.empty())
    {
        return cv::Mat();
    }

    double min_v = 0.0;
    double max_v = 0.0;
    cv::minMaxLoc(depth, &min_v, &max_v);

    cv::Mat normalized;
    if (max_v - min_v > 0.0)
    {
        depth.convertTo(normalized, CV_8U, 255.0 / (max_v - min_v), -255.0 * min_v / (max_v - min_v));
    }
    else
    {
        normalized = cv::Mat::zeros(depth.size(), CV_8U);
    }

    cv::Mat bgr;
    if (grayscale)
    {
        cv::cvtColor(normalized, bgr, cv::COLOR_GRAY2BGR);
    }
    else
    {
        cv::applyColorMap(normalized, bgr, cv::COLORMAP_TURBO);
    }
    return bgr;
}

double camera_fps()
{
    const auto now = std::chrono::steady_clock::now();
    const auto cutoff = now - std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                                  std::chrono::duration<double>(kFpsWindowSec));
    std::lock_guard<std::mutex> lock(g_camera_mutex);
    while (!g_camera_timestamps.empty() && g_camera_timestamps.front() < cutoff)
    {
        g_camera_timestamps.pop_front();
    }
    return static_cast<double>(g_camera_timestamps.size()) / kFpsWindowSec;
}

void set_fps_window_title(double fps)
{
    char title[128];
    std::snprintf(title, sizeof(title), "Depth Anything v2 Demo - %.1f FPS", fps);
    cv::setWindowTitle(kWindowName, title);
}

} // namespace

int main(int argc, char **argv)
{
    install_crash_handlers();
    set_crash_phase(kPhaseParseArgs);

    Options options;
    if (!parse_args(argc, argv, &options))
    {
        print_usage(argv[0]);
        return 1;
    }

    try
    {
        set_crash_phase(kPhaseCheckModel);
        if (!file_exists(options.model_path))
        {
            std::cerr << "Error: model file not found: " << options.model_path << std::endl;
            return 1;
        }
        print_startup_diagnostics(options);

        set_crash_phase(kPhaseOpenCamera);
        configure_camera_controls(options.camera_index, options.disable_dynamic_framerate);
        cv::VideoCapture cap(options.camera_index, camera_backend_api(options.camera_backend));
        if (options.camera_buffer_size > 0)
        {
            cap.set(cv::CAP_PROP_BUFFERSIZE, options.camera_buffer_size);
        }
        if (!options.camera_fourcc.empty())
        {
            cap.set(cv::CAP_PROP_FOURCC, fourcc_from_string(options.camera_fourcc));
        }
        cap.set(cv::CAP_PROP_FRAME_WIDTH, options.width);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, options.height);
        cap.set(cv::CAP_PROP_FPS, options.fps);

        if (!cap.isOpened())
        {
            std::cerr << "Error: Could not open camera index " << options.camera_index << std::endl;
            return 1;
        }

        if (options.camera_only)
        {
            std::cout << "Start Camera Capture Benchmark. Press Ctrl+C to exit." << std::endl;
            std::cout << "Camera request: /dev/video" << options.camera_index
                      << " " << options.width << "x" << options.height
                      << " @ " << options.fps << " FPS"
                      << ", backend=" << options.camera_backend << std::endl;
            std::cout << "Actual camera: "
                      << static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH)) << "x"
                      << static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT))
                      << " @ " << cap.get(cv::CAP_PROP_FPS) << " FPS"
                      << ", FOURCC=" << fourcc_to_string(cap.get(cv::CAP_PROP_FOURCC))
                      << ", backend_id=" << cap.get(cv::CAP_PROP_BACKEND)
                      << std::endl;

            PipelineTimingSnapshot timing_snapshot;
            uint64_t last_camera_frames = 0;
            auto last_stats_print = std::chrono::steady_clock::now();
            std::chrono::steady_clock::time_point last_frame_ts;
            std::chrono::steady_clock::time_point last_loop_end;

            while (true)
            {
                cv::Mat frame;
                const auto read_start = std::chrono::steady_clock::now();
                if (last_loop_end.time_since_epoch().count() != 0)
                {
                    record_timing(g_timing.camera_loop_gap, last_loop_end, read_start);
                }

                const auto grab_start = read_start;
                if (!cap.grab())
                {
                    std::cerr << "Error: Could not grab camera frame." << std::endl;
                    break;
                }
                const auto grab_end = std::chrono::steady_clock::now();
                record_timing(g_timing.camera_grab, grab_start, grab_end);

                const auto retrieve_start = grab_end;
                if (!cap.retrieve(frame))
                {
                    std::cerr << "Error: Could not retrieve camera frame." << std::endl;
                    break;
                }
                const auto retrieve_end = std::chrono::steady_clock::now();
                record_timing(g_timing.camera_retrieve, retrieve_start, retrieve_end);
                record_timing(g_timing.camera_read, read_start, retrieve_end);

                if (last_frame_ts.time_since_epoch().count() != 0)
                {
                    const auto interval_us = std::chrono::duration_cast<std::chrono::microseconds>(
                                                 retrieve_end - last_frame_ts)
                                                 .count();
                    if (interval_us >= 0)
                    {
                        record_timing_us(g_timing.camera_interval, static_cast<uint64_t>(interval_us));
                    }
                }
                last_frame_ts = retrieve_end;
                g_camera_frames.fetch_add(1, std::memory_order_relaxed);
                last_loop_end = std::chrono::steady_clock::now();

                const auto now = std::chrono::steady_clock::now();
                if (now - last_stats_print >= std::chrono::seconds(1))
                {
                    const double stats_elapsed_sec = std::chrono::duration<double>(now - last_stats_print).count();
                    last_stats_print = now;
                    const uint64_t camera_frames = g_camera_frames.load(std::memory_order_relaxed);
                    const double camera_rate = rate_since_last(camera_frames, &last_camera_frames, stats_elapsed_sec);

                    const double camera_read_ms = average_ms_since_last(g_timing.camera_read, &timing_snapshot.camera_read);
                    const double camera_grab_ms = average_ms_since_last(g_timing.camera_grab, &timing_snapshot.camera_grab);
                    const double camera_retrieve_ms = average_ms_since_last(g_timing.camera_retrieve, &timing_snapshot.camera_retrieve);
                    const double camera_interval_ms = average_ms_since_last(g_timing.camera_interval, &timing_snapshot.camera_interval);
                    const double camera_loop_gap_ms = average_ms_since_last(g_timing.camera_loop_gap, &timing_snapshot.camera_loop_gap);

                    std::cout << std::fixed << std::setprecision(2)
                              << "CameraOnly: frames=" << camera_frames
                              << " rate=" << camera_rate
                              << " read=" << camera_read_ms
                              << " grab=" << camera_grab_ms
                              << " retrieve=" << camera_retrieve_ms
                              << " interval=" << camera_interval_ms
                              << " loop_gap=" << camera_loop_gap_ms
                              << " frame_size=" << frame.cols << "x" << frame.rows
                              << std::endl;
                }
            }

            cap.release();
            return 0;
        }

        set_crash_phase(kPhaseCreateEngine);
        AsyncDepthAnything async_depth(options.model_path, options.queue_size);

        set_crash_phase(kPhaseDisplayLoop);
        const std::pair<int, int> screen = screen_size();
        double depth_fps_display = 0.0;
        double camera_fps_display = 0.0;
        auto last_fps_update = std::chrono::steady_clock::now();
        auto last_stats_print = std::chrono::steady_clock::now();
        PipelineTimingSnapshot timing_snapshot;
        uint64_t last_camera_frames = 0;
        uint64_t last_submitted_frames = 0;
        uint64_t last_skipped_frames = 0;
        uint64_t last_completed_frames = 0;
        uint64_t last_displayed_frames = 0;

        std::cout << "Start Async Processing. Press 'q' or ESC to exit." << std::endl;
        std::cout << "Camera request: /dev/video" << options.camera_index
                  << " " << options.width << "x" << options.height
                  << " @ " << options.fps << " FPS"
                  << ", backend=" << options.camera_backend << std::endl;
        std::cout << "Actual camera: "
                  << static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH)) << "x"
                  << static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT))
                  << " @ " << cap.get(cv::CAP_PROP_FPS) << " FPS"
                  << ", FOURCC=" << fourcc_to_string(cap.get(cv::CAP_PROP_FOURCC))
                  << ", backend_id=" << cap.get(cv::CAP_PROP_BACKEND)
                  << std::endl;
        std::cout << "Model: " << options.model_path << std::endl;

        cv::namedWindow(kWindowName, cv::WINDOW_NORMAL);

        bool window_fullscreen = false;
        bool launcher_ready = false;
        std::atomic<bool> stop_capture(false);
        std::exception_ptr capture_error = nullptr;
        std::exception_ptr inference_error = nullptr;

        std::thread capture_thread([&]() {
            set_crash_phase(kPhaseCaptureThread);
            try
            {
                uint64_t seq = 0;
                std::chrono::steady_clock::time_point last_frame_ts;
                std::chrono::steady_clock::time_point last_loop_end;
                while (!stop_capture.load(std::memory_order_relaxed))
                {
                    cv::Mat frame;
                    const auto read_start = std::chrono::steady_clock::now();
                    if (last_loop_end.time_since_epoch().count() != 0)
                    {
                        record_timing(g_timing.camera_loop_gap, last_loop_end, read_start);
                    }
                    const auto grab_start = read_start;
                    if (!cap.grab())
                    {
                        stop_capture.store(true, std::memory_order_relaxed);
                        g_camera_cv.notify_all();
                        break;
                    }
                    const auto grab_end = std::chrono::steady_clock::now();
                    record_timing(g_timing.camera_grab, grab_start, grab_end);

                    const auto retrieve_start = grab_end;
                    if (!cap.retrieve(frame))
                    {
                        stop_capture.store(true, std::memory_order_relaxed);
                        g_camera_cv.notify_all();
                        break;
                    }
                    const auto retrieve_end = std::chrono::steady_clock::now();
                    record_timing(g_timing.camera_retrieve, retrieve_start, retrieve_end);
                    record_timing(g_timing.camera_read, read_start, retrieve_end);

                    if (last_frame_ts.time_since_epoch().count() != 0)
                    {
                        const auto interval_us = std::chrono::duration_cast<std::chrono::microseconds>(
                                                     retrieve_end - last_frame_ts)
                                                     .count();
                        if (interval_us >= 0)
                        {
                            record_timing_us(g_timing.camera_interval, static_cast<uint64_t>(interval_us));
                        }
                    }
                    last_frame_ts = retrieve_end;

                    {
                        ScopedTimer store_timer(g_timing.camera_store);
                        const auto lock_wait_start = std::chrono::steady_clock::now();
                        std::unique_lock<std::mutex> lock(g_camera_mutex);
                        record_timing(g_timing.camera_lock_wait, lock_wait_start);
                        g_latest_camera.frame = frame.clone();
                        g_latest_camera.seq = ++seq;
                        g_camera_timestamps.push_back(std::chrono::steady_clock::now());
                    }
                    g_camera_frames.fetch_add(1, std::memory_order_relaxed);
                    g_camera_cv.notify_one();
                    last_loop_end = std::chrono::steady_clock::now();
                }
            }
            catch (...)
            {
                capture_error = std::current_exception();
                stop_capture.store(true, std::memory_order_relaxed);
                g_camera_cv.notify_all();
            }
        });

        std::thread inference_thread([&]() {
            set_crash_phase(kPhaseInferenceThread);
            uint64_t last_seq = 0;
            try
            {
                while (true)
                {
                    cv::Mat frame;
                    {
                        std::unique_lock<std::mutex> lock(g_camera_mutex);
                        const auto wait_start = std::chrono::steady_clock::now();
                        g_camera_cv.wait(lock, [&]() {
                            return stop_capture.load(std::memory_order_relaxed) ||
                                   g_latest_camera.seq != last_seq;
                        });
                        record_timing(g_timing.inference_wait, wait_start);
                        if (stop_capture.load(std::memory_order_relaxed) && g_latest_camera.seq == last_seq)
                        {
                            break;
                        }
                        const auto copy_start = std::chrono::steady_clock::now();
                        frame = g_latest_camera.frame.clone();
                        last_seq = g_latest_camera.seq;
                        record_timing(g_timing.inference_copy, copy_start);
                    }

                    if (!frame.empty())
                    {
                        if (!async_depth.try_submit(frame))
                        {
                            g_skipped_frames.fetch_add(1, std::memory_order_relaxed);
                        }
                    }
                }
            }
            catch (...)
            {
                inference_error = std::current_exception();
                stop_capture.store(true, std::memory_order_relaxed);
                g_camera_cv.notify_all();
            }
        });

        while (!stop_capture.load(std::memory_order_relaxed))
        {
            const auto now = std::chrono::steady_clock::now();
            const double since_update = std::chrono::duration<double>(now - last_fps_update).count();
            if (since_update >= kFpsUpdateIntervalSec)
            {
                depth_fps_display = async_depth.completion_fps();
                camera_fps_display = camera_fps();
                last_fps_update = now;
            }

            DepthResult result;
            if (async_depth.pop_next_result(&result))
            {
                if (!result.depth.empty())
                {
                    {
                        ScopedTimer display_timer(g_timing.display_total);

                        cv::Mat frame_show;
                        {
                            ScopedTimer post_timer(g_timing.display_postprocess);
                            cv::Mat depth_colored = create_depth_map(result.depth, options.grayscale);
                            cv::Mat display_content;
                            if (options.side_by_side)
                            {
                                cv::Mat depth_scaled;
                                cv::resize(depth_colored, depth_scaled, result.original_bgr.size(), 0.0, 0.0, cv::INTER_LINEAR);
                                cv::hconcat(result.original_bgr, depth_scaled, display_content);
                            }
                            else
                            {
                                display_content = depth_colored;
                            }

                            frame_show = letterbox_to_screen(display_content, screen.first, screen.second, options.margin_bgr);
                            draw_fps_overlay(frame_show, depth_fps_display);
                            draw_model_name_overlay(frame_show, options.model_path);
                            draw_frame_id_overlay(frame_show, result.frame_id, result.latency_ms);
                        }

                        {
                            ScopedTimer show_timer(g_timing.display_show);
                            cv::imshow(kWindowName, frame_show);
                            set_fps_window_title(depth_fps_display);

                            if (!window_fullscreen)
                            {
                                cv::setWindowProperty(kWindowName, cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
                                window_fullscreen = true;
                            }
                            if (!launcher_ready)
                            {
                                notify_launcher_ready();
                                launcher_ready = true;
                            }
                        }
                    }
                }
            }

            if (now - last_stats_print >= std::chrono::seconds(1))
            {
                const double stats_elapsed_sec = std::chrono::duration<double>(now - last_stats_print).count();
                last_stats_print = now;
                const uint64_t camera_frames = g_camera_frames.load(std::memory_order_relaxed);
                const uint64_t submitted_frames = g_submitted_frames.load(std::memory_order_relaxed);
                const uint64_t skipped_frames = g_skipped_frames.load(std::memory_order_relaxed);
                const uint64_t completed_frames = g_completed_frames.load(std::memory_order_relaxed);
                const uint64_t displayed_frames = g_displayed_frames.load(std::memory_order_relaxed);

                const double camera_rate = rate_since_last(camera_frames, &last_camera_frames, stats_elapsed_sec);
                const double submitted_rate = rate_since_last(submitted_frames, &last_submitted_frames, stats_elapsed_sec);
                const double skipped_rate = rate_since_last(skipped_frames, &last_skipped_frames, stats_elapsed_sec);
                const double completed_rate = rate_since_last(completed_frames, &last_completed_frames, stats_elapsed_sec);
                const double displayed_rate = rate_since_last(displayed_frames, &last_displayed_frames, stats_elapsed_sec);

                const double camera_read_ms = average_ms_since_last(g_timing.camera_read, &timing_snapshot.camera_read);
                const double camera_grab_ms = average_ms_since_last(g_timing.camera_grab, &timing_snapshot.camera_grab);
                const double camera_retrieve_ms = average_ms_since_last(g_timing.camera_retrieve, &timing_snapshot.camera_retrieve);
                const double camera_interval_ms = average_ms_since_last(g_timing.camera_interval, &timing_snapshot.camera_interval);
                const double camera_loop_gap_ms = average_ms_since_last(g_timing.camera_loop_gap, &timing_snapshot.camera_loop_gap);
                const double camera_store_ms = average_ms_since_last(g_timing.camera_store, &timing_snapshot.camera_store);
                const double camera_lock_wait_ms = average_ms_since_last(g_timing.camera_lock_wait, &timing_snapshot.camera_lock_wait);
                const double inference_wait_ms = average_ms_since_last(g_timing.inference_wait, &timing_snapshot.inference_wait);
                const double inference_copy_ms = average_ms_since_last(g_timing.inference_copy, &timing_snapshot.inference_copy);
                const double submit_total_ms = average_ms_since_last(g_timing.submit_total, &timing_snapshot.submit_total);
                const double preprocess_ms = average_ms_since_last(g_timing.preprocess, &timing_snapshot.preprocess);
                const double run_async_ms = average_ms_since_last(g_timing.run_async, &timing_snapshot.run_async);
                const double callback_total_ms = average_ms_since_last(g_timing.callback_total, &timing_snapshot.callback_total);
                const double callback_tensor_ms = average_ms_since_last(g_timing.callback_tensor, &timing_snapshot.callback_tensor);
                const double display_total_ms = average_ms_since_last(g_timing.display_total, &timing_snapshot.display_total);
                const double display_postprocess_ms = average_ms_since_last(g_timing.display_postprocess, &timing_snapshot.display_postprocess);
                const double display_show_ms = average_ms_since_last(g_timing.display_show, &timing_snapshot.display_show);
                const double wait_key_ms = average_ms_since_last(g_timing.wait_key, &timing_snapshot.wait_key);

                std::cout << std::fixed << std::setprecision(2);
                std::cout << "Stats: camera=" << g_camera_frames.load(std::memory_order_relaxed)
                          << " submitted=" << g_submitted_frames.load(std::memory_order_relaxed)
                          << " skipped_full=" << g_skipped_frames.load(std::memory_order_relaxed)
                          << " completed=" << g_completed_frames.load(std::memory_order_relaxed)
                          << " displayed=" << g_displayed_frames.load(std::memory_order_relaxed)
                          << " next_display_id=" << async_depth.next_display_id()
                          << " inflight=" << async_depth.in_flight()
                          << " camera_fps=" << camera_fps_display
                          << " depth_fps=" << depth_fps_display
                          << std::endl;
                std::cout << "Rates/s: camera=" << camera_rate
                          << " submitted=" << submitted_rate
                          << " skipped_full=" << skipped_rate
                          << " completed=" << completed_rate
                          << " displayed=" << displayed_rate
                          << std::endl;
                std::cout << "Timing avg ms: camera_read=" << camera_read_ms
                          << " camera_grab=" << camera_grab_ms
                          << " camera_retrieve=" << camera_retrieve_ms
                          << " camera_interval=" << camera_interval_ms
                          << " camera_loop_gap=" << camera_loop_gap_ms
                          << " camera_store=" << camera_store_ms
                          << " camera_lock_wait=" << camera_lock_wait_ms
                          << " inference_wait=" << inference_wait_ms
                          << " inference_copy=" << inference_copy_ms
                          << " submit_total=" << submit_total_ms
                          << " preprocess=" << preprocess_ms
                          << " run_async_call=" << run_async_ms
                          << " callback_total=" << callback_total_ms
                          << " callback_tensor=" << callback_tensor_ms
                          << " display_total=" << display_total_ms
                          << " display_post=" << display_postprocess_ms
                          << " display_show=" << display_show_ms
                          << " wait_key=" << wait_key_ms
                          << std::endl;
            }

            const auto wait_key_start = std::chrono::steady_clock::now();
            const int key = cv::waitKey(1) & 0xff;
            record_timing(g_timing.wait_key, wait_key_start);
            if (key == 'q' || key == 27)
            {
                cv::waitKey(500);
                stop_capture.store(true, std::memory_order_relaxed);
                g_camera_cv.notify_all();
                break;
            }
        }

        set_crash_phase(kPhaseShutdown);
        stop_capture.store(true, std::memory_order_relaxed);
        g_camera_cv.notify_all();
        if (capture_thread.joinable())
        {
            capture_thread.join();
        }
        if (inference_thread.joinable())
        {
            inference_thread.join();
        }

        async_depth.wait_all();

        if (capture_error)
        {
            std::rethrow_exception(capture_error);
        }
        if (inference_error)
        {
            std::rethrow_exception(inference_error);
        }

        cap.release();
        cv::destroyAllWindows();
        std::cout << "\nFinished." << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error during " << crash_phase_name(g_crash_phase) << ": " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
