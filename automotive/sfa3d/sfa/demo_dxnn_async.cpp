#include "dxrt/dxrt_api.h"
#include "dxrt/extern/cxxopts.hpp"

#include <opencv2/opencv.hpp>
#include <QApplication>
#include <QCloseEvent>
#include <QColor>
#include <QEventLoop>
#include <QFont>
#include <QFontDatabase>
#include <QFontInfo>
#include <QFontMetrics>
#include <QImage>
#include <QKeyEvent>
#include <QPainter>
#include <QPoint>
#include <QPushButton>
#include <QRect>
#include <QResizeEvent>
#include <QSize>
#include <QString>
#include <QWidget>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <optional>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr int kMaxInferenceQueue = 6;
constexpr int kBevWidth = 608;
constexpr int kBevHeight = 608;
constexpr int kHmWidth = 152;
constexpr int kHmHeight = 152;
constexpr int kNumClasses = 3;
constexpr int kDefaultDetectionPaletteIndex = 0;  // --color 1 (aurora)
constexpr double kBBoxLineWidthScale = 0.8;
constexpr int kDownRatio = 4;
constexpr float kMinX = 0.0F;
constexpr float kMaxX = 50.0F;
constexpr float kMinY = -25.0F;
constexpr float kMaxY = 25.0F;
constexpr float kMinZ = -2.73F;
constexpr float kMaxZ = 1.27F;
constexpr float kBoundSizeX = kMaxX - kMinX;
constexpr float kBoundSizeY = kMaxY - kMinY;
constexpr float kDiscretization = kBoundSizeX / static_cast<float>(kBevHeight);
constexpr const char* kWindowName = "SFA3D - Front vs Back (DXNN Async C++)";
constexpr const char* kDefaultDemoCheckpoint = "../checkpoints/fpn_resnet_18/fpn_resnet_18_epoch_300.pth";
constexpr const char* kTitleText =
    "Super Fast and Accurate 3D Object Detection based on 3D LiDAR Point Clouds (SFA3D)";
constexpr size_t kMaxPreparedQueue = 4;
constexpr auto kPreparedQueuePollInterval = std::chrono::milliseconds(1);

struct FontPreset {
    const char* name;
    const char* description;
    std::array<const char*, 3> preferred_families;
};

const std::array<FontPreset, 3> kFontPresets = {{
    {"sans", "clean UI sans-serif (recommended)", {"Noto Sans", "Ubuntu Sans", "DejaVu Sans"}},
    {"serif", "editorial serif", {"Noto Serif", "Liberation Serif", "DejaVu Serif"}},
    {"mono", "technical monospace", {"Noto Sans Mono", "Liberation Mono", "DejaVu Sans Mono"}},
}};

QString gUiFontFamily = "DejaVu Sans";

void notifyLauncherReady() {
    const char* path = std::getenv("DX_LAUNCHER_READY_FILE");
    if (path == nullptr || *path == '\0') {
        return;
    }
    std::ofstream ready(path, std::ios::trunc);
    if (ready) {
        ready << "ready\n";
    }
}

std::string fontPresetNames() {
    std::ostringstream names;
    for (size_t i = 0; i < kFontPresets.size(); ++i) {
        if (i > 0) {
            names << ", ";
        }
        names << (i + 1) << "=" << kFontPresets[i].name
              << " (" << kFontPresets[i].description << ")";
    }
    return names.str();
}

int fontPresetIndex(std::string value) {
    if (value.size() == 1 && value[0] >= '1' &&
        value[0] < '1' + static_cast<int>(kFontPresets.size())) {
        return value[0] - '1';
    }
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    for (size_t i = 0; i < kFontPresets.size(); ++i) {
        if (value == kFontPresets[i].name) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

QString resolveFontFamily(int preset_index) {
    const FontPreset& preset = kFontPresets.at(static_cast<size_t>(preset_index));
    const QStringList installed = QFontDatabase().families();
    for (const char* candidate : preset.preferred_families) {
        const QString requested = QString::fromUtf8(candidate);
        for (const QString& family : installed) {
            if (family.compare(requested, Qt::CaseInsensitive) == 0) {
                return family;
            }
        }
    }

    if (std::string(preset.name) == "mono") {
        return QFontDatabase::systemFont(QFontDatabase::FixedFont).family();
    }
    QFont fallback(std::string(preset.name) == "serif" ? "serif" : "sans-serif");
    fallback.setStyleHint(std::string(preset.name) == "serif" ? QFont::Serif : QFont::SansSerif);
    return QFontInfo(fallback).family();
}

struct DetectionPalette {
    const char* name;
    std::array<cv::Scalar, kNumClasses> class_colors;
    cv::Scalar heading;
};

// OpenCV colors use BGR ordering. Class order: Pedestrian, Car, Cyclist.
const std::array<DetectionPalette, 5> kDetectionPalettes = {{
    {"aurora",
     {cv::Scalar(255, 230, 64), cv::Scalar(92, 238, 118), cv::Scalar(255, 70, 224)},
     cv::Scalar(98, 234, 255)},
    {"sunset",
     {cv::Scalar(54, 178, 255), cv::Scalar(98, 234, 255), cv::Scalar(86, 124, 255)},
     cv::Scalar(120, 210, 255)},
    {"ocean",
     {cv::Scalar(255, 214, 72), cv::Scalar(210, 238, 88), cv::Scalar(255, 124, 88)},
     cv::Scalar(106, 252, 214)},
    {"neon",
     {cv::Scalar(178, 255, 42), cv::Scalar(255, 74, 238), cv::Scalar(70, 240, 126)},
     cv::Scalar(92, 255, 238)},
    {"graphite",
     {cv::Scalar(236, 222, 186), cv::Scalar(146, 226, 255), cv::Scalar(184, 164, 255)},
     cv::Scalar(210, 228, 238)},
}};

std::string detectionPaletteNames() {
    std::ostringstream names;
    for (size_t i = 0; i < kDetectionPalettes.size(); ++i) {
        if (i > 0) {
            names << ", ";
        }
        names << (i + 1) << "=" << kDetectionPalettes[i].name;
    }
    return names.str();
}

int detectionPaletteIndex(std::string value) {
    if (value.size() == 1 && value[0] >= '1' &&
        value[0] < '1' + static_cast<int>(kDetectionPalettes.size())) {
        return value[0] - '1';
    }

    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    for (size_t i = 0; i < kDetectionPalettes.size(); ++i) {
        if (value == kDetectionPalettes[i].name) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

// UI palette aligned with paddle-ocr/cam-ppocr-v6.
const cv::Scalar kUiBg(30, 30, 30);
const cv::Scalar kUiPanel(38, 37, 37);
const cv::Scalar kUiCard(48, 45, 45);
const cv::Scalar kUiBorder(70, 63, 63);
const cv::Scalar kUiAccent(204, 122, 0);
const cv::Scalar kUiAccentDark(158, 90, 0);
const cv::Scalar kUiText(245, 244, 244);
const cv::Scalar kUiTextDim(170, 161, 161);
const cv::Scalar kUiTrack(62, 58, 57);

struct Boundary {
    float min_x;
    float max_x;
    float min_y;
    float max_y;
    float min_z;
    float max_z;
};

const Boundary kFrontBoundary{kMinX, kMaxX, kMinY, kMaxY, kMinZ, kMaxZ};
const Boundary kBackBoundary{-50.0F, 0.0F, kMinY, kMaxY, kMinZ, kMaxZ};

struct Config {
    std::string saved_fn = "fpn_resnet_18";
    std::string arch = "fpn_resnet_18";
    fs::path pretrained_path;
    std::string foldername = "2011_09_26_drive_0014_sync";
    int top_k = 50;
    bool no_cuda = false;
    int gpu_idx = 0;
    float peak_thresh = 0.2F;
    std::string output_format = "image";
    int output_width = 608;
    bool full_screen = false;
    bool exit_btn = false;
    int color_palette_index = kDefaultDetectionPaletteIndex;
    int font_preset_index = 0;
    bool loop = false;
    bool debug = false;
    double timing_cpu_scale = 1.0;

    fs::path repo_root;
    fs::path dataset_dir;
    fs::path calib_path;
};

using Clock = std::chrono::steady_clock;

double elapsedMs(Clock::time_point start, Clock::time_point end = Clock::now()) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

struct StageTimings {
    double load_ms = 0.0;
    double make_bev_ms = 0.0;
    double prepare_input_ms = 0.0;
    double submit_ms = 0.0;
    double callback_copy_ms = 0.0;
    double decode_ms = 0.0;
    double render_ms = 0.0;

    double cpuTotalMs() const {
        return load_ms + make_bev_ms + prepare_input_ms + submit_ms + callback_copy_ms + decode_ms + render_ms;
    }
};

struct LidarPoint {
    float x = 0.0F;
    float y = 0.0F;
    float z = 0.0F;
    float intensity = 0.0F;
};

struct BevMap {
    int channels = 3;
    int height = kBevHeight;
    int width = kBevWidth;
    std::vector<float> data;

    BevMap() : data(static_cast<size_t>(channels * height * width), 0.0F) {}

    float& at(int c, int y, int x) {
        return data[(static_cast<size_t>(c) * height + y) * width + x];
    }

    float at(int c, int y, int x) const {
        return data[(static_cast<size_t>(c) * height + y) * width + x];
    }
};

struct Detection {
    float score = 0.0F;
    float x = 0.0F;
    float y = 0.0F;
    float z = 0.0F;
    float h = 0.0F;
    float w = 0.0F;
    float l = 0.0F;
    float yaw = 0.0F;
};

using DetectionsByClass = std::array<std::vector<Detection>, kNumClasses>;

struct KittiDetection {
    int cls_id = -1;
    float x = 0.0F;
    float y = 0.0F;
    float z = 0.0F;
    float h = 0.0F;
    float w = 0.0F;
    float l = 0.0F;
    float yaw = 0.0F;
};

struct SampleResult {
    int64_t sequence_idx = 0;
    int dataset_idx = 0;
    fs::path img_path;
    BevMap front_bevmap;
    BevMap back_bevmap;
    cv::Mat img_rgb;
    std::optional<DetectionsByClass> front_detections;
    std::optional<DetectionsByClass> back_detections;
    double inference_fps = 0.0;
    StageTimings timings;

    bool isComplete() const {
        return front_detections.has_value() && back_detections.has_value();
    }
};

struct PreparedSample {
    SampleResult sample;
    std::vector<uint8_t> front_input;
    std::vector<uint8_t> back_input;
};

struct InputSpec {
    std::vector<int64_t> shape;
    dxrt::DataType dtype = dxrt::DataType::NONE_TYPE;
    size_t byte_size = 0;
};

struct CopiedTensor {
    std::string name;
    std::vector<int64_t> shape;
    dxrt::DataType type = dxrt::DataType::NONE_TYPE;
    std::vector<uint8_t> bytes;
};

struct AsyncJob {
    int64_t sequence_idx = 0;
    std::string side;
    std::vector<uint8_t> input_data;
};

struct CompletedJob {
    int64_t sequence_idx = 0;
    std::string side;
    std::vector<CopiedTensor> outputs;
    std::chrono::steady_clock::time_point completed_at;
    double callback_copy_ms = 0.0;
    std::string error;
};

struct PreparedItem {
    enum class Type { Sample, Error, Stop };

    Type type = Type::Stop;
    PreparedSample sample;
    std::string message;
    std::string detail;

    static PreparedItem sampleItem(PreparedSample prepared_sample) {
        PreparedItem item;
        item.type = Type::Sample;
        item.sample = std::move(prepared_sample);
        return item;
    }

    static PreparedItem errorItem(std::string msg, std::string detail_msg) {
        PreparedItem item;
        item.type = Type::Error;
        item.message = std::move(msg);
        item.detail = std::move(detail_msg);
        return item;
    }

    static PreparedItem stopItem() {
        PreparedItem item;
        item.type = Type::Stop;
        return item;
    }
};

struct RenderItem {
    enum class Type { Sample, Error, Stop };

    Type type = Type::Stop;
    SampleResult sample;
    std::string message;
    std::string detail;

    static RenderItem sampleItem(SampleResult sample_result) {
        RenderItem item;
        item.type = Type::Sample;
        item.sample = std::move(sample_result);
        return item;
    }

    static RenderItem errorItem(std::string msg, std::string detail_msg) {
        RenderItem item;
        item.type = Type::Error;
        item.message = std::move(msg);
        item.detail = std::move(detail_msg);
        return item;
    }

    static RenderItem stopItem() {
        RenderItem item;
        item.type = Type::Stop;
        return item;
    }
};

template <typename T>
class BlockingQueue {
public:
    explicit BlockingQueue(size_t max_size = 0) : max_size_(max_size) {}

    void push(T value) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (max_size_ > 0) {
            not_full_.wait(lock, [&] { return closed_ || queue_.size() < max_size_; });
        }
        if (closed_) {
            return;
        }
        queue_.push(std::move(value));
        not_empty_.notify_one();
    }

    bool pop(T& value) {
        std::unique_lock<std::mutex> lock(mutex_);
        not_empty_.wait(lock, [&] { return closed_ || !queue_.empty(); });
        if (queue_.empty()) {
            return false;
        }
        value = std::move(queue_.front());
        queue_.pop();
        not_full_.notify_one();
        return true;
    }

    template <typename Rep, typename Period>
    bool popFor(T& value, const std::chrono::duration<Rep, Period>& timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!not_empty_.wait_for(lock, timeout, [&] { return closed_ || !queue_.empty(); })) {
            return false;
        }
        if (queue_.empty()) {
            return false;
        }
        value = std::move(queue_.front());
        queue_.pop();
        not_full_.notify_one();
        return true;
    }

    void close() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            closed_ = true;
        }
        not_empty_.notify_all();
        not_full_.notify_all();
    }

private:
    size_t max_size_ = 0;
    std::queue<T> queue_;
    bool closed_ = false;
    std::mutex mutex_;
    std::condition_variable not_empty_;
    std::condition_variable not_full_;
};

std::string shapeToString(const std::vector<int64_t>& shape) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) {
            oss << ", ";
        }
        oss << shape[i];
    }
    oss << "]";
    return oss.str();
}

std::string shellQuote(const std::string& text) {
    std::string quoted = "'";
    for (char ch : text) {
        if (ch == '\'') {
            quoted += "'\\''";
        } else {
            quoted += ch;
        }
    }
    quoted += "'";
    return quoted;
}

bool commandExists(const std::string& command) {
    const std::string check = "command -v " + command + " >/dev/null 2>&1";
    return std::system(check.c_str()) == 0;
}

void runShellCommand(const std::string& command, const std::string& description) {
    const int rc = std::system(command.c_str());
    if (rc != 0) {
        throw std::runtime_error(description + " failed with exit code " + std::to_string(rc));
    }
}

std::string stackMessage(const std::exception& exc) {
    return exc.what();
}

void ensureDirectory(const fs::path& path) {
    std::error_code ec;
    fs::create_directories(path, ec);
    if (ec) {
        throw std::runtime_error("Failed to create directory " + path.string() + ": " + ec.message());
    }
}

bool datasetReady(const Config& config) {
    const fs::path data_root = config.dataset_dir / config.foldername / config.foldername.substr(0, 10) / config.foldername;
    return fs::exists(data_root / "image_02" / "data") && fs::exists(data_root / "velodyne_points" / "data");
}

void downloadAndUnzip(const Config& config) {
    if (datasetReady(config)) {
        return;
    }

    ensureDirectory(config.dataset_dir);
    const std::string parent = config.foldername.size() > 5
        ? config.foldername.substr(0, config.foldername.size() - 5)
        : config.foldername;
    const std::string url =
        "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/" + parent + "/" + config.foldername + ".zip";
    const fs::path zip_path = config.dataset_dir / (config.foldername + ".zip");

    if (fs::exists(zip_path)) {
        std::cout << "The dataset have been downloaded" << std::endl;
    } else {
        std::cout << "\nDownloading data for demonstration..." << std::endl;
        if (commandExists("wget")) {
            runShellCommand("wget -O " + shellQuote(zip_path.string()) + " " + shellQuote(url), "wget");
        } else if (commandExists("curl")) {
            runShellCommand("curl -L -o " + shellQuote(zip_path.string()) + " " + shellQuote(url), "curl");
        } else {
            throw std::runtime_error("Neither wget nor curl is available to download the demo dataset");
        }
    }

    std::cout << "\nUnzipping the downloaded data..." << std::endl;
    ensureDirectory(config.dataset_dir / config.foldername);
    runShellCommand(
        "unzip -q " + shellQuote(zip_path.string()) + " -d " + shellQuote((config.dataset_dir / config.foldername).string()),
        "unzip");
}

Config parseArgs(int argc, char* argv[]) {
    Config config;
    config.repo_root = fs::absolute(fs::path(SFA3D_REPO_ROOT));

    std::string pretrained_path = kDefaultDemoCheckpoint;
    std::string color = kDetectionPalettes[kDefaultDetectionPaletteIndex].name;
    std::string font = kFontPresets[0].name;
    bool full_screen_dash = false;

    cxxopts::Options options("demo_dxnn_async_cpp", "SFA3D DXNN async demo implemented in C++");
    options.add_options()
        ("saved_fn", "Accepted for legacy CLI compatibility",
            cxxopts::value<std::string>(config.saved_fn)->default_value("fpn_resnet_18"))
        ("a,arch", "The model architecture name",
            cxxopts::value<std::string>(config.arch)->default_value("fpn_resnet_18"))
        ("pretrained_path", "Path to a .dxnn model",
            cxxopts::value<std::string>(pretrained_path)->default_value(kDefaultDemoCheckpoint))
        ("foldername", "KITTI raw demo folder name",
            cxxopts::value<std::string>(config.foldername)->default_value("2011_09_26_drive_0014_sync"))
        ("K", "Number of top K center points",
            cxxopts::value<int>(config.top_k)->default_value("50"))
        ("no_cuda", "Accepted for legacy CLI compatibility",
            cxxopts::value<bool>(config.no_cuda)->default_value("false"))
        ("gpu_idx", "Accepted for legacy CLI compatibility",
            cxxopts::value<int>(config.gpu_idx)->default_value("0"))
        ("peak_thresh", "Peak threshold for decoded detections",
            cxxopts::value<float>(config.peak_thresh)->default_value("0.2"))
        ("output_format", "Accepted for legacy CLI compatibility",
            cxxopts::value<std::string>(config.output_format)->default_value("image"))
        ("output-width", "Accepted for legacy CLI compatibility",
            cxxopts::value<int>(config.output_width)->default_value("608"))
        ("full_screen", "Display the OpenCV window in full-screen mode",
            cxxopts::value<bool>(config.full_screen)->default_value("false"))
        ("full-screen", "Alias for --full_screen",
            cxxopts::value<bool>(full_screen_dash)->default_value("false"))
        ("exit-btn", "Draw a small exit button at the top-right of the display",
            cxxopts::value<bool>(config.exit_btn)->default_value("false"))
        ("color", "3D BBox palette: " + detectionPaletteNames(),
            cxxopts::value<std::string>(color)->default_value(
                kDetectionPalettes[kDefaultDetectionPaletteIndex].name))
        ("font", "UI font: " + fontPresetNames(),
            cxxopts::value<std::string>(font)->default_value(kFontPresets[0].name))
        ("loop", "Loop the input sequence forever",
            cxxopts::value<bool>(config.loop)->default_value("false"))
        ("debug", "Enable debug logs and stage timing instrumentation",
            cxxopts::value<bool>(config.debug)->default_value("false"))
        ("timing-cpu-scale", "Scale CPU-side timing totals for target CPU estimation in debug logs",
            cxxopts::value<double>(config.timing_cpu_scale)->default_value("1.0"))
        ("h,help", "Print usage");

    const auto result = options.parse(argc, argv);
    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        std::exit(0);
    }

    config.full_screen = config.full_screen || full_screen_dash;
    config.color_palette_index = detectionPaletteIndex(color);
    if (config.color_palette_index < 0) {
        throw std::runtime_error("--color expects one of: " + detectionPaletteNames());
    }
    config.font_preset_index = fontPresetIndex(font);
    if (config.font_preset_index < 0) {
        throw std::runtime_error("--font expects one of: " + fontPresetNames());
    }
    if (config.timing_cpu_scale <= 0.0) {
        throw std::runtime_error("--timing-cpu-scale must be greater than 0");
    }
    if (pretrained_path == kDefaultDemoCheckpoint) {
        config.pretrained_path = config.repo_root / "sfa3d_608x608_q-lite.dxnn";
    } else {
        config.pretrained_path = fs::path(pretrained_path);
    }

    config.dataset_dir = config.repo_root / "dataset" / "kitti" / "demo";
    config.calib_path = config.repo_root / "dataset" / "kitti" / "demo" / "calib.txt";

    return config;
}

std::vector<LidarPoint> readLidarFile(const fs::path& lidar_file) {
    std::ifstream input(lidar_file, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error("Failed to open lidar file: " + lidar_file.string());
    }
    const auto size = input.tellg();
    if (size < 0 || static_cast<uint64_t>(size) % (sizeof(float) * 4ULL) != 0) {
        throw std::runtime_error("Invalid lidar file size: " + lidar_file.string());
    }
    input.seekg(0, std::ios::beg);

    std::vector<float> raw(static_cast<size_t>(size) / sizeof(float));
    input.read(reinterpret_cast<char*>(raw.data()), size);
    if (!input) {
        throw std::runtime_error("Failed to read lidar file: " + lidar_file.string());
    }

    std::vector<LidarPoint> points;
    points.reserve(raw.size() / 4);
    for (size_t i = 0; i < raw.size(); i += 4) {
        points.push_back({raw[i], raw[i + 1], raw[i + 2], raw[i + 3]});
    }
    return points;
}

struct CellInfo {
    int count = 0;
    float top_z = -std::numeric_limits<float>::infinity();
    float top_intensity = 0.0F;
};

int normalizeNumpyIndex(int index, int size) {
    if (index < 0) {
        index += size;
    }
    return index;
}

void updateDenseBevCell(std::vector<CellInfo>& cells, const Boundary& boundary, const LidarPoint& point) {
    if (point.x < boundary.min_x || point.x > boundary.max_x ||
        point.y < boundary.min_y || point.y > boundary.max_y ||
        point.z < boundary.min_z || point.z > boundary.max_z) {
        return;
    }

    constexpr int temp_height = kBevHeight + 1;
    constexpr int temp_width = kBevWidth + 1;
    const float half_width = static_cast<float>(temp_width) / 2.0F;
    int row = normalizeNumpyIndex(static_cast<int>(std::floor(point.x / kDiscretization)), temp_height);
    int col = normalizeNumpyIndex(static_cast<int>(std::floor(point.y / kDiscretization) + half_width), temp_width);
    if (row < 0 || row >= temp_height || col < 0 || col >= temp_width) {
        return;
    }

    auto& cell = cells[static_cast<size_t>(row * temp_width + col)];
    cell.count += 1;
    const float adjusted_z = point.z - boundary.min_z;
    if (adjusted_z > cell.top_z) {
        cell.top_z = adjusted_z;
        cell.top_intensity = point.intensity;
    }
}

BevMap cellsToBEVMap(const std::vector<CellInfo>& cells, const Boundary& boundary) {
    constexpr int temp_width = kBevWidth + 1;
    const float max_height = std::abs(boundary.max_z - boundary.min_z);
    BevMap bev;
    for (int y = 0; y < kBevHeight; ++y) {
        for (int x = 0; x < kBevWidth; ++x) {
            const auto& cell = cells[static_cast<size_t>(y * temp_width + x)];
            if (cell.count <= 0) {
                continue;
            }
            bev.at(2, y, x) = std::min(1.0F, std::log(static_cast<float>(cell.count) + 1.0F) / std::log(64.0F));
            bev.at(1, y, x) = cell.top_z / max_height;
            bev.at(0, y, x) = cell.top_intensity;
        }
    }
    return bev;
}

BevMap flipBevMap(const BevMap& src);

std::pair<BevMap, BevMap> makeFrontBackBEVMaps(const std::vector<LidarPoint>& lidar) {
    constexpr int temp_height = kBevHeight + 1;
    constexpr int temp_width = kBevWidth + 1;
    std::vector<CellInfo> front_cells(static_cast<size_t>(temp_height * temp_width));
    std::vector<CellInfo> back_cells(static_cast<size_t>(temp_height * temp_width));

    for (const auto& point : lidar) {
        updateDenseBevCell(front_cells, kFrontBoundary, point);
        updateDenseBevCell(back_cells, kBackBoundary, point);
    }

    return {cellsToBEVMap(front_cells, kFrontBoundary), flipBevMap(cellsToBEVMap(back_cells, kBackBoundary))};
}

BevMap flipBevMap(const BevMap& src) {
    BevMap dst;
    for (int c = 0; c < src.channels; ++c) {
        for (int y = 0; y < src.height; ++y) {
            for (int x = 0; x < src.width; ++x) {
                dst.at(c, y, x) = src.at(c, src.height - 1 - y, src.width - 1 - x);
            }
        }
    }
    return dst;
}

class DemoKittiDataset {
public:
    explicit DemoKittiDataset(const Config& config)
        : config_(config),
          dataset_root_(config.dataset_dir / config.foldername / config.foldername.substr(0, 10) / config.foldername),
          image_dir_(dataset_root_ / "image_02" / "data"),
          lidar_dir_(dataset_root_ / "velodyne_points" / "data") {
        if (!fs::exists(lidar_dir_)) {
            throw std::runtime_error("Missing lidar directory: " + lidar_dir_.string());
        }

        std::vector<fs::path> lidar_files;
        for (const auto& entry : fs::directory_iterator(lidar_dir_)) {
            if (entry.is_regular_file() && entry.path().extension() == ".bin") {
                lidar_files.push_back(entry.path());
            }
        }
        std::sort(lidar_files.begin(), lidar_files.end());

        sample_ids_.reserve(lidar_files.size());
        for (const auto& path : lidar_files) {
            sample_ids_.push_back(std::stoi(path.stem().string()));
        }

        if (sample_ids_.empty()) {
            throw std::runtime_error("No lidar samples found in: " + lidar_dir_.string());
        }
    }

    size_t size() const {
        return sample_ids_.size();
    }

    SampleResult loadBevmapFrontVsBack(size_t index, int64_t sequence_idx) const {
        const int sample_id = sample_ids_.at(index);

        Clock::time_point load_start;
        if (config_.debug) {
            load_start = Clock::now();
        }
        const fs::path img_path = image_dir_ / (formatSampleId(sample_id) + ".png");
        cv::Mat img_bgr = cv::imread(img_path.string(), cv::IMREAD_COLOR);
        if (img_bgr.empty()) {
            throw std::runtime_error("Failed to read image: " + img_path.string());
        }
        cv::Mat img_rgb;
        cv::cvtColor(img_bgr, img_rgb, cv::COLOR_BGR2RGB);

        const fs::path lidar_path = lidar_dir_ / (formatSampleId(sample_id) + ".bin");
        const auto lidar = readLidarFile(lidar_path);
        double load_ms = 0.0;
        if (config_.debug) {
            load_ms = elapsedMs(load_start);
        }

        Clock::time_point bev_start;
        if (config_.debug) {
            bev_start = Clock::now();
        }
        auto bev_maps = makeFrontBackBEVMaps(lidar);

        SampleResult sample;
        sample.sequence_idx = sequence_idx;
        sample.dataset_idx = sample_id;
        sample.img_path = img_path;
        sample.front_bevmap = std::move(bev_maps.first);
        sample.back_bevmap = std::move(bev_maps.second);
        sample.img_rgb = std::move(img_rgb);
        if (config_.debug) {
            sample.timings.load_ms = load_ms;
            sample.timings.make_bev_ms = elapsedMs(bev_start);
        }
        return sample;
    }

private:
    static std::string formatSampleId(int sample_id) {
        std::ostringstream oss;
        oss << std::setw(10) << std::setfill('0') << sample_id;
        return oss.str();
    }

    const Config& config_;
    fs::path dataset_root_;
    fs::path image_dir_;
    fs::path lidar_dir_;
    std::vector<int> sample_ids_;
};

size_t numElements(const std::vector<int64_t>& shape) {
    size_t total = 1;
    for (const auto dim : shape) {
        if (dim <= 0) {
            throw std::runtime_error("Dynamic or invalid tensor shape is not supported: " + shapeToString(shape));
        }
        total *= static_cast<size_t>(dim);
    }
    return total;
}

template <typename T>
void writeScalar(std::vector<uint8_t>& buffer, size_t index, T value) {
    const size_t offset = index * sizeof(T);
    if (offset + sizeof(T) > buffer.size()) {
        throw std::runtime_error("Input buffer write exceeds allocated tensor size");
    }
    std::memcpy(buffer.data() + offset, &value, sizeof(T));
}

template <typename T>
void writeIntegerInput(std::vector<uint8_t>& buffer, size_t index, float value, float max_value) {
    const float scaled = std::clamp(value * 255.0F, 0.0F, max_value);
    writeScalar<T>(buffer, index, static_cast<T>(scaled));
}

uint8_t toInputUint8(float value) {
    const float scaled = value * 255.0F;
    if (scaled <= 0.0F) {
        return 0;
    }
    if (scaled >= 255.0F) {
        return 255;
    }
    return static_cast<uint8_t>(scaled);
}

InputSpec getInputSpec(const dxrt::InferenceEngine& engine) {
    const auto input_tensors = engine.GetInputs();
    if (input_tensors.size() != 1) {
        throw std::runtime_error("This demo expects a single-input DXNN model, but got " +
                                 std::to_string(input_tensors.size()) + " inputs");
    }

    const auto& input = input_tensors.front();
    InputSpec spec;
    spec.shape = input.shape();
    spec.dtype = input.type();
    spec.byte_size = static_cast<size_t>(input.size_in_bytes());
    if (spec.byte_size == 0) {
        throw std::runtime_error("DXNN input tensor has empty size");
    }
    return spec;
}

std::vector<uint8_t> prepareDxnnInput(const InputSpec& input, const BevMap& bevmap) {
    const auto& shape = input.shape;
    const auto dtype = input.dtype;
    if (shape.size() != 4) {
        throw std::runtime_error("Expected 4D DXNN input shape, got " + shapeToString(shape));
    }

    const bool nchw = shape[1] == bevmap.channels && shape[2] == bevmap.height && shape[3] == bevmap.width;
    const bool nhwc = shape[1] == bevmap.height && shape[2] == bevmap.width && shape[3] == bevmap.channels;
    if (!nchw && !nhwc) {
        throw std::runtime_error("Cannot map BEV shape [3, 608, 608] to DXNN input shape " + shapeToString(shape));
    }

    const size_t elements = numElements(shape);
    std::vector<uint8_t> buffer(input.byte_size, 0);
    if (buffer.empty()) {
        throw std::runtime_error("DXNN input tensor has empty size");
    }

    if (dtype == dxrt::DataType::UINT8) {
        const size_t plane = static_cast<size_t>(bevmap.height * bevmap.width);
        if (buffer.size() != plane * static_cast<size_t>(bevmap.channels)) {
            throw std::runtime_error("UINT8 input byte size does not match BEV tensor shape");
        }
        if (nhwc) {
            for (size_t i = 0; i < plane; ++i) {
                for (int c = 0; c < bevmap.channels; ++c) {
                    buffer[i * static_cast<size_t>(bevmap.channels) + static_cast<size_t>(c)] =
                        toInputUint8(bevmap.data[static_cast<size_t>(c) * plane + i]);
                }
            }
        } else {
            for (int c = 0; c < bevmap.channels; ++c) {
                const size_t channel_offset = static_cast<size_t>(c) * plane;
                for (size_t i = 0; i < plane; ++i) {
                    buffer[channel_offset + i] = toInputUint8(bevmap.data[channel_offset + i]);
                }
            }
        }
        return buffer;
    }

    auto write_value = [&](size_t out_index, float value) {
        switch (dtype) {
        case dxrt::DataType::UINT8:
            writeIntegerInput<uint8_t>(buffer, out_index, value, static_cast<float>(std::numeric_limits<uint8_t>::max()));
            break;
        case dxrt::DataType::INT8:
            writeIntegerInput<int8_t>(buffer, out_index, value, static_cast<float>(std::numeric_limits<int8_t>::max()));
            break;
        case dxrt::DataType::UINT16:
            writeIntegerInput<uint16_t>(buffer, out_index, value, static_cast<float>(std::numeric_limits<uint16_t>::max()));
            break;
        case dxrt::DataType::INT16:
            writeIntegerInput<int16_t>(buffer, out_index, value, static_cast<float>(std::numeric_limits<int16_t>::max()));
            break;
        case dxrt::DataType::INT32:
            writeIntegerInput<int32_t>(buffer, out_index, value, static_cast<float>(std::numeric_limits<int32_t>::max()));
            break;
        case dxrt::DataType::FLOAT:
            writeScalar<float>(buffer, out_index, value);
            break;
        default:
            throw std::runtime_error("Unsupported DXNN input dtype: " + dxrt::DataTypeToString(dtype));
        }
    };

    if (dtype == dxrt::DataType::FLOAT && buffer.size() != elements * sizeof(float)) {
        throw std::runtime_error("Input byte size does not match float tensor shape");
    }

    if (nchw) {
        for (int c = 0; c < bevmap.channels; ++c) {
            for (int y = 0; y < bevmap.height; ++y) {
                for (int x = 0; x < bevmap.width; ++x) {
                    const size_t out_index = (static_cast<size_t>(c) * bevmap.height + y) * bevmap.width + x;
                    write_value(out_index, bevmap.at(c, y, x));
                }
            }
        }
    } else {
        for (int y = 0; y < bevmap.height; ++y) {
            for (int x = 0; x < bevmap.width; ++x) {
                for (int c = 0; c < bevmap.channels; ++c) {
                    const size_t out_index = (static_cast<size_t>(y) * bevmap.width + x) * bevmap.channels + c;
                    write_value(out_index, bevmap.at(c, y, x));
                }
            }
        }
    }

    return buffer;
}

std::vector<float> tensorToFloatVector(const CopiedTensor& tensor) {
    if (tensor.type != dxrt::DataType::FLOAT) {
        throw std::runtime_error("Expected float output tensor " + tensor.name +
                                 ", got " + dxrt::DataTypeToString(tensor.type));
    }
    if (tensor.bytes.size() % sizeof(float) != 0) {
        throw std::runtime_error("Output tensor byte size is not divisible by sizeof(float): " + tensor.name);
    }
    std::vector<float> values(tensor.bytes.size() / sizeof(float));
    std::memcpy(values.data(), tensor.bytes.data(), tensor.bytes.size());
    return values;
}

float sigmoid(float value) {
    return 1.0F / (1.0F + std::exp(-value));
}

struct OutputTensors {
    std::vector<float> hm_cen;
    std::vector<float> cen_offset;
    std::vector<float> direction;
    std::vector<float> z_coor;
    std::vector<float> dim;
};

OutputTensors mapOutputs(const std::vector<CopiedTensor>& outputs) {
    static const std::array<std::string, 5> fallback = {"dim", "z_coor", "cen_offset", "hm_cen", "direction"};

    std::map<std::string, const CopiedTensor*> by_name;
    for (const auto& output : outputs) {
        if (output.name == "hm_cen" || output.name == "cen_offset" || output.name == "direction" ||
            output.name == "z_coor" || output.name == "dim") {
            by_name[output.name] = &output;
        }
    }

    if (by_name.size() != fallback.size() && outputs.size() == fallback.size()) {
        by_name.clear();
        for (size_t i = 0; i < outputs.size(); ++i) {
            by_name[fallback[i]] = &outputs[i];
        }
    }

    for (const auto& name : {"hm_cen", "cen_offset", "direction", "z_coor", "dim"}) {
        if (by_name.find(name) == by_name.end()) {
            throw std::runtime_error(std::string("Missing DXNN output: ") + name);
        }
    }

    OutputTensors mapped;
    mapped.hm_cen = tensorToFloatVector(*by_name.at("hm_cen"));
    mapped.cen_offset = tensorToFloatVector(*by_name.at("cen_offset"));
    mapped.direction = tensorToFloatVector(*by_name.at("direction"));
    mapped.z_coor = tensorToFloatVector(*by_name.at("z_coor"));
    mapped.dim = tensorToFloatVector(*by_name.at("dim"));

    const size_t expected = static_cast<size_t>(kHmHeight * kHmWidth);
    if (mapped.hm_cen.size() != expected * 3 || mapped.cen_offset.size() != expected * 2 ||
        mapped.direction.size() != expected * 2 || mapped.z_coor.size() != expected ||
        mapped.dim.size() != expected * 3) {
        throw std::runtime_error("Unexpected DXNN output tensor size");
    }
    return mapped;
}

size_t chwIndex(int channel, int y, int x, int height = kHmHeight, int width = kHmWidth) {
    return (static_cast<size_t>(channel) * height + y) * width + x;
}

DetectionsByClass decodeOutputs(const Config& config, const std::vector<CopiedTensor>& outputs) {
    auto tensors = mapOutputs(outputs);
    for (auto& value : tensors.hm_cen) {
        value = sigmoid(value);
    }
    for (auto& value : tensors.cen_offset) {
        value = sigmoid(value);
    }

    struct Candidate {
        float score = 0.0F;
        int cls = 0;
        int y = 0;
        int x = 0;
    };

    std::vector<Candidate> candidates;
    candidates.reserve(static_cast<size_t>(kNumClasses * kHmHeight * kHmWidth));

    for (int cls = 0; cls < kNumClasses; ++cls) {
        for (int y = 0; y < kHmHeight; ++y) {
            for (int x = 0; x < kHmWidth; ++x) {
                const float value = tensors.hm_cen[chwIndex(cls, y, x)];
                bool keep = true;
                for (int dy = -1; dy <= 1 && keep; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        const int yy = y + dy;
                        const int xx = x + dx;
                        if (yy < 0 || yy >= kHmHeight || xx < 0 || xx >= kHmWidth) {
                            continue;
                        }
                        if (tensors.hm_cen[chwIndex(cls, yy, xx)] > value) {
                            keep = false;
                            break;
                        }
                    }
                }
                if (keep) {
                    candidates.push_back({value, cls, y, x});
                }
            }
        }
    }

    const auto by_score_desc = [](const Candidate& lhs, const Candidate& rhs) {
        if (lhs.score != rhs.score) {
            return lhs.score > rhs.score;
        }
        if (lhs.cls != rhs.cls) {
            return lhs.cls < rhs.cls;
        }
        if (lhs.y != rhs.y) {
            return lhs.y < rhs.y;
        }
        return lhs.x < rhs.x;
    };

    if (candidates.size() > static_cast<size_t>(config.top_k)) {
        std::nth_element(candidates.begin(), candidates.begin() + config.top_k, candidates.end(), by_score_desc);
        candidates.resize(static_cast<size_t>(config.top_k));
    }
    std::sort(candidates.begin(), candidates.end(), by_score_desc);

    DetectionsByClass detections;
    for (const auto& candidate : candidates) {
        if (candidate.score <= config.peak_thresh) {
            continue;
        }

        const int y = candidate.y;
        const int x = candidate.x;
        const float center_x = static_cast<float>(x) + tensors.cen_offset[chwIndex(0, y, x)];
        const float center_y = static_cast<float>(y) + tensors.cen_offset[chwIndex(1, y, x)];
        const float dir_sin = tensors.direction[chwIndex(0, y, x)];
        const float dir_cos = tensors.direction[chwIndex(1, y, x)];

        Detection det;
        det.score = candidate.score;
        det.x = center_x * static_cast<float>(kDownRatio);
        det.y = center_y * static_cast<float>(kDownRatio);
        det.z = tensors.z_coor[chwIndex(0, y, x)];
        det.h = tensors.dim[chwIndex(0, y, x)];
        det.w = tensors.dim[chwIndex(1, y, x)] / kBoundSizeY * static_cast<float>(kBevWidth);
        det.l = tensors.dim[chwIndex(2, y, x)] / kBoundSizeX * static_cast<float>(kBevHeight);
        det.yaw = std::atan2(dir_sin, dir_cos);
        detections[static_cast<size_t>(candidate.cls)].push_back(det);
    }

    return detections;
}

std::vector<cv::Point> getCorners(float x, float y, float w, float l, float yaw) {
    const float cos_yaw = std::cos(yaw);
    const float sin_yaw = std::sin(yaw);
    std::vector<cv::Point2f> corners(4);
    corners[0] = {x - w / 2.0F * cos_yaw - l / 2.0F * sin_yaw,
                  y - w / 2.0F * sin_yaw + l / 2.0F * cos_yaw};
    corners[1] = {x - w / 2.0F * cos_yaw + l / 2.0F * sin_yaw,
                  y - w / 2.0F * sin_yaw - l / 2.0F * cos_yaw};
    corners[2] = {x + w / 2.0F * cos_yaw + l / 2.0F * sin_yaw,
                  y + w / 2.0F * sin_yaw - l / 2.0F * cos_yaw};
    corners[3] = {x + w / 2.0F * cos_yaw - l / 2.0F * sin_yaw,
                  y + w / 2.0F * sin_yaw + l / 2.0F * cos_yaw};

    std::vector<cv::Point> result;
    result.reserve(corners.size());
    for (const auto& corner : corners) {
        result.emplace_back(static_cast<int>(corner.x), static_cast<int>(corner.y));
    }
    return result;
}

void blendBBoxLine(cv::Mat& image, cv::Point from, cv::Point to,
                   const cv::Scalar& color, int thickness, double opacity) {
    if (image.empty() || thickness <= 0 || opacity <= 0.0) {
        return;
    }

    const cv::Rect image_rect(0, 0, image.cols, image.rows);
    if (!cv::clipLine(image_rect, from, to)) {
        return;
    }

    const int padding = thickness + 2;
    const int left = std::max(0, std::min(from.x, to.x) - padding);
    const int top = std::max(0, std::min(from.y, to.y) - padding);
    const int right = std::min(image.cols - 1, std::max(from.x, to.x) + padding);
    const int bottom = std::min(image.rows - 1, std::max(from.y, to.y) + padding);
    const cv::Rect bounds(left, top, right - left + 1, bottom - top + 1);
    cv::Mat target = image(bounds);
    cv::Mat overlay = target.clone();
    const cv::Point offset = bounds.tl();
    cv::line(overlay, from - offset, to - offset, color, thickness, cv::LINE_AA);
    cv::addWeighted(overlay, opacity, target, 1.0 - opacity, 0.0, target);
}

void drawBBoxLine(cv::Mat& image, const cv::Point& from, const cv::Point& to,
                  const cv::Scalar& color, int original_thickness = 2) {
    const double scaled_thickness = original_thickness * kBBoxLineWidthScale;
    const int inner_thickness = static_cast<int>(std::floor(scaled_thickness));
    const int outer_thickness = static_cast<int>(std::ceil(scaled_thickness));
    const double outer_opacity = scaled_thickness - inner_thickness;

    if (outer_thickness > inner_thickness && outer_opacity > 0.0) {
        blendBBoxLine(image, from, to, color, outer_thickness, outer_opacity);
    }
    if (inner_thickness > 0) {
        cv::line(image, from, to, color, inner_thickness, cv::LINE_AA);
    }
}

void drawBBoxPolyline(cv::Mat& image, const std::vector<cv::Point>& points,
                      const cv::Scalar& color) {
    if (points.size() < 2) {
        return;
    }
    for (size_t i = 0; i < points.size(); ++i) {
        drawBBoxLine(image, points[i], points[(i + 1) % points.size()], color);
    }
}

void drawRotatedBox(cv::Mat& img, float x, float y, float w, float l, float yaw,
                    const cv::Scalar& color, const cv::Scalar& heading_color) {
    const auto corners = getCorners(x, y, w, l, yaw);
    drawBBoxPolyline(img, corners, color);
    drawBBoxLine(img, corners[0], corners[3], heading_color);
}

void drawPredictions(cv::Mat& img, const DetectionsByClass& detections,
                     const DetectionPalette& palette) {
    for (int cls = 0; cls < kNumClasses; ++cls) {
        for (const auto& det : detections[static_cast<size_t>(cls)]) {
            drawRotatedBox(img, det.x, det.y, det.w, det.l, det.yaw,
                           palette.class_colors[static_cast<size_t>(cls)], palette.heading);
        }
    }
}

std::vector<KittiDetection> convertDetToRealValues(const DetectionsByClass& detections) {
    std::vector<KittiDetection> converted;
    for (int cls = 0; cls < kNumClasses; ++cls) {
        for (const auto& det : detections[static_cast<size_t>(cls)]) {
            KittiDetection kd;
            kd.cls_id = cls;
            const float lidar_yaw = -det.yaw;
            kd.x = det.y / static_cast<float>(kBevHeight) * kBoundSizeX + kMinX;
            kd.y = det.x / static_cast<float>(kBevWidth) * kBoundSizeY + kMinY;
            kd.z = det.z + kMinZ;
            kd.h = det.h;
            kd.w = det.w / static_cast<float>(kBevWidth) * kBoundSizeY;
            kd.l = det.l / static_cast<float>(kBevHeight) * kBoundSizeX;
            kd.yaw = lidar_yaw;
            converted.push_back(kd);
        }
    }
    return converted;
}

struct Calibration {
    std::array<float, 12> p2{};
    std::array<float, 9> r0{};
    std::array<float, 12> v2c{};

    explicit Calibration(const fs::path& calib_path) {
        std::ifstream input(calib_path);
        if (!input) {
            throw std::runtime_error("Failed to open calibration file: " + calib_path.string());
        }

        std::string line;
        while (std::getline(input, line)) {
            std::istringstream iss(line);
            std::string key;
            iss >> key;
            if (!key.empty() && key.back() == ':') {
                key.pop_back();
            }

            if (key == "P2") {
                for (auto& value : p2) {
                    iss >> value;
                }
            } else if (key == "R0_rect") {
                for (auto& value : r0) {
                    iss >> value;
                }
            } else if (key == "Tr_velo_to_cam") {
                for (auto& value : v2c) {
                    iss >> value;
                }
            }
        }
    }
};

std::array<float, 3> lidarToCamera(float x, float y, float z, const Calibration& calib) {
    const float ref_x = calib.v2c[0] * x + calib.v2c[1] * y + calib.v2c[2] * z + calib.v2c[3];
    const float ref_y = calib.v2c[4] * x + calib.v2c[5] * y + calib.v2c[6] * z + calib.v2c[7];
    const float ref_z = calib.v2c[8] * x + calib.v2c[9] * y + calib.v2c[10] * z + calib.v2c[11];

    return {
        calib.r0[0] * ref_x + calib.r0[1] * ref_y + calib.r0[2] * ref_z,
        calib.r0[3] * ref_x + calib.r0[4] * ref_y + calib.r0[5] * ref_z,
        calib.r0[6] * ref_x + calib.r0[7] * ref_y + calib.r0[8] * ref_z,
    };
}

std::vector<KittiDetection> lidarToCameraBox(const std::vector<KittiDetection>& boxes, const Calibration& calib) {
    std::vector<KittiDetection> converted;
    converted.reserve(boxes.size());
    for (auto box : boxes) {
        const auto xyz = lidarToCamera(box.x, box.y, box.z, calib);
        box.x = xyz[0];
        box.y = xyz[1];
        box.z = xyz[2];
        box.yaw = -box.yaw - static_cast<float>(CV_PI) / 2.0F;
        converted.push_back(box);
    }
    return converted;
}

std::array<std::array<float, 3>, 8> computeBox3d(const KittiDetection& box) {
    const float c = std::cos(box.yaw);
    const float s = std::sin(box.yaw);
    const std::array<float, 8> x_corners = {
        box.l / 2.0F, box.l / 2.0F, -box.l / 2.0F, -box.l / 2.0F,
        box.l / 2.0F, box.l / 2.0F, -box.l / 2.0F, -box.l / 2.0F,
    };
    const std::array<float, 8> y_corners = {0.0F, 0.0F, 0.0F, 0.0F, -box.h, -box.h, -box.h, -box.h};
    const std::array<float, 8> z_corners = {
        box.w / 2.0F, -box.w / 2.0F, -box.w / 2.0F, box.w / 2.0F,
        box.w / 2.0F, -box.w / 2.0F, -box.w / 2.0F, box.w / 2.0F,
    };

    std::array<std::array<float, 3>, 8> corners{};
    for (size_t i = 0; i < corners.size(); ++i) {
        corners[i][0] = c * x_corners[i] + s * z_corners[i] + box.x;
        corners[i][1] = y_corners[i] + box.y;
        corners[i][2] = -s * x_corners[i] + c * z_corners[i] + box.z;
    }
    return corners;
}

std::vector<cv::Point> projectToImage(const std::array<std::array<float, 3>, 8>& corners, const Calibration& calib) {
    std::vector<cv::Point> projected;
    projected.reserve(corners.size());
    for (const auto& point : corners) {
        const float u = calib.p2[0] * point[0] + calib.p2[1] * point[1] + calib.p2[2] * point[2] + calib.p2[3];
        const float v = calib.p2[4] * point[0] + calib.p2[5] * point[1] + calib.p2[6] * point[2] + calib.p2[7];
        const float w = calib.p2[8] * point[0] + calib.p2[9] * point[1] + calib.p2[10] * point[2] + calib.p2[11];
        projected.emplace_back(static_cast<int>(u / w), static_cast<int>(v / w));
    }
    return projected;
}

void drawBox3d(cv::Mat& image, const std::vector<cv::Point>& corners, const cv::Scalar& color) {
    static const int edges[12][2] = {
        {0, 1}, {1, 2}, {2, 3}, {3, 0},
        {4, 5}, {5, 6}, {6, 7}, {7, 4},
        {0, 4}, {1, 5}, {2, 6}, {3, 7},
    };

    for (const auto& edge : edges) {
        drawBBoxLine(image, corners[edge[0]], corners[edge[1]], color);
    }
    drawBBoxLine(image, corners[0], corners[5], color, 1);
    drawBBoxLine(image, corners[1], corners[4], color, 1);
}

void showRgbImageWithBoxes(cv::Mat& img_bgr, const std::vector<KittiDetection>& labels,
                           const Calibration& calib, const DetectionPalette& palette) {
    for (const auto& label : labels) {
        if (label.z < 2.0F || label.cls_id < 0) {
            continue;
        }
        const auto corners_3d = computeBox3d(label);
        const auto corners_2d = projectToImage(corners_3d, calib);
        drawBox3d(img_bgr, corners_2d, palette.class_colors[static_cast<size_t>(label.cls_id)]);
    }
}

cv::Mat bevToMat(const BevMap& bevmap) {
    cv::Mat img(bevmap.height, bevmap.width, CV_8UC3);
    for (int y = 0; y < bevmap.height; ++y) {
        for (int x = 0; x < bevmap.width; ++x) {
            auto& pixel = img.at<cv::Vec3b>(y, x);
            for (int c = 0; c < bevmap.channels; ++c) {
                const float value = std::clamp(bevmap.at(c, y, x) * 255.0F, 0.0F, 255.0F);
                pixel[c] = static_cast<uint8_t>(value);
            }
        }
    }
    return img;
}

int uiPixelSize(double scale) {
    return std::max(9, static_cast<int>(std::round(scale * 30.0)));
}

QFont makeUiFont(int pixel_size, int thickness = 1) {
    QFont font(gUiFontFamily);
    font.setPixelSize(std::max(8, pixel_size));
    font.setWeight(thickness >= 2 ? QFont::DemiBold : QFont::Medium);
    font.setStyleStrategy(QFont::PreferAntialias);
    font.setHintingPreference(QFont::PreferFullHinting);
    return font;
}

struct UiTextMetrics {
    int width = 0;
    int height = 0;
    int ascent = 0;
    int descent = 0;
};

UiTextMetrics measureUiText(const std::string& text, int pixel_size, int thickness = 1) {
    const QFontMetrics metrics(makeUiFont(pixel_size, thickness));
    return {metrics.horizontalAdvance(QString::fromStdString(text)), metrics.height(),
            metrics.ascent(), metrics.descent()};
}

QColor qtColor(const cv::Scalar& bgr) {
    return QColor(static_cast<int>(bgr[2]), static_cast<int>(bgr[1]), static_cast<int>(bgr[0]));
}

void drawUiText(cv::Mat& canvas, const std::string& text, cv::Point baseline,
                int pixel_size, const cv::Scalar& color, int thickness = 1,
                bool shadow = false) {
    QImage image(canvas.data, canvas.cols, canvas.rows, static_cast<int>(canvas.step),
                 QImage::Format_BGR888);
    QPainter painter(&image);
    if (!painter.isActive()) {
        throw std::runtime_error("Qt text painter could not render into the OpenCV frame");
    }
    painter.setRenderHint(QPainter::TextAntialiasing, true);
    painter.setFont(makeUiFont(pixel_size, thickness));
    const QString qtext = QString::fromStdString(text);
    if (shadow) {
        painter.setPen(QColor(0, 0, 0, 210));
        painter.drawText(QPoint(baseline.x + 1, baseline.y + 1), qtext);
    }
    painter.setPen(qtColor(color));
    painter.drawText(QPoint(baseline.x, baseline.y), qtext);
}

void writeCredit(cv::Mat& img, cv::Point org_author, cv::Point org_fps, double fps) {
    const int pad_x = 14;
    const int pad_y = 8;
    const int baseline_y = org_author.y;

    const std::string source_text = "SFA3D DXNN Async";
    constexpr int source_font_size = 15;
    const UiTextMetrics source_size = measureUiText(source_text, source_font_size);
    const cv::Rect source_card(org_author.x - pad_x,
                               baseline_y - source_size.ascent - pad_y,
                               source_size.width + pad_x * 2,
                               source_size.height + pad_y * 2);
    cv::rectangle(img, source_card, kUiCard, cv::FILLED);
    cv::rectangle(img, source_card, kUiBorder, 1, cv::LINE_AA);
    cv::line(img, cv::Point(source_card.x, source_card.y),
             cv::Point(source_card.x + source_card.width, source_card.y),
             kUiAccent, 2, cv::LINE_AA);
    drawUiText(img, source_text, cv::Point(org_author.x, baseline_y), source_font_size, kUiText);

    std::ostringstream oss;
    oss << "Inference " << std::fixed << std::setprecision(1) << fps << " FPS";
    const std::string fps_text = oss.str();
    constexpr int fps_font_size = 16;
    const UiTextMetrics fps_size = measureUiText(fps_text, fps_font_size);
    const cv::Rect fps_card(org_fps.x - pad_x,
                            org_fps.y - fps_size.ascent - pad_y,
                            fps_size.width + pad_x * 2,
                            fps_size.height + pad_y * 2);
    cv::rectangle(img, fps_card, kUiCard, cv::FILLED);
    cv::rectangle(img, fps_card, kUiBorder, 1, cv::LINE_AA);
    cv::line(img, cv::Point(fps_card.x, fps_card.y),
             cv::Point(fps_card.x + fps_card.width, fps_card.y),
             kUiAccentDark, 2, cv::LINE_AA);
    drawUiText(img, fps_text, org_fps, fps_font_size, kUiText);
}

void prependTitleBar(cv::Mat& img, bool reserve_exit_button_space) {
    constexpr int title_height = 76;
    constexpr int side_padding = 36;
    constexpr int exit_button_reserve = 64;

    cv::Mat title_bar(title_height, img.cols, img.type(), kUiPanel);
    cv::rectangle(title_bar, cv::Rect(0, 0, img.cols, title_height), kUiBorder, 1, cv::LINE_AA);
    cv::line(title_bar, cv::Point(0, title_height - 2), cv::Point(img.cols, title_height - 2),
             kUiAccent, 2, cv::LINE_AA);

    int font_size = 22;
    const int thickness = 2;
    const int reserved_right = reserve_exit_button_space ? exit_button_reserve : 0;
    const int available_width = std::max(160, img.cols - side_padding * 2 - reserved_right);

    UiTextMetrics text_size = measureUiText(kTitleText, font_size, thickness);
    while (text_size.width > available_width && font_size > 13) {
        --font_size;
        text_size = measureUiText(kTitleText, font_size, thickness);
    }

    const int text_area_right = img.cols - reserved_right;
    const int x = std::max(side_padding, (text_area_right - text_size.width) / 2);
    const int y = (title_height - text_size.height) / 2 + text_size.ascent - 2;
    drawUiText(title_bar, kTitleText, cv::Point(x, y), font_size, kUiText, thickness);

    cv::Mat titled;
    cv::vconcat(title_bar, img, titled);
    img = std::move(titled);
}

int countDetections(const DetectionsByClass& detections) {
    int total = 0;
    for (const auto& class_detections : detections) {
        total += static_cast<int>(class_detections.size());
    }
    return total;
}

float bestScore(const DetectionsByClass& detections) {
    float best = 0.0F;
    for (const auto& class_detections : detections) {
        for (const auto& det : class_detections) {
            best = std::max(best, det.score);
        }
    }
    return best;
}

void drawPanelText(cv::Mat& canvas, const std::string& text, cv::Point org, double scale,
                   const cv::Scalar& color, int thickness = 1) {
    drawUiText(canvas, text, org, uiPixelSize(scale), color, thickness, true);
}

void drawHudSectionTitle(cv::Mat& canvas, const std::string& title, int x, int y, int width) {
    drawPanelText(canvas, title, cv::Point(x, y), 0.46, kUiAccent, 1);
    cv::line(canvas, cv::Point(x, y + 12), cv::Point(x + width, y + 12),
             kUiBorder, 1, cv::LINE_AA);
}

int drawHudMetric(cv::Mat& canvas, const std::string& label, const std::string& value, int x, int y, int width) {
    drawPanelText(canvas, label, cv::Point(x, y), 0.38, kUiTextDim, 1);
    const UiTextMetrics value_size = measureUiText(value, uiPixelSize(0.5));
    drawPanelText(canvas, value, cv::Point(x + width - value_size.width, y), 0.5, kUiText, 1);
    return y + 34;
}

void drawHudPill(cv::Mat& canvas, const std::string& text, cv::Rect rect, const cv::Scalar& accent) {
    cv::rectangle(canvas, rect, kUiCard, cv::FILLED);
    cv::rectangle(canvas, rect, accent, 1, cv::LINE_AA);
    const UiTextMetrics text_size = measureUiText(text, uiPixelSize(0.42));
    const int text_x = rect.x + std::max(6, (rect.width - text_size.width) / 2);
    const int text_y = rect.y + (rect.height + text_size.height) / 2 - 3;
    drawPanelText(canvas, text, cv::Point(text_x, text_y), 0.42, kUiText, 1);
}

void drawDetectionBar(cv::Mat& canvas, int x, int y, int width, const std::string& label,
                      int front_count, int back_count, const cv::Scalar& color) {
    const int total = front_count + back_count;
    const int max_count = std::max(1, total);
    const int bar_width = width - 96;
    const int front_width = static_cast<int>(std::round(static_cast<double>(bar_width) * front_count / max_count));
    const int back_width = static_cast<int>(std::round(static_cast<double>(bar_width) * back_count / max_count));

    drawPanelText(canvas, label, cv::Point(x, y), 0.4, kUiText, 1);
    cv::Rect track(x + 88, y - 12, bar_width, 10);
    cv::rectangle(canvas, track, kUiTrack, cv::FILLED);
    if (front_width > 0) {
        cv::rectangle(canvas, cv::Rect(track.x, track.y, std::min(front_width, track.width), track.height), color, cv::FILLED);
    }
    if (back_width > 0) {
        cv::rectangle(canvas, cv::Rect(track.x, track.y + 12, std::min(back_width, track.width), track.height),
                      color * 0.55, cv::FILLED);
    }

    std::ostringstream counts;
    counts << front_count << "/" << back_count;
    drawPanelText(canvas, counts.str(), cv::Point(x + width - 44, y + 8), 0.36, kUiTextDim, 1);
}

void fillFullscreenSidePanels(cv::Mat& img, const Config& config, const SampleResult& sample, double display_fps) {
    const auto& palette = kDetectionPalettes.at(static_cast<size_t>(config.color_palette_index));
    constexpr double target_aspect = 16.0 / 9.0;
    const int min_panel_width = 280;
    const int target_width = std::max(static_cast<int>(std::round(static_cast<double>(img.rows) * target_aspect)),
                                      img.cols + min_panel_width * 2);
    if (target_width <= img.cols) {
        return;
    }

    cv::Mat canvas(img.rows, target_width, img.type(), kUiBg);

    const int left_panel_width = (target_width - img.cols) / 2;
    const int center_x = left_panel_width;
    img.copyTo(canvas(cv::Rect(center_x, 0, img.cols, img.rows)));

    cv::line(canvas, cv::Point(center_x - 1, 0), cv::Point(center_x - 1, canvas.rows),
             kUiBorder, 1, cv::LINE_AA);
    cv::line(canvas, cv::Point(center_x + img.cols, 0), cv::Point(center_x + img.cols, canvas.rows),
             kUiBorder, 1, cv::LINE_AA);

    const int margin = 32;
    const int panel_width = std::max(180, left_panel_width - margin * 2);
    cv::rectangle(canvas, cv::Rect(margin - 14, 50, panel_width + 28, canvas.rows - 100),
                  kUiPanel, cv::FILLED);
    cv::rectangle(canvas, cv::Rect(margin - 14, 50, panel_width + 28, canvas.rows - 100),
                  kUiBorder, 1, cv::LINE_AA);
    int y = 78;

    drawHudSectionTitle(canvas, "RUN STATUS", margin, y, panel_width);
    y += 44;
    std::ostringstream fps;
    fps << std::fixed << std::setprecision(1) << display_fps;
    y = drawHudMetric(canvas, "Frame", std::to_string(sample.sequence_idx), margin, y, panel_width);
    y = drawHudMetric(canvas, "Sample", std::to_string(sample.dataset_idx), margin, y, panel_width);
    y = drawHudMetric(canvas, "Inference FPS", fps.str(), margin, y, panel_width);
    y = drawHudMetric(canvas, "Input", "608 x 608", margin, y, panel_width);
    y = drawHudMetric(canvas, "Async Queue", std::to_string(kMaxInferenceQueue), margin, y, panel_width);
    y += 18;

    drawHudSectionTitle(canvas, "MODEL", margin, y, panel_width);
    y += 44;
    const std::string model_name = config.pretrained_path.filename().string();
    drawPanelText(canvas, model_name, cv::Point(margin, y), 0.38, kUiText, 1);
    y += 42;
    drawHudPill(canvas, "DXNN", cv::Rect(margin, y, 86, 30), kUiAccent);
    drawHudPill(canvas, config.loop ? "LOOP ON" : "LOOP OFF", cv::Rect(margin + 98, y, 112, 30),
                config.loop ? cv::Scalar(85, 190, 115) : kUiBorder);

    const int right_x = center_x + img.cols + margin;
    const int right_panel_width = std::max(180, canvas.cols - right_x - margin);
    cv::rectangle(canvas, cv::Rect(right_x - 14, 50, right_panel_width + 28, canvas.rows - 100),
                  kUiPanel, cv::FILLED);
    cv::rectangle(canvas, cv::Rect(right_x - 14, 50, right_panel_width + 28, canvas.rows - 100),
                  kUiBorder, 1, cv::LINE_AA);
    int right_y = 78;
    const auto& front = *sample.front_detections;
    const auto& back = *sample.back_detections;
    const std::array<std::string, kNumClasses> class_names = {"Pedestrian", "Car", "Cyclist"};

    drawHudSectionTitle(canvas, "DETECTIONS", right_x, right_y, right_panel_width);
    right_y += 44;
    right_y = drawHudMetric(canvas, "Front", std::to_string(countDetections(front)), right_x, right_y, right_panel_width);
    right_y = drawHudMetric(canvas, "Back", std::to_string(countDetections(back)), right_x, right_y, right_panel_width);
    std::ostringstream conf;
    conf << std::fixed << std::setprecision(2) << std::max(bestScore(front), bestScore(back));
    right_y = drawHudMetric(canvas, "Top Score", conf.str(), right_x, right_y, right_panel_width);
    right_y += 18;

    drawHudSectionTitle(canvas, "CLASS SUMMARY", right_x, right_y, right_panel_width);
    right_y += 42;
    for (int cls = 0; cls < kNumClasses; ++cls) {
        const cv::Scalar& class_color = palette.class_colors[static_cast<size_t>(cls)];
        cv::circle(canvas, cv::Point(right_x + 8, right_y - 7), 5, class_color, cv::FILLED, cv::LINE_AA);
        drawDetectionBar(canvas, right_x + 20, right_y, right_panel_width - 20, class_names[static_cast<size_t>(cls)],
                         static_cast<int>(front[static_cast<size_t>(cls)].size()),
                         static_cast<int>(back[static_cast<size_t>(cls)].size()),
                         class_color);
        right_y += 56;
    }

    right_y += 14;
    drawHudSectionTitle(canvas, "ORIENTATION", right_x, right_y, right_panel_width);
    right_y += 52;
    cv::Point center(right_x + right_panel_width / 2, right_y + 64);
    cv::line(canvas, cv::Point(center.x, center.y + 44), cv::Point(center.x, center.y - 46),
             kUiAccent, 2, cv::LINE_AA);
    cv::line(canvas, cv::Point(center.x, center.y - 46), cv::Point(center.x - 10, center.y - 28),
             kUiAccent, 2, cv::LINE_AA);
    cv::line(canvas, cv::Point(center.x, center.y - 46), cv::Point(center.x + 10, center.y - 28),
             kUiAccent, 2, cv::LINE_AA);
    drawPanelText(canvas, "FRONT", cv::Point(center.x - 34, center.y - 58), 0.42, kUiText, 1);
    drawPanelText(canvas, "BACK", cv::Point(center.x - 26, center.y + 68), 0.42, kUiTextDim, 1);

    img = std::move(canvas);
}

class QtFrameViewer : public QWidget {
public:
    QtFrameViewer(const Config& config, std::atomic<bool>& stop_requested)
        : config_(config),
          stop_requested_(stop_requested) {
        setWindowTitle(kWindowName);
        setAutoFillBackground(false);
        setAttribute(Qt::WA_OpaquePaintEvent, true);
        setAttribute(Qt::WA_NoSystemBackground, true);
        setFocusPolicy(Qt::StrongFocus);
        setMinimumSize(QSize(960, 540));
        setStyleSheet("background-color: #1e1e1e;");

        if (config_.exit_btn) {
            exit_button_ = new QPushButton("X", this);
            exit_button_->setFixedSize(32, 28);
            exit_button_->setFocusPolicy(Qt::NoFocus);
            exit_button_->setToolTip("Exit");
            exit_button_->setStyleSheet(QString(R"(
                QPushButton {
                    color: #f4f4f5;
                    background-color: #2d2d30;
                    border: 1px solid #3f3f46;
                    border-radius: 6px;
                    font-size: 13px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #3a3a3d;
                    color: #ffffff;
                }
                QPushButton:pressed {
                    background-color: #005a9e;
                }
            )"));
            connect(exit_button_, &QPushButton::clicked, this, [this] {
                requestStop();
                close();
            });
            positionExitButton();
            exit_button_->show();
        }
    }

    void showFrame(cv::Mat frame) {
        frame_ = std::move(frame);
        if (!shown_) {
            shown_ = true;
            if (config_.full_screen) {
                showFullScreen();
            } else {
                resize(QSize(frame_.cols, frame_.rows));
                show();
            }
            positionExitButton();
        }
        update();
        QApplication::processEvents(QEventLoop::AllEvents, 1);
        if (!launcher_ready_notified_) {
            notifyLauncherReady();
            launcher_ready_notified_ = true;
        }
    }

    bool isClosed() const {
        return closed_;
    }

protected:
    void paintEvent(QPaintEvent*) override {
        QPainter painter(this);
        painter.setRenderHint(QPainter::Antialiasing, false);
        painter.setRenderHint(QPainter::SmoothPixmapTransform, false);

        if (frame_.empty()) {
            painter.fillRect(rect(), QColor(30, 30, 30));
            return;
        }

        const QImage image(frame_.data, frame_.cols, frame_.rows,
                           static_cast<int>(frame_.step), QImage::Format_BGR888);
        target_rect_ = targetRectFor(image.size());
        if (target_rect_ != rect()) {
            painter.fillRect(rect(), QColor(30, 30, 30));
            painter.fillRect(target_rect_.adjusted(-1, -1, 1, 1), QColor(63, 63, 70));
        }
        painter.drawImage(target_rect_, image);
    }

    void keyPressEvent(QKeyEvent* event) override {
        if (event->key() == Qt::Key_Escape || event->key() == Qt::Key_Q) {
            requestStop();
            close();
            return;
        }
        QWidget::keyPressEvent(event);
    }

    void closeEvent(QCloseEvent* event) override {
        requestStop();
        closed_ = true;
        QWidget::closeEvent(event);
    }

    void resizeEvent(QResizeEvent* event) override {
        QWidget::resizeEvent(event);
        positionExitButton();
    }

private:
    QRect targetRectFor(const QSize& image_size) const {
        QSize scaled = image_size;
        scaled.scale(size(), Qt::KeepAspectRatio);
        return QRect(QPoint((width() - scaled.width()) / 2, (height() - scaled.height()) / 2), scaled);
    }

    void requestStop() {
        stop_requested_.store(true);
        closed_ = true;
    }

    void positionExitButton() {
        if (exit_button_ == nullptr) {
            return;
        }
        constexpr int margin = 18;
        exit_button_->move(std::max(0, width() - exit_button_->width() - margin), margin);
        exit_button_->raise();
    }

    const Config& config_;
    std::atomic<bool>& stop_requested_;
    QPushButton* exit_button_ = nullptr;
    cv::Mat frame_;
    QRect target_rect_;
    bool shown_ = false;
    bool launcher_ready_notified_ = false;
    bool closed_ = false;
};

cv::Mat renderSampleImage(const Config& config, const Calibration& calib, const SampleResult& sample, double display_fps) {
    if (!sample.front_detections || !sample.back_detections) {
        throw std::runtime_error("Cannot render incomplete sample");
    }

    const auto& palette = kDetectionPalettes.at(static_cast<size_t>(config.color_palette_index));

    cv::Mat front_bevmap = bevToMat(sample.front_bevmap);
    cv::resize(front_bevmap, front_bevmap, cv::Size(kBevWidth, kBevHeight));
    drawPredictions(front_bevmap, *sample.front_detections, palette);
    cv::rotate(front_bevmap, front_bevmap, cv::ROTATE_90_COUNTERCLOCKWISE);

    cv::Mat back_bevmap = bevToMat(sample.back_bevmap);
    cv::resize(back_bevmap, back_bevmap, cv::Size(kBevWidth, kBevHeight));
    drawPredictions(back_bevmap, *sample.back_detections, palette);
    cv::rotate(back_bevmap, back_bevmap, cv::ROTATE_90_CLOCKWISE);

    cv::Mat full_bev;
    cv::hconcat(back_bevmap, front_bevmap, full_bev);

    cv::Mat img_bgr;
    cv::cvtColor(sample.img_rgb, img_bgr, cv::COLOR_RGB2BGR);

    auto kitti_dets = convertDetToRealValues(*sample.front_detections);
    if (!kitti_dets.empty()) {
        kitti_dets = lidarToCameraBox(kitti_dets, calib);
        showRgbImageWithBoxes(img_bgr, kitti_dets, calib, palette);
    }
    cv::resize(img_bgr, img_bgr, cv::Size(kBevWidth * 2, 375));

    cv::Mat out_img;
    cv::vconcat(img_bgr, full_bev, out_img);
    writeCredit(out_img, cv::Point(50, 410), cv::Point(900, 410), display_fps);
    prependTitleBar(out_img, config.exit_btn && !config.full_screen);
    if (config.full_screen) {
        fillFullscreenSidePanels(out_img, config, sample, display_fps);
    }

    return out_img;
}

void printDxnnIoInfo(dxrt::InferenceEngine& engine) {
    std::cout << "DXNN inputs:" << std::endl;
    for (const auto& tensor : engine.GetInputs()) {
        std::cout << "  - " << tensor.name() << ": shape=" << shapeToString(tensor.shape())
                  << ", dtype=" << dxrt::DataTypeToString(tensor.type()) << std::endl;
    }
    std::cout << "DXNN outputs:" << std::endl;
    for (const auto& tensor : engine.GetOutputs()) {
        std::cout << "  - " << tensor.name() << ": shape=" << shapeToString(tensor.shape())
                  << ", dtype=" << dxrt::DataTypeToString(tensor.type()) << std::endl;
    }
    std::cout << "Async work queue: " << kMaxInferenceQueue << "\n" << std::endl;
}

void printTimingLog(const Config& config, const SampleResult& sample) {
    if (!config.debug) {
        return;
    }

    const auto& t = sample.timings;
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2)
        << "[Timing] frame=" << sample.sequence_idx
        << " sample=" << sample.dataset_idx
        << " load=" << t.load_ms << "ms"
        << " makeBEV=" << t.make_bev_ms << "ms"
        << " prepareInput=" << t.prepare_input_ms << "ms"
        << " RunAsyncSubmit=" << t.submit_ms << "ms"
        << " callbackCopy=" << t.callback_copy_ms << "ms"
        << " decode=" << t.decode_ms << "ms"
        << " render=" << t.render_ms << "ms"
        << " cpuTotal=" << t.cpuTotalMs() << "ms";
    if (config.timing_cpu_scale != 1.0) {
        oss << " targetCpuEst(x" << config.timing_cpu_scale << ")="
            << (t.cpuTotalMs() * config.timing_cpu_scale) << "ms";
    }
    std::cout << oss.str() << std::endl;
}

class PreprocessWorker {
public:
    PreprocessWorker(const Config& config, const DemoKittiDataset& dataset, InputSpec input_spec,
                     BlockingQueue<PreparedItem>& prepared_queue, std::atomic<bool>& stop_requested)
        : config_(config),
          dataset_(dataset),
          input_spec_(std::move(input_spec)),
          prepared_queue_(prepared_queue),
          stop_requested_(stop_requested) {}

    void operator()() {
        run();
    }

private:
    void run() {
        try {
            int64_t sequence_idx = 0;
            do {
                for (size_t sample_idx = 0; sample_idx < dataset_.size(); ++sample_idx) {
                    if (stop_requested_.load()) {
                        break;
                    }
                    prepared_queue_.push(PreparedItem::sampleItem(prepareSample(sample_idx, sequence_idx++)));
                }
            } while (config_.loop && !stop_requested_.load());
        } catch (const std::exception& exc) {
            prepared_queue_.push(PreparedItem::errorItem(exc.what(), stackMessage(exc)));
        } catch (...) {
            prepared_queue_.push(PreparedItem::errorItem("Unknown preprocess exception", "Unknown preprocess exception"));
        }

        prepared_queue_.push(PreparedItem::stopItem());
    }

    PreparedSample prepareSample(size_t sample_idx, int64_t sequence_idx) const {
        PreparedSample prepared;
        prepared.sample = dataset_.loadBevmapFrontVsBack(sample_idx, sequence_idx);

        Clock::time_point prepare_start;
        if (config_.debug) {
            prepare_start = Clock::now();
        }
        prepared.front_input = prepareDxnnInput(input_spec_, prepared.sample.front_bevmap);
        prepared.back_input = prepareDxnnInput(input_spec_, prepared.sample.back_bevmap);
        if (config_.debug) {
            prepared.sample.timings.prepare_input_ms = elapsedMs(prepare_start);
        }
        return prepared;
    }

    const Config& config_;
    const DemoKittiDataset& dataset_;
    InputSpec input_spec_;
    BlockingQueue<PreparedItem>& prepared_queue_;
    std::atomic<bool>& stop_requested_;
};

class AsyncInferenceWorker {
public:
    AsyncInferenceWorker(const Config& config, const DemoKittiDataset& dataset,
                         BlockingQueue<RenderItem>& render_queue, std::atomic<bool>& stop_requested)
        : config_(config),
          dataset_(dataset),
          render_queue_(render_queue),
          stop_requested_(stop_requested) {}

    void operator()() {
        run();
    }

private:
    void run() {
        try {
            dxrt::InferenceOption option;
            option.devices = {0};
            option.boundOption = dxrt::InferenceOption::BOUND_OPTION::NPU_ALL;
            option.bufferCount = kMaxInferenceQueue;

#if defined(DXRT_NFH_ACCELERATION_AVAILABLE) || defined(DXRT_CPU_OP_ACCELERATION_AVAILABLE)
            auto& config = dxrt::Configuration::GetInstance();
#ifdef DXRT_NFH_ACCELERATION_AVAILABLE
            config.SetEnable(dxrt::Configuration::ITEM::NFH_ACCELERATION, true);
#endif
#ifdef DXRT_CPU_OP_ACCELERATION_AVAILABLE
            config.SetEnable(dxrt::Configuration::ITEM::CPU_OP_ACCELERATION, true);
#endif
#endif

            dxrt::InferenceEngine engine(config_.pretrained_path.string(), option);
            if (config_.debug) {
                std::cout << "Loaded DXNN model from " << config_.pretrained_path << "\n" << std::endl;
                printDxnnIoInfo(engine);
            }
            const InputSpec input_spec = getInputSpec(engine);

            const auto output_names = engine.GetOutputTensorNames();
            engine.RegisterCallback([this, output_names](dxrt::TensorPtrs& outputs, void* user_arg) -> int {
                return callback(outputs, user_arg, output_names);
            });

            BlockingQueue<PreparedItem> prepared_queue(kMaxPreparedQueue);
            PreprocessWorker preprocess_worker(config_, dataset_, input_spec, prepared_queue, stop_requested_);
            std::thread preprocess_thread(std::ref(preprocess_worker));

            try {
                consumePreparedSamples(engine, prepared_queue);
            } catch (...) {
                stop_requested_.store(true);
                prepared_queue.close();
                if (preprocess_thread.joinable()) {
                    preprocess_thread.join();
                }
                engine.RegisterCallback({});
                throw;
            }

            prepared_queue.close();
            if (preprocess_thread.joinable()) {
                preprocess_thread.join();
            }
            engine.RegisterCallback({});
        } catch (const std::exception& exc) {
            render_queue_.push(RenderItem::errorItem(exc.what(), stackMessage(exc)));
        } catch (...) {
            render_queue_.push(RenderItem::errorItem("Unknown worker exception", "Unknown worker exception"));
        }

        render_queue_.push(RenderItem::stopItem());
    }

    int callback(dxrt::TensorPtrs& outputs, void* user_arg, const std::vector<std::string>& output_names) {
        std::unique_ptr<AsyncJob> job(static_cast<AsyncJob*>(user_arg));
        CompletedJob completed;
        completed.completed_at = std::chrono::steady_clock::now();

        try {
            if (!job) {
                throw std::runtime_error("DXRT callback received null user argument");
            }
            completed.sequence_idx = job->sequence_idx;
            completed.side = job->side;
            completed.outputs.reserve(outputs.size());
            Clock::time_point copy_start;
            if (config_.debug) {
                copy_start = Clock::now();
            }
            for (size_t i = 0; i < outputs.size(); ++i) {
                const auto& tensor = outputs[i];
                CopiedTensor copied;
                copied.name = tensor->name();
                if ((copied.name.empty() || copied.name.find('/') != std::string::npos) && i < output_names.size()) {
                    copied.name = output_names[i];
                }
                copied.shape = tensor->shape();
                copied.type = tensor->type();
                const size_t bytes = static_cast<size_t>(tensor->size_in_bytes());
                copied.bytes.resize(bytes);
                if (bytes > 0) {
                    std::memcpy(copied.bytes.data(), tensor->data(), bytes);
                }
                completed.outputs.push_back(std::move(copied));
            }
            if (config_.debug) {
                completed.callback_copy_ms = elapsedMs(copy_start);
            }
        } catch (const std::exception& exc) {
            completed.error = exc.what();
        } catch (...) {
            completed.error = "Unknown DXRT callback exception";
        }

        completion_queue_.push(std::move(completed));
        return completed.error.empty() ? 0 : -1;
    }

    void consumePreparedSamples(dxrt::InferenceEngine& engine, BlockingQueue<PreparedItem>& prepared_queue) {
        bool preprocess_done = false;
        while (!preprocess_done || outstanding_jobs_ > 0) {
            if (!preprocess_done && !stop_requested_.load()) {
                PreparedItem item;
                if (prepared_queue.popFor(item, kPreparedQueuePollInterval)) {
                    if (item.type == PreparedItem::Type::Stop) {
                        preprocess_done = true;
                    } else if (item.type == PreparedItem::Type::Error) {
                        std::cerr << item.detail << std::endl;
                        throw std::runtime_error(item.message);
                    } else {
                        submitPreparedSample(engine, std::move(item.sample));
                    }
                    continue;
                }
            } else {
                preprocess_done = true;
            }

            if (outstanding_jobs_ > 0) {
                processOneCompletion();
            }
        }
    }

    void submitPreparedSample(dxrt::InferenceEngine& engine, PreparedSample prepared) {
        const int64_t sequence_idx = prepared.sample.sequence_idx;
        auto inserted = samples_.emplace(sequence_idx, std::move(prepared.sample));
        auto& stored_sample = inserted.first->second;

        stored_sample.timings.submit_ms +=
            submitJob(engine, stored_sample.sequence_idx, "front", std::move(prepared.front_input));
        stored_sample.timings.submit_ms +=
            submitJob(engine, stored_sample.sequence_idx, "back", std::move(prepared.back_input));
    }

    double submitJob(dxrt::InferenceEngine& engine, int64_t sequence_idx, const std::string& side,
                     std::vector<uint8_t> input_data) {
        while (outstanding_jobs_ >= kMaxInferenceQueue) {
            processOneCompletion();
        }

        auto* job = new AsyncJob{sequence_idx, side, std::move(input_data)};
        try {
            if (!infer_start_time_.has_value()) {
                infer_start_time_ = std::chrono::steady_clock::now();
            }
            Clock::time_point submit_start;
            if (config_.debug) {
                submit_start = Clock::now();
            }
            engine.RunAsync(job->input_data.data(), job);
            double submit_ms = 0.0;
            if (config_.debug) {
                submit_ms = elapsedMs(submit_start);
            }
            ++outstanding_jobs_;
            return submit_ms;
        } catch (...) {
            delete job;
            throw;
        }
    }

    void processOneCompletion() {
        CompletedJob completed;
        if (!completion_queue_.pop(completed)) {
            throw std::runtime_error("Completion queue closed unexpectedly");
        }
        if (!completed.error.empty()) {
            throw std::runtime_error("DXNN async callback failed: " + completed.error);
        }

        auto sample_it = samples_.find(completed.sequence_idx);
        if (sample_it == samples_.end()) {
            throw std::runtime_error("Completed job references unknown sample sequence " +
                                     std::to_string(completed.sequence_idx));
        }
        auto& sample = sample_it->second;
        if (config_.debug) {
            sample.timings.callback_copy_ms += completed.callback_copy_ms;
        }

        DetectionsByClass detections;
        if (config_.debug) {
            const auto decode_start = Clock::now();
            detections = decodeOutputs(config_, completed.outputs);
            sample.timings.decode_ms += elapsedMs(decode_start);
        } else {
            detections = decodeOutputs(config_, completed.outputs);
        }
        ++completed_inferences_;

        const auto start = infer_start_time_.value_or(completed.completed_at);
        const std::chrono::duration<double> elapsed = completed.completed_at - start;
        const double seconds = std::max(elapsed.count(), 1e-6);
        const double inference_fps = static_cast<double>(completed_inferences_) / seconds;
        sample.inference_fps = inference_fps;

        if (completed.side == "front") {
            sample.front_detections = std::move(detections);
        } else {
            sample.back_detections = std::move(detections);
        }

        --outstanding_jobs_;

        if (sample.isComplete()) {
            render_queue_.push(RenderItem::sampleItem(std::move(sample)));
            samples_.erase(sample_it);
        }
    }

    const Config& config_;
    const DemoKittiDataset& dataset_;
    BlockingQueue<RenderItem>& render_queue_;
    std::atomic<bool>& stop_requested_;
    BlockingQueue<CompletedJob> completion_queue_;
    std::map<int64_t, SampleResult> samples_;
    int outstanding_jobs_ = 0;
    std::optional<std::chrono::steady_clock::time_point> infer_start_time_;
    int completed_inferences_ = 0;
};

void renderLoop(const Config& config, BlockingQueue<RenderItem>& render_queue, std::atomic<bool>& stop_requested) {
    int64_t next_sequence_idx = 0;
    std::map<int64_t, SampleResult> pending_samples;
    bool worker_done = false;
    const Calibration calib(config.calib_path);
    QtFrameViewer viewer(config, stop_requested);

    try {
        while (true) {
            if (!worker_done && !stop_requested.load()) {
                RenderItem item;
                if (!render_queue.pop(item)) {
                    worker_done = true;
                } else if (item.type == RenderItem::Type::Stop) {
                    worker_done = true;
                } else if (item.type == RenderItem::Type::Error) {
                    std::cerr << item.detail << std::endl;
                    throw std::runtime_error(item.message);
                } else {
                    pending_samples[item.sample.sequence_idx] = std::move(item.sample);
                }
            }

            while (!stop_requested.load()) {
                auto sample_it = pending_samples.find(next_sequence_idx);
                if (sample_it == pending_samples.end()) {
                    break;
                }

                Clock::time_point render_start;
                if (config.debug) {
                    render_start = Clock::now();
                }
                cv::Mat out_img = renderSampleImage(config, calib, sample_it->second, sample_it->second.inference_fps);

                viewer.showFrame(std::move(out_img));
                if (viewer.isClosed()) {
                    stop_requested.store(true);
                }
                if (config.debug) {
                    sample_it->second.timings.render_ms = elapsedMs(render_start);
                    printTimingLog(config, sample_it->second);
                }

                pending_samples.erase(sample_it);
                ++next_sequence_idx;
            }

            if (stop_requested.load()) {
                break;
            }
            if (worker_done && pending_samples.empty()) {
                break;
            }
        }
    } catch (...) {
        stop_requested.store(true);
        throw;
    }

    stop_requested.store(stop_requested.load());
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        Config config = parseArgs(argc, argv);
        std::cout << "3D BBox palette: "
                  << kDetectionPalettes.at(static_cast<size_t>(config.color_palette_index)).name
                  << std::endl;
        int qt_argc = 1;
        QApplication app(qt_argc, argv);
        QApplication::setApplicationName("SFA3D DXNN Demo");
        gUiFontFamily = resolveFontFamily(config.font_preset_index);
        QApplication::setFont(QFont(gUiFontFamily));
        std::cout << "UI font: "
                  << kFontPresets.at(static_cast<size_t>(config.font_preset_index)).name
                  << " (resolved family: " << gUiFontFamily.toStdString() << ")" << std::endl;

        downloadAndUnzip(config);

        if (!fs::is_regular_file(config.pretrained_path)) {
            throw std::runtime_error("No file at " + config.pretrained_path.string());
        }

        DemoKittiDataset demo_dataset(config);
        BlockingQueue<RenderItem> render_queue;
        std::atomic<bool> stop_requested{false};

        AsyncInferenceWorker worker(config, demo_dataset, render_queue, stop_requested);
        std::thread inference_thread(std::ref(worker));

        try {
            renderLoop(config, render_queue, stop_requested);
        } catch (...) {
            stop_requested.store(true);
            render_queue.close();
            inference_thread.join();
            throw;
        }

        stop_requested.store(true);
        inference_thread.join();
        return 0;
    } catch (const cxxopts::exceptions::exception& exc) {
        std::cerr << "Error parsing arguments: " << exc.what() << std::endl;
        return 1;
    } catch (const dxrt::Exception& exc) {
        std::cerr << "dxrt::Exception: " << exc.what() << std::endl;
        return 1;
    } catch (const std::exception& exc) {
        std::cerr << "Error: " << exc.what() << std::endl;
        return 1;
    }
}
