#include <QtCore/QCoreApplication>
#include <QtCore/QMetaObject>
#include <QtCore/QString>
#include <QtGui/QColor>
#include <QtGui/QFont>
#include <QtGui/QLinearGradient>
#include <QtGui/QImage>
#include <QtGui/QKeyEvent>
#include <QtGui/QMouseEvent>
#include <QtGui/QPainter>
#include <QtGui/QPen>
#include <QtWidgets/QApplication>
#include <QtWidgets/QWidget>

#include <dxrt/dxrt_api.h>
#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace {

constexpr const char* kDefaultPalmModelPath = "assets/hand-detector_192x192.dxnn";
constexpr const char* kDefaultLandmarkModelPath = "assets/HandLandmarkLite.dxnn";
constexpr const char* kDefaultPoseModelPath = "assets/yolo26s-pose.dxnn";
constexpr int kPalmWidth = 192;
constexpr int kPalmHeight = 192;
constexpr int kPalmChannels = 3;
constexpr int kPalmAnchorCount = 2016;
constexpr int kPalmBoxCoords = 18;
constexpr int kPalmKeypointCount = 7;
constexpr int kLandmarkWidth = 224;
constexpr int kLandmarkHeight = 224;
constexpr int kLandmarkChannels = 3;
constexpr int kHandLandmarkCount = 21;
constexpr int kCoordsPerLandmark = 3;
constexpr float kDefaultPalmConfidence = 0.2f;
constexpr float kDefaultLandmarkConfidence = 0.5f;
constexpr float kDefaultPoseConfidence = 0.3f;
constexpr float kDefaultNmsThreshold = 0.3f;
constexpr float kDefaultPoseNmsThreshold = 0.45f;
constexpr int kDefaultMaxHands = 8;
constexpr int kDefaultCameraWidth = 640;
constexpr int kDefaultCameraHeight = 480;
constexpr int kDefaultCameraFps = 30;
constexpr float kPi = 3.14159265358979323846f;
constexpr int kWristCenterKeypoint = 0;
constexpr int kMiddleFingerMcpKeypoint = 2;
constexpr float kHandRoiTargetAngle = kPi * 0.5f;
constexpr float kHandRoiScaleX = 2.6f;
constexpr float kHandRoiScaleY = 2.6f;
constexpr float kHandRoiShiftX = 0.0f;
constexpr float kHandRoiShiftY = -0.5f;
constexpr std::size_t kFrameQueueDepth = 2;
constexpr std::size_t kRenderQueueDepth = 3;
constexpr std::uint64_t kResultHistoryFrames = 90;
constexpr int kQueuePollMs = 20;
constexpr int kRenderResultWaitMs = 6;

struct Options {
    std::string palm_model_path = kDefaultPalmModelPath;
    std::string landmark_model_path = kDefaultLandmarkModelPath;
    std::string pose_model_path = kDefaultPoseModelPath;
    bool use_camera = true;
    int camera_index = 0;
    std::string video_path;
    int camera_width = kDefaultCameraWidth;
    int camera_height = kDefaultCameraHeight;
    int camera_fps = kDefaultCameraFps;
    int max_hands = kDefaultMaxHands;
    float palm_confidence = kDefaultPalmConfidence;
    float landmark_confidence = kDefaultLandmarkConfidence;
    float pose_confidence = kDefaultPoseConfidence;
    float nms_threshold = kDefaultNmsThreshold;
    float pose_nms_threshold = kDefaultPoseNmsThreshold;
    bool fullscreen = true;
    bool loop_video = false;
    bool keep_aspect = true;
    bool show_palm_overlay = true;
    bool show_pose_overlay = true;
    bool save_output = false;
    bool show_exit_button = false;
};

struct Anchor {
    float x_center = 0.0f;
    float y_center = 0.0f;
    float w = 1.0f;
    float h = 1.0f;
};

struct PreprocessInfo {
    float scale_x = 1.0f;
    float scale_y = 1.0f;
    float pad_x = 0.0f;
    float pad_y = 0.0f;
};

struct PosePreprocessInfo {
    int pad_x = 0;
    int pad_y = 0;
    float scale = 1.0f;
    int original_width = 0;
    int original_height = 0;
    int input_width = 0;
    int input_height = 0;
};

struct PoseKeypoint {
    float x = 0.0f;
    float y = 0.0f;
    float confidence = 0.0f;
};

struct PoseResult {
    std::vector<float> box;
    float confidence = 0.0f;
    std::vector<PoseKeypoint> keypoints;
};

struct NormalizedRect {
    float x_center = 0.0f;
    float y_center = 0.0f;
    float width = 0.0f;
    float height = 0.0f;
    float rotation = 0.0f;
};

struct PalmCandidate {
    cv::Rect2f box;
    std::array<cv::Point2f, kPalmKeypointCount> keypoints{};
    float score = 0.0f;
};

struct PalmDetection {
    cv::Rect2f box;
    std::array<cv::Point2f, kPalmKeypointCount> keypoints{};
    NormalizedRect hand_roi;
    float score = 0.0f;
};

struct LandmarkPoint {
    cv::Point2f image;
    float z = 0.0f;
};

struct HandLandmarkResult {
    PalmDetection palm;
    std::array<LandmarkPoint, kHandLandmarkCount> landmarks{};
    std::array<cv::Point3f, kHandLandmarkCount> world_landmarks{};
    std::string handedness = "Unknown";
    float handedness_score = 0.0f;
    float confidence = 0.0f;
};

struct LandmarkCrop {
    cv::Mat crop_to_frame;
};

struct LandmarkCollector {
    std::mutex mutex;
    std::condition_variable cv;
    std::vector<HandLandmarkResult> results;
    int expected = 0;
    int completed = 0;
};

struct LandmarkJob {
    std::shared_ptr<std::vector<std::uint8_t>> input;
    PalmDetection palm;
    LandmarkCrop crop;
    LandmarkCollector* collector = nullptr;
    float confidence_threshold = kDefaultLandmarkConfidence;
};

struct PoseCollector {
    std::mutex mutex;
    std::condition_variable cv;
    std::vector<PoseResult> results;
    bool completed = false;
    double inference_ms = 0.0;
};

struct PoseJob {
    std::shared_ptr<std::vector<std::uint8_t>> input;
    PosePreprocessInfo prep;
    PoseCollector* collector = nullptr;
    float score_threshold = kDefaultPoseConfidence;
    float nms_threshold = kDefaultPoseNmsThreshold;
    int num_keypoints = 17;
    std::chrono::steady_clock::time_point submit_time;
};

using FrameId = std::uint64_t;

struct FramePacket {
    FrameId id = 0;
    cv::Mat frame;
    std::chrono::steady_clock::time_point captured_at;
};

struct PosePacket {
    FrameId frame_id = 0;
    std::vector<PoseResult> poses;
    double inference_ms = 0.0;
};

struct HandPacket {
    FrameId frame_id = 0;
    std::vector<PalmDetection> palms;
    std::vector<HandLandmarkResult> hands;
};

struct ResultBundle {
    bool has_pose = false;
    bool pose_exact = false;
    PosePacket pose;
    bool has_hand = false;
    bool hand_exact = false;
    HandPacket hand;
};

template <typename T>
class BoundedQueue {
public:
    explicit BoundedQueue(std::size_t max_size)
        : max_size_(std::max<std::size_t>(1, max_size)) {}

    bool push(T item) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (closed_) return false;
            while (items_.size() >= max_size_) {
                items_.pop_front();
            }
            items_.push_back(std::move(item));
        }
        cv_.notify_one();
        return true;
    }

    bool pop_for(T* out, std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!cv_.wait_for(lock, timeout, [this]() { return closed_ || !items_.empty(); })) {
            return false;
        }
        if (items_.empty()) return false;
        *out = std::move(items_.front());
        items_.pop_front();
        return true;
    }

    void close() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            closed_ = true;
        }
        cv_.notify_all();
    }

    bool closed() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return closed_;
    }

private:
    const std::size_t max_size_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<T> items_;
    bool closed_ = false;
};

struct PipelineQueues {
    BoundedQueue<FramePacket> capture;
    BoundedQueue<FramePacket> pose;
    BoundedQueue<FramePacket> hand;
    BoundedQueue<FramePacket> render;

    PipelineQueues()
        : capture(kFrameQueueDepth),
          pose(kFrameQueueDepth),
          hand(kFrameQueueDepth),
          render(kRenderQueueDepth) {}

    void close_all() {
        capture.close();
        pose.close();
        hand.close();
        render.close();
    }
};

class ResultStore {
public:
    void set_pose(PosePacket packet) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            latest_pose_ = packet;
            has_latest_pose_ = true;
            pose_results_[packet.frame_id] = std::move(packet);
        }
        cv_.notify_all();
    }

    void set_hand(HandPacket packet) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            latest_hand_ = packet;
            has_latest_hand_ = true;
            hand_results_[packet.frame_id] = std::move(packet);
        }
        cv_.notify_all();
    }

    ResultBundle wait_for(FrameId frame_id,
                          bool need_pose,
                          bool need_hand,
                          std::chrono::milliseconds timeout) {
        const auto deadline = std::chrono::steady_clock::now() + timeout;
        std::unique_lock<std::mutex> lock(mutex_);
        auto ready = [&]() {
            return (!need_pose || pose_results_.find(frame_id) != pose_results_.end()) &&
                   (!need_hand || hand_results_.find(frame_id) != hand_results_.end());
        };
        if (!ready()) {
            cv_.wait_until(lock, deadline, ready);
        }

        ResultBundle bundle;
        const auto pose_it = pose_results_.find(frame_id);
        if (pose_it != pose_results_.end()) {
            bundle.pose = pose_it->second;
            bundle.has_pose = true;
            bundle.pose_exact = true;
        } else if (has_latest_pose_ && latest_pose_.frame_id <= frame_id) {
            bundle.pose = latest_pose_;
            bundle.has_pose = true;
        }

        const auto hand_it = hand_results_.find(frame_id);
        if (hand_it != hand_results_.end()) {
            bundle.hand = hand_it->second;
            bundle.has_hand = true;
            bundle.hand_exact = true;
        } else if (has_latest_hand_ && latest_hand_.frame_id <= frame_id) {
            bundle.hand = latest_hand_;
            bundle.has_hand = true;
        }
        return bundle;
    }

    void prune_before(FrameId min_frame_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto it = pose_results_.begin(); it != pose_results_.end();) {
            if (it->first < min_frame_id) {
                it = pose_results_.erase(it);
            } else {
                ++it;
            }
        }
        for (auto it = hand_results_.begin(); it != hand_results_.end();) {
            if (it->first < min_frame_id) {
                it = hand_results_.erase(it);
            } else {
                ++it;
            }
        }
    }

private:
    std::mutex mutex_;
    std::condition_variable cv_;
    std::unordered_map<FrameId, PosePacket> pose_results_;
    std::unordered_map<FrameId, HandPacket> hand_results_;
    PosePacket latest_pose_;
    HandPacket latest_hand_;
    bool has_latest_pose_ = false;
    bool has_latest_hand_ = false;
};

bool file_exists(const std::string& path) {
    std::ifstream in(path.c_str(), std::ios::binary);
    return static_cast<bool>(in);
}

bool parse_int(const std::string& value, int* out) {
    try {
        std::size_t used = 0;
        const int parsed = std::stoi(value, &used);
        if (used != value.size()) return false;
        *out = parsed;
        return true;
    } catch (...) {
        return false;
    }
}

bool parse_float(const std::string& value, float* out) {
    try {
        std::size_t used = 0;
        const float parsed = std::stof(value, &used);
        if (used != value.size()) return false;
        *out = parsed;
        return true;
    } catch (...) {
        return false;
    }
}

std::string next_output_video_path() {
    for (int index = 0; index < 1000; ++index) {
        std::ostringstream path;
        path << "output-" << std::setw(2) << std::setfill('0') << index << ".mp4";
        if (!file_exists(path.str())) return path.str();
    }
    throw std::runtime_error("failed to find available output-XX.mp4 filename");
}

double select_output_fps(const Options& options, double source_fps) {
    if (source_fps > 1.0 && source_fps < 240.0) return source_fps;
    if (options.use_camera && options.camera_fps > 0) return static_cast<double>(options.camera_fps);
    return static_cast<double>(kDefaultCameraFps);
}

void print_usage(const char* argv0) {
    std::cout
        << "Usage: " << argv0 << " [OPTIONS]\n"
        << "  --palm-model <PATH>         Palm detector .dxnn (default: "
        << kDefaultPalmModelPath << ")\n"
        << "  --landmark-model <PATH>     Hand landmark .dxnn (default: "
        << kDefaultLandmarkModelPath << ")\n"
        << "  --pose-model <PATH>         YOLO pose .dxnn (default: "
        << kDefaultPoseModelPath << ")\n"
        << "  -c, --camera [INDEX]        Use webcam (default index: 0)\n"
        << "  -v, --video <PATH>          Use video file input\n"
        << "      --max-hands <N>         Max hands per frame (default: 8)\n"
        << "      --width <N>             Requested webcam width (default: 640)\n"
        << "      --height <N>            Requested webcam height (default: 480)\n"
        << "      --fps <N>               Requested webcam FPS (default: 30)\n"
        << "      --palm-conf <FLOAT>     Palm confidence threshold (default: 0.2)\n"
        << "      --landmark-conf <FLOAT> Landmark presence threshold (default: 0.5)\n"
        << "      --pose-conf <FLOAT>     Pose confidence threshold (default: 0.3)\n"
        << "      --nms <FLOAT>           Palm weighted NMS IoU threshold (default: 0.3)\n"
        << "      --pose-nms <FLOAT>      Pose NMS IoU threshold (default: 0.45)\n"
        << "      --loop                  Loop video input\n"
        << "      --stretch               Resize palm input without letterbox\n"
        << "      --show-palm             Draw palm detector bbox/ROI overlay (default)\n"
        << "      --hide-palm             Hide palm detector overlay\n"
        << "      --landmark-only         Same as --hide-palm\n"
        << "      --show-pose             Draw pose estimation overlay (default)\n"
        << "      --hide-pose             Hide pose estimation overlay\n"
        << "      --exit-btn              Show a clickable exit button at the top-right\n"
        << "  -s, --save                  Save rendered result video as output-00.mp4, output-01.mp4, ...\n"
        << "      --windowed              Show a window instead of fullscreen\n"
        << "      --fullscreen            Force fullscreen (default)\n"
        << "  -h, --help                  Show this help\n";
}

bool parse_args(int argc, char** argv, Options* options) {
    for (int i = 1; i < argc;) {
        const std::string arg(argv[i++]);
        auto require_value = [&](const char* name) -> const char* {
            if (i >= argc) {
                std::cerr << "Error: missing value for " << name << std::endl;
                return nullptr;
            }
            return argv[i++];
        };

        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (arg == "--palm-model") {
            const char* value = require_value(arg.c_str());
            if (value == nullptr) return false;
            options->palm_model_path = value;
        } else if (arg == "--landmark-model") {
            const char* value = require_value(arg.c_str());
            if (value == nullptr) return false;
            options->landmark_model_path = value;
        } else if (arg == "--pose-model") {
            const char* value = require_value(arg.c_str());
            if (value == nullptr) return false;
            options->pose_model_path = value;
        } else if (arg == "-c" || arg == "--camera") {
            options->use_camera = true;
            options->video_path.clear();
            if (i < argc && argv[i][0] != '-') {
                int parsed = 0;
                if (!parse_int(argv[i], &parsed) || parsed < 0) {
                    std::cerr << "Error: " << arg << " expects a non-negative camera index" << std::endl;
                    return false;
                }
                options->camera_index = parsed;
                ++i;
            }
        } else if (arg == "-v" || arg == "--video") {
            const char* value = require_value(arg.c_str());
            if (value == nullptr) return false;
            options->use_camera = false;
            options->video_path = value;
        } else if (arg == "--max-hands") {
            const char* value = require_value(arg.c_str());
            if (value == nullptr || !parse_int(value, &options->max_hands) ||
                options->max_hands <= 0) {
                std::cerr << "Error: --max-hands expects a positive integer" << std::endl;
                return false;
            }
        } else if (arg == "--width") {
            const char* value = require_value(arg.c_str());
            if (value == nullptr || !parse_int(value, &options->camera_width) ||
                options->camera_width <= 0) {
                std::cerr << "Error: --width expects a positive integer" << std::endl;
                return false;
            }
        } else if (arg == "--height") {
            const char* value = require_value(arg.c_str());
            if (value == nullptr || !parse_int(value, &options->camera_height) ||
                options->camera_height <= 0) {
                std::cerr << "Error: --height expects a positive integer" << std::endl;
                return false;
            }
        } else if (arg == "--fps") {
            const char* value = require_value(arg.c_str());
            if (value == nullptr || !parse_int(value, &options->camera_fps) ||
                options->camera_fps <= 0) {
                std::cerr << "Error: --fps expects a positive integer" << std::endl;
                return false;
            }
        } else if (arg == "--palm-conf") {
            const char* value = require_value(arg.c_str());
            if (value == nullptr || !parse_float(value, &options->palm_confidence) ||
                options->palm_confidence < 0.0f || options->palm_confidence > 1.0f) {
                std::cerr << "Error: --palm-conf expects a float in [0, 1]" << std::endl;
                return false;
            }
        } else if (arg == "--landmark-conf") {
            const char* value = require_value(arg.c_str());
            if (value == nullptr || !parse_float(value, &options->landmark_confidence) ||
                options->landmark_confidence < 0.0f || options->landmark_confidence > 1.0f) {
                std::cerr << "Error: --landmark-conf expects a float in [0, 1]" << std::endl;
                return false;
            }
        } else if (arg == "--pose-conf") {
            const char* value = require_value(arg.c_str());
            if (value == nullptr || !parse_float(value, &options->pose_confidence) ||
                options->pose_confidence < 0.0f || options->pose_confidence > 1.0f) {
                std::cerr << "Error: --pose-conf expects a float in [0, 1]" << std::endl;
                return false;
            }
        } else if (arg == "--nms") {
            const char* value = require_value(arg.c_str());
            if (value == nullptr || !parse_float(value, &options->nms_threshold) ||
                options->nms_threshold < 0.0f || options->nms_threshold > 1.0f) {
                std::cerr << "Error: --nms expects a float in [0, 1]" << std::endl;
                return false;
            }
        } else if (arg == "--pose-nms") {
            const char* value = require_value(arg.c_str());
            if (value == nullptr || !parse_float(value, &options->pose_nms_threshold) ||
                options->pose_nms_threshold < 0.0f || options->pose_nms_threshold > 1.0f) {
                std::cerr << "Error: --pose-nms expects a float in [0, 1]" << std::endl;
                return false;
            }
        } else if (arg == "--loop") {
            options->loop_video = true;
        } else if (arg == "--stretch") {
            options->keep_aspect = false;
        } else if (arg == "--show-palm") {
            options->show_palm_overlay = true;
        } else if (arg == "--hide-palm" || arg == "--landmark-only") {
            options->show_palm_overlay = false;
        } else if (arg == "--show-pose") {
            options->show_pose_overlay = true;
        } else if (arg == "--hide-pose") {
            options->show_pose_overlay = false;
        } else if (arg == "--exit-btn") {
            options->show_exit_button = true;
        } else if (arg == "-s" || arg == "--save") {
            options->save_output = true;
        } else if (arg == "--windowed") {
            options->fullscreen = false;
        } else if (arg == "--fullscreen") {
            options->fullscreen = true;
        } else {
            std::cerr << "Error: unknown option '" << arg << "'\n";
            print_usage(argv[0]);
            return false;
        }
    }
    return true;
}

void notify_launcher_ready() {
    const char* path = std::getenv("DX_LAUNCHER_READY_FILE");
    if (path == nullptr || *path == '\0') return;
    std::ofstream out(path, std::ios::app);
    if (out) out << "ready\n";
}

float sigmoid(float value) {
    value = std::max(-100.0f, std::min(100.0f, value));
    return 1.0f / (1.0f + std::exp(-value));
}

float clamp_float(float value, float low, float high) {
    return std::max(low, std::min(high, value));
}

float normalize_radians(float angle) {
    return angle - 2.0f * kPi * std::floor((angle + kPi) / (2.0f * kPi));
}

std::vector<Anchor> generate_palm_anchors() {
    constexpr int kNumLayers = 4;
    constexpr float kMinScale = 0.1484375f;
    constexpr float kMaxScale = 0.75f;
    constexpr float kAnchorOffsetX = 0.5f;
    constexpr float kAnchorOffsetY = 0.5f;
    constexpr float kInterpolatedScaleAspectRatio = 1.0f;
    const int strides[kNumLayers] = {8, 16, 16, 16};

    auto calculate_scale = [&](int stride_index) {
        if (kNumLayers == 1) return (kMinScale + kMaxScale) * 0.5f;
        return kMinScale +
               (kMaxScale - kMinScale) * static_cast<float>(stride_index) /
                   static_cast<float>(kNumLayers - 1);
    };

    std::vector<Anchor> anchors;
    int layer_id = 0;
    while (layer_id < kNumLayers) {
        std::vector<float> anchor_widths;
        std::vector<float> anchor_heights;
        std::vector<float> aspect_ratios;
        std::vector<float> scales;
        int last_same_stride_layer = layer_id;

        while (last_same_stride_layer < kNumLayers &&
               strides[last_same_stride_layer] == strides[layer_id]) {
            const float scale = calculate_scale(last_same_stride_layer);
            aspect_ratios.push_back(1.0f);
            scales.push_back(scale);
            const float next_scale =
                last_same_stride_layer == kNumLayers - 1
                    ? 1.0f
                    : calculate_scale(last_same_stride_layer + 1);
            aspect_ratios.push_back(kInterpolatedScaleAspectRatio);
            scales.push_back(std::sqrt(scale * next_scale));
            ++last_same_stride_layer;
        }

        for (std::size_t i = 0; i < aspect_ratios.size(); ++i) {
            const float ratio_sqrt = std::sqrt(aspect_ratios[i]);
            anchor_heights.push_back(scales[i] / ratio_sqrt);
            anchor_widths.push_back(scales[i] * ratio_sqrt);
        }

        const int feature_map_height =
            static_cast<int>(std::ceil(static_cast<float>(kPalmHeight) / strides[layer_id]));
        const int feature_map_width =
            static_cast<int>(std::ceil(static_cast<float>(kPalmWidth) / strides[layer_id]));

        for (int y = 0; y < feature_map_height; ++y) {
            for (int x = 0; x < feature_map_width; ++x) {
                for (std::size_t anchor_id = 0; anchor_id < anchor_widths.size(); ++anchor_id) {
                    (void)anchor_id;
                    anchors.push_back(Anchor{
                        (static_cast<float>(x) + kAnchorOffsetX) /
                            static_cast<float>(feature_map_width),
                        (static_cast<float>(y) + kAnchorOffsetY) /
                            static_cast<float>(feature_map_height),
                        1.0f,
                        1.0f,
                    });
                }
            }
        }
        layer_id = last_same_stride_layer;
    }
    return anchors;
}

float intersection_area(const cv::Rect2f& a, const cv::Rect2f& b) {
    const float x1 = std::max(a.x, b.x);
    const float y1 = std::max(a.y, b.y);
    const float x2 = std::min(a.x + a.width, b.x + b.width);
    const float y2 = std::min(a.y + a.height, b.y + b.height);
    return std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
}

float intersection_over_union(const cv::Rect2f& a, const cv::Rect2f& b) {
    const float inter = intersection_area(a, b);
    const float area_a = std::max(0.0f, a.width) * std::max(0.0f, a.height);
    const float area_b = std::max(0.0f, b.width) * std::max(0.0f, b.height);
    const float denom = area_a + area_b - inter;
    return denom > 0.0f ? inter / denom : 0.0f;
}

PalmCandidate weighted_candidate(const std::vector<PalmCandidate>& detections,
                                 const std::vector<int>& cluster_indices) {
    const PalmCandidate& top = detections[cluster_indices.front()];
    PalmCandidate out = top;
    float total_score = 0.0f;
    float xmin = 0.0f;
    float ymin = 0.0f;
    float xmax = 0.0f;
    float ymax = 0.0f;
    std::array<cv::Point2f, kPalmKeypointCount> keypoints{};

    for (int idx : cluster_indices) {
        const PalmCandidate& candidate = detections[idx];
        const float weight = candidate.score;
        total_score += weight;
        xmin += candidate.box.x * weight;
        ymin += candidate.box.y * weight;
        xmax += (candidate.box.x + candidate.box.width) * weight;
        ymax += (candidate.box.y + candidate.box.height) * weight;
        for (int k = 0; k < kPalmKeypointCount; ++k) {
            keypoints[k].x += candidate.keypoints[k].x * weight;
            keypoints[k].y += candidate.keypoints[k].y * weight;
        }
    }

    if (total_score <= 0.0f) return out;
    xmin /= total_score;
    ymin /= total_score;
    xmax /= total_score;
    ymax /= total_score;
    out.box = cv::Rect2f(xmin, ymin, std::max(0.0f, xmax - xmin),
                         std::max(0.0f, ymax - ymin));
    for (int k = 0; k < kPalmKeypointCount; ++k) {
        out.keypoints[k].x = keypoints[k].x / total_score;
        out.keypoints[k].y = keypoints[k].y / total_score;
    }
    return out;
}

std::vector<PalmCandidate> weighted_non_max_suppression(std::vector<PalmCandidate> detections,
                                                        float threshold,
                                                        int max_detections) {
    std::sort(detections.begin(), detections.end(), [](const PalmCandidate& a,
                                                       const PalmCandidate& b) {
        return a.score > b.score;
    });

    std::vector<int> remaining;
    remaining.reserve(detections.size());
    for (int i = 0; i < static_cast<int>(detections.size()); ++i) remaining.push_back(i);

    std::vector<PalmCandidate> selected;
    selected.reserve(static_cast<std::size_t>(max_detections));
    while (!remaining.empty() && static_cast<int>(selected.size()) < max_detections) {
        const int top_idx = remaining.front();
        std::vector<int> cluster;
        std::vector<int> next_remaining;
        for (int idx : remaining) {
            if (intersection_over_union(detections[idx].box, detections[top_idx].box) >
                threshold) {
                cluster.push_back(idx);
            } else {
                next_remaining.push_back(idx);
            }
        }
        selected.push_back(weighted_candidate(detections, cluster));
        remaining = std::move(next_remaining);
    }
    return selected;
}

std::vector<std::uint8_t> preprocess_palm_frame(const cv::Mat& bgr,
                                               bool keep_aspect,
                                               PreprocessInfo* info) {
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    cv::Mat input(kPalmHeight, kPalmWidth, CV_8UC3, cv::Scalar(0, 0, 0));

    if (keep_aspect) {
        const float scale = std::min(static_cast<float>(kPalmWidth) / rgb.cols,
                                     static_cast<float>(kPalmHeight) / rgb.rows);
        const int resized_w = std::max(1, static_cast<int>(std::round(rgb.cols * scale)));
        const int resized_h = std::max(1, static_cast<int>(std::round(rgb.rows * scale)));
        const int pad_x = (kPalmWidth - resized_w) / 2;
        const int pad_y = (kPalmHeight - resized_h) / 2;
        cv::Mat resized;
        cv::resize(rgb, resized, cv::Size(resized_w, resized_h), 0.0, 0.0, cv::INTER_LINEAR);
        resized.copyTo(input(cv::Rect(pad_x, pad_y, resized_w, resized_h)));
        info->scale_x = scale;
        info->scale_y = scale;
        info->pad_x = static_cast<float>(pad_x);
        info->pad_y = static_cast<float>(pad_y);
    } else {
        cv::resize(rgb, input, cv::Size(kPalmWidth, kPalmHeight), 0.0, 0.0, cv::INTER_LINEAR);
        info->scale_x = static_cast<float>(kPalmWidth) / rgb.cols;
        info->scale_y = static_cast<float>(kPalmHeight) / rgb.rows;
        info->pad_x = 0.0f;
        info->pad_y = 0.0f;
    }

    if (!input.isContinuous()) input = input.clone();
    const std::size_t bytes = input.total() * input.elemSize();
    std::vector<std::uint8_t> tensor(bytes);
    std::memcpy(tensor.data(), input.data, bytes);
    return tensor;
}

dxrt::Tensor* find_float_tensor(const dxrt::TensorPtrs& outputs,
                                int anchor_count,
                                int last_dim) {
    for (const auto& tensor : outputs) {
        if (tensor->type() != dxrt::DataType::FLOAT) continue;
        const auto& shape = tensor->shape();
        if (shape.size() == 3 && shape[0] == 1 && shape[1] == anchor_count &&
            shape[2] == last_dim) {
            return tensor.get();
        }
    }
    return nullptr;
}

NormalizedRect make_hand_roi(const cv::Rect2f& box,
                             const std::array<cv::Point2f, kPalmKeypointCount>& keypoints,
                             const cv::Size& frame_size) {
    const float image_w = static_cast<float>(frame_size.width);
    const float image_h = static_cast<float>(frame_size.height);

    NormalizedRect roi;
    roi.x_center = (box.x + box.width * 0.5f) / image_w;
    roi.y_center = (box.y + box.height * 0.5f) / image_h;
    roi.width = box.width / image_w;
    roi.height = box.height / image_h;

    const cv::Point2f& start = keypoints[kWristCenterKeypoint];
    const cv::Point2f& end = keypoints[kMiddleFingerMcpKeypoint];
    roi.rotation =
        normalize_radians(kHandRoiTargetAngle - std::atan2(-(end.y - start.y), end.x - start.x));

    float width = roi.width;
    float height = roi.height;
    const float x_shift = (image_w * width * kHandRoiShiftX * std::cos(roi.rotation) -
                           image_h * height * kHandRoiShiftY * std::sin(roi.rotation)) /
                          image_w;
    const float y_shift = (image_w * width * kHandRoiShiftX * std::sin(roi.rotation) +
                           image_h * height * kHandRoiShiftY * std::cos(roi.rotation)) /
                          image_h;
    roi.x_center += x_shift;
    roi.y_center += y_shift;

    const float long_side = std::max(width * image_w, height * image_h);
    width = long_side / image_w;
    height = long_side / image_h;
    roi.width = width * kHandRoiScaleX;
    roi.height = height * kHandRoiScaleY;
    return roi;
}

std::array<cv::Point2f, 4> hand_roi_corners(const NormalizedRect& roi,
                                            const cv::Size& frame_size) {
    const float cx = roi.x_center * frame_size.width;
    const float cy = roi.y_center * frame_size.height;
    const float w = roi.width * frame_size.width;
    const float h = roi.height * frame_size.height;
    const float cos_r = std::cos(roi.rotation);
    const float sin_r = std::sin(roi.rotation);
    const std::array<cv::Point2f, 4> local = {
        cv::Point2f(-w * 0.5f, -h * 0.5f),
        cv::Point2f(w * 0.5f, -h * 0.5f),
        cv::Point2f(w * 0.5f, h * 0.5f),
        cv::Point2f(-w * 0.5f, h * 0.5f),
    };
    std::array<cv::Point2f, 4> corners{};
    for (std::size_t i = 0; i < local.size(); ++i) {
        corners[i].x = cx + local[i].x * cos_r - local[i].y * sin_r;
        corners[i].y = cy + local[i].x * sin_r + local[i].y * cos_r;
    }
    return corners;
}

std::vector<PalmDetection> decode_palm_detections(const dxrt::TensorPtrs& outputs,
                                                  const std::vector<Anchor>& anchors,
                                                  const PreprocessInfo& prep,
                                                  const cv::Size& frame_size,
                                                  float confidence,
                                                  float nms_threshold,
                                                  int max_hands) {
    const int anchor_count = static_cast<int>(anchors.size());
    dxrt::Tensor* box_tensor = find_float_tensor(outputs, anchor_count, kPalmBoxCoords);
    dxrt::Tensor* score_tensor = find_float_tensor(outputs, anchor_count, 1);
    if (box_tensor == nullptr || score_tensor == nullptr) {
        throw std::runtime_error("unexpected palm outputs: expected FLOAT [1,2016,18] and [1,2016,1]");
    }

    const float* raw_boxes = static_cast<const float*>(box_tensor->data());
    const float* raw_scores = static_cast<const float*>(score_tensor->data());
    std::vector<PalmCandidate> candidates;
    candidates.reserve(64);

    auto model_to_frame_x = [&](float normalized_x) {
        const float model_x = normalized_x * static_cast<float>(kPalmWidth);
        return (model_x - prep.pad_x) / prep.scale_x;
    };
    auto model_to_frame_y = [&](float normalized_y) {
        const float model_y = normalized_y * static_cast<float>(kPalmHeight);
        return (model_y - prep.pad_y) / prep.scale_y;
    };

    for (int i = 0; i < anchor_count; ++i) {
        const float score = sigmoid(raw_scores[i]);
        if (score < confidence) continue;

        const float* box = raw_boxes + static_cast<std::size_t>(i) * kPalmBoxCoords;
        const Anchor& anchor = anchors[static_cast<std::size_t>(i)];
        const float x_center = box[0] / static_cast<float>(kPalmWidth) * anchor.w + anchor.x_center;
        const float y_center = box[1] / static_cast<float>(kPalmHeight) * anchor.h + anchor.y_center;
        const float box_w = box[2] / static_cast<float>(kPalmWidth) * anchor.w;
        const float box_h = box[3] / static_cast<float>(kPalmHeight) * anchor.h;
        if (box_w <= 0.0f || box_h <= 0.0f) continue;

        PalmCandidate candidate;
        candidate.box = cv::Rect2f(x_center - box_w * 0.5f, y_center - box_h * 0.5f,
                                   box_w, box_h);
        candidate.score = score;
        for (int k = 0; k < kPalmKeypointCount; ++k) {
            const int offset = 4 + k * 2;
            candidate.keypoints[k].x =
                box[offset] / static_cast<float>(kPalmWidth) * anchor.w + anchor.x_center;
            candidate.keypoints[k].y =
                box[offset + 1] / static_cast<float>(kPalmHeight) * anchor.h + anchor.y_center;
        }
        candidates.push_back(candidate);
    }

    std::vector<PalmCandidate> weighted =
        weighted_non_max_suppression(std::move(candidates), nms_threshold, max_hands);
    std::vector<PalmDetection> detections;
    detections.reserve(weighted.size());
    for (const auto& candidate : weighted) {
        const float x1 = model_to_frame_x(candidate.box.x);
        const float y1 = model_to_frame_y(candidate.box.y);
        const float x2 = model_to_frame_x(candidate.box.x + candidate.box.width);
        const float y2 = model_to_frame_y(candidate.box.y + candidate.box.height);
        const cv::Rect2f frame_box(std::min(x1, x2), std::min(y1, y2),
                                   std::abs(x2 - x1), std::abs(y2 - y1));
        if (frame_box.width < 2.0f || frame_box.height < 2.0f) continue;

        PalmDetection det;
        det.box.x = clamp_float(frame_box.x, 0.0f, static_cast<float>(frame_size.width - 1));
        det.box.y = clamp_float(frame_box.y, 0.0f, static_cast<float>(frame_size.height - 1));
        const float right = clamp_float(frame_box.x + frame_box.width, 0.0f,
                                        static_cast<float>(frame_size.width - 1));
        const float bottom = clamp_float(frame_box.y + frame_box.height, 0.0f,
                                         static_cast<float>(frame_size.height - 1));
        det.box.width = std::max(0.0f, right - det.box.x);
        det.box.height = std::max(0.0f, bottom - det.box.y);
        det.score = candidate.score;
        for (int k = 0; k < kPalmKeypointCount; ++k) {
            det.keypoints[k].x = model_to_frame_x(candidate.keypoints[k].x);
            det.keypoints[k].y = model_to_frame_y(candidate.keypoints[k].y);
        }
        det.hand_roi = make_hand_roi(frame_box, det.keypoints, frame_size);
        detections.push_back(det);
    }
    return detections;
}

std::shared_ptr<std::vector<std::uint8_t>> make_landmark_input(const cv::Mat& frame,
                                                               const NormalizedRect& roi,
                                                               LandmarkCrop* crop_info) {
    const auto src = hand_roi_corners(roi, frame.size());
    const std::array<cv::Point2f, 4> dst = {
        cv::Point2f(0.0f, 0.0f),
        cv::Point2f(static_cast<float>(kLandmarkWidth - 1), 0.0f),
        cv::Point2f(static_cast<float>(kLandmarkWidth - 1),
                    static_cast<float>(kLandmarkHeight - 1)),
        cv::Point2f(0.0f, static_cast<float>(kLandmarkHeight - 1)),
    };
    const cv::Mat frame_to_crop = cv::getPerspectiveTransform(src.data(), dst.data());
    crop_info->crop_to_frame = cv::getPerspectiveTransform(dst.data(), src.data());

    cv::Mat crop_bgr;
    cv::warpPerspective(frame, crop_bgr, frame_to_crop,
                        cv::Size(kLandmarkWidth, kLandmarkHeight), cv::INTER_LINEAR,
                        cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    cv::Mat crop_rgb;
    cv::cvtColor(crop_bgr, crop_rgb, cv::COLOR_BGR2RGB);
    if (!crop_rgb.isContinuous()) crop_rgb = crop_rgb.clone();

    auto tensor = std::make_shared<std::vector<std::uint8_t>>(crop_rgb.total() * crop_rgb.elemSize());
    std::memcpy(tensor->data(), crop_rgb.data, tensor->size());
    return tensor;
}

HandLandmarkResult parse_landmark_outputs(const dxrt::TensorPtrs& outputs,
                                          const LandmarkJob& job) {
    if (outputs.size() < 3) {
        throw std::runtime_error("unexpected landmark outputs: expected at least 3 tensors");
    }
    const float* landmarks = static_cast<const float*>(outputs[0]->data());
    const float presence_score = static_cast<const float*>(outputs[1]->data())[0];
    const float handedness_score = static_cast<const float*>(outputs[2]->data())[0];
    const float* world = outputs.size() > 3 ? static_cast<const float*>(outputs[3]->data()) : nullptr;

    HandLandmarkResult result;
    result.palm = job.palm;
    result.handedness_score = handedness_score;
    result.handedness = handedness_score > 0.5f ? "Right" : "Left";
    result.confidence = presence_score;

    std::vector<cv::Point2f> crop_points;
    crop_points.reserve(kHandLandmarkCount);
    for (int i = 0; i < kHandLandmarkCount; ++i) {
        const int offset = i * kCoordsPerLandmark;
        crop_points.emplace_back(landmarks[offset + 0], landmarks[offset + 1]);
        result.landmarks[i].z = landmarks[offset + 2];
        if (world != nullptr) {
            result.world_landmarks[i] = cv::Point3f(world[offset + 0], world[offset + 1],
                                                    world[offset + 2]);
        }
    }

    std::vector<cv::Point2f> frame_points;
    cv::perspectiveTransform(crop_points, frame_points, job.crop.crop_to_frame);
    for (int i = 0; i < kHandLandmarkCount; ++i) {
        result.landmarks[i].image = frame_points[static_cast<std::size_t>(i)];
    }
    return result;
}

void landmark_callback(dxrt::TensorPtrs& outputs, void* user_data) {
    std::unique_ptr<LandmarkJob> job(static_cast<LandmarkJob*>(user_data));
    if (!job || job->collector == nullptr) return;

    try {
        HandLandmarkResult result = parse_landmark_outputs(outputs, *job);
        if (result.confidence >= job->confidence_threshold) {
            std::lock_guard<std::mutex> lock(job->collector->mutex);
            job->collector->results.push_back(std::move(result));
        }
    } catch (const std::exception& e) {
        std::cerr << "Landmark postprocess error: " << e.what() << std::endl;
    }

    {
        std::lock_guard<std::mutex> lock(job->collector->mutex);
        ++job->collector->completed;
    }
    job->collector->cv.notify_one();
}

std::vector<HandLandmarkResult> run_landmark_async(dxrt::InferenceEngine& landmark_engine,
                                                   const cv::Mat& frame,
                                                   const std::vector<PalmDetection>& palms,
                                                   float confidence_threshold) {
    LandmarkCollector collector;
    collector.expected = static_cast<int>(palms.size());
    if (palms.empty()) return {};

    for (const auto& palm : palms) {
        auto job = std::make_unique<LandmarkJob>();
        job->palm = palm;
        job->collector = &collector;
        job->confidence_threshold = confidence_threshold;
        job->input = make_landmark_input(frame, palm.hand_roi, &job->crop);
        void* input_ptr = job->input->data();
        void* user_data = job.release();
        landmark_engine.RunAsync(input_ptr, user_data);
    }

    std::unique_lock<std::mutex> lock(collector.mutex);
    collector.cv.wait(lock, [&collector]() { return collector.completed >= collector.expected; });
    std::sort(collector.results.begin(), collector.results.end(),
              [](const HandLandmarkResult& a, const HandLandmarkResult& b) {
                  return a.confidence > b.confidence;
              });
    return std::move(collector.results);
}

void parse_model_input_shape(const std::vector<int64_t>& shape,
                             int* width,
                             int* height,
                             int* channels) {
    if (shape.size() >= 4) {
        const bool nhwc = shape[3] == 3 || shape[3] == 1;
        if (nhwc) {
            *height = static_cast<int>(shape[1]);
            *width = static_cast<int>(shape[2]);
            *channels = static_cast<int>(shape[3]);
        } else {
            *channels = static_cast<int>(shape[1]);
            *height = static_cast<int>(shape[2]);
            *width = static_cast<int>(shape[3]);
        }
    } else if (shape.size() == 3) {
        *height = static_cast<int>(shape[1]);
        *width = static_cast<int>(shape[2]);
        *channels = 3;
    } else if (shape.size() >= 2) {
        *height = static_cast<int>(shape[0]);
        *width = static_cast<int>(shape[1]);
        *channels = 3;
    } else {
        *width = 0;
        *height = 0;
        *channels = 0;
    }
}

std::shared_ptr<std::vector<std::uint8_t>> make_pose_input(const cv::Mat& frame,
                                                           int input_width,
                                                           int input_height,
                                                           PosePreprocessInfo* prep) {
    if (frame.empty()) {
        throw std::runtime_error("empty frame for pose input");
    }

    prep->original_width = frame.cols;
    prep->original_height = frame.rows;
    prep->input_width = input_width;
    prep->input_height = input_height;
    prep->scale = std::min(static_cast<float>(input_width) / static_cast<float>(frame.cols),
                           static_cast<float>(input_height) / static_cast<float>(frame.rows));
    const int resized_width = std::max(1, cvRound(static_cast<float>(frame.cols) * prep->scale));
    const int resized_height = std::max(1, cvRound(static_cast<float>(frame.rows) * prep->scale));
    prep->pad_x = (input_width - resized_width) / 2;
    prep->pad_y = (input_height - resized_height) / 2;

    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(resized_width, resized_height), 0.0, 0.0,
               cv::INTER_LINEAR);
    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

    cv::Mat input(input_height, input_width, CV_8UC3, cv::Scalar(114, 114, 114));
    rgb.copyTo(input(cv::Rect(prep->pad_x, prep->pad_y, resized_width, resized_height)));
    if (!input.isContinuous()) input = input.clone();

    auto tensor = std::make_shared<std::vector<std::uint8_t>>(input.total() * input.elemSize());
    std::memcpy(tensor->data(), input.data, tensor->size());
    return tensor;
}

std::vector<PoseResult> parse_pose_outputs(const dxrt::TensorPtrs& outputs,
                                           const PosePreprocessInfo& prep,
                                           float score_threshold,
                                           float nms_threshold,
                                           int num_keypoints) {
    std::vector<PoseResult> results;
    if (outputs.empty()) return results;

    const auto& tensor = outputs[0];
    const auto shape = tensor->shape();
    const float* raw_data = static_cast<const float*>(tensor->data());
    if (raw_data == nullptr || shape.size() < 2) return results;

    int channels = 0;
    int count = 0;
    bool needs_transpose = false;
    if (shape.size() == 3) {
        if (shape[1] <= shape[2]) {
            channels = static_cast<int>(shape[1]);
            count = static_cast<int>(shape[2]);
            needs_transpose = true;
        } else {
            count = static_cast<int>(shape[1]);
            channels = static_cast<int>(shape[2]);
            needs_transpose = false;
        }
    } else {
        if (shape[0] <= shape[1]) {
            channels = static_cast<int>(shape[0]);
            count = static_cast<int>(shape[1]);
            needs_transpose = true;
        } else {
            count = static_cast<int>(shape[0]);
            channels = static_cast<int>(shape[1]);
            needs_transpose = false;
        }
    }
    if (channels < 5 || count <= 0) return results;

    std::vector<float> transposed;
    const float* row_data = raw_data;
    if (needs_transpose) {
        transposed.resize(static_cast<std::size_t>(count) * static_cast<std::size_t>(channels));
        for (int c = 0; c < channels; ++c) {
            for (int n = 0; n < count; ++n) {
                transposed[static_cast<std::size_t>(n) * channels + c] = raw_data[c * count + n];
            }
        }
        row_data = transposed.data();
    }

    std::vector<cv::Rect> nms_boxes;
    std::vector<float> nms_scores;
    std::vector<int> candidate_indices;
    for (int i = 0; i < count; ++i) {
        const float* row = row_data + static_cast<std::size_t>(i) * channels;
        const float score = row[4];
        if (score < score_threshold) continue;

        float x = 0.0f;
        float y = 0.0f;
        float w = 0.0f;
        float h = 0.0f;
        if (needs_transpose) {
            x = row[0] - row[2] * 0.5f;
            y = row[1] - row[3] * 0.5f;
            w = row[2];
            h = row[3];
        } else {
            x = row[0];
            y = row[1];
            w = row[2] - row[0];
            h = row[3] - row[1];
        }
        if (w <= 1.0f || h <= 1.0f) continue;

        nms_boxes.emplace_back(cvRound(x), cvRound(y), cvRound(w), cvRound(h));
        nms_scores.push_back(score);
        candidate_indices.push_back(i);
    }
    if (nms_boxes.empty()) return results;

    std::vector<int> keep;
    cv::dnn::NMSBoxes(nms_boxes, nms_scores, score_threshold, nms_threshold, keep);
    for (int keep_index : keep) {
        const int row_index = candidate_indices[static_cast<std::size_t>(keep_index)];
        const float* row = row_data + static_cast<std::size_t>(row_index) * channels;

        float x1 = 0.0f;
        float y1 = 0.0f;
        float x2 = 0.0f;
        float y2 = 0.0f;
        if (needs_transpose) {
            x1 = row[0] - row[2] * 0.5f;
            y1 = row[1] - row[3] * 0.5f;
            x2 = row[0] + row[2] * 0.5f;
            y2 = row[1] + row[3] * 0.5f;
        } else {
            x1 = row[0];
            y1 = row[1];
            x2 = row[2];
            y2 = row[3];
        }

        auto scale_x = [&prep](float value) {
            return clamp_float((value - static_cast<float>(prep.pad_x)) / prep.scale,
                               0.0f, static_cast<float>(prep.original_width - 1));
        };
        auto scale_y = [&prep](float value) {
            return clamp_float((value - static_cast<float>(prep.pad_y)) / prep.scale,
                               0.0f, static_cast<float>(prep.original_height - 1));
        };

        PoseResult pose;
        pose.box = {scale_x(x1), scale_y(y1), scale_x(x2), scale_y(y2)};
        pose.confidence = nms_scores[static_cast<std::size_t>(keep_index)];

        const int compact_channels = 5 + num_keypoints * 3;
        const int kp_offset = (!needs_transpose && channels >= compact_channels + 1) ? 6 : 5;
        if (channels >= kp_offset + num_keypoints * 3) {
            const float* kp_data = row + kp_offset;
            pose.keypoints.reserve(static_cast<std::size_t>(num_keypoints));
            for (int kp = 0; kp < num_keypoints; ++kp) {
                pose.keypoints.push_back(PoseKeypoint{
                    scale_x(kp_data[kp * 3]),
                    scale_y(kp_data[kp * 3 + 1]),
                    kp_data[kp * 3 + 2],
                });
            }
        }
        results.push_back(std::move(pose));
    }

    std::sort(results.begin(), results.end(), [](const PoseResult& a, const PoseResult& b) {
        return a.confidence > b.confidence;
    });
    return results;
}

void pose_callback(dxrt::TensorPtrs& outputs, void* user_data) {
    std::unique_ptr<PoseJob> job(static_cast<PoseJob*>(user_data));
    if (!job || job->collector == nullptr) return;

    std::vector<PoseResult> results;
    try {
        results = parse_pose_outputs(outputs, job->prep, job->score_threshold,
                                     job->nms_threshold, job->num_keypoints);
    } catch (const std::exception& e) {
        std::cerr << "Pose postprocess error: " << e.what() << std::endl;
    }

    const auto done_time = std::chrono::steady_clock::now();
    {
        std::lock_guard<std::mutex> lock(job->collector->mutex);
        job->collector->results = std::move(results);
        job->collector->inference_ms =
            std::chrono::duration<double, std::milli>(done_time - job->submit_time).count();
        job->collector->completed = true;
    }
    job->collector->cv.notify_one();
}

std::unique_ptr<PoseCollector> submit_pose_async(dxrt::InferenceEngine& pose_engine,
                                                 const cv::Mat& frame,
                                                 int input_width,
                                                 int input_height,
                                                 float score_threshold,
                                                 float nms_threshold) {
    auto collector = std::make_unique<PoseCollector>();
    auto job = std::make_unique<PoseJob>();
    job->collector = collector.get();
    job->score_threshold = score_threshold;
    job->nms_threshold = nms_threshold;
    job->input = make_pose_input(frame, input_width, input_height, &job->prep);
    job->submit_time = std::chrono::steady_clock::now();
    void* input_ptr = job->input->data();
    pose_engine.RunAsync(input_ptr, job.release());
    return collector;
}

std::vector<PoseResult> wait_pose_result(PoseCollector* collector, double* inference_ms) {
    std::unique_lock<std::mutex> lock(collector->mutex);
    collector->cv.wait(lock, [collector]() { return collector->completed; });
    if (inference_ms != nullptr) {
        *inference_ms = collector->inference_ms;
    }
    return std::move(collector->results);
}

std::string format_float(float value, int precision) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << value;
    return oss.str();
}

void draw_filled_rounded_rect(cv::Mat& image,
                              const cv::Rect& rect,
                              int radius,
                              const cv::Scalar& color) {
    const cv::Rect bounds(0, 0, image.cols, image.rows);
    const cv::Rect clipped = rect & bounds;
    if (clipped.empty()) return;

    const int r = std::max(0, std::min(radius, std::min(clipped.width, clipped.height) / 2));
    if (r == 0) {
        cv::rectangle(image, clipped, color, cv::FILLED, cv::LINE_AA);
        return;
    }

    cv::rectangle(image,
                  cv::Rect(clipped.x + r, clipped.y, clipped.width - 2 * r, clipped.height),
                  color, cv::FILLED, cv::LINE_AA);
    cv::rectangle(image,
                  cv::Rect(clipped.x, clipped.y + r, clipped.width, clipped.height - 2 * r),
                  color, cv::FILLED, cv::LINE_AA);
    cv::circle(image, cv::Point(clipped.x + r, clipped.y + r), r, color, cv::FILLED, cv::LINE_AA);
    cv::circle(image, cv::Point(clipped.x + clipped.width - r - 1, clipped.y + r), r,
               color, cv::FILLED, cv::LINE_AA);
    cv::circle(image, cv::Point(clipped.x + r, clipped.y + clipped.height - r - 1), r,
               color, cv::FILLED, cv::LINE_AA);
    cv::circle(image,
               cv::Point(clipped.x + clipped.width - r - 1, clipped.y + clipped.height - r - 1),
               r, color, cv::FILLED, cv::LINE_AA);
}

void blend_filled_rounded_rect(cv::Mat& frame,
                               const cv::Rect& rect,
                               int radius,
                               const cv::Scalar& color,
                               double alpha) {
    const cv::Rect bounds(0, 0, frame.cols, frame.rows);
    const cv::Rect clipped = rect & bounds;
    if (clipped.empty()) return;

    cv::Mat roi = frame(clipped);
    cv::Mat overlay = roi.clone();
    draw_filled_rounded_rect(overlay, cv::Rect(0, 0, clipped.width, clipped.height),
                             radius, color);
    cv::addWeighted(overlay, alpha, roi, 1.0 - alpha, 0.0, roi);
}

void draw_label(cv::Mat& frame,
                const std::string& text,
                const cv::Point& origin,
                double font_scale,
                int thickness,
                const cv::Scalar& fg,
                const cv::Scalar& bg) {
    int baseline = 0;
    const cv::Size text_size =
        cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, font_scale, thickness, &baseline);
    const int pad_x = std::max(4, text_size.height / 4);
    const int pad_y = std::max(3, text_size.height / 5);
    const int x = std::max(0, origin.x);
    const int y = std::max(text_size.height + pad_y * 2, origin.y);
    const cv::Rect bg_rect(
        x,
        y - text_size.height - baseline - pad_y * 2,
        std::min(frame.cols - x, text_size.width + pad_x * 2),
        text_size.height + baseline + pad_y * 2);

    if (bg_rect.width > 0 && bg_rect.height > 0) {
        blend_filled_rounded_rect(frame, bg_rect, bg_rect.height / 2, bg, 0.58);
        cv::putText(frame, text, cv::Point(x + pad_x, y - baseline - pad_y),
                    cv::FONT_HERSHEY_SIMPLEX, font_scale, fg, thickness, cv::LINE_AA);
    }
}

cv::Point to_pixel(const cv::Point2f& point) {
    return cv::Point(cvRound(point.x), cvRound(point.y));
}

cv::Scalar landmark_point_color(int index) {
    static const std::array<cv::Scalar, kHandLandmarkCount> colors = {
        cv::Scalar(255, 255, 32),   // wrist: electric cyan
        cv::Scalar(255, 56, 255),   // thumb: neon magenta
        cv::Scalar(255, 72, 255),
        cv::Scalar(255, 88, 255),
        cv::Scalar(255, 112, 255),
        cv::Scalar(48, 255, 255),   // index: neon yellow
        cv::Scalar(64, 255, 255),
        cv::Scalar(80, 255, 255),
        cv::Scalar(96, 255, 255),
        cv::Scalar(72, 255, 64),    // middle: laser green
        cv::Scalar(88, 255, 80),
        cv::Scalar(104, 255, 96),
        cv::Scalar(120, 255, 112),
        cv::Scalar(255, 160, 48),   // ring: electric blue
        cv::Scalar(255, 176, 64),
        cv::Scalar(255, 192, 80),
        cv::Scalar(255, 208, 96),
        cv::Scalar(255, 64, 216),   // pinky: hot violet
        cv::Scalar(255, 80, 224),
        cv::Scalar(255, 96, 232),
        cv::Scalar(255, 112, 240),
    };
    return colors[std::max(0, std::min(index, kHandLandmarkCount - 1))];
}

void blend_circle(cv::Mat& frame,
                  const cv::Point& center,
                  int radius,
                  const cv::Scalar& color,
                  double alpha) {
    const cv::Rect bounds(0, 0, frame.cols, frame.rows);
    const cv::Rect circle_rect(center.x - radius, center.y - radius,
                               radius * 2 + 1, radius * 2 + 1);
    const cv::Rect clipped = circle_rect & bounds;
    if (clipped.empty()) return;

    cv::Mat roi = frame(clipped);
    cv::Mat overlay = roi.clone();
    cv::circle(overlay, cv::Point(center.x - clipped.x, center.y - clipped.y), radius,
               color, cv::FILLED, cv::LINE_AA);
    cv::addWeighted(overlay, alpha, roi, 1.0 - alpha, 0.0, roi);
}

void draw_neon_point(cv::Mat& frame,
                     const cv::Point& center,
                     int radius,
                     const cv::Scalar& color) {
    const int point_radius = std::max(1, cvRound(static_cast<float>(radius + 1) * 0.8f));
    const int raw_glow_radius = std::max(radius * 3, radius + 5);
    const int raw_aura_radius = std::max(radius * 2, radius + 3);
    const int scaled_glow_radius =
        cvRound(static_cast<float>(raw_glow_radius) * 0.8f);
    const int scaled_aura_radius =
        cvRound(static_cast<float>(raw_aura_radius) * 0.8f);
    const int glow_radius =
        point_radius + std::max(1, cvRound(static_cast<float>(
                                 std::max(0, scaled_glow_radius - point_radius)) * 0.5f));
    const int aura_radius =
        point_radius + std::max(1, cvRound(static_cast<float>(
                                 std::max(0, scaled_aura_radius - point_radius)) * 0.5f));
    blend_circle(frame, center, glow_radius, color, 0.11);
    blend_circle(frame, center, aura_radius, color, 0.19);

    cv::circle(frame, center, point_radius, color, cv::FILLED, cv::LINE_AA);
    cv::circle(frame, center, std::max(1, cvRound(static_cast<float>(point_radius) * 0.45f)),
               cv::Scalar(255, 255, 255),
               cv::FILLED, cv::LINE_AA);
}

void draw_hand_landmarks(cv::Mat& frame, const HandLandmarkResult& hand, int radius) {
    static const int connections[][2] = {
        {0, 1}, {1, 2}, {2, 3}, {3, 4},      {0, 5},  {5, 6},  {6, 7},
        {7, 8}, {0, 9}, {9, 10}, {10, 11},   {11, 12}, {0, 13}, {13, 14},
        {14, 15}, {15, 16}, {0, 17}, {17, 18}, {18, 19}, {19, 20},
        {5, 9}, {9, 13}, {13, 17},
    };
    cv::Mat line_overlay = frame.clone();
    for (const auto& connection : connections) {
        const cv::Point2f& a = hand.landmarks[connection[0]].image;
        const cv::Point2f& b = hand.landmarks[connection[1]].image;
        cv::line(line_overlay, to_pixel(a), to_pixel(b), cv::Scalar(246, 248, 250), 1,
                 cv::LINE_AA);
    }
    cv::addWeighted(line_overlay, 0.5, frame, 0.5, 0.0, frame);

    for (int i = 0; i < kHandLandmarkCount; ++i) {
        const cv::Point center = to_pixel(hand.landmarks[i].image);
        const int point_radius = std::max(3, cvRound(static_cast<float>(radius) * 0.9f));
        draw_neon_point(frame, center, point_radius, landmark_point_color(i));
    }
}

void draw_pose_results(cv::Mat& frame, const std::vector<PoseResult>& poses) {
    static const std::array<std::pair<int, int>, 19> skeleton = {{
        {15, 13}, {13, 11}, {16, 14}, {14, 12}, {11, 12},
        {5, 11}, {6, 12}, {5, 6},
        {5, 7}, {6, 8}, {7, 9}, {8, 10},
        {1, 2}, {0, 1}, {0, 2}, {1, 3}, {2, 4},
        {3, 5}, {4, 6},
    }};
    static const std::array<cv::Scalar, 19> limb_colors = {{
        cv::Scalar(72, 220, 255), cv::Scalar(72, 220, 255),
        cv::Scalar(72, 220, 255), cv::Scalar(72, 220, 255),
        cv::Scalar(255, 82, 232), cv::Scalar(255, 82, 232),
        cv::Scalar(255, 82, 232), cv::Scalar(48, 226, 255),
        cv::Scalar(48, 226, 255), cv::Scalar(48, 226, 255),
        cv::Scalar(48, 226, 255), cv::Scalar(48, 226, 255),
        cv::Scalar(92, 255, 132), cv::Scalar(92, 255, 132),
        cv::Scalar(92, 255, 132), cv::Scalar(92, 255, 132),
        cv::Scalar(92, 255, 132), cv::Scalar(92, 255, 132),
        cv::Scalar(92, 255, 132),
    }};
    static const std::array<cv::Scalar, 17> keypoint_colors = {{
        cv::Scalar(92, 255, 132), cv::Scalar(92, 255, 132),
        cv::Scalar(92, 255, 132), cv::Scalar(92, 255, 132),
        cv::Scalar(92, 255, 132), cv::Scalar(48, 226, 255),
        cv::Scalar(48, 226, 255), cv::Scalar(48, 226, 255),
        cv::Scalar(48, 226, 255), cv::Scalar(48, 226, 255),
        cv::Scalar(48, 226, 255), cv::Scalar(72, 220, 255),
        cv::Scalar(72, 220, 255), cv::Scalar(72, 220, 255),
        cv::Scalar(72, 220, 255), cv::Scalar(72, 220, 255),
        cv::Scalar(72, 220, 255),
    }};

    const int base = std::max(1, std::min(frame.cols, frame.rows));
    const int line_thickness = std::max(1, base / 540);
    const int point_radius = std::max(2, base / 260);
    const float kp_threshold = 0.3f;

    for (const auto& pose : poses) {
        if (pose.box.size() >= 4) {
            const cv::Point pt1(cvRound(pose.box[0]), cvRound(pose.box[1]));
            const cv::Point pt2(cvRound(pose.box[2]), cvRound(pose.box[3]));
            cv::rectangle(frame, pt1, pt2, cv::Scalar(92, 255, 132), line_thickness,
                          cv::LINE_AA);
            draw_label(frame, "person " + format_float(pose.confidence, 2),
                       cv::Point(pt1.x, pt1.y - 6), 0.48, 1,
                       cv::Scalar(238, 244, 248), cv::Scalar(14, 22, 30));
        }

        for (std::size_t i = 0; i < skeleton.size(); ++i) {
            const int a_idx = skeleton[i].first;
            const int b_idx = skeleton[i].second;
            if (a_idx >= static_cast<int>(pose.keypoints.size()) ||
                b_idx >= static_cast<int>(pose.keypoints.size())) {
                continue;
            }
            const PoseKeypoint& a = pose.keypoints[static_cast<std::size_t>(a_idx)];
            const PoseKeypoint& b = pose.keypoints[static_cast<std::size_t>(b_idx)];
            if (a.confidence < kp_threshold || b.confidence < kp_threshold) continue;
            cv::line(frame, cv::Point(cvRound(a.x), cvRound(a.y)),
                     cv::Point(cvRound(b.x), cvRound(b.y)), limb_colors[i],
                     line_thickness, cv::LINE_AA);
        }

        for (std::size_t i = 0; i < pose.keypoints.size(); ++i) {
            const PoseKeypoint& kp = pose.keypoints[i];
            if (kp.confidence < kp_threshold) continue;
            const cv::Point center(cvRound(kp.x), cvRound(kp.y));
            const cv::Scalar color = keypoint_colors[i % keypoint_colors.size()];
            cv::circle(frame, center, point_radius + 1, cv::Scalar(4, 8, 12),
                       cv::FILLED, cv::LINE_AA);
            cv::circle(frame, center, point_radius, color, cv::FILLED, cv::LINE_AA);
            cv::circle(frame, center, std::max(1, point_radius / 2),
                       cv::Scalar(255, 255, 255), cv::FILLED, cv::LINE_AA);
        }
    }
}

void draw_results(cv::Mat& frame,
                  const std::vector<PalmDetection>& palms,
                  const std::vector<HandLandmarkResult>& hands,
                  const std::vector<PoseResult>& poses,
                  bool show_palm_overlay) {
    const int base = std::max(1, std::min(frame.cols, frame.rows));
    const int box_thickness = std::max(2, base / 280);
    const int text_thickness = std::max(1, base / 720);
    const double font_scale = std::max(0.45, std::min(0.95, base / 900.0));

    draw_pose_results(frame, poses);

    if (show_palm_overlay) {
        for (const auto& palm : palms) {
            cv::rectangle(frame, palm.box, cv::Scalar(48, 220, 255), box_thickness, cv::LINE_AA);
            const auto corners = hand_roi_corners(palm.hand_roi, frame.size());
            for (std::size_t i = 0; i < corners.size(); ++i) {
                cv::line(frame, corners[i], corners[(i + 1) % corners.size()],
                         cv::Scalar(255, 160, 70), std::max(1, box_thickness - 1), cv::LINE_AA);
            }
        }
    }

    for (const auto& hand : hands) {
        draw_hand_landmarks(frame, hand, std::max(2, box_thickness));
        const cv::Point label_pos(static_cast<int>(hand.landmarks[0].image.x),
                                  static_cast<int>(hand.landmarks[0].image.y) - 8);
        draw_label(frame,
                   hand.handedness + " " + format_float(hand.confidence, 2),
                   label_pos, font_scale * 0.8, text_thickness,
                   cv::Scalar(238, 244, 248), cv::Scalar(20, 26, 34));
    }

}

QImage mat_to_qimage(const cv::Mat& bgr) {
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    QImage image(rgb.data, rgb.cols, rgb.rows, static_cast<int>(rgb.step), QImage::Format_RGB888);
    return image.copy();
}

struct HudMetrics {
    std::size_t hand_count = 0;
    std::size_t palm_count = 0;
    std::size_t pose_count = 0;
    double fps = 0.0;
    double pose_ms = 0.0;
    int max_hands = 0;
    bool show_palm_overlay = true;
    bool show_pose_overlay = true;
};

class FrameView : public QWidget {
public:
    explicit FrameView(QWidget* parent = nullptr) : QWidget(parent) {
        setWindowTitle("DEEPX M1 Hand Landmark + Pose Detection");
        setFocusPolicy(Qt::StrongFocus);
    }

    void setExitButtonVisible(bool visible) {
        show_exit_button_ = visible;
        update();
    }

    void setFrame(const QImage& image, const HudMetrics& metrics) {
        frame_ = image;
        metrics_ = metrics;
        update();
    }

protected:
    void paintEvent(QPaintEvent*) override {
        QPainter painter(this);
        painter.fillRect(rect(), Qt::black);

        if (!frame_.isNull()) {
            QSize target_size = frame_.size();
            target_size.scale(size(), Qt::KeepAspectRatio);
            const QPoint top_left(0, (height() - target_size.height()) / 2);
            painter.drawImage(QRect(top_left, target_size), frame_);

            const QRect panel_rect(target_size.width(), 0,
                                   std::max(0, width() - target_size.width()), height());
            drawSidePanel(painter, panel_rect);
        }

        drawExitButton(painter);

        if (!frame_.isNull() && !launcher_ready_notified_) {
            notify_launcher_ready();
            launcher_ready_notified_ = true;
        }
    }

    void keyPressEvent(QKeyEvent* event) override {
        if (event->key() == Qt::Key_Escape || event->key() == Qt::Key_Q) {
            QCoreApplication::quit();
            return;
        }
        if (event->key() == Qt::Key_F) {
            isFullScreen() ? showNormal() : showFullScreen();
            return;
        }
        QWidget::keyPressEvent(event);
    }

    void mousePressEvent(QMouseEvent* event) override {
        if (show_exit_button_ && event->button() == Qt::LeftButton &&
            exitButtonRect().contains(event->pos())) {
            event->accept();
            QCoreApplication::quit();
            return;
        }
        QWidget::mousePressEvent(event);
    }

private:
    QRect exitButtonRect() const {
        constexpr int kButtonWidth = 32;
        constexpr int kButtonHeight = 28;
        constexpr int kMargin = 14;
        return QRect(width() - kButtonWidth - kMargin, kMargin, kButtonWidth, kButtonHeight);
    }

    void drawExitButton(QPainter& painter) const {
        if (!show_exit_button_) return;

        painter.setRenderHint(QPainter::Antialiasing, true);
        const QRect r = exitButtonRect();

        painter.setPen(QPen(QColor(60, 60, 60, 180), 1));
        painter.setBrush(QColor(45, 45, 48, 145));
        painter.drawRoundedRect(r, 6, 6);

        QFont font = painter.font();
        font.setPixelSize(13);
        font.setBold(true);
        painter.setFont(font);
        painter.setPen(QColor(204, 204, 204, 170));
        painter.drawText(r, Qt::AlignCenter, "X");
    }

    void drawMetricCard(QPainter& painter,
                        const QRect& rect,
                        const QString& label,
                        const QString& value,
                        const QColor& accent) const {
        if (rect.width() <= 0 || rect.height() <= 0) return;

        painter.setPen(Qt::NoPen);
        painter.setBrush(QColor(17, 24, 34, 232));
        painter.drawRoundedRect(rect, 8, 8);
        painter.setBrush(accent);
        painter.drawRoundedRect(QRect(rect.x(), rect.y(), 4, rect.height()), 2, 2);

        QFont label_font = painter.font();
        label_font.setPixelSize(12);
        label_font.setBold(true);
        painter.setFont(label_font);
        painter.setPen(QColor(152, 164, 180));
        painter.drawText(rect.adjusted(18, 12, -16, 0), Qt::AlignLeft | Qt::AlignTop, label);

        QFont value_font = painter.font();
        value_font.setPixelSize(30);
        value_font.setBold(true);
        painter.setFont(value_font);
        painter.setPen(QColor(244, 248, 252));
        painter.drawText(rect.adjusted(18, 0, -16, -14), Qt::AlignLeft | Qt::AlignBottom, value);
    }

    void drawSidePanel(QPainter& painter, const QRect& panel_rect) const {
        if (panel_rect.width() < 180 || panel_rect.height() < 360) return;

        painter.setRenderHint(QPainter::Antialiasing, true);
        QLinearGradient bg(panel_rect.topLeft(), panel_rect.bottomRight());
        bg.setColorAt(0.0, QColor(6, 10, 17));
        bg.setColorAt(0.55, QColor(12, 18, 27));
        bg.setColorAt(1.0, QColor(7, 13, 20));
        painter.fillRect(panel_rect, bg);

        painter.setPen(QPen(QColor(42, 236, 255, 150), 1));
        painter.drawLine(panel_rect.topLeft(), panel_rect.bottomLeft());

        const int margin = std::max(24, std::min(42, panel_rect.width() / 10));
        const QRect content = panel_rect.adjusted(margin, 42, -margin, -36);
        int y = content.top();

        QFont eyebrow = painter.font();
        eyebrow.setPixelSize(28);
        eyebrow.setBold(true);
        painter.setFont(eyebrow);
        painter.setPen(QColor(69, 241, 255));
        painter.drawText(QRect(content.left(), y, content.width(), 28),
                         Qt::AlignLeft | Qt::AlignVCenter, "DEEPX M1");
        y += 32;

        QFont title = painter.font();
        title.setPixelSize(42);
        title.setBold(true);
        painter.setFont(title);
        painter.setPen(QColor(245, 248, 252));
        painter.drawText(QRect(content.left(), y, content.width(), 42),
                         Qt::AlignLeft | Qt::AlignVCenter, "HANDS + POSE");
        y += 50;

        QFont subtitle = painter.font();
        subtitle.setPixelSize(20);
        subtitle.setBold(false);
        painter.setFont(subtitle);
        painter.setPen(QColor(154, 166, 180));
        painter.drawText(QRect(content.left(), y, content.width(), 22),
                         Qt::AlignLeft | Qt::AlignVCenter, "21 hand points + YOLO26s pose");
        y += 42;

        painter.setPen(QPen(QColor(255, 255, 255, 42), 1));
        painter.drawLine(content.left(), y, content.right(), y);
        y += 32;

        const int gap = 14;
        const int card_h = std::max(72, std::min(104, (content.bottom() - y - 86 - gap * 2) / 3));
        drawMetricCard(painter, QRect(content.left(), y, content.width(), card_h),
                       "HANDS",
                       QString::fromStdString(std::to_string(metrics_.hand_count)) + "/" +
                           QString::number(metrics_.max_hands),
                       QColor(83, 255, 134));
        y += card_h + gap;
        drawMetricCard(painter, QRect(content.left(), y, content.width(), card_h),
                       "PERSONS", QString::fromStdString(std::to_string(metrics_.pose_count)),
                       QColor(255, 217, 74));
        y += card_h + gap;
        drawMetricCard(painter, QRect(content.left(), y, content.width(), card_h),
                       "FPS", QString::number(metrics_.fps, 'f', 1),
                       QColor(47, 221, 255));
    }

    QImage frame_;
    HudMetrics metrics_;
    bool show_exit_button_ = false;
    bool launcher_ready_notified_ = false;
};

class FpsCounter {
public:
    double update() {
        ++frames_;
        const auto now = std::chrono::steady_clock::now();
        const std::chrono::duration<double> elapsed = now - last_;
        if (elapsed.count() >= 1.0) {
            fps_ = static_cast<double>(frames_) / elapsed.count();
            frames_ = 0;
            last_ = now;
        }
        return fps_;
    }

private:
    int frames_ = 0;
    double fps_ = 0.0;
    std::chrono::steady_clock::time_point last_ = std::chrono::steady_clock::now();
};

void request_quit_from_worker() {
    QMetaObject::invokeMethod(QCoreApplication::instance(), []() { QCoreApplication::quit(); },
                              Qt::QueuedConnection);
}

void publish_frame(FrameView* view, const cv::Mat& frame, const HudMetrics& metrics) {
    QImage image = mat_to_qimage(frame);
    QMetaObject::invokeMethod(view, [view, image, metrics]() { view->setFrame(image, metrics); },
                              Qt::QueuedConnection);
}

cv::Mat center_crop_wide_frame_to_4_3(const cv::Mat& frame) {
    if (frame.empty() || frame.cols * 3 <= frame.rows * 4) {
        return frame;
    }

    const int target_width = std::max(1, frame.rows * 4 / 3);
    const int x = std::max(0, (frame.cols - target_width) / 2);
    return frame(cv::Rect(x, 0, target_width, frame.rows));
}

void validate_hand_model_shapes(dxrt::InferenceEngine& palm_engine,
                                dxrt::InferenceEngine& landmark_engine) {
    const auto palm_input = palm_engine.GetInputs().front();
    const auto palm_shape = palm_input.shape();
    if (palm_shape.size() != 4 || palm_shape[1] != kPalmHeight || palm_shape[2] != kPalmWidth ||
        palm_shape[3] != kPalmChannels || palm_input.type() != dxrt::DataType::UINT8) {
        throw std::runtime_error("palm model input must be UINT8 [1,192,192,3]");
    }
    const auto lm_input = landmark_engine.GetInputs().front();
    const auto lm_shape = lm_input.shape();
    if (lm_shape.size() != 4 || lm_shape[1] != kLandmarkHeight ||
        lm_shape[2] != kLandmarkWidth || lm_shape[3] != kLandmarkChannels ||
        lm_input.type() != dxrt::DataType::UINT8) {
        throw std::runtime_error("landmark model input must be UINT8 [1,224,224,3]");
    }
}

void validate_pose_model_shape(dxrt::InferenceEngine& pose_engine,
                               int* pose_input_width,
                               int* pose_input_height) {
    const auto pose_input = pose_engine.GetInputs().front();
    int pose_channels = 0;
    parse_model_input_shape(pose_input.shape(), pose_input_width, pose_input_height,
                            &pose_channels);
    if (*pose_input_width <= 0 || *pose_input_height <= 0 || pose_channels != 3 ||
        pose_input.type() != dxrt::DataType::UINT8) {
        throw std::runtime_error("pose model input must be UINT8 image tensor with 3 channels");
    }
}

void stop_pipeline(PipelineQueues* queues, std::atomic<bool>* running) {
    running->store(false);
    queues->close_all();
}

void capture_loop(const Options& options,
                  cv::VideoCapture* cap,
                  PipelineQueues* queues,
                  std::atomic<bool>* running,
                  bool pace_video,
                  std::chrono::duration<double> frame_interval) {
    try {
        FrameId next_frame_id = 1;
        auto next_frame_time = std::chrono::steady_clock::now();
        while (running->load()) {
            cv::Mat frame;
            if (!cap->read(frame) || frame.empty()) {
                if (!options.use_camera && options.loop_video) {
                    cap->set(cv::CAP_PROP_POS_FRAMES, 0.0);
                    continue;
                }
                break;
            }

            FramePacket packet;
            packet.id = next_frame_id++;
            const cv::Mat cropped = center_crop_wide_frame_to_4_3(frame);
            if (options.use_camera) {
                cv::flip(cropped, packet.frame, 1);
            } else {
                packet.frame = cropped.clone();
            }
            packet.captured_at = std::chrono::steady_clock::now();
            queues->capture.push(std::move(packet));

            if (pace_video) {
                next_frame_time +=
                    std::chrono::duration_cast<std::chrono::steady_clock::duration>(frame_interval);
                std::this_thread::sleep_until(next_frame_time);
                if (next_frame_time < std::chrono::steady_clock::now() - std::chrono::seconds(1)) {
                    next_frame_time = std::chrono::steady_clock::now();
                }
            }
        }
        queues->capture.close();
    } catch (const std::exception& e) {
        std::cerr << "Capture error: " << e.what() << std::endl;
        stop_pipeline(queues, running);
        request_quit_from_worker();
    }
}

void dispatch_loop(const Options& options, PipelineQueues* queues, std::atomic<bool>* running) {
    const auto poll = std::chrono::milliseconds(kQueuePollMs);
    while (true) {
        if (!running->load() && !queues->capture.closed()) break;

        FramePacket packet;
        if (!queues->capture.pop_for(&packet, poll)) {
            if (queues->capture.closed() || !running->load()) break;
            continue;
        }

        if (options.show_pose_overlay) {
            queues->pose.push(FramePacket{packet.id, packet.frame.clone(), packet.captured_at});
        }
        queues->hand.push(FramePacket{packet.id, packet.frame.clone(), packet.captured_at});
        queues->render.push(std::move(packet));
    }

    queues->pose.close();
    queues->hand.close();
    queues->render.close();
}

void pose_loop(const Options& options,
               BoundedQueue<FramePacket>* pose_queue,
               ResultStore* results,
               PipelineQueues* queues,
               std::atomic<bool>* running) {
    try {
        dxrt::InferenceEngine pose_engine(options.pose_model_path);
        int pose_input_width = 0;
        int pose_input_height = 0;
        validate_pose_model_shape(pose_engine, &pose_input_width, &pose_input_height);
        pose_engine.RegisterCallback([](dxrt::TensorPtrs& outputs, void* user_data) -> int {
            pose_callback(outputs, user_data);
            return 0;
        });

        const auto poll = std::chrono::milliseconds(kQueuePollMs);
        while (true) {
            if (!running->load() && !pose_queue->closed()) break;

            FramePacket packet;
            if (!pose_queue->pop_for(&packet, poll)) {
                if (pose_queue->closed() || !running->load()) break;
                continue;
            }

            double inference_ms = 0.0;
            std::vector<PoseResult> poses;
            auto collector = submit_pose_async(pose_engine, packet.frame, pose_input_width,
                                               pose_input_height, options.pose_confidence,
                                               options.pose_nms_threshold);
            poses = wait_pose_result(collector.get(), &inference_ms);
            results->set_pose(PosePacket{packet.id, std::move(poses), inference_ms});
        }
    } catch (const dxrt::Exception& e) {
        std::cerr << "Pose DXRT error: " << e.what() << std::endl;
        stop_pipeline(queues, running);
        request_quit_from_worker();
    } catch (const std::exception& e) {
        std::cerr << "Pose error: " << e.what() << std::endl;
        stop_pipeline(queues, running);
        request_quit_from_worker();
    }
}

void hand_loop(const Options& options,
               BoundedQueue<FramePacket>* hand_queue,
               ResultStore* results,
               PipelineQueues* queues,
               std::atomic<bool>* running) {
    try {
        dxrt::InferenceEngine palm_engine(options.palm_model_path);
        dxrt::InferenceEngine landmark_engine(options.landmark_model_path);
        validate_hand_model_shapes(palm_engine, landmark_engine);
        landmark_engine.RegisterCallback([](dxrt::TensorPtrs& outputs, void* user_data) -> int {
            landmark_callback(outputs, user_data);
            return 0;
        });

        const auto anchors = generate_palm_anchors();
        if (anchors.size() != static_cast<std::size_t>(kPalmAnchorCount)) {
            throw std::runtime_error("internal palm anchor generation failed");
        }

        const auto poll = std::chrono::milliseconds(kQueuePollMs);
        while (true) {
            if (!running->load() && !hand_queue->closed()) break;

            FramePacket packet;
            if (!hand_queue->pop_for(&packet, poll)) {
                if (hand_queue->closed() || !running->load()) break;
                continue;
            }

            PreprocessInfo prep;
            std::vector<std::uint8_t> palm_input =
                preprocess_palm_frame(packet.frame, options.keep_aspect, &prep);
            auto palm_outputs = palm_engine.Run(palm_input.data());
            std::vector<PalmDetection> palms =
                decode_palm_detections(palm_outputs, anchors, prep, packet.frame.size(),
                                       options.palm_confidence, options.nms_threshold,
                                       options.max_hands);
            std::vector<HandLandmarkResult> hands =
                run_landmark_async(landmark_engine, packet.frame, palms,
                                   options.landmark_confidence);
            results->set_hand(HandPacket{packet.id, std::move(palms), std::move(hands)});
        }
    } catch (const dxrt::Exception& e) {
        std::cerr << "Hand DXRT error: " << e.what() << std::endl;
        stop_pipeline(queues, running);
        request_quit_from_worker();
    } catch (const std::exception& e) {
        std::cerr << "Hand error: " << e.what() << std::endl;
        stop_pipeline(queues, running);
        request_quit_from_worker();
    }
}

void render_loop(const Options& options,
                 FrameView* view,
                 BoundedQueue<FramePacket>* render_queue,
                 ResultStore* results,
                 PipelineQueues* queues,
                 std::atomic<bool>* running,
                 double output_fps,
                 const std::string& output_path) {
    try {
        FpsCounter fps_counter;
        cv::VideoWriter output_writer;
        const auto poll = std::chrono::milliseconds(kQueuePollMs);
        const auto result_wait = std::chrono::milliseconds(kRenderResultWaitMs);

        while (true) {
            if (!running->load() && !render_queue->closed()) break;

            FramePacket packet;
            if (!render_queue->pop_for(&packet, poll)) {
                if (render_queue->closed() || !running->load()) break;
                continue;
            }

            ResultBundle bundle =
                results->wait_for(packet.id, options.show_pose_overlay, true, result_wait);

            std::vector<PalmDetection> palms;
            std::vector<HandLandmarkResult> hands;
            if (bundle.has_hand) {
                palms = bundle.hand.palms;
                hands = bundle.hand.hands;
            }

            std::vector<PoseResult> poses;
            double pose_ms = 0.0;
            if (options.show_pose_overlay && bundle.has_pose) {
                poses = bundle.pose.poses;
                pose_ms = bundle.pose.inference_ms;
            }

            const double fps = fps_counter.update();
            HudMetrics metrics;
            metrics.hand_count = hands.size();
            metrics.palm_count = palms.size();
            metrics.pose_count = poses.size();
            metrics.fps = fps;
            metrics.pose_ms = pose_ms;
            metrics.max_hands = options.max_hands;
            metrics.show_palm_overlay = options.show_palm_overlay;
            metrics.show_pose_overlay = options.show_pose_overlay;
            draw_results(packet.frame, palms, hands, poses, options.show_palm_overlay);
            if (options.save_output) {
                if (!output_writer.isOpened()) {
                    output_writer.open(output_path,
                                       cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                                       output_fps, packet.frame.size(), true);
                    if (!output_writer.isOpened()) {
                        throw std::runtime_error("failed to open output video file: " + output_path);
                    }
                }
                output_writer.write(packet.frame);
            }
            publish_frame(view, packet.frame, metrics);

            const FrameId keep_from =
                packet.id > kResultHistoryFrames ? packet.id - kResultHistoryFrames : 0;
            results->prune_before(keep_from);
        }
    } catch (const std::exception& e) {
        std::cerr << "Render error: " << e.what() << std::endl;
        stop_pipeline(queues, running);
        request_quit_from_worker();
    }
}

void run_detection_loop(const Options& options, FrameView* view, std::atomic<bool>* running) {
    try {
        cv::VideoCapture cap;
        if (options.use_camera) {
            cap.open(options.camera_index, cv::CAP_V4L2);
            cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
            cap.set(cv::CAP_PROP_FRAME_WIDTH, options.camera_width);
            cap.set(cv::CAP_PROP_FRAME_HEIGHT, options.camera_height);
            cap.set(cv::CAP_PROP_FPS, options.camera_fps);
        } else {
            cap.open(options.video_path);
        }
        if (!cap.isOpened()) {
            throw std::runtime_error(options.use_camera ? "failed to open webcam"
                                                        : "failed to open video file");
        }

        const double source_fps = cap.get(cv::CAP_PROP_FPS);
        const bool pace_video = !options.use_camera && source_fps > 1.0 && source_fps < 240.0;
        const auto frame_interval =
            pace_video ? std::chrono::duration<double>(1.0 / source_fps)
                       : std::chrono::duration<double>(0.0);
        const std::string output_path = options.save_output ? next_output_video_path() : "";
        const double output_fps = select_output_fps(options, source_fps);
        if (options.save_output) {
            std::cout << "Saving rendered output to " << output_path
                      << " at " << std::fixed << std::setprecision(1) << output_fps
                      << " FPS" << std::endl;
        }

        PipelineQueues queues;
        ResultStore results;

        std::thread capture_thread(capture_loop, std::cref(options), &cap, &queues, running,
                                   pace_video, frame_interval);
        std::thread dispatch_thread(dispatch_loop, std::cref(options), &queues, running);
        std::thread hand_thread(hand_loop, std::cref(options), &queues.hand, &results, &queues,
                                running);
        std::thread pose_thread;
        if (options.show_pose_overlay) {
            pose_thread = std::thread(pose_loop, std::cref(options), &queues.pose, &results,
                                      &queues, running);
        } else {
            queues.pose.close();
        }
        std::thread render_thread(render_loop, std::cref(options), view, &queues.render,
                                  &results, &queues, running, output_fps, output_path);

        if (capture_thread.joinable()) capture_thread.join();
        if (dispatch_thread.joinable()) dispatch_thread.join();
        if (hand_thread.joinable()) hand_thread.join();
        if (pose_thread.joinable()) pose_thread.join();
        if (render_thread.joinable()) render_thread.join();

        if (running->load()) request_quit_from_worker();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        request_quit_from_worker();
    }
}

}  // namespace

int main(int argc, char** argv) {
    Options options;
    if (!parse_args(argc, argv, &options)) return 1;
    if (!file_exists(options.palm_model_path)) {
        std::cerr << "Error: palm model file not found: " << options.palm_model_path << std::endl;
        return 1;
    }
    if (!file_exists(options.landmark_model_path)) {
        std::cerr << "Error: landmark model file not found: " << options.landmark_model_path << std::endl;
        return 1;
    }
    if (!file_exists(options.pose_model_path)) {
        std::cerr << "Error: pose model file not found: " << options.pose_model_path << std::endl;
        return 1;
    }
    if (!options.use_camera && !file_exists(options.video_path)) {
        std::cerr << "Error: video file not found: " << options.video_path << std::endl;
        return 1;
    }

    QApplication app(argc, argv);
    FrameView view;
    view.setExitButtonVisible(options.show_exit_button);
    if (options.fullscreen) {
        view.showFullScreen();
    } else {
        view.resize(1280, 720);
        view.show();
    }

    std::atomic<bool> running(true);
    std::thread worker(run_detection_loop, options, &view, &running);
    QObject::connect(&app, &QCoreApplication::aboutToQuit, [&running]() {
        running.store(false);
    });

    const int rc = app.exec();
    running.store(false);
    if (worker.joinable()) worker.join();
    return rc;
}
