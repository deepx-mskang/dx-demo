#include <QtCore/QCoreApplication>
#include <QtCore/QMetaObject>
#include <QtCore/QString>
#include <QtGui/QColor>
#include <QtGui/QFont>
#include <QtGui/QLinearGradient>
#include <QtGui/QImage>
#include <QtGui/QKeyEvent>
#include <QtGui/QPainter>
#include <QtGui/QPen>
#include <QtWidgets/QApplication>
#include <QtWidgets/QWidget>

#include <dxrt/dxrt_api.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr const char* kDefaultModelPath = "assets/hand-detector_192x192.dxnn";
constexpr int kModelWidth = 192;
constexpr int kModelHeight = 192;
constexpr int kModelChannels = 3;
constexpr int kPalmAnchorCount = 2016;
constexpr int kPalmBoxCoords = 18;
constexpr int kPalmKeypointCount = 7;
constexpr float kDefaultConfidence = 0.2f;
constexpr float kDefaultNmsThreshold = 0.3f;
constexpr int kDefaultMaxDetections = 8;
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

struct Options {
    std::string model_path = kDefaultModelPath;
    bool use_camera = true;
    int camera_index = 0;
    std::string video_path;
    int camera_width = kDefaultCameraWidth;
    int camera_height = kDefaultCameraHeight;
    int camera_fps = kDefaultCameraFps;
    float confidence = kDefaultConfidence;
    float nms_threshold = kDefaultNmsThreshold;
    int max_detections = kDefaultMaxDetections;
    bool fullscreen = true;
    bool loop_video = false;
    bool keep_aspect = true;
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

struct NormalizedRect {
    float x_center = 0.0f;
    float y_center = 0.0f;
    float width = 0.0f;
    float height = 0.0f;
    float rotation = 0.0f;
};

struct PalmCandidate {
    cv::Rect2f box;
    std::array<cv::Point2f, kPalmKeypointCount> keypoints;
    float score = 0.0f;
};

struct Detection {
    cv::Rect2f box;
    std::array<cv::Point2f, kPalmKeypointCount> keypoints;
    NormalizedRect hand_roi;
    float score = 0.0f;
};

bool file_exists(const std::string& path) {
    std::ifstream in(path.c_str(), std::ios::binary);
    return static_cast<bool>(in);
}

bool parse_int(const std::string& value, int* out) {
    try {
        std::size_t used = 0;
        const int parsed = std::stoi(value, &used);
        if (used != value.size()) {
            return false;
        }
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
        if (used != value.size()) {
            return false;
        }
        *out = parsed;
        return true;
    } catch (...) {
        return false;
    }
}

void print_usage(const char* argv0) {
    std::cout
        << "Usage: " << argv0 << " [OPTIONS]\n"
        << "  -m, --model <PATH>          .dxnn model path (default: "
        << kDefaultModelPath << ")\n"
        << "  -c, --camera [INDEX]        Use webcam (default index: 0)\n"
        << "  -v, --video <PATH>          Use video file input\n"
        << "      --width <N>             Requested webcam width (default: "
        << kDefaultCameraWidth << ")\n"
        << "      --height <N>            Requested webcam height (default: "
        << kDefaultCameraHeight << ")\n"
        << "      --fps <N>               Requested webcam FPS (default: "
        << kDefaultCameraFps << ")\n"
        << "      --conf <FLOAT>          Confidence threshold (default: 0.2)\n"
        << "      --nms <FLOAT>           Weighted NMS IoU threshold (default: 0.3)\n"
        << "      --max-detections <N>    Max boxes to draw (default: 8)\n"
        << "      --loop                  Loop video input\n"
        << "      --stretch               Resize input without letterbox\n"
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
        } else if (arg == "-m" || arg == "--model") {
            const char* value = require_value(arg.c_str());
            if (value == nullptr) {
                return false;
            }
            options->model_path = value;
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
            if (value == nullptr) {
                return false;
            }
            options->use_camera = false;
            options->video_path = value;
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
        } else if (arg == "--conf") {
            const char* value = require_value(arg.c_str());
            if (value == nullptr || !parse_float(value, &options->confidence) ||
                options->confidence < 0.0f || options->confidence > 1.0f) {
                std::cerr << "Error: --conf expects a float in [0, 1]" << std::endl;
                return false;
            }
        } else if (arg == "--nms") {
            const char* value = require_value(arg.c_str());
            if (value == nullptr || !parse_float(value, &options->nms_threshold) ||
                options->nms_threshold < 0.0f || options->nms_threshold > 1.0f) {
                std::cerr << "Error: --nms expects a float in [0, 1]" << std::endl;
                return false;
            }
        } else if (arg == "--max-detections") {
            const char* value = require_value(arg.c_str());
            if (value == nullptr || !parse_int(value, &options->max_detections) ||
                options->max_detections <= 0) {
                std::cerr << "Error: --max-detections expects a positive integer" << std::endl;
                return false;
            }
        } else if (arg == "--loop") {
            options->loop_video = true;
        } else if (arg == "--stretch") {
            options->keep_aspect = false;
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

    if (!options->use_camera && options->video_path.empty()) {
        std::cerr << "Error: --video requires a path" << std::endl;
        return false;
    }
    return true;
}

void notify_launcher_ready() {
    const char* path = std::getenv("DX_LAUNCHER_READY_FILE");
    if (path == nullptr || *path == '\0') {
        return;
    }

    std::ofstream out(path, std::ios::app);
    if (out) {
        out << "ready\n";
    }
}

float sigmoid(float value) {
    value = std::max(-100.0f, std::min(100.0f, value));
    return 1.0f / (1.0f + std::exp(-value));
}

float clamp_float(float value, float low, float high) {
    return std::max(low, std::min(high, value));
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
        if (kNumLayers == 1) {
            return (kMinScale + kMaxScale) * 0.5f;
        }
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
            scales.push_back(std::sqrt(scale * next_scale));
            aspect_ratios.push_back(kInterpolatedScaleAspectRatio);
            ++last_same_stride_layer;
        }

        for (std::size_t i = 0; i < aspect_ratios.size(); ++i) {
            const float ratio_sqrt = std::sqrt(aspect_ratios[i]);
            anchor_heights.push_back(scales[i] / ratio_sqrt);
            anchor_widths.push_back(scales[i] * ratio_sqrt);
        }

        const int feature_map_height =
            static_cast<int>(std::ceil(static_cast<float>(kModelHeight) / strides[layer_id]));
        const int feature_map_width =
            static_cast<int>(std::ceil(static_cast<float>(kModelWidth) / strides[layer_id]));

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

cv::Rect2f clip_rect(const cv::Rect2f& rect, int width, int height) {
    const float x1 = clamp_float(rect.x, 0.0f, static_cast<float>(width - 1));
    const float y1 = clamp_float(rect.y, 0.0f, static_cast<float>(height - 1));
    const float x2 = clamp_float(rect.x + rect.width, 0.0f, static_cast<float>(width - 1));
    const float y2 = clamp_float(rect.y + rect.height, 0.0f, static_cast<float>(height - 1));
    return cv::Rect2f(x1, y1, std::max(0.0f, x2 - x1), std::max(0.0f, y2 - y1));
}

float normalize_radians(float angle) {
    return angle - 2.0f * kPi * std::floor((angle + kPi) / (2.0f * kPi));
}

float intersection_area(const cv::Rect2f& a, const cv::Rect2f& b) {
    const float x1 = std::max(a.x, b.x);
    const float y1 = std::max(a.y, b.y);
    const float x2 = std::min(a.x + a.width, b.x + b.width);
    const float y2 = std::min(a.y + a.height, b.y + b.height);
    const float inter_w = std::max(0.0f, x2 - x1);
    const float inter_h = std::max(0.0f, y2 - y1);
    return inter_w * inter_h;
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

    if (total_score <= 0.0f) {
        return out;
    }

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
    for (int i = 0; i < static_cast<int>(detections.size()); ++i) {
        remaining.push_back(i);
    }

    std::vector<PalmCandidate> selected;
    selected.reserve(static_cast<std::size_t>(max_detections));
    while (!remaining.empty() && static_cast<int>(selected.size()) < max_detections) {
        const int top_idx = remaining.front();
        std::vector<int> cluster;
        std::vector<int> next_remaining;
        cluster.reserve(remaining.size());
        next_remaining.reserve(remaining.size());

        for (int idx : remaining) {
            if (intersection_over_union(detections[idx].box, detections[top_idx].box) >
                threshold) {
                cluster.push_back(idx);
            } else {
                next_remaining.push_back(idx);
            }
        }

        selected.push_back(weighted_candidate(detections, cluster));
        if (next_remaining.size() == remaining.size()) {
            break;
        }
        remaining = std::move(next_remaining);
    }
    return selected;
}

std::vector<std::uint8_t> preprocess_frame(const cv::Mat& bgr,
                                           bool keep_aspect,
                                           PreprocessInfo* info) {
    if (bgr.empty()) {
        throw std::runtime_error("empty frame");
    }

    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

    cv::Mat input(kModelHeight, kModelWidth, CV_8UC3, cv::Scalar(0, 0, 0));
    if (keep_aspect) {
        const float scale = std::min(static_cast<float>(kModelWidth) / static_cast<float>(rgb.cols),
                                     static_cast<float>(kModelHeight) / static_cast<float>(rgb.rows));
        const int resized_w = std::max(1, static_cast<int>(std::round(rgb.cols * scale)));
        const int resized_h = std::max(1, static_cast<int>(std::round(rgb.rows * scale)));
        const int pad_x = (kModelWidth - resized_w) / 2;
        const int pad_y = (kModelHeight - resized_h) / 2;

        cv::Mat resized;
        cv::resize(rgb, resized, cv::Size(resized_w, resized_h), 0.0, 0.0, cv::INTER_LINEAR);
        resized.copyTo(input(cv::Rect(pad_x, pad_y, resized_w, resized_h)));

        info->scale_x = scale;
        info->scale_y = scale;
        info->pad_x = static_cast<float>(pad_x);
        info->pad_y = static_cast<float>(pad_y);
    } else {
        cv::resize(rgb, input, cv::Size(kModelWidth, kModelHeight), 0.0, 0.0, cv::INTER_LINEAR);
        info->scale_x = static_cast<float>(kModelWidth) / static_cast<float>(rgb.cols);
        info->scale_y = static_cast<float>(kModelHeight) / static_cast<float>(rgb.rows);
        info->pad_x = 0.0f;
        info->pad_y = 0.0f;
    }

    if (!input.isContinuous()) {
        input = input.clone();
    }

    const std::size_t bytes =
        static_cast<std::size_t>(input.rows) * static_cast<std::size_t>(input.step);
    std::vector<std::uint8_t> tensor(bytes);
    std::memcpy(tensor.data(), input.data, bytes);
    return tensor;
}

dxrt::Tensor* find_float_tensor(const dxrt::TensorPtrs& outputs,
                                int anchor_count,
                                int last_dim) {
    for (const auto& tensor : outputs) {
        if (tensor->type() != dxrt::DataType::FLOAT) {
            continue;
        }
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
    const float rotation = roi.rotation;
    const float x_shift = (image_w * width * kHandRoiShiftX * std::cos(rotation) -
                           image_h * height * kHandRoiShiftY * std::sin(rotation)) /
                          image_w;
    const float y_shift = (image_w * width * kHandRoiShiftX * std::sin(rotation) +
                           image_h * height * kHandRoiShiftY * std::cos(rotation)) /
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
    const float cx = roi.x_center * static_cast<float>(frame_size.width);
    const float cy = roi.y_center * static_cast<float>(frame_size.height);
    const float w = roi.width * static_cast<float>(frame_size.width);
    const float h = roi.height * static_cast<float>(frame_size.height);
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

[[maybe_unused]] cv::Mat crop_hand_roi_for_landmark(const cv::Mat& frame,
                                                    const NormalizedRect& roi,
                                                    const cv::Size& output_size) {
    const auto corners = hand_roi_corners(roi, frame.size());
    const std::array<cv::Point2f, 4> dst = {
        cv::Point2f(0.0f, 0.0f),
        cv::Point2f(static_cast<float>(output_size.width - 1), 0.0f),
        cv::Point2f(static_cast<float>(output_size.width - 1),
                    static_cast<float>(output_size.height - 1)),
        cv::Point2f(0.0f, static_cast<float>(output_size.height - 1)),
    };

    const cv::Mat transform = cv::getPerspectiveTransform(corners.data(), dst.data());
    cv::Mat cropped;
    cv::warpPerspective(frame, cropped, transform, output_size, cv::INTER_LINEAR,
                        cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    return cropped;
}

std::vector<Detection> decode_palm_detections(const dxrt::TensorPtrs& outputs,
                                              const std::vector<Anchor>& anchors,
                                              const PreprocessInfo& prep,
                                              const cv::Size& frame_size,
                                              float confidence,
                                              float nms_threshold,
                                              int max_detections) {
    const int anchor_count = static_cast<int>(anchors.size());
    dxrt::Tensor* box_tensor = find_float_tensor(outputs, anchor_count, kPalmBoxCoords);
    dxrt::Tensor* score_tensor = find_float_tensor(outputs, anchor_count, 1);
    if (box_tensor == nullptr || score_tensor == nullptr) {
        throw std::runtime_error("unexpected model outputs: expected FLOAT [1,2016,18] and [1,2016,1]");
    }

    const float* raw_boxes = static_cast<const float*>(box_tensor->data());
    const float* raw_scores = static_cast<const float*>(score_tensor->data());
    std::vector<PalmCandidate> candidates;
    candidates.reserve(64);

    auto model_to_frame_x = [&](float normalized_x) {
        const float model_x = normalized_x * static_cast<float>(kModelWidth);
        return (model_x - prep.pad_x) / prep.scale_x;
    };
    auto model_to_frame_y = [&](float normalized_y) {
        const float model_y = normalized_y * static_cast<float>(kModelHeight);
        return (model_y - prep.pad_y) / prep.scale_y;
    };

    for (int i = 0; i < anchor_count; ++i) {
        const float score = sigmoid(raw_scores[i]);
        if (score < confidence) {
            continue;
        }

        const float* box = raw_boxes + static_cast<std::size_t>(i) * kPalmBoxCoords;
        const Anchor& anchor = anchors[static_cast<std::size_t>(i)];

        const float x_center = box[0] / static_cast<float>(kModelWidth) * anchor.w + anchor.x_center;
        const float y_center = box[1] / static_cast<float>(kModelHeight) * anchor.h + anchor.y_center;
        const float box_w = box[2] / static_cast<float>(kModelWidth) * anchor.w;
        const float box_h = box[3] / static_cast<float>(kModelHeight) * anchor.h;
        cv::Rect2f candidate_box(x_center - box_w * 0.5f, y_center - box_h * 0.5f,
                                 box_w, box_h);

        if (candidate_box.width <= 0.0f || candidate_box.height <= 0.0f) {
            continue;
        }

        PalmCandidate candidate;
        candidate.box = candidate_box;
        candidate.score = score;
        for (int k = 0; k < kPalmKeypointCount; ++k) {
            const int offset = 4 + k * 2;
            candidate.keypoints[k].x =
                box[offset] / static_cast<float>(kModelWidth) * anchor.w + anchor.x_center;
            candidate.keypoints[k].y =
                box[offset + 1] / static_cast<float>(kModelHeight) * anchor.h + anchor.y_center;
        }
        candidates.push_back(candidate);
    }

    std::vector<PalmCandidate> weighted =
        weighted_non_max_suppression(std::move(candidates), nms_threshold, max_detections);

    std::vector<Detection> detections;
    detections.reserve(weighted.size());
    for (const auto& candidate : weighted) {
        const float x1 = model_to_frame_x(candidate.box.x);
        const float y1 = model_to_frame_y(candidate.box.y);
        const float x2 = model_to_frame_x(candidate.box.x + candidate.box.width);
        const float y2 = model_to_frame_y(candidate.box.y + candidate.box.height);

        const float left = std::min(x1, x2);
        const float top = std::min(y1, y2);
        const float right = std::max(x1, x2);
        const float bottom = std::max(y1, y2);
        const cv::Rect2f frame_box(left, top, right - left, bottom - top);
        cv::Rect2f clipped = clip_rect(frame_box, frame_size.width, frame_size.height);
        if (clipped.width < 2.0f || clipped.height < 2.0f) {
            continue;
        }

        Detection det;
        det.box = clipped;
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

std::string format_float(float value, int precision) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << value;
    return oss.str();
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
        cv::rectangle(frame, bg_rect, bg, cv::FILLED, cv::LINE_AA);
        cv::putText(frame, text, cv::Point(x + pad_x, y - baseline - pad_y),
                    cv::FONT_HERSHEY_SIMPLEX, font_scale, fg, thickness, cv::LINE_AA);
    }
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
    const int point_radius = std::max(1, cvRound(static_cast<float>(radius + 1) * 0.2f));
    const int raw_glow_radius = std::max(radius * 3, radius + 2);
    const int raw_aura_radius = std::max(radius * 2, radius + 1);
    const int scaled_glow_radius =
        cvRound(static_cast<float>(raw_glow_radius) * 0.2f);
    const int scaled_aura_radius =
        cvRound(static_cast<float>(raw_aura_radius) * 0.2f);
    const int glow_radius =
        point_radius + std::max(1, cvRound(static_cast<float>(
                                 std::max(0, scaled_glow_radius - point_radius)) * 0.5f));
    const int aura_radius =
        point_radius + std::max(1, cvRound(static_cast<float>(
                                 std::max(0, scaled_aura_radius - point_radius)) * 0.5f));
    blend_circle(frame, center, glow_radius, color, 0.11);
    blend_circle(frame, center, aura_radius, color, 0.19);

    cv::circle(frame, center, point_radius, color, cv::FILLED, cv::LINE_AA);
    cv::circle(frame, center, std::max(1, cvRound(static_cast<float>(point_radius) * 0.2f)),
               cv::Scalar(255, 255, 255),
               cv::FILLED, cv::LINE_AA);
}

void draw_detections(cv::Mat& frame, const std::vector<Detection>& detections) {
    const int base = std::max(1, std::min(frame.cols, frame.rows));
    const int box_thickness = std::max(1, base / 280);
    const int text_thickness = std::max(1, base / 720);
    const double font_scale = std::max(0.45, std::min(0.95, base / 900.0));
    const cv::Scalar box_color(48, 220, 255);
    const cv::Scalar roi_color(80, 255, 120);
    const cv::Scalar keypoint_color(255, 255, 32);
    const cv::Scalar text_fg(20, 24, 28);
    const cv::Scalar label_bg(48, 220, 255);
    const int keypoint_radius = std::max(3, box_thickness + 1);

    for (const auto& det : detections) {
        cv::rectangle(frame, det.box, box_color, box_thickness, cv::LINE_AA);
        const auto corners = hand_roi_corners(det.hand_roi, frame.size());
        for (std::size_t i = 0; i < corners.size(); ++i) {
            cv::line(frame, corners[i], corners[(i + 1) % corners.size()], roi_color,
                     std::max(1, box_thickness - 1), cv::LINE_AA);
        }
        for (const auto& point : det.keypoints) {
            draw_neon_point(frame, cv::Point(cvRound(point.x), cvRound(point.y)),
                            keypoint_radius, keypoint_color);
        }
        const std::string label = "hand " + format_float(det.score, 2);
        const int label_x = static_cast<int>(std::round(det.box.x));
        const int label_y = static_cast<int>(std::round(det.box.y)) - 4;
        draw_label(frame, label, cv::Point(label_x, label_y), font_scale, text_thickness,
                   text_fg, label_bg);
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
    double fps = 0.0;
    double npu_ms = 0.0;
    float confidence = 0.0f;
};

class FrameView : public QWidget {
public:
    explicit FrameView(QWidget* parent = nullptr) : QWidget(parent) {
        setWindowTitle("DEEPX M1 Hand Detector");
        setFocusPolicy(Qt::StrongFocus);
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
        if (frame_.isNull()) {
            return;
        }

        QSize target_size = frame_.size();
        target_size.scale(size(), Qt::KeepAspectRatio);
        const QPoint top_left(0, (height() - target_size.height()) / 2);
        painter.drawImage(QRect(top_left, target_size), frame_);

        const QRect panel_rect(target_size.width(), 0,
                               std::max(0, width() - target_size.width()), height());
        drawSidePanel(painter, panel_rect);

        if (!launcher_ready_notified_) {
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

private:
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
        eyebrow.setPixelSize(13);
        eyebrow.setBold(true);
        painter.setFont(eyebrow);
        painter.setPen(QColor(69, 241, 255));
        painter.drawText(QRect(content.left(), y, content.width(), 18),
                         Qt::AlignLeft | Qt::AlignVCenter, "DEEPX M1");
        y += 28;

        QFont title = painter.font();
        title.setPixelSize(30);
        title.setBold(true);
        painter.setFont(title);
        painter.setPen(QColor(245, 248, 252));
        painter.drawText(QRect(content.left(), y, content.width(), 40),
                         Qt::AlignLeft | Qt::AlignVCenter, "HAND DETECTOR");
        y += 44;

        QFont subtitle = painter.font();
        subtitle.setPixelSize(14);
        subtitle.setBold(false);
        painter.setFont(subtitle);
        painter.setPen(QColor(154, 166, 180));
        painter.drawText(QRect(content.left(), y, content.width(), 22),
                         Qt::AlignLeft | Qt::AlignVCenter, "Realtime palm tracking");
        y += 42;

        painter.setPen(QPen(QColor(255, 255, 255, 42), 1));
        painter.drawLine(content.left(), y, content.right(), y);
        y += 32;

        const int gap = 14;
        const int card_h = std::max(72, std::min(104, (content.bottom() - y - 86 - gap * 2) / 3));
        drawMetricCard(painter, QRect(content.left(), y, content.width(), card_h),
                       "HANDS", QString::fromStdString(std::to_string(metrics_.hand_count)),
                       QColor(83, 255, 134));
        y += card_h + gap;
        drawMetricCard(painter, QRect(content.left(), y, content.width(), card_h),
                       "FPS", QString::number(metrics_.fps, 'f', 1),
                       QColor(47, 221, 255));
        y += card_h + gap;
        drawMetricCard(painter, QRect(content.left(), y, content.width(), card_h),
                       "NPU", QString::number(metrics_.npu_ms, 'f', 2) + " ms",
                       QColor(255, 217, 74));
    }

    QImage frame_;
    HudMetrics metrics_;
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

void run_detection_loop(const Options& options, FrameView* view, std::atomic<bool>* running) {
    try {
        dxrt::InferenceEngine engine(options.model_path);
        const auto anchors = generate_palm_anchors();
        if (anchors.size() != static_cast<std::size_t>(kPalmAnchorCount)) {
            throw std::runtime_error("internal anchor generation failed");
        }

        const auto input_shape = engine.GetInputs().front().shape();
        if (input_shape.size() != 4 || input_shape[1] != kModelHeight ||
            input_shape[2] != kModelWidth || input_shape[3] != kModelChannels ||
            engine.GetInputs().front().type() != dxrt::DataType::UINT8) {
            throw std::runtime_error("model input must be UINT8 [1,192,192,3]");
        }

        cv::VideoCapture cap;
        if (options.use_camera) {
            cap.open(options.camera_index, cv::CAP_ANY);
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
        auto next_frame_time = std::chrono::steady_clock::now();
        FpsCounter fps_counter;

        while (running->load()) {
            cv::Mat frame;
            if (!cap.read(frame) || frame.empty()) {
                if (!options.use_camera && options.loop_video) {
                    cap.set(cv::CAP_PROP_POS_FRAMES, 0.0);
                    continue;
                }
                break;
            }
            frame = center_crop_wide_frame_to_4_3(frame);

            PreprocessInfo prep;
            std::vector<std::uint8_t> input =
                preprocess_frame(frame, options.keep_aspect, &prep);
            auto outputs = engine.Run(input.data());
            std::vector<Detection> detections =
                decode_palm_detections(outputs, anchors, prep, frame.size(), options.confidence,
                                       options.nms_threshold, options.max_detections);

            const double fps = fps_counter.update();
            const double npu_ms = static_cast<double>(engine.GetNpuInferenceTime()) / 1000.0;
            HudMetrics metrics;
            metrics.hand_count = detections.size();
            metrics.fps = fps;
            metrics.npu_ms = npu_ms;
            metrics.confidence = options.confidence;
            draw_detections(frame, detections);
            publish_frame(view, frame, metrics);

            if (pace_video) {
                next_frame_time +=
                    std::chrono::duration_cast<std::chrono::steady_clock::duration>(frame_interval);
                std::this_thread::sleep_until(next_frame_time);
                if (next_frame_time < std::chrono::steady_clock::now() - std::chrono::seconds(1)) {
                    next_frame_time = std::chrono::steady_clock::now();
                }
            }
        }

        if (running->load()) {
            request_quit_from_worker();
        }
    } catch (const dxrt::Exception& e) {
        std::cerr << "DXRT error: " << e.what() << std::endl;
        request_quit_from_worker();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        request_quit_from_worker();
    }
}

}  // namespace

int main(int argc, char** argv) {
    Options options;
    if (!parse_args(argc, argv, &options)) {
        return 1;
    }
    if (!file_exists(options.model_path)) {
        std::cerr << "Error: model file not found: " << options.model_path << std::endl;
        return 1;
    }
    if (!options.use_camera && !file_exists(options.video_path)) {
        std::cerr << "Error: video file not found: " << options.video_path << std::endl;
        return 1;
    }

    QApplication app(argc, argv);
    FrameView view;
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
    if (worker.joinable()) {
        worker.join();
    }
    return rc;
}
