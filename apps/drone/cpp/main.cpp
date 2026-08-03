#include <dxrt/dxrt_api.h>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#include <QApplication>
#include <QCoreApplication>
#include <QFont>
#include <QFontMetrics>
#include <QImage>
#include <QKeyEvent>
#include <QMouseEvent>
#include <QPainter>
#include <QTimer>
#include <QWidget>

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr int kTemplateSize = 112;
constexpr int kSearchSize = 224;
constexpr float kTemplateFactor = 2.0f;
constexpr float kSearchFactor = 4.5f;
constexpr int kDefaultUpdateInterval = 200;
constexpr float kTemplateUpdateThreshold = 0.5f;
constexpr float kMaxScoreDecay = 1.0f;
constexpr float kClipMargin = 10.0f;
constexpr std::array<float, 3> kImageMean = {0.485f, 0.456f, 0.406f};
constexpr std::array<float, 3> kImageStd = {0.229f, 0.224f, 0.225f};
constexpr float kMinTrackingScore = 0.15f;
constexpr float kMaxFrameScaleChange = 1.20f;
constexpr float kMaxCenterShiftFactor = 0.75f;
constexpr float kMinCenterShiftPixels = 12.0f;
constexpr float kGoodTrackingScore = 0.5f;
constexpr int kExitButtonWidth = 32;
constexpr int kExitButtonHeight = 28;
constexpr int kExitButtonMargin = 14;

void notify_launcher_ready()
{
    const char* path = std::getenv("DX_LAUNCHER_READY_FILE");
    if (path == nullptr || *path == '\0') {
        return;
    }
    std::ofstream ready(path, std::ios::trunc);
    if (ready) {
        ready << "ready\n";
    }
}

enum class Backend {
    Onnx,
    Dxnn,
};

struct AppOptions {
    Backend backend = Backend::Onnx;
    std::string backend_name = "onnx";
    std::string model_path = "mixformer_sim.onnx";
    std::string video_path = "video1.mp4";
    bool use_camera = false;
    int camera_index = 0;
    int camera_width = 0;
    int camera_height = 0;
    int camera_fps = 0;
    int camera_buffer_size = 0;
    std::string camera_fourcc;
    bool full_screen = false;
    bool show_exit_button = false;
    bool loop = false;
    bool debug = false;
    float min_track_score = kMinTrackingScore;
};

struct TensorOutput {
    std::vector<int64_t> shape;
    std::vector<float> values;
};

struct CropTensor {
    std::vector<float> tensor;
    float resize_factor = 1.0f;
};

struct InputView {
    const float* data = nullptr;
    std::size_t count = 0;
    std::array<int64_t, 4> shape{};
};

std::string to_lower(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

std::string backend_label(Backend backend)
{
    return backend == Backend::Dxnn ? "DXNN" : "ONNX";
}

std::uint64_t element_count(const std::vector<int64_t>& shape)
{
    if (shape.empty()) {
        return 0;
    }
    std::uint64_t count = 1;
    for (const auto dim : shape) {
        if (dim <= 0) {
            continue;
        }
        count *= static_cast<std::uint64_t>(dim);
    }
    return count;
}

float sigmoid(float x)
{
    if (x >= 0.0f) {
        const float z = std::exp(-x);
        return 1.0f / (1.0f + z);
    }
    const float z = std::exp(x);
    return z / (1.0f + z);
}

int fourcc_from_string(const std::string& code)
{
    return cv::VideoWriter::fourcc(code[0], code[1], code[2], code[3]);
}

std::string fourcc_text(int fourcc)
{
    std::string text(4, ' ');
    for (int i = 0; i < 4; ++i) {
        const unsigned char ch = static_cast<unsigned char>((fourcc >> (8 * i)) & 0xff);
        text[static_cast<std::size_t>(i)] = std::isprint(ch) ? static_cast<char>(ch) : '?';
    }
    return text;
}

void print_usage(const char* argv0)
{
    std::cout
        << "Usage: " << argv0 << " --model <PATH> [OPTIONS]\n"
        << "      --backend onnx|dxnn     Inference backend (default: onnx)\n"
        << "      --model <PATH>          Path to model file (required)\n"
        << "      --video <PATH>          Video file input path\n"
        << "      --camera [INDEX]        Use camera input (default index: 0)\n"
        << "      --width <N>             Camera width; requires --camera\n"
        << "      --height <N>            Camera height; requires --camera\n"
        << "      --fps <N>               Camera FPS request; requires --camera\n"
        << "      --camera-buffer-size <N>\n"
        << "                              Set OpenCV camera buffer size when N > 0\n"
        << "      --camera-fourcc CODE    Request camera pixel format, e.g. MJPG or YUYV\n"
        << "      --full_screen           Show the GUI in fullscreen mode\n"
        << "      --exit-btn              Show a clickable exit button at the top-right\n"
        << "      --loop                  Loop video input\n"
        << "      --min-track-score <F>   Reject predictions below this score, 0..1\n"
        << "                              (default: " << kMinTrackingScore << ")\n"
        << "      --debug                 Print tracking anomaly logs\n"
        << "  -h, --help                  Show this help\n"
        << "\n"
        << "Example:\n"
        << "  " << argv0 << " --backend dxnn --model assets/mixformer_sim.dxnn --video assets/drone_test.mp4\n";
}

std::string require_arg_value(int& index, int argc, char** argv)
{
    if (index + 1 >= argc) {
        throw std::runtime_error(std::string("missing value for ") + argv[index]);
    }
    ++index;
    return argv[index];
}

float require_float_arg(int& index, int argc, char** argv)
{
    const std::string name = argv[index];
    const std::string value = require_arg_value(index, argc, argv);
    std::size_t parsed = 0;
    float result = 0.0f;
    try {
        result = std::stof(value, &parsed);
    } catch (const std::exception&) {
        throw std::runtime_error("invalid numeric value for " + name + ": " + value);
    }
    if (parsed != value.size() || !std::isfinite(result)) {
        throw std::runtime_error("invalid numeric value for " + name + ": " + value);
    }
    return result;
}

int require_int_arg(int& index, int argc, char** argv)
{
    const std::string name = argv[index];
    const std::string value = require_arg_value(index, argc, argv);
    std::size_t parsed = 0;
    int result = 0;
    try {
        result = std::stoi(value, &parsed);
    } catch (const std::exception&) {
        throw std::runtime_error("invalid integer value for " + name + ": " + value);
    }
    if (parsed != value.size()) {
        throw std::runtime_error("invalid integer value for " + name + ": " + value);
    }
    return result;
}

AppOptions parse_args(int argc, char** argv)
{
    AppOptions options;
    bool video_requested = false;
    bool camera_requested = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        }
        if (arg == "--backend") {
            options.backend_name = to_lower(require_arg_value(i, argc, argv));
            if (options.backend_name == "onnx") {
                options.backend = Backend::Onnx;
            } else if (options.backend_name == "dxnn") {
                options.backend = Backend::Dxnn;
            } else {
                throw std::runtime_error("unsupported backend: " + options.backend_name);
            }
        } else if (arg == "--model") {
            options.model_path = require_arg_value(i, argc, argv);
        } else if (arg == "--video") {
            options.video_path = require_arg_value(i, argc, argv);
            video_requested = true;
        } else if (arg == "--camera") {
            options.use_camera = true;
            camera_requested = true;
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                options.camera_index = require_int_arg(i, argc, argv);
            }
        } else if (arg == "--width") {
            options.camera_width = require_int_arg(i, argc, argv);
        } else if (arg == "--height") {
            options.camera_height = require_int_arg(i, argc, argv);
        } else if (arg == "--fps") {
            options.camera_fps = require_int_arg(i, argc, argv);
        } else if (arg == "--camera-buffer-size") {
            options.camera_buffer_size = require_int_arg(i, argc, argv);
        } else if (arg == "--camera-fourcc") {
            options.camera_fourcc = require_arg_value(i, argc, argv);
        } else if (arg == "--full_screen") {
            options.full_screen = true;
        } else if (arg == "--exit-btn") {
            options.show_exit_button = true;
        } else if (arg == "--loop") {
            options.loop = true;
        } else if (arg == "--min-track-score") {
            options.min_track_score = require_float_arg(i, argc, argv);
        } else if (arg == "--debug") {
            options.debug = true;
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }

    if (video_requested && camera_requested) {
        throw std::runtime_error("--video and --camera cannot be used together");
    }
    if (options.camera_index < 0) {
        throw std::runtime_error("camera index must be non-negative");
    }
    if (!options.use_camera &&
        (options.camera_width > 0 || options.camera_height > 0 || options.camera_fps > 0 ||
         options.camera_buffer_size > 0 || !options.camera_fourcc.empty())) {
        throw std::runtime_error("camera options require --camera");
    }
    if (options.camera_width < 0 || options.camera_height < 0 ||
        options.camera_fps < 0 || options.camera_buffer_size < 0) {
        throw std::runtime_error("camera width, height, fps and buffer size must be non-negative");
    }
    if (!options.camera_fourcc.empty() && options.camera_fourcc.size() != 4) {
        throw std::runtime_error("--camera-fourcc expects a 4-character code: " +
                                 options.camera_fourcc);
    }
    if (!(options.min_track_score >= 0.0f) || options.min_track_score > 1.0f) {
        throw std::runtime_error("--min-track-score must be within 0..1");
    }
    return options;
}

QImage mat_to_qimage(const cv::Mat& bgr)
{
    if (bgr.empty()) {
        return {};
    }
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    QImage image(rgb.data, rgb.cols, rgb.rows, static_cast<int>(rgb.step), QImage::Format_RGB888);
    return image.copy();
}

cv::Rect2f normalized_rect(const cv::Point2f& a, const cv::Point2f& b)
{
    const float x1 = std::min(a.x, b.x);
    const float y1 = std::min(a.y, b.y);
    const float x2 = std::max(a.x, b.x);
    const float y2 = std::max(a.y, b.y);
    return cv::Rect2f(x1, y1, x2 - x1, y2 - y1);
}

class MixFormerV2Tracker {
public:
    MixFormerV2Tracker(const std::string& model_path,
                       Backend backend,
                       float min_track_score,
                       bool debug)
        : backend_(backend),
          min_track_score_(min_track_score),
          debug_(debug),
          ort_env_(ORT_LOGGING_LEVEL_WARNING, "mixformer_v2")
    {
        if (backend_ == Backend::Onnx) {
            init_onnx(model_path);
        } else {
            init_dxnn(model_path);
        }
        std::cout << "[Info] Input normalization: ImageNet mean/std; min tracking score="
                  << min_track_score_ << std::endl;
    }

    void init(const cv::Mat& image, const cv::Rect2f& init_bbox)
    {
        state_ = init_bbox;
        frame_id_ = 0;
        max_pred_score_ = -1.0f;
        last_pred_score_ = -1.0f;
        last_image_width_ = image.cols;
        last_image_height_ = image.rows;

        template_tensor_ = sample_target(image, state_, kTemplateFactor, kTemplateSize).tensor;
        online_template_tensor_ = template_tensor_;
        online_max_template_tensor_ = template_tensor_;
    }

    float last_pred_score() const
    {
        return last_pred_score_;
    }

    float min_track_score() const
    {
        return min_track_score_;
    }

    cv::Rect update(const cv::Mat& image)
    {
        ++frame_id_;

        const CropTensor search_crop = sample_target(image, state_, kSearchFactor, kSearchSize);
        const std::vector<TensorOutput> outputs = run(search_crop.tensor);

        std::array<float, 4> pred_box{};
        if (!find_pred_box(outputs, pred_box)) {
            throw std::runtime_error("could not find Bounding Box output with last dimension 4");
        }
        const float pred_score = find_pred_score(outputs);
        const std::array<float, 4> raw_pred_box = pred_box;
        const cv::Rect2f previous_state = state_;

        for (float& value : pred_box) {
            value = value * static_cast<float>(kSearchSize) / search_crop.resize_factor;
        }

        const cv::Rect2f mapped_candidate = map_box_back(pred_box, search_crop.resize_factor);
        const cv::Rect2f candidate = stabilize_prediction(mapped_candidate, pred_score);
        state_ = clip_box(candidate,
                          image.rows,
                          image.cols,
                          kClipMargin);

        last_pred_score_ = pred_score;
        debug_log_prediction(image, raw_pred_box, previous_state, mapped_candidate, pred_score);

        update_online_template(image, pred_score);

        return cv::Rect(static_cast<int>(state_.x),
                        static_cast<int>(state_.y),
                        static_cast<int>(state_.width),
                        static_cast<int>(state_.height));
    }

private:
    static void print_box(std::ostream& output, const cv::Rect2f& box)
    {
        output << '[' << box.x << ',' << box.y << ',' << box.width << ',' << box.height << ']';
    }

    static void print_box(std::ostream& output, const std::array<float, 4>& box)
    {
        output << '[' << box[0] << ',' << box[1] << ',' << box[2] << ',' << box[3] << ']';
    }

    // The raw head output is cxcywh normalized to the search crop, so anything far
    // outside the crop or wider than the crop itself means the head lost the target.
    static bool is_raw_range_suspicious(const std::array<float, 4>& raw_pred_box)
    {
        return raw_pred_box[0] < -0.25f || raw_pred_box[0] > 1.25f ||
               raw_pred_box[1] < -0.25f || raw_pred_box[1] > 1.25f ||
               raw_pred_box[2] <= 0.0f || raw_pred_box[2] > 1.5f ||
               raw_pred_box[3] <= 0.0f || raw_pred_box[3] > 1.5f;
    }

    void debug_log_prediction(const cv::Mat& image,
                              const std::array<float, 4>& raw_pred_box,
                              const cv::Rect2f& previous_state,
                              const cv::Rect2f& mapped_candidate,
                              float pred_score)
    {
        if (!debug_) {
            return;
        }

        std::vector<std::string> flags;
        if (image.cols != last_image_width_ || image.rows != last_image_height_) {
            flags.emplace_back("frame-size-change");
        }
        last_image_width_ = image.cols;
        last_image_height_ = image.rows;

        if (!std::isfinite(pred_score)) {
            flags.emplace_back("non-finite-score");
        } else if (pred_score < 0.0f) {
            flags.emplace_back("missing-score");
        } else if (pred_score < min_track_score_) {
            flags.emplace_back("low-score");
        }
        if (is_raw_range_suspicious(raw_pred_box)) {
            flags.emplace_back("raw-range");
        }

        float width_ratio = -1.0f;
        float height_ratio = -1.0f;
        float area_ratio = -1.0f;
        float center_shift = -1.0f;
        float occupancy = -1.0f;
        const float frame_area = static_cast<float>(image.cols) * image.rows;
        const bool comparable = previous_state.width > 0.0f && previous_state.height > 0.0f &&
                                std::isfinite(mapped_candidate.width) &&
                                std::isfinite(mapped_candidate.height) &&
                                frame_area > 0.0f;
        if (comparable) {
            width_ratio = mapped_candidate.width / previous_state.width;
            height_ratio = mapped_candidate.height / previous_state.height;
            const float previous_area = previous_state.width * previous_state.height;
            area_ratio = mapped_candidate.width * mapped_candidate.height / previous_area;

            const float previous_cx = previous_state.x + 0.5f * previous_state.width;
            const float previous_cy = previous_state.y + 0.5f * previous_state.height;
            const float candidate_cx = mapped_candidate.x + 0.5f * mapped_candidate.width;
            const float candidate_cy = mapped_candidate.y + 0.5f * mapped_candidate.height;
            center_shift = std::hypot(candidate_cx - previous_cx, candidate_cy - previous_cy) /
                           std::max(previous_state.width, previous_state.height);

            occupancy = state_.width * state_.height / frame_area;

            if (width_ratio < 0.5f || width_ratio > 2.0f ||
                height_ratio < 0.5f || height_ratio > 2.0f) {
                flags.emplace_back("scale-jump");
            }
            if (area_ratio < (1.0f / 3.0f) || area_ratio > 3.0f) {
                flags.emplace_back("area-jump");
            }
            if (center_shift > 2.0f) {
                flags.emplace_back("center-jump");
            }
            if (occupancy > 0.5f && previous_area / frame_area <= 0.5f) {
                flags.emplace_back("large-frame-box");
            }
        }

        if (flags.empty()) {
            return;
        }

        std::cout << "[Debug][Anomaly] frame=" << frame_id_ << " flags=";
        for (std::size_t i = 0; i < flags.size(); ++i) {
            if (i > 0) {
                std::cout << ',';
            }
            std::cout << flags[i];
        }
        std::cout << " score=" << pred_score
                  << " frame_size=" << image.cols << 'x' << image.rows
                  << " raw=";
        print_box(std::cout, raw_pred_box);
        std::cout << " prev=";
        print_box(std::cout, previous_state);
        std::cout << " mapped=";
        print_box(std::cout, mapped_candidate);
        std::cout << " final=";
        print_box(std::cout, state_);
        std::cout << " ratios=[w:" << width_ratio
                  << ",h:" << height_ratio
                  << ",area:" << area_ratio
                  << ",shift:" << center_shift
                  << ",occupancy:" << occupancy << ']' << std::endl;
    }

    void init_onnx(const std::string& model_path)
    {
        std::cout << "[Info] Loading ONNX model from: " << model_path << std::endl;

        ort_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        const auto providers = Ort::GetAvailableProviders();
        const bool has_cuda = std::find(providers.begin(), providers.end(),
                                        "CUDAExecutionProvider") != providers.end();
        if (has_cuda) {
            try {
                OrtCUDAProviderOptions cuda_options{};
                ort_options_.AppendExecutionProvider_CUDA(cuda_options);
                std::cout << "[Info] ONNX Runtime provider: CUDAExecutionProvider" << std::endl;
            } catch (const Ort::Exception& e) {
                std::cout << "[Warn] CUDAExecutionProvider failed; using CPUExecutionProvider: "
                          << e.what() << std::endl;
            }
        } else {
            std::cout << "[Info] ONNX Runtime provider: CPUExecutionProvider" << std::endl;
        }

        ort_session_ = std::make_unique<Ort::Session>(ort_env_, model_path.c_str(), ort_options_);

        Ort::AllocatorWithDefaultOptions allocator;
        const std::size_t input_count = ort_session_->GetInputCount();
        const std::size_t output_count = ort_session_->GetOutputCount();

        for (std::size_t i = 0; i < input_count; ++i) {
            auto name = ort_session_->GetInputNameAllocated(i, allocator);
            ort_input_names_.emplace_back(name.get());
        }
        for (std::size_t i = 0; i < output_count; ++i) {
            auto name = ort_session_->GetOutputNameAllocated(i, allocator);
            ort_output_names_.emplace_back(name.get());
        }
        refresh_ort_name_ptrs();

        if (ort_input_names_.size() < 3) {
            throw std::runtime_error("ONNX model must have template, online_template, and search inputs");
        }
    }

    void init_dxnn(const std::string& model_path)
    {
        std::cout << "[Info] Loading DXNN model for DEEPX NPU from: " << model_path << std::endl;
        dx_engine_ = std::make_unique<dxrt::InferenceEngine>(model_path);

        dx_input_names_ = dx_engine_->GetInputTensorNames();
        std::cout << "[Info] DXNN input order:";
        for (const auto& name : dx_input_names_) {
            std::cout << ' ' << name;
        }
        std::cout << std::endl;

        if (dx_input_names_.size() < 3) {
            throw std::runtime_error("DXNN model must have template, online_template, and search inputs");
        }
    }

    void refresh_ort_name_ptrs()
    {
        ort_input_name_ptrs_.clear();
        ort_output_name_ptrs_.clear();
        for (const auto& name : ort_input_names_) {
            ort_input_name_ptrs_.push_back(name.c_str());
        }
        for (const auto& name : ort_output_names_) {
            ort_output_name_ptrs_.push_back(name.c_str());
        }
    }

    CropTensor sample_target(const cv::Mat& image,
                             const cv::Rect2f& target_box,
                             float search_area_factor,
                             int output_size) const
    {
        if (image.empty()) {
            throw std::runtime_error("empty image");
        }
        if (target_box.width <= 0.0f || target_box.height <= 0.0f) {
            throw std::runtime_error("too small bounding box");
        }

        const int crop_size = static_cast<int>(
            std::ceil(std::sqrt(target_box.width * target_box.height) * search_area_factor));
        if (crop_size < 1) {
            throw std::runtime_error("too small bounding box");
        }

        const int x1 = static_cast<int>(
            std::round(target_box.x + 0.5f * target_box.width - 0.5f * crop_size));
        const int y1 = static_cast<int>(
            std::round(target_box.y + 0.5f * target_box.height - 0.5f * crop_size));
        const int x2 = x1 + crop_size;
        const int y2 = y1 + crop_size;

        const int pad_left = std::max(0, -x1);
        const int pad_top = std::max(0, -y1);
        const int pad_right = std::max(0, x2 - image.cols + 1);
        const int pad_bottom = std::max(0, y2 - image.rows + 1);

        const int crop_x1 = x1 + pad_left;
        const int crop_y1 = y1 + pad_top;
        const int crop_x2 = x2 - pad_right;
        const int crop_y2 = y2 - pad_bottom;

        const int roi_x = std::clamp(crop_x1, 0, image.cols);
        const int roi_y = std::clamp(crop_y1, 0, image.rows);
        const int roi_right = std::clamp(crop_x2, 0, image.cols);
        const int roi_bottom = std::clamp(crop_y2, 0, image.rows);

        const cv::Rect roi(roi_x, roi_y, roi_right - roi_x, roi_bottom - roi_y);
        if (roi.width <= 0 || roi.height <= 0) {
            throw std::runtime_error("empty crop");
        }

        cv::Mat cropped = image(roi);
        cv::Mat padded;
        cv::copyMakeBorder(cropped,
                           padded,
                           pad_top,
                           pad_bottom,
                           pad_left,
                           pad_right,
                           cv::BORDER_CONSTANT,
                           cv::Scalar(0, 0, 0));

        cv::Mat resized;
        cv::resize(padded, resized, cv::Size(output_size, output_size));
        cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
        resized.convertTo(resized, CV_32FC3, 1.0 / 255.0);

        const int plane_size = output_size * output_size;
        std::vector<float> tensor(static_cast<std::size_t>(3 * plane_size));

        for (int y = 0; y < output_size; ++y) {
            const auto* row = resized.ptr<cv::Vec3f>(y);
            for (int x = 0; x < output_size; ++x) {
                const int offset = y * output_size + x;
                for (int c = 0; c < 3; ++c) {
                    tensor[static_cast<std::size_t>(c * plane_size + offset)] =
                        (row[x][c] - kImageMean[static_cast<std::size_t>(c)]) /
                        kImageStd[static_cast<std::size_t>(c)];
                }
            }
        }

        return {std::move(tensor), static_cast<float>(output_size) / crop_size};
    }

    cv::Rect2f map_box_back(const std::array<float, 4>& pred_box, float resize_factor) const
    {
        const float cx_prev = state_.x + 0.5f * state_.width;
        const float cy_prev = state_.y + 0.5f * state_.height;
        const float half_side = 0.5f * static_cast<float>(kSearchSize) / resize_factor;

        const float cx_real = pred_box[0] + (cx_prev - half_side);
        const float cy_real = pred_box[1] + (cy_prev - half_side);
        return cv::Rect2f(cx_real - 0.5f * pred_box[2],
                          cy_real - 0.5f * pred_box[3],
                          pred_box[2],
                          pred_box[3]);
    }

    cv::Rect2f stabilize_prediction(const cv::Rect2f& candidate, float pred_score) const
    {
        const bool valid = std::isfinite(candidate.x) &&
                           std::isfinite(candidate.y) &&
                           std::isfinite(candidate.width) &&
                           std::isfinite(candidate.height) &&
                           candidate.width > 0.0f &&
                           candidate.height > 0.0f;
        if (!valid || (pred_score >= 0.0f && pred_score < min_track_score_)) {
            return state_;
        }

        if (frame_id_ <= 1) {
            return candidate;
        }

        const float candidate_center_x = candidate.x + 0.5f * candidate.width;
        const float candidate_center_y = candidate.y + 0.5f * candidate.height;
        const float previous_center_x = state_.x + 0.5f * state_.width;
        const float previous_center_y = state_.y + 0.5f * state_.height;

        const float width = std::clamp(candidate.width,
                                       state_.width / kMaxFrameScaleChange,
                                       state_.width * kMaxFrameScaleChange);
        const float height = std::clamp(candidate.height,
                                        state_.height / kMaxFrameScaleChange,
                                        state_.height * kMaxFrameScaleChange);

        const float dx = candidate_center_x - previous_center_x;
        const float dy = candidate_center_y - previous_center_y;
        const float distance = std::hypot(dx, dy);
        const float max_shift = std::max(
            kMinCenterShiftPixels,
            kMaxCenterShiftFactor * std::hypot(state_.width, state_.height));

        float center_x = candidate_center_x;
        float center_y = candidate_center_y;
        if (distance > max_shift) {
            const float ratio = max_shift / distance;
            center_x = previous_center_x + dx * ratio;
            center_y = previous_center_y + dy * ratio;
        }

        return cv::Rect2f(center_x - 0.5f * width,
                          center_y - 0.5f * height,
                          width,
                          height);
    }

    cv::Rect2f clip_box(const cv::Rect2f& box, int image_h, int image_w, float margin) const
    {
        float x1 = box.x;
        float y1 = box.y;
        float x2 = box.x + box.width;
        float y2 = box.y + box.height;

        x1 = std::min(std::max(0.0f, x1), static_cast<float>(image_w) - margin);
        x2 = std::min(std::max(margin, x2), static_cast<float>(image_w));
        y1 = std::min(std::max(0.0f, y1), static_cast<float>(image_h) - margin);
        y2 = std::min(std::max(margin, y2), static_cast<float>(image_h));

        return cv::Rect2f(x1,
                          y1,
                          std::max(margin, x2 - x1),
                          std::max(margin, y2 - y1));
    }

    void update_online_template(const cv::Mat& image, float pred_score)
    {
        if (pred_score >= 0.0f) {
            max_pred_score_ *= kMaxScoreDecay;
            if (pred_score > kTemplateUpdateThreshold && pred_score > max_pred_score_) {
                online_max_template_tensor_ =
                    sample_target(image, state_, kTemplateFactor, kTemplateSize).tensor;
                max_pred_score_ = pred_score;
            }
        }

        if (frame_id_ > 0 && frame_id_ % kDefaultUpdateInterval == 0) {
            online_template_tensor_ = online_max_template_tensor_;
            max_pred_score_ = -1.0f;
            online_max_template_tensor_ = template_tensor_;
        }
    }

    InputView input_for_name(const std::string& name,
                             std::size_t index,
                             const std::vector<float>& search_tensor) const
    {
        const std::string lower = to_lower(name);
        const bool is_online = lower.find("online") != std::string::npos;
        const bool is_search = lower.find("search") != std::string::npos;
        const bool is_template = lower.find("template") != std::string::npos;

        if (is_online) {
            return {online_template_tensor_.data(),
                    online_template_tensor_.size(),
                    {1, 3, kTemplateSize, kTemplateSize}};
        }
        if (is_search) {
            return {search_tensor.data(),
                    search_tensor.size(),
                    {1, 3, kSearchSize, kSearchSize}};
        }
        if (is_template) {
            return {template_tensor_.data(),
                    template_tensor_.size(),
                    {1, 3, kTemplateSize, kTemplateSize}};
        }

        if (index == 0) {
            return {template_tensor_.data(),
                    template_tensor_.size(),
                    {1, 3, kTemplateSize, kTemplateSize}};
        }
        if (index == 1) {
            return {online_template_tensor_.data(),
                    online_template_tensor_.size(),
                    {1, 3, kTemplateSize, kTemplateSize}};
        }
        return {search_tensor.data(), search_tensor.size(), {1, 3, kSearchSize, kSearchSize}};
    }

    std::vector<TensorOutput> run(const std::vector<float>& search_tensor)
    {
        return backend_ == Backend::Onnx ? run_onnx(search_tensor) : run_dxnn(search_tensor);
    }

    std::vector<TensorOutput> run_onnx(const std::vector<float>& search_tensor)
    {
        if (!ort_session_) {
            throw std::runtime_error("ONNX Runtime session is not initialized");
        }

        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::vector<Ort::Value> input_values;
        input_values.reserve(ort_input_names_.size());

        for (std::size_t i = 0; i < ort_input_names_.size(); ++i) {
            const InputView input = input_for_name(ort_input_names_[i], i, search_tensor);
            input_values.emplace_back(Ort::Value::CreateTensor<float>(
                memory_info,
                const_cast<float*>(input.data),
                input.count,
                input.shape.data(),
                input.shape.size()));
        }

        Ort::RunOptions run_options;
        std::vector<Ort::Value> ort_outputs = ort_session_->Run(
            run_options,
            ort_input_name_ptrs_.data(),
            input_values.data(),
            input_values.size(),
            ort_output_name_ptrs_.data(),
            ort_output_name_ptrs_.size());

        std::vector<TensorOutput> outputs;
        outputs.reserve(ort_outputs.size());
        for (auto& output : ort_outputs) {
            if (!output.IsTensor()) {
                continue;
            }
            auto info = output.GetTensorTypeAndShapeInfo();
            const std::vector<int64_t> shape = info.GetShape();
            const std::size_t count = info.GetElementCount();
            const float* data = output.GetTensorData<float>();
            outputs.push_back({shape, std::vector<float>(data, data + count)});
        }
        return outputs;
    }

    std::vector<TensorOutput> run_dxnn(const std::vector<float>& search_tensor)
    {
        if (!dx_engine_) {
            throw std::runtime_error("DXRT InferenceEngine is not initialized");
        }

        dxrt::TensorPtrs tensors;
        if (!dx_input_names_.empty()) {
            std::map<std::string, void*> input_map;
            for (std::size_t i = 0; i < dx_input_names_.size(); ++i) {
                const InputView input = input_for_name(dx_input_names_[i], i, search_tensor);
                input_map[dx_input_names_[i]] = const_cast<float*>(input.data);
            }
            tensors = dx_engine_->RunMultiInput(input_map);
        } else {
            std::vector<void*> inputs = {
                template_tensor_.data(),
                online_template_tensor_.data(),
                const_cast<float*>(search_tensor.data()),
            };
            tensors = dx_engine_->RunMultiInput(inputs);
        }

        std::vector<TensorOutput> outputs;
        outputs.reserve(tensors.size());
        for (const auto& tensor : tensors) {
            if (!tensor || tensor->type() != dxrt::DataType::FLOAT) {
                continue;
            }
            const std::vector<int64_t> shape = tensor->shape();
            const auto count = static_cast<std::size_t>(element_count(shape));
            const float* data = static_cast<const float*>(tensor->data());
            outputs.push_back({shape, std::vector<float>(data, data + count)});
        }
        return outputs;
    }

    bool find_pred_box(const std::vector<TensorOutput>& outputs, std::array<float, 4>& box) const
    {
        for (const auto& output : outputs) {
            const bool shape_matches = !output.shape.empty() && output.shape.back() == 4;
            if ((shape_matches || output.values.size() == 4) && output.values.size() >= 4) {
                const std::size_t box_count = output.values.size() / 4;
                if (box_count == 0) {
                    continue;
                }
                box = {0.0f, 0.0f, 0.0f, 0.0f};
                for (std::size_t i = 0; i < box_count; ++i) {
                    for (std::size_t j = 0; j < box.size(); ++j) {
                        box[j] += output.values[i * 4 + j];
                    }
                }
                for (float& value : box) {
                    value /= static_cast<float>(box_count);
                }
                return true;
            }
        }
        return false;
    }

    float find_pred_score(const std::vector<TensorOutput>& outputs) const
    {
        for (const auto& output : outputs) {
            const bool is_box_output = !output.shape.empty() && output.shape.back() == 4;
            if (!is_box_output && output.values.size() == 1) {
                return sigmoid(output.values.front());
            }
        }
        return -1.0f;
    }

    Backend backend_ = Backend::Onnx;
    cv::Rect2f state_{0.0f, 0.0f, 0.0f, 0.0f};
    int frame_id_ = 0;
    float max_pred_score_ = -1.0f;
    float last_pred_score_ = -1.0f;
    float min_track_score_ = kMinTrackingScore;
    bool debug_ = false;
    int last_image_width_ = 0;
    int last_image_height_ = 0;
    std::vector<float> template_tensor_;
    std::vector<float> online_template_tensor_;
    std::vector<float> online_max_template_tensor_;

    Ort::Env ort_env_;
    Ort::SessionOptions ort_options_;
    std::unique_ptr<Ort::Session> ort_session_;
    std::vector<std::string> ort_input_names_;
    std::vector<std::string> ort_output_names_;
    std::vector<const char*> ort_input_name_ptrs_;
    std::vector<const char*> ort_output_name_ptrs_;

    std::unique_ptr<dxrt::InferenceEngine> dx_engine_;
    std::vector<std::string> dx_input_names_;
};

class TrackingWindow : public QWidget {
public:
    TrackingWindow(AppOptions options, std::unique_ptr<MixFormerV2Tracker> tracker)
        : options_(std::move(options)),
          tracker_(std::move(tracker))
    {
        setWindowTitle(QString("MixFormerV2 Tracking (%1)").arg(QString::fromStdString(backend_label(options_.backend))));
        setFocusPolicy(Qt::StrongFocus);
        setMouseTracking(true);
        setMinimumSize(640, 360);

        open_input();

        double fps = cap_.get(cv::CAP_PROP_FPS);
        if (!std::isfinite(fps) || fps <= 1.0) {
            fps = 30.0;
        }
        frame_interval_ms_ = std::max(1, static_cast<int>(std::round(1000.0 / fps)));

        if (!cap_.read(current_frame_) || current_frame_.empty()) {
            throw std::runtime_error("cannot read first frame from " + input_description());
        }
        frame_image_ = mat_to_qimage(current_frame_);

        std::cout << "[" << backend_label(options_.backend)
                  << " Mode] Drag to select the object to track; tracking starts on release."
                  << std::endl;

        connect(&timer_, &QTimer::timeout, this, [this]() {
            process_next_frame();
        });
        timer_.setTimerType(Qt::PreciseTimer);
    }

protected:
    void paintEvent(QPaintEvent*) override
    {
        QPainter painter(this);
        painter.fillRect(rect(), QColor(8, 10, 12));

        if (frame_image_.isNull()) {
            return;
        }

        const QRectF image_rect = image_draw_rect();
        painter.drawImage(image_rect, frame_image_);
        painter.setRenderHint(QPainter::Antialiasing, true);

        draw_tracking_overlay(painter, image_rect);
        draw_selection_overlay(painter, image_rect);
        draw_hud(painter, image_rect);
        draw_exit_button(painter);
        if (!launcher_ready_notified_) {
            notify_launcher_ready();
            launcher_ready_notified_ = true;
        }
    }

    void mousePressEvent(QMouseEvent* event) override
    {
        if (options_.show_exit_button &&
            event->button() == Qt::LeftButton &&
            exit_button_rect().contains(event->pos())) {
            QCoreApplication::quit();
            return;
        }

        if (mode_ != Mode::Selecting || event->button() != Qt::LeftButton) {
            QWidget::mousePressEvent(event);
            return;
        }

        cv::Point2f image_point;
        if (!widget_to_image(event->pos(), image_point)) {
            return;
        }
        dragging_ = true;
        drag_start_ = image_point;
        drag_current_ = image_point;
        selected_roi_ = {};
        update();
    }

    void mouseMoveEvent(QMouseEvent* event) override
    {
        if (mode_ == Mode::Selecting && dragging_) {
            widget_to_image_clamped(event->pos(), drag_current_);
            selected_roi_ = normalized_rect(drag_start_, drag_current_);
            update();
            return;
        }
        QWidget::mouseMoveEvent(event);
    }

    void mouseReleaseEvent(QMouseEvent* event) override
    {
        if (mode_ == Mode::Selecting && dragging_ && event->button() == Qt::LeftButton) {
            widget_to_image_clamped(event->pos(), drag_current_);
            selected_roi_ = normalized_rect(drag_start_, drag_current_);
            dragging_ = false;
            start_tracking();
            return;
        }
        QWidget::mouseReleaseEvent(event);
    }

    void keyPressEvent(QKeyEvent* event) override
    {
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
    enum class Mode {
        Selecting,
        Tracking,
        Finished,
        Error,
    };

    std::string input_description() const
    {
        if (options_.use_camera) {
            return "camera " + std::to_string(options_.camera_index);
        }
        return "video: " + options_.video_path;
    }

    void open_input()
    {
        cap_.release();
        if (!options_.use_camera) {
            if (!cap_.open(options_.video_path)) {
                throw std::runtime_error("cannot open video: " + options_.video_path);
            }
            return;
        }

        if (!cap_.open(options_.camera_index)) {
            throw std::runtime_error("cannot open camera: " +
                                     std::to_string(options_.camera_index));
        }

        if (options_.camera_buffer_size > 0) {
            cap_.set(cv::CAP_PROP_BUFFERSIZE, options_.camera_buffer_size);
        }
        if (!options_.camera_fourcc.empty()) {
            cap_.set(cv::CAP_PROP_FOURCC,
                     static_cast<double>(fourcc_from_string(options_.camera_fourcc)));
        }
        if (options_.camera_width > 0) {
            cap_.set(cv::CAP_PROP_FRAME_WIDTH, options_.camera_width);
        }
        if (options_.camera_height > 0) {
            cap_.set(cv::CAP_PROP_FRAME_HEIGHT, options_.camera_height);
        }
        if (options_.camera_fps > 0) {
            cap_.set(cv::CAP_PROP_FPS, options_.camera_fps);
        }

        // Backends silently ignore unsupported requests, so report what actually took effect.
        const int actual_fourcc = static_cast<int>(std::lround(cap_.get(cv::CAP_PROP_FOURCC)));
        const int actual_width = static_cast<int>(std::lround(cap_.get(cv::CAP_PROP_FRAME_WIDTH)));
        const int actual_height = static_cast<int>(std::lround(cap_.get(cv::CAP_PROP_FRAME_HEIGHT)));
        const double actual_fps = cap_.get(cv::CAP_PROP_FPS);
        std::cout << "[Camera] index=" << options_.camera_index
                  << " format=" << fourcc_text(actual_fourcc)
                  << " resolution=" << actual_width << 'x' << actual_height
                  << " fps=" << actual_fps << std::endl;

        if (!options_.camera_fourcc.empty() &&
            actual_fourcc != fourcc_from_string(options_.camera_fourcc)) {
            std::cerr << "[Warn] Camera kept format " << fourcc_text(actual_fourcc)
                      << " instead of the requested " << options_.camera_fourcc << '.' << std::endl;
        }
        if ((options_.camera_width > 0 && actual_width != options_.camera_width) ||
            (options_.camera_height > 0 && actual_height != options_.camera_height)) {
            std::cerr << "[Warn] Camera adjusted the requested resolution to "
                      << actual_width << 'x' << actual_height << '.' << std::endl;
        }
    }

    QRectF exit_button_rect() const
    {
        const int x = std::max(0, width() - kExitButtonWidth - kExitButtonMargin);
        return QRectF(x, kExitButtonMargin, kExitButtonWidth, kExitButtonHeight);
    }

    void draw_exit_button(QPainter& painter) const
    {
        if (!options_.show_exit_button) {
            return;
        }

        const QRectF button_rect = exit_button_rect();
        painter.save();
        painter.setRenderHint(QPainter::Antialiasing, true);
        painter.setPen(QPen(QColor(60, 60, 60), 1));
        painter.setBrush(QColor(48, 45, 45, 230));
        painter.drawRoundedRect(button_rect, 6, 6);

        QFont font = painter.font();
        font.setPixelSize(13);
        font.setBold(true);
        painter.setFont(font);
        painter.setPen(QColor(204, 204, 204));
        painter.drawText(button_rect, Qt::AlignCenter, "X");
        painter.restore();
    }

    QRectF image_draw_rect() const
    {
        QSize target_size = frame_image_.size();
        target_size.scale(size(), Qt::KeepAspectRatio);
        const QPointF top_left((width() - target_size.width()) * 0.5,
                               (height() - target_size.height()) * 0.5);
        return QRectF(top_left, QSizeF(target_size));
    }

    bool widget_to_image(const QPoint& widget_point, cv::Point2f& image_point) const
    {
        const QRectF image_rect = image_draw_rect();
        if (!image_rect.contains(widget_point)) {
            return false;
        }
        widget_to_image_clamped(widget_point, image_point);
        return true;
    }

    void widget_to_image_clamped(const QPoint& widget_point, cv::Point2f& image_point) const
    {
        const QRectF image_rect = image_draw_rect();
        const double x = std::clamp(static_cast<double>(widget_point.x()), image_rect.left(), image_rect.right());
        const double y = std::clamp(static_cast<double>(widget_point.y()), image_rect.top(), image_rect.bottom());
        const double sx = frame_image_.width() / image_rect.width();
        const double sy = frame_image_.height() / image_rect.height();
        image_point.x = static_cast<float>((x - image_rect.left()) * sx);
        image_point.y = static_cast<float>((y - image_rect.top()) * sy);
    }

    QRectF image_to_widget_rect(const cv::Rect2f& image_rect, const QRectF& draw_rect) const
    {
        const double sx = draw_rect.width() / frame_image_.width();
        const double sy = draw_rect.height() / frame_image_.height();
        return QRectF(draw_rect.left() + image_rect.x * sx,
                      draw_rect.top() + image_rect.y * sy,
                      image_rect.width * sx,
                      image_rect.height * sy);
    }

    void draw_tracking_overlay(QPainter& painter, const QRectF& draw_rect) const
    {
        if (mode_ != Mode::Tracking && mode_ != Mode::Finished) {
            return;
        }
        if (latest_bbox_.width <= 0 || latest_bbox_.height <= 0) {
            return;
        }

        painter.save();
        const QRectF bbox_rect = image_to_widget_rect(latest_bbox_, draw_rect);
        painter.setPen(QPen(QColor(38, 230, 118), 3));
        painter.setBrush(Qt::NoBrush);
        painter.drawRect(bbox_rect);

        // The score readout is the field diagnostic for CSP-1726: it shows at a glance
        // whether the head is confident and whether the values span a plausible range.
        const float score = tracker_->last_pred_score();
        if (score >= 0.0f && std::isfinite(score)) {
            QColor accent(255, 91, 91);
            if (score >= kGoodTrackingScore) {
                accent = QColor(38, 230, 118);
            } else if (score >= tracker_->min_track_score()) {
                accent = QColor(255, 190, 72);
            }

            const QString score_text = QString::number(score, 'f', 3);
            QFont score_font = painter.font();
            score_font.setPixelSize(16);
            score_font.setWeight(QFont::DemiBold);
            painter.setFont(score_font);

            const QFontMetrics metrics(score_font);
            const qreal pill_width =
                std::max<qreal>(58.0, metrics.horizontalAdvance(score_text) + 22.0);
            constexpr qreal pill_height = 30.0;
            qreal pill_x = bbox_rect.center().x() - pill_width * 0.5;
            pill_x = std::clamp(pill_x,
                                draw_rect.left() + 4.0,
                                std::max(draw_rect.left() + 4.0,
                                         draw_rect.right() - pill_width - 4.0));
            const qreal pill_y = std::max(draw_rect.top() + 4.0,
                                          bbox_rect.top() - pill_height - 7.0);
            const QRectF pill_rect(pill_x, pill_y, pill_width, pill_height);

            painter.setPen(Qt::NoPen);
            painter.setBrush(QColor(0, 0, 0, 105));
            painter.drawRoundedRect(pill_rect.translated(0.0, 3.0), 9.0, 9.0);
            painter.setPen(QPen(accent, 1.5));
            painter.setBrush(QColor(13, 17, 20, 225));
            painter.drawRoundedRect(pill_rect, 9.0, 9.0);
            painter.setPen(QColor(245, 249, 247));
            painter.drawText(pill_rect, Qt::AlignCenter, score_text);
        }
        painter.restore();
    }

    void draw_selection_overlay(QPainter& painter, const QRectF& draw_rect) const
    {
        if (mode_ != Mode::Selecting) {
            return;
        }

        cv::Rect2f roi = selected_roi_;
        if (dragging_) {
            roi = normalized_rect(drag_start_, drag_current_);
        }
        if (roi.width <= 0.0f || roi.height <= 0.0f) {
            return;
        }

        painter.setPen(QPen(QColor(255, 214, 88), 2));
        painter.setBrush(QColor(255, 214, 88, 42));
        painter.drawRect(image_to_widget_rect(roi, draw_rect));
    }

    void draw_hud(QPainter& painter, const QRectF& draw_rect) const
    {
        const QRectF panel(draw_rect.left() + 16.0, draw_rect.top() + 14.0, 360.0, 72.0);
        painter.setPen(Qt::NoPen);
        painter.setBrush(QColor(0, 0, 0, 150));
        painter.drawRoundedRect(panel, 8, 8);

        QFont title_font = painter.font();
        title_font.setPixelSize(20);
        title_font.setBold(true);
        painter.setFont(title_font);
        painter.setPen(QColor(235, 250, 241));
        painter.drawText(panel.adjusted(14, 10, -12, -36),
                         Qt::AlignLeft | Qt::AlignVCenter,
                         QString("MixFormerV2 (%1)").arg(QString::fromStdString(backend_label(options_.backend))));

        QFont status_font = painter.font();
        status_font.setPixelSize(14);
        status_font.setBold(false);
        painter.setFont(status_font);
        painter.setPen(QColor(188, 205, 196));

        QString status;
        switch (mode_) {
        case Mode::Selecting:
            status = "Drag over a target to start tracking";
            break;
        case Mode::Tracking:
            status = QString("Tracking  |  %1 FPS").arg(display_fps_, 0, 'f', 1);
            break;
        case Mode::Finished:
            status = "Finished";
            break;
        case Mode::Error:
            status = status_text_;
            break;
        }
        painter.drawText(panel.adjusted(14, 34, -12, -10), Qt::AlignLeft | Qt::AlignVCenter, status);
    }

    void start_tracking()
    {
        if (selected_roi_.width < 2.0f || selected_roi_.height < 2.0f) {
            std::cout << "Invalid bounding box. Select a larger target region." << std::endl;
            update();
            return;
        }

        tracker_->init(current_frame_, selected_roi_);
        latest_bbox_ = selected_roi_;
        mode_ = Mode::Tracking;
        last_frame_time_ = std::chrono::steady_clock::now();
        timer_.start(frame_interval_ms_);
        update();
    }

    void process_next_frame()
    {
        if (mode_ != Mode::Tracking) {
            return;
        }

        try {
            cv::Mat frame;
            if (!cap_.read(frame) || frame.empty()) {
                if (options_.use_camera) {
                    throw std::runtime_error("cannot read frame from " + input_description());
                }
                if (options_.loop) {
                    restart_tracking_loop();
                    return;
                }
                timer_.stop();
                mode_ = Mode::Finished;
                update();
                return;
            }

            const cv::Rect bbox = tracker_->update(frame);
            latest_bbox_ = cv::Rect2f(static_cast<float>(bbox.x),
                                      static_cast<float>(bbox.y),
                                      static_cast<float>(bbox.width),
                                      static_cast<float>(bbox.height));
            current_frame_ = frame;
            frame_image_ = mat_to_qimage(current_frame_);
            update_fps();
            update();
        } catch (const std::exception& e) {
            timer_.stop();
            mode_ = Mode::Error;
            status_text_ = QString::fromStdString(e.what());
            std::cerr << "Error: " << e.what() << std::endl;
            update();
        }
    }

    void restart_tracking_loop()
    {
        cv::Mat first_frame;
        cap_.set(cv::CAP_PROP_POS_FRAMES, 0);
        if (!cap_.read(first_frame) || first_frame.empty()) {
            cap_.release();
            if (!cap_.open(options_.video_path) ||
                !cap_.read(first_frame) || first_frame.empty()) {
                throw std::runtime_error("cannot restart video loop: " + options_.video_path);
            }
        }

        current_frame_ = std::move(first_frame);
        frame_image_ = mat_to_qimage(current_frame_);
        tracker_->init(current_frame_, selected_roi_);
        latest_bbox_ = selected_roi_;
        display_fps_ = 0.0;
        last_frame_time_ = std::chrono::steady_clock::now();
        std::cout << "[Loop] Restarted video from the first frame." << std::endl;
        update();
    }

    void update_fps()
    {
        const auto now = std::chrono::steady_clock::now();
        const double seconds = std::chrono::duration<double>(now - last_frame_time_).count();
        last_frame_time_ = now;
        if (seconds > 0.0) {
            const double instant = 1.0 / seconds;
            if (display_fps_ <= 0.0) {
                display_fps_ = instant;
            } else {
                display_fps_ = display_fps_ * 0.88 + instant * 0.12;
            }
        }
    }

    AppOptions options_;
    std::unique_ptr<MixFormerV2Tracker> tracker_;
    cv::VideoCapture cap_;
    cv::Mat current_frame_;
    QImage frame_image_;
    QTimer timer_;

    Mode mode_ = Mode::Selecting;
    int frame_interval_ms_ = 33;
    bool dragging_ = false;
    bool launcher_ready_notified_ = false;
    cv::Point2f drag_start_{0.0f, 0.0f};
    cv::Point2f drag_current_{0.0f, 0.0f};
    cv::Rect2f selected_roi_{};
    cv::Rect2f latest_bbox_{};
    QString status_text_;
    double display_fps_ = 0.0;
    std::chrono::steady_clock::time_point last_frame_time_ = std::chrono::steady_clock::now();
};

}  // namespace

int main(int argc, char** argv)
{
    try {
        AppOptions options = parse_args(argc, argv);
        const bool full_screen = options.full_screen;

        if (!fs::exists(options.model_path)) {
            throw std::runtime_error("model file not found: " + options.model_path);
        }

        QApplication app(argc, argv);

        auto tracker = std::make_unique<MixFormerV2Tracker>(options.model_path,
                                                            options.backend,
                                                            options.min_track_score,
                                                            options.debug);
        TrackingWindow window(std::move(options), std::move(tracker));

        if (full_screen) {
            window.showFullScreen();
        } else {
            window.resize(1280, 720);
            window.show();
        }

        return app.exec();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        print_usage(argv[0]);
        return 1;
    }
}
