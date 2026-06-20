#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cctype>
#include <cstring>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <new>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <dxrt/dxrt_api.h>
#include <opencv2/opencv.hpp>

#include <QApplication>
#include <QCloseEvent>
#include <QElapsedTimer>
#include <QEventLoop>
#include <QImage>
#include <QKeyEvent>
#include <QMouseEvent>
#include <QPainter>
#include <QResizeEvent>
#include <QThread>
#include <QWidget>

namespace
{

constexpr const char *kDefaultModel = "assets/yolopv2_384x640_q-lite.dxnn";
constexpr const char *kDefaultVideo = "assets/video-test-3.mp4";
constexpr const char *kWindowName = "YOLOPv2 DXNN Qt5 Async";
constexpr int kDefaultImageSize = 640;
constexpr int kDefaultClasses = 80;
constexpr int kDefaultQueueMax = 6;
constexpr int kAnchorsPerLayer = 3;
constexpr int kMaxDetections = 300;
constexpr int kMaxNmsCandidates = 30000;
constexpr int kDefaultPaletteIndex = 0;

struct RenderPalette
{
    const char *name = "";
    cv::Scalar bbox;
    cv::Scalar bbox_corner;
    cv::Scalar bbox_shadow;
    cv::Scalar drive;
    cv::Scalar lane;
    cv::Scalar status_panel;
    cv::Scalar status_accent;
    cv::Scalar status_border;
    cv::Scalar status_label;
    cv::Scalar status_value;
};

const std::array<RenderPalette, 5> kRenderPalettes = {{
    {"aurora",
     cv::Scalar(255, 230, 64), cv::Scalar(255, 96, 220), cv::Scalar(24, 38, 44),
     cv::Scalar(92, 238, 118), cv::Scalar(255, 70, 224),
     cv::Scalar(12, 18, 22), cv::Scalar(255, 230, 64), cv::Scalar(58, 72, 78),
     cv::Scalar(160, 186, 194), cv::Scalar(248, 252, 255)},
    {"sunset",
     cv::Scalar(54, 178, 255), cv::Scalar(98, 234, 255), cv::Scalar(36, 24, 24),
     cv::Scalar(70, 210, 126), cv::Scalar(86, 124, 255),
     cv::Scalar(24, 18, 20), cv::Scalar(54, 178, 255), cv::Scalar(82, 76, 92),
     cv::Scalar(178, 180, 194), cv::Scalar(255, 250, 244)},
    {"ocean",
     cv::Scalar(255, 214, 72), cv::Scalar(106, 252, 214), cv::Scalar(24, 34, 42),
     cv::Scalar(210, 238, 88), cv::Scalar(255, 124, 88),
     cv::Scalar(16, 23, 28), cv::Scalar(255, 214, 72), cv::Scalar(72, 88, 94),
     cv::Scalar(168, 194, 202), cv::Scalar(250, 254, 255)},
    {"neon",
     cv::Scalar(178, 255, 42), cv::Scalar(255, 74, 238), cv::Scalar(18, 26, 18),
     cv::Scalar(70, 240, 126), cv::Scalar(255, 64, 232),
     cv::Scalar(14, 16, 18), cv::Scalar(178, 255, 42), cv::Scalar(62, 82, 60),
     cv::Scalar(172, 202, 168), cv::Scalar(248, 255, 242)},
    {"graphite",
     cv::Scalar(236, 222, 186), cv::Scalar(146, 226, 255), cv::Scalar(28, 30, 34),
     cv::Scalar(146, 210, 168), cv::Scalar(184, 164, 255),
     cv::Scalar(20, 22, 24), cv::Scalar(236, 222, 186), cv::Scalar(76, 78, 84),
     cv::Scalar(174, 180, 184), cv::Scalar(248, 248, 244)},
}};

const std::array<std::array<cv::Size2f, kAnchorsPerLayer>, 3> kAnchorGrid = {{
    {cv::Size2f(12.0f, 16.0f), cv::Size2f(19.0f, 36.0f), cv::Size2f(40.0f, 28.0f)},
    {cv::Size2f(36.0f, 75.0f), cv::Size2f(76.0f, 55.0f), cv::Size2f(72.0f, 146.0f)},
    {cv::Size2f(142.0f, 110.0f), cv::Size2f(192.0f, 243.0f), cv::Size2f(459.0f, 401.0f)},
}};

const std::array<const char *, kDefaultClasses> kCocoClassNames = {{
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
    "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase",
    "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
}};

struct Options
{
    std::string model_path = kDefaultModel;
    std::string source = kDefaultVideo;
    bool use_camera = false;
    int camera_index = 0;
    int img_size = kDefaultImageSize;
    int num_classes = kDefaultClasses;
    float conf_thres = 0.30f;
    float iou_thres = 0.45f;
    bool agnostic_nms = false;
    bool no_display = false;
    bool show_exit_button = false;
    bool print_detections = false;
    bool save_video = false;
    std::string save_path;
    int resize_width = 1280;
    int resize_height = 720;
    int delay = 1;
    int log_interval = 30;
    int max_frames = 0;
    int queue_max = kDefaultQueueMax;
    bool loop = false;
    int palette_index = kDefaultPaletteIndex;
};

struct AverageMeter
{
    double sum = 0.0;
    int count = 0;

    void update(double value)
    {
        sum += value;
        ++count;
    }

    double avg() const
    {
        return count > 0 ? sum / static_cast<double>(count) : 0.0;
    }
};

struct LetterboxInfo
{
    double ratio = 1.0;
    double pad_w = 0.0;
    double pad_h = 0.0;
    int left = 0;
    int right = 0;
    int top = 0;
    int bottom = 0;
    int net_w = 0;
    int net_h = 0;
    cv::Size original_size;
};

struct InputInfo
{
    std::vector<int64_t> shape;
    dxrt::DataType type = dxrt::DataType::NONE_TYPE;
    int net_w = 0;
    int net_h = 0;
    bool nchw = true;
    std::size_t elements = 0;
};

struct TensorView
{
    dxrt::Tensor *tensor = nullptr;
    const float *data = nullptr;
    std::string name;
    int channels = 0;
    int height = 0;
    int width = 0;
    bool nchw = true;
};

struct Detection
{
    cv::Rect2f box;
    float confidence = 0.0f;
    int class_id = 0;
};

struct AsyncJob
{
    int frame_id = 0;
    int job_id = -1;
    cv::Mat frame;
    LetterboxInfo letterbox;
    std::vector<std::uint8_t> u8_input;
    std::vector<float> float_input;
    void *input_ptr = nullptr;
    std::chrono::steady_clock::time_point submit_time;
};

bool file_exists(const std::string &path)
{
    std::ifstream in(path.c_str(), std::ios::binary);
    return static_cast<bool>(in);
}

std::string join_path(const std::string &a, const std::string &b)
{
    if (a.empty())
    {
        return b;
    }
    if (a.back() == '/')
    {
        return a + b;
    }
    return a + "/" + b;
}

std::string resolve_file_path(const std::string &path)
{
    if (file_exists(path))
    {
        return path;
    }
#ifdef YOLOPV2_ROOT_DIR
    const std::string rooted = join_path(YOLOPV2_ROOT_DIR, path);
    if (file_exists(rooted))
    {
        return rooted;
    }
#endif
    return path;
}

bool parse_int(const std::string &s, int *out)
{
    if (out == nullptr)
    {
        return false;
    }
    try
    {
        std::size_t used = 0;
        const int v = std::stoi(s, &used);
        if (used != s.size())
        {
            return false;
        }
        *out = v;
        return true;
    }
    catch (...)
    {
        return false;
    }
}

bool parse_float(const std::string &s, float *out)
{
    if (out == nullptr)
    {
        return false;
    }
    try
    {
        std::size_t used = 0;
        const float v = std::stof(s, &used);
        if (used != s.size())
        {
            return false;
        }
        *out = v;
        return true;
    }
    catch (...)
    {
        return false;
    }
}

bool is_integer_string(const std::string &s)
{
    if (s.empty())
    {
        return false;
    }
    std::size_t start = s[0] == '-' ? 1 : 0;
    if (start >= s.size())
    {
        return false;
    }
    for (std::size_t i = start; i < s.size(); ++i)
    {
        if (!std::isdigit(static_cast<unsigned char>(s[i])))
        {
            return false;
        }
    }
    return true;
}

std::string detection_class_name(int class_id)
{
    if (class_id == 3)
    {
        return "car";
    }

    if (class_id >= 0 && class_id < static_cast<int>(kCocoClassNames.size()))
    {
        return kCocoClassNames[static_cast<std::size_t>(class_id)];
    }

    std::ostringstream fallback;
    fallback << "class_" << class_id;
    return fallback.str();
}

std::string lower_ascii(std::string value)
{
    for (char &ch : value)
    {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    return value;
}

std::string palette_names()
{
    std::ostringstream names;
    for (std::size_t i = 0; i < kRenderPalettes.size(); ++i)
    {
        if (i > 0)
        {
            names << ", ";
        }
        names << kRenderPalettes[i].name;
    }
    return names.str();
}

int palette_index_by_name(const std::string &name)
{
    int numeric_index = 0;
    if (is_integer_string(name) && parse_int(name, &numeric_index) &&
        numeric_index >= 1 && numeric_index <= static_cast<int>(kRenderPalettes.size()))
    {
        return numeric_index - 1;
    }

    const std::string needle = lower_ascii(name);
    for (std::size_t i = 0; i < kRenderPalettes.size(); ++i)
    {
        if (needle == kRenderPalettes[i].name)
        {
            return static_cast<int>(i);
        }
    }
    return -1;
}

void print_usage(const char *argv0)
{
    std::cout
        << "Usage: " << argv0 << " [OPTIONS]\n"
        << "  -m, --model PATH           DXNN model path (default: " << kDefaultModel << ")\n"
        << "  -s, --source PATH          Video path or camera index (default: " << kDefaultVideo << ")\n"
        << "  -v, --video PATH           Use video input\n"
        << "      --camera INDEX         Use camera input\n"
        << "      --img-size N           Model input size fallback (default: 640)\n"
        << "      --conf-thres FLOAT     Detection confidence threshold (default: 0.30)\n"
        << "      --iou-thres FLOAT      Detection NMS IoU threshold (default: 0.45)\n"
        << "      --classes N            Number of detection classes (default: 80)\n"
        << "      --agnostic-nms         Suppress boxes across classes\n"
        << "      --resize W H           Resize frames before inference/display (default: 1280 720)\n"
        << "      --no-resize            Keep source frame size\n"
        << "      --delay MS             Qt event delay for video (default: 1)\n"
        << "      --log-interval N       Print timing every N frames; 0 disables (default: 30)\n"
        << "      --max-frames N         Stop after N frames; 0 means all frames\n"
        << "      --loop                 Loop video input until stopped or max-frames is reached\n"
        << "      --color NAME|1-5       Output palette: " << palette_names()
        << " (default: " << kRenderPalettes[kDefaultPaletteIndex].name << ")\n"
        << "      --queue-max N          Maximum in-flight async requests (default: " << kDefaultQueueMax << ")\n"
        << "      --max-inflight N       Alias for --queue-max\n"
        << "      --save PATH            Save annotated video\n"
        << "      --print-detections     Print boxes per frame\n"
        << "      --no-display           Do not open the GUI window\n"
        << "      --exit-btn             Show a small exit button at the top-right\n"
        << "  -h, --help                 Show this help\n";
}

bool parse_args(int argc, char **argv, Options *opt)
{
    if (opt == nullptr)
    {
        return false;
    }

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
            opt->model_path = value;
        }
        else if (arg == "-s" || arg == "--source")
        {
            const char *value = require_value(arg.c_str());
            if (value == nullptr)
            {
                return false;
            }
            opt->source = value;
            opt->use_camera = is_integer_string(opt->source);
            if (opt->use_camera && !parse_int(opt->source, &opt->camera_index))
            {
                return false;
            }
        }
        else if (arg == "-v" || arg == "--video")
        {
            const char *value = require_value(arg.c_str());
            if (value == nullptr)
            {
                return false;
            }
            opt->source = value;
            opt->use_camera = false;
        }
        else if (arg == "--camera")
        {
            const char *value = require_value(arg.c_str());
            if (value == nullptr || !parse_int(value, &opt->camera_index) || opt->camera_index < 0)
            {
                std::cerr << "Error: --camera expects a non-negative integer" << std::endl;
                return false;
            }
            opt->use_camera = true;
        }
        else if (arg == "--img-size")
        {
            const char *value = require_value(arg.c_str());
            if (value == nullptr || !parse_int(value, &opt->img_size) || opt->img_size <= 0)
            {
                std::cerr << "Error: --img-size expects a positive integer" << std::endl;
                return false;
            }
        }
        else if (arg == "--conf-thres")
        {
            const char *value = require_value(arg.c_str());
            if (value == nullptr || !parse_float(value, &opt->conf_thres) || opt->conf_thres < 0.0f ||
                opt->conf_thres > 1.0f)
            {
                std::cerr << "Error: --conf-thres expects a value in [0, 1]" << std::endl;
                return false;
            }
        }
        else if (arg == "--iou-thres")
        {
            const char *value = require_value(arg.c_str());
            if (value == nullptr || !parse_float(value, &opt->iou_thres) || opt->iou_thres < 0.0f ||
                opt->iou_thres > 1.0f)
            {
                std::cerr << "Error: --iou-thres expects a value in [0, 1]" << std::endl;
                return false;
            }
        }
        else if (arg == "--classes")
        {
            const char *value = require_value(arg.c_str());
            if (value == nullptr || !parse_int(value, &opt->num_classes) || opt->num_classes <= 0)
            {
                std::cerr << "Error: --classes expects a positive integer" << std::endl;
                return false;
            }
        }
        else if (arg == "--agnostic-nms")
        {
            opt->agnostic_nms = true;
        }
        else if (arg == "--resize")
        {
            const char *w = require_value(arg.c_str());
            const char *h = require_value(arg.c_str());
            if (w == nullptr || h == nullptr ||
                !parse_int(w, &opt->resize_width) || !parse_int(h, &opt->resize_height) ||
                opt->resize_width <= 0 || opt->resize_height <= 0)
            {
                std::cerr << "Error: --resize expects positive WIDTH HEIGHT" << std::endl;
                return false;
            }
        }
        else if (arg == "--no-resize")
        {
            opt->resize_width = 0;
            opt->resize_height = 0;
        }
        else if (arg == "--delay")
        {
            const char *value = require_value(arg.c_str());
            if (value == nullptr || !parse_int(value, &opt->delay) || opt->delay < 0)
            {
                std::cerr << "Error: --delay expects a non-negative integer" << std::endl;
                return false;
            }
        }
        else if (arg == "--log-interval")
        {
            const char *value = require_value(arg.c_str());
            if (value == nullptr || !parse_int(value, &opt->log_interval) || opt->log_interval < 0)
            {
                std::cerr << "Error: --log-interval expects a non-negative integer" << std::endl;
                return false;
            }
        }
        else if (arg == "--max-frames")
        {
            const char *value = require_value(arg.c_str());
            if (value == nullptr || !parse_int(value, &opt->max_frames) || opt->max_frames < 0)
            {
                std::cerr << "Error: --max-frames expects a non-negative integer" << std::endl;
                return false;
            }
        }
        else if (arg == "--queue-max" || arg == "--max-inflight")
        {
            const char *value = require_value(arg.c_str());
            if (value == nullptr || !parse_int(value, &opt->queue_max) || opt->queue_max <= 0)
            {
                std::cerr << "Error: " << arg << " expects a positive integer" << std::endl;
                return false;
            }
        }
        else if (arg == "--loop")
        {
            opt->loop = true;
        }
        else if (arg == "--color")
        {
            const char *value = require_value(arg.c_str());
            const int palette_index = value == nullptr ? -1 : palette_index_by_name(value);
            if (palette_index < 0)
            {
                std::cerr << "Error: --color expects one of: " << palette_names() << std::endl;
                return false;
            }
            opt->palette_index = palette_index;
        }
        else if (arg == "--save")
        {
            const char *value = require_value(arg.c_str());
            if (value == nullptr)
            {
                return false;
            }
            opt->save_video = true;
            opt->save_path = value;
        }
        else if (arg == "--print-detections")
        {
            opt->print_detections = true;
        }
        else if (arg == "--no-display")
        {
            opt->no_display = true;
        }
        else if (arg == "--exit-btn")
        {
            opt->show_exit_button = true;
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

void print_shape(const std::vector<int64_t> &shape)
{
    std::cout << "[";
    for (std::size_t i = 0; i < shape.size(); ++i)
    {
        if (i > 0)
        {
            std::cout << ",";
        }
        std::cout << shape[i];
    }
    std::cout << "]";
}

void print_tensors(const std::string &title, const dxrt::Tensors &tensors)
{
    std::cout << title << " count=" << tensors.size() << std::endl;
    for (std::size_t i = 0; i < tensors.size(); ++i)
    {
        std::cout << "  #" << i << " name=" << tensors[i].name()
                  << " type=" << tensors[i].type() << " shape=";
        print_shape(tensors[i].shape());
        std::cout << std::endl;
    }
}

void print_output_ptrs_once(const dxrt::TensorPtrs &outputs)
{
    static bool printed = false;
    if (printed)
    {
        return;
    }
    printed = true;

    std::cout << "Run outputs count=" << outputs.size() << std::endl;
    for (std::size_t i = 0; i < outputs.size(); ++i)
    {
        if (!outputs[i])
        {
            continue;
        }
        std::cout << "  #" << i << " name=" << outputs[i]->name()
                  << " type=" << outputs[i]->type() << " shape=";
        print_shape(outputs[i]->shape());
        std::cout << std::endl;
    }
}

InputInfo analyze_input(dxrt::InferenceEngine &engine, int fallback_size)
{
    const dxrt::Tensors inputs = engine.GetInputs();
    if (inputs.empty())
    {
        throw std::runtime_error("model has no input tensor");
    }

    InputInfo info;
    info.shape = inputs.front().shape();
    info.type = inputs.front().type();
    info.elements = shape_element_count(info.shape);

    if (info.shape.size() == 4 && info.shape[1] == 3)
    {
        info.nchw = true;
        info.net_h = static_cast<int>(info.shape[2]);
        info.net_w = static_cast<int>(info.shape[3]);
    }
    else if (info.shape.size() == 4 && info.shape[3] == 3)
    {
        info.nchw = false;
        info.net_h = static_cast<int>(info.shape[1]);
        info.net_w = static_cast<int>(info.shape[2]);
    }
    else
    {
        info.nchw = false;
        info.net_w = fallback_size;
        info.net_h = fallback_size;
    }

    if (info.net_w <= 0 || info.net_h <= 0)
    {
        throw std::runtime_error("invalid model input shape");
    }
    if (info.type != dxrt::DataType::UINT8 && info.type != dxrt::DataType::FLOAT)
    {
        throw std::runtime_error("only UINT8 and FLOAT input tensors are supported");
    }

    return info;
}

cv::Mat letterbox(const cv::Mat &frame, int net_w, int net_h, LetterboxInfo *info)
{
    if (frame.empty())
    {
        return cv::Mat();
    }

    const int src_w = frame.cols;
    const int src_h = frame.rows;
    const double r = std::min(static_cast<double>(net_w) / static_cast<double>(src_w),
                              static_cast<double>(net_h) / static_cast<double>(src_h));
    const int new_w = std::max(1, static_cast<int>(std::round(src_w * r)));
    const int new_h = std::max(1, static_cast<int>(std::round(src_h * r)));

    const double dw = (static_cast<double>(net_w - new_w)) / 2.0;
    const double dh = (static_cast<double>(net_h - new_h)) / 2.0;
    const int top = static_cast<int>(std::round(dh - 0.1));
    const int bottom = static_cast<int>(std::round(dh + 0.1));
    const int left = static_cast<int>(std::round(dw - 0.1));
    const int right = static_cast<int>(std::round(dw + 0.1));

    cv::Mat resized;
    if (new_w != src_w || new_h != src_h)
    {
        cv::resize(frame, resized, cv::Size(new_w, new_h), 0.0, 0.0, cv::INTER_LINEAR);
    }
    else
    {
        resized = frame;
    }

    cv::Mat out;
    cv::copyMakeBorder(resized, out, top, bottom, left, right, cv::BORDER_CONSTANT,
                       cv::Scalar(114, 114, 114));

    if (info != nullptr)
    {
        info->ratio = r;
        info->pad_w = dw;
        info->pad_h = dh;
        info->left = left;
        info->right = right;
        info->top = top;
        info->bottom = bottom;
        info->net_w = net_w;
        info->net_h = net_h;
        info->original_size = frame.size();
    }

    return out;
}

void prepare_input(const cv::Mat &frame, const InputInfo &input_info,
                   std::vector<std::uint8_t> *u8_input,
                   std::vector<float> *float_input,
                   void **run_ptr,
                   LetterboxInfo *letterbox_info)
{
    if (u8_input == nullptr || float_input == nullptr || run_ptr == nullptr)
    {
        throw std::runtime_error("invalid input buffers");
    }

    cv::Mat boxed = letterbox(frame, input_info.net_w, input_info.net_h, letterbox_info);
    cv::Mat rgb;
    cv::cvtColor(boxed, rgb, cv::COLOR_BGR2RGB);

    const std::size_t image_elements = static_cast<std::size_t>(input_info.net_w) *
                                       static_cast<std::size_t>(input_info.net_h) * 3U;
    const std::size_t elements = std::max(input_info.elements, image_elements);

    if (input_info.type == dxrt::DataType::UINT8)
    {
        u8_input->assign(elements, 0);
        if (input_info.nchw)
        {
            const int plane = input_info.net_w * input_info.net_h;
            for (int y = 0; y < input_info.net_h; ++y)
            {
                const cv::Vec3b *row = rgb.ptr<cv::Vec3b>(y);
                for (int x = 0; x < input_info.net_w; ++x)
                {
                    for (int c = 0; c < 3; ++c)
                    {
                        (*u8_input)[static_cast<std::size_t>(c * plane + y * input_info.net_w + x)] = row[x][c];
                    }
                }
            }
        }
        else
        {
            if (!rgb.isContinuous())
            {
                rgb = rgb.clone();
            }
            std::memcpy(u8_input->data(), rgb.data, image_elements);
        }
        *run_ptr = u8_input->data();
    }
    else
    {
        float_input->assign(elements, 0.0f);
        if (input_info.nchw)
        {
            const int plane = input_info.net_w * input_info.net_h;
            for (int y = 0; y < input_info.net_h; ++y)
            {
                const cv::Vec3b *row = rgb.ptr<cv::Vec3b>(y);
                for (int x = 0; x < input_info.net_w; ++x)
                {
                    for (int c = 0; c < 3; ++c)
                    {
                        (*float_input)[static_cast<std::size_t>(c * plane + y * input_info.net_w + x)] =
                            static_cast<float>(row[x][c]) / 255.0f;
                    }
                }
            }
        }
        else
        {
            std::size_t idx = 0;
            for (int y = 0; y < input_info.net_h; ++y)
            {
                const cv::Vec3b *row = rgb.ptr<cv::Vec3b>(y);
                for (int x = 0; x < input_info.net_w; ++x)
                {
                    for (int c = 0; c < 3; ++c)
                    {
                        (*float_input)[idx++] = static_cast<float>(row[x][c]) / 255.0f;
                    }
                }
            }
        }
        *run_ptr = float_input->data();
    }
}

float sigmoid(float x)
{
    if (x >= 0.0f)
    {
        const float z = std::exp(-x);
        return 1.0f / (1.0f + z);
    }
    const float z = std::exp(x);
    return z / (1.0f + z);
}

bool make_tensor_view(dxrt::Tensor *tensor, TensorView *view)
{
    if (tensor == nullptr || view == nullptr || tensor->data() == nullptr ||
        tensor->type() != dxrt::DataType::FLOAT)
    {
        return false;
    }

    const std::vector<int64_t> &shape = tensor->shape();
    if (shape.size() != 4)
    {
        return false;
    }

    TensorView out;
    out.tensor = tensor;
    out.data = static_cast<const float *>(tensor->data());
    out.name = tensor->name();

    if (shape[1] > 0 && shape[2] > 0 && shape[3] > 0 &&
        (shape[1] <= 2 || shape[1] >= 16))
    {
        out.nchw = true;
        out.channels = static_cast<int>(shape[1]);
        out.height = static_cast<int>(shape[2]);
        out.width = static_cast<int>(shape[3]);
    }
    else if (shape[1] > 0 && shape[2] > 0 && shape[3] > 0)
    {
        out.nchw = false;
        out.height = static_cast<int>(shape[1]);
        out.width = static_cast<int>(shape[2]);
        out.channels = static_cast<int>(shape[3]);
    }
    else
    {
        return false;
    }

    if (out.channels <= 0 || out.height <= 0 || out.width <= 0)
    {
        return false;
    }

    *view = out;
    return true;
}

float tensor_value(const TensorView &view, int c, int y, int x)
{
    if (view.nchw)
    {
        const std::size_t idx = (static_cast<std::size_t>(c) * view.height +
                                 static_cast<std::size_t>(y)) *
                                    view.width +
                                static_cast<std::size_t>(x);
        return view.data[idx];
    }

    const std::size_t idx = (static_cast<std::size_t>(y) * view.width +
                             static_cast<std::size_t>(x)) *
                                view.channels +
                            static_cast<std::size_t>(c);
    return view.data[idx];
}

bool is_detection_tensor(const TensorView &view, int num_classes, int net_w, int net_h)
{
    const int attrs = 5 + num_classes;
    const int valid_channels = attrs * kAnchorsPerLayer;
    if (view.channels < valid_channels)
    {
        return false;
    }

    const std::array<int, 3> strides = {8, 16, 32};
    for (int stride : strides)
    {
        if (net_w % stride == 0 && net_h % stride == 0 &&
            view.width == net_w / stride && view.height == net_h / stride)
        {
            return true;
        }
    }
    return false;
}

int detection_layer_index(const TensorView &view, int net_w, int net_h)
{
    if (net_w % 8 == 0 && net_h % 8 == 0 &&
        view.width == net_w / 8 && view.height == net_h / 8)
    {
        return 0;
    }
    if (net_w % 16 == 0 && net_h % 16 == 0 &&
        view.width == net_w / 16 && view.height == net_h / 16)
    {
        return 1;
    }
    return 2;
}

void decode_detection_layer(const TensorView &view, const LetterboxInfo &lb,
                            const Options &opt, std::vector<Detection> *detections)
{
    if (detections == nullptr)
    {
        return;
    }

    const int attrs = 5 + opt.num_classes;
    const int valid_channels = attrs * kAnchorsPerLayer;
    if (view.channels < valid_channels)
    {
        return;
    }

    const int layer = detection_layer_index(view, lb.net_w, lb.net_h);
    const float stride_x = static_cast<float>(lb.net_w) / static_cast<float>(view.width);
    const float stride_y = static_cast<float>(lb.net_h) / static_cast<float>(view.height);

    for (int a = 0; a < kAnchorsPerLayer; ++a)
    {
        const int base = a * attrs;
        const cv::Size2f anchor = kAnchorGrid[static_cast<std::size_t>(layer)][static_cast<std::size_t>(a)];
        for (int gy = 0; gy < view.height; ++gy)
        {
            for (int gx = 0; gx < view.width; ++gx)
            {
                const float obj = sigmoid(tensor_value(view, base + 4, gy, gx));
                if (obj < opt.conf_thres)
                {
                    continue;
                }

                float best_cls_conf = 0.0f;
                int best_cls = 0;
                for (int cls = 0; cls < opt.num_classes; ++cls)
                {
                    const float cls_conf = sigmoid(tensor_value(view, base + 5 + cls, gy, gx));
                    if (cls_conf > best_cls_conf)
                    {
                        best_cls_conf = cls_conf;
                        best_cls = cls;
                    }
                }

                const float score = obj * best_cls_conf;
                if (score < opt.conf_thres)
                {
                    continue;
                }

                const float sx = sigmoid(tensor_value(view, base + 0, gy, gx));
                const float sy = sigmoid(tensor_value(view, base + 1, gy, gx));
                const float sw = sigmoid(tensor_value(view, base + 2, gy, gx));
                const float sh = sigmoid(tensor_value(view, base + 3, gy, gx));

                const float cx = (sx * 2.0f - 0.5f + static_cast<float>(gx)) * stride_x;
                const float cy = (sy * 2.0f - 0.5f + static_cast<float>(gy)) * stride_y;
                const float bw = std::pow(sw * 2.0f, 2.0f) * anchor.width;
                const float bh = std::pow(sh * 2.0f, 2.0f) * anchor.height;

                float x1 = (cx - bw * 0.5f - static_cast<float>(lb.pad_w)) / static_cast<float>(lb.ratio);
                float y1 = (cy - bh * 0.5f - static_cast<float>(lb.pad_h)) / static_cast<float>(lb.ratio);
                float x2 = (cx + bw * 0.5f - static_cast<float>(lb.pad_w)) / static_cast<float>(lb.ratio);
                float y2 = (cy + bh * 0.5f - static_cast<float>(lb.pad_h)) / static_cast<float>(lb.ratio);

                x1 = std::max(0.0f, std::min(x1, static_cast<float>(lb.original_size.width - 1)));
                y1 = std::max(0.0f, std::min(y1, static_cast<float>(lb.original_size.height - 1)));
                x2 = std::max(0.0f, std::min(x2, static_cast<float>(lb.original_size.width - 1)));
                y2 = std::max(0.0f, std::min(y2, static_cast<float>(lb.original_size.height - 1)));

                if (x2 <= x1 + 1.0f || y2 <= y1 + 1.0f)
                {
                    continue;
                }

                Detection det;
                det.box = cv::Rect2f(cv::Point2f(x1, y1), cv::Point2f(x2, y2));
                det.confidence = score;
                det.class_id = best_cls;
                detections->push_back(det);
            }
        }
    }
}

float intersection_over_union(const cv::Rect2f &a, const cv::Rect2f &b)
{
    const float x1 = std::max(a.x, b.x);
    const float y1 = std::max(a.y, b.y);
    const float x2 = std::min(a.x + a.width, b.x + b.width);
    const float y2 = std::min(a.y + a.height, b.y + b.height);
    const float inter_w = std::max(0.0f, x2 - x1);
    const float inter_h = std::max(0.0f, y2 - y1);
    const float inter = inter_w * inter_h;
    const float uni = a.area() + b.area() - inter;
    return uni > 0.0f ? inter / uni : 0.0f;
}

std::vector<Detection> nms(std::vector<Detection> detections, float iou_thres, bool agnostic)
{
    std::sort(detections.begin(), detections.end(),
              [](const Detection &a, const Detection &b) { return a.confidence > b.confidence; });

    if (detections.size() > static_cast<std::size_t>(kMaxNmsCandidates))
    {
        detections.resize(kMaxNmsCandidates);
    }

    std::vector<Detection> kept;
    std::vector<bool> removed(detections.size(), false);
    for (std::size_t i = 0; i < detections.size(); ++i)
    {
        if (removed[i])
        {
            continue;
        }
        kept.push_back(detections[i]);
        if (kept.size() >= static_cast<std::size_t>(kMaxDetections))
        {
            break;
        }

        for (std::size_t j = i + 1; j < detections.size(); ++j)
        {
            if (removed[j])
            {
                continue;
            }
            if (!agnostic && detections[i].class_id != detections[j].class_id)
            {
                continue;
            }
            if (intersection_over_union(detections[i].box, detections[j].box) > iou_thres)
            {
                removed[j] = true;
            }
        }
    }
    return kept;
}

cv::Mat extract_channel_crop(const TensorView &view, int channel, const cv::Rect &roi)
{
    cv::Mat out(roi.height, roi.width, CV_32F);
    for (int y = 0; y < roi.height; ++y)
    {
        float *dst = out.ptr<float>(y);
        const int src_y = roi.y + y;
        for (int x = 0; x < roi.width; ++x)
        {
            dst[x] = tensor_value(view, channel, src_y, roi.x + x);
        }
    }
    return out;
}

cv::Rect segmentation_crop_roi(const TensorView &view, const LetterboxInfo &lb)
{
    const double sx = static_cast<double>(view.width) / static_cast<double>(lb.net_w);
    const double sy = static_cast<double>(view.height) / static_cast<double>(lb.net_h);

    int left = static_cast<int>(std::round(lb.pad_w * sx - 0.1));
    int right = static_cast<int>(std::round(lb.pad_w * sx + 0.1));
    int top = static_cast<int>(std::round(lb.pad_h * sy - 0.1));
    int bottom = static_cast<int>(std::round(lb.pad_h * sy + 0.1));

    left = std::max(0, std::min(left, view.width - 1));
    top = std::max(0, std::min(top, view.height - 1));
    right = std::max(0, std::min(right, view.width - left - 1));
    bottom = std::max(0, std::min(bottom, view.height - top - 1));

    const int width = std::max(1, view.width - left - right);
    const int height = std::max(1, view.height - top - bottom);
    return cv::Rect(left, top, width, height);
}

cv::Mat make_drive_mask(const TensorView &view, const LetterboxInfo &lb)
{
    if (view.channels < 2)
    {
        return cv::Mat();
    }

    const cv::Rect roi = segmentation_crop_roi(view, lb);
    cv::Mat bg = extract_channel_crop(view, 0, roi);
    cv::Mat drive = extract_channel_crop(view, 1, roi);

    cv::Mat bg_resized;
    cv::Mat drive_resized;
    cv::resize(bg, bg_resized, lb.original_size, 0.0, 0.0, cv::INTER_LINEAR);
    cv::resize(drive, drive_resized, lb.original_size, 0.0, 0.0, cv::INTER_LINEAR);

    cv::Mat mask(lb.original_size, CV_8U, cv::Scalar(0));
    for (int y = 0; y < mask.rows; ++y)
    {
        const float *bg_row = bg_resized.ptr<float>(y);
        const float *drive_row = drive_resized.ptr<float>(y);
        std::uint8_t *mask_row = mask.ptr<std::uint8_t>(y);
        for (int x = 0; x < mask.cols; ++x)
        {
            mask_row[x] = drive_row[x] > bg_row[x] ? 1 : 0;
        }
    }
    return mask;
}

cv::Mat make_lane_mask(const TensorView &view, const LetterboxInfo &lb)
{
    if (view.channels < 1)
    {
        return cv::Mat();
    }

    const cv::Rect roi = segmentation_crop_roi(view, lb);
    cv::Mat lane = extract_channel_crop(view, 0, roi);
    cv::Mat lane_resized;
    cv::resize(lane, lane_resized, lb.original_size, 0.0, 0.0, cv::INTER_LINEAR);

    cv::Mat mask(lb.original_size, CV_8U, cv::Scalar(0));
    for (int y = 0; y < mask.rows; ++y)
    {
        const float *lane_row = lane_resized.ptr<float>(y);
        std::uint8_t *mask_row = mask.ptr<std::uint8_t>(y);
        for (int x = 0; x < mask.cols; ++x)
        {
            mask_row[x] = lane_row[x] >= 0.5f ? 1 : 0;
        }
    }
    return mask;
}

cv::Vec3b scalar_to_vec3b(const cv::Scalar &color)
{
    return cv::Vec3b(cv::saturate_cast<std::uint8_t>(color[0]),
                     cv::saturate_cast<std::uint8_t>(color[1]),
                     cv::saturate_cast<std::uint8_t>(color[2]));
}

void overlay_segmentation(cv::Mat &frame, const cv::Mat &drive_mask, const cv::Mat &lane_mask,
                          const RenderPalette &palette)
{
    if (frame.empty())
    {
        return;
    }

    const cv::Vec3b drive_color = scalar_to_vec3b(palette.drive);
    const cv::Vec3b lane_color = scalar_to_vec3b(palette.lane);
    for (int y = 0; y < frame.rows; ++y)
    {
        cv::Vec3b *row = frame.ptr<cv::Vec3b>(y);
        const std::uint8_t *drive = drive_mask.empty() ? nullptr : drive_mask.ptr<std::uint8_t>(y);
        const std::uint8_t *lane = lane_mask.empty() ? nullptr : lane_mask.ptr<std::uint8_t>(y);
        for (int x = 0; x < frame.cols; ++x)
        {
            bool has_color = false;
            cv::Vec3b color(0, 0, 0);
            if (drive != nullptr && drive[x] != 0)
            {
                color = drive_color;
                has_color = true;
            }
            if (lane != nullptr && lane[x] != 0)
            {
                color = lane_color;
                has_color = true;
            }
            if (has_color)
            {
                for (int c = 0; c < 3; ++c)
                {
                    row[x][c] = static_cast<std::uint8_t>(row[x][c] * 0.5f + color[c] * 0.5f);
                }
            }
        }
    }
}

void draw_corner_line(cv::Mat &frame, const cv::Point &from, const cv::Point &to,
                      const cv::Scalar &color, int thickness)
{
    cv::line(frame, from, to, color, thickness, cv::LINE_AA);
}

void draw_detections(cv::Mat &frame, const std::vector<Detection> &detections,
                     const RenderPalette &palette)
{
    std::vector<cv::Rect> boxes;
    boxes.reserve(detections.size());
    for (const Detection &det : detections)
    {
        cv::Rect rect(cv::Point(static_cast<int>(std::round(det.box.x)),
                                static_cast<int>(std::round(det.box.y))),
                      cv::Point(static_cast<int>(std::round(det.box.x + det.box.width)),
                                static_cast<int>(std::round(det.box.y + det.box.height))));
        rect &= cv::Rect(0, 0, frame.cols, frame.rows);
        if (rect.empty())
        {
            continue;
        }

        boxes.push_back(rect);
    }

    if (boxes.empty())
    {
        return;
    }

    const int thickness = std::max(1, std::min(frame.cols, frame.rows) / 720);
    const int glow_thickness = thickness + 2;
    const int shadow_thickness = thickness + 1;
    cv::Mat glow = frame.clone();
    for (const cv::Rect &rect : boxes)
    {
        cv::rectangle(glow, rect, palette.bbox, glow_thickness, cv::LINE_AA);
    }
    cv::addWeighted(glow, 0.22, frame, 0.78, 0.0, frame);

    for (const cv::Rect &rect : boxes)
    {
        cv::rectangle(frame, rect, palette.bbox_shadow, shadow_thickness, cv::LINE_AA);
        cv::rectangle(frame, rect, palette.bbox, thickness, cv::LINE_AA);

        const int x1 = rect.x;
        const int y1 = rect.y;
        const int x2 = rect.x + rect.width - 1;
        const int y2 = rect.y + rect.height - 1;
        const int corner_len = std::max(10, std::min(42, std::min(rect.width, rect.height) / 3));
        const int corner_thickness = thickness;

        draw_corner_line(frame, cv::Point(x1, y1), cv::Point(std::min(x1 + corner_len, x2), y1),
                         palette.bbox_corner, corner_thickness);
        draw_corner_line(frame, cv::Point(x1, y1), cv::Point(x1, std::min(y1 + corner_len, y2)),
                         palette.bbox_corner, corner_thickness);
        draw_corner_line(frame, cv::Point(x2, y1), cv::Point(std::max(x2 - corner_len, x1), y1),
                         palette.bbox_corner, corner_thickness);
        draw_corner_line(frame, cv::Point(x2, y1), cv::Point(x2, std::min(y1 + corner_len, y2)),
                         palette.bbox_corner, corner_thickness);
        draw_corner_line(frame, cv::Point(x1, y2), cv::Point(std::min(x1 + corner_len, x2), y2),
                         palette.bbox_corner, corner_thickness);
        draw_corner_line(frame, cv::Point(x1, y2), cv::Point(x1, std::max(y2 - corner_len, y1)),
                         palette.bbox_corner, corner_thickness);
        draw_corner_line(frame, cv::Point(x2, y2), cv::Point(std::max(x2 - corner_len, x1), y2),
                         palette.bbox_corner, corner_thickness);
        draw_corner_line(frame, cv::Point(x2, y2), cv::Point(x2, std::max(y2 - corner_len, y1)),
                         palette.bbox_corner, corner_thickness);
    }
}

void draw_status(cv::Mat &frame, double fps, const RenderPalette &palette)
{
    const int font = cv::FONT_HERSHEY_SIMPLEX;
    const double label_scale = 0.46;
    const double value_scale = 0.78;
    const int label_thickness = 1;
    const int value_thickness = 2;

    std::ostringstream value;
    value << std::fixed << std::setprecision(1) << fps;

    int label_baseline = 0;
    int value_baseline = 0;
    const cv::Size label_size = cv::getTextSize("FPS", font, label_scale, label_thickness, &label_baseline);
    const cv::Size value_size = cv::getTextSize(value.str(), font, value_scale, value_thickness, &value_baseline);
    const int pad_x = 14;
    const int pad_y = 10;
    const int gap = 9;
    const int box_w = std::min(frame.cols - 16, pad_x * 2 + label_size.width + gap + value_size.width);
    const int box_h = std::min(frame.rows - 16, pad_y * 2 + std::max(label_size.height, value_size.height));
    if (box_w <= 0 || box_h <= 0 || frame.empty())
    {
        return;
    }

    const cv::Rect panel(12, 12, box_w, box_h);
    cv::Mat overlay = frame.clone();
    cv::rectangle(overlay, panel, palette.status_panel, -1, cv::LINE_AA);
    cv::addWeighted(overlay, 0.62, frame, 0.38, 0.0, frame);

    cv::rectangle(frame, cv::Rect(panel.x, panel.y, 4, panel.height), palette.status_accent, -1, cv::LINE_AA);
    cv::rectangle(frame, panel, palette.status_border, 1, cv::LINE_AA);

    const int center_y = panel.y + panel.height / 2;
    const int label_x = panel.x + pad_x;
    const int label_y = center_y + label_size.height / 2;
    const int value_x = label_x + label_size.width + gap;
    const int value_y = center_y + value_size.height / 2;
    cv::putText(frame, "FPS", cv::Point(label_x, label_y), font, label_scale,
                palette.status_label, label_thickness, cv::LINE_AA);
    cv::putText(frame, value.str(), cv::Point(value_x, value_y), font, value_scale,
                palette.status_value, value_thickness, cv::LINE_AA);
}

void collect_outputs(const dxrt::TensorPtrs &outputs, const Options &opt, const InputInfo &input_info,
                     std::vector<TensorView> *det_layers,
                     TensorView *drive,
                     TensorView *lane)
{
    if (det_layers == nullptr || drive == nullptr || lane == nullptr)
    {
        return;
    }

    det_layers->clear();
    *drive = TensorView();
    *lane = TensorView();

    for (const auto &ptr : outputs)
    {
        TensorView view;
        if (!ptr || !make_tensor_view(ptr.get(), &view))
        {
            continue;
        }

        if (is_detection_tensor(view, opt.num_classes, input_info.net_w, input_info.net_h))
        {
            det_layers->push_back(view);
        }
        else if (view.channels == 2 && drive->data == nullptr)
        {
            *drive = view;
        }
        else if (view.channels == 1 && lane->data == nullptr)
        {
            *lane = view;
        }
    }

    std::sort(det_layers->begin(), det_layers->end(),
              [](const TensorView &a, const TensorView &b) { return a.width > b.width; });
}

void print_detections(int frame_id, const std::vector<Detection> &detections)
{
    for (const Detection &det : detections)
    {
        std::cout << "frame " << (frame_id + 1)
                  << " cls=" << detection_class_name(det.class_id)
                  << " conf=" << std::fixed << std::setprecision(4) << det.confidence
                  << " box=[" << det.box.x << "," << det.box.y << ","
                  << det.box.x + det.box.width << "," << det.box.y + det.box.height << "]"
                  << std::endl;
    }
}

std::string default_save_path(const Options &opt)
{
    if (!opt.save_path.empty())
    {
        return opt.save_path;
    }
    if (opt.use_camera)
    {
        return "camera_dxnn_async.mp4";
    }
    return "video_dxnn_async.mp4";
}

class FullscreenVideoWidget : public QWidget
{
public:
    explicit FullscreenVideoWidget(bool show_exit_button, QWidget *parent = nullptr)
        : QWidget(parent), show_exit_button_(show_exit_button)
    {
        setWindowTitle(kWindowName);
        setWindowFlags(Qt::Window | Qt::FramelessWindowHint);
        setAttribute(Qt::WA_OpaquePaintEvent, true);
        setAutoFillBackground(false);
        setFocusPolicy(Qt::StrongFocus);
        setMouseTracking(true);
    }

    void setFrame(const cv::Mat &bgr)
    {
        if (bgr.empty() || bgr.type() != CV_8UC3)
        {
            frame_image_ = QImage();
            update();
            return;
        }

        ensureFrameBuffer(bgr.cols, bgr.rows);
#if QT_VERSION >= QT_VERSION_CHECK(5, 14, 0)
        for (int y = 0; y < bgr.rows; ++y)
        {
            std::memcpy(frame_image_.scanLine(y), bgr.ptr(y), static_cast<std::size_t>(bgr.cols) * 3U);
        }
#else
        cv::Mat rgb_view(bgr.rows, bgr.cols, CV_8UC3, frame_image_.bits(), frame_image_.bytesPerLine());
        cv::cvtColor(bgr, rgb_view, cv::COLOR_BGR2RGB);
#endif
        update();
    }

    bool stopRequested() const
    {
        return stop_requested_;
    }

protected:
    void paintEvent(QPaintEvent *) override
    {
        QPainter painter(this);
        painter.fillRect(rect(), QColor(0, 0, 0));

        if (frame_image_.isNull())
        {
            return;
        }

        const QSize target_size = frame_image_.size().scaled(size(), Qt::KeepAspectRatio);
        const QRect target((width() - target_size.width()) / 2,
                           (height() - target_size.height()) / 2,
                           target_size.width(),
                           target_size.height());

        painter.drawImage(target, frame_image_);
        painter.setPen(QColor(255, 255, 255, 36));
        painter.drawRect(target.adjusted(0, 0, -1, -1));
        if (show_exit_button_)
        {
            drawCloseButton(painter);
        }
    }

    void resizeEvent(QResizeEvent *event) override
    {
        QWidget::resizeEvent(event);
        positionCloseButton();
    }

    void mouseMoveEvent(QMouseEvent *event) override
    {
        const bool hover = show_exit_button_ && close_button_rect_.contains(event->pos());
        if (hover != close_hover_)
        {
            close_hover_ = hover;
            update(close_button_rect_.adjusted(-1, -1, 1, 1));
        }
        QWidget::mouseMoveEvent(event);
    }

    void leaveEvent(QEvent *event) override
    {
        if (close_hover_)
        {
            close_hover_ = false;
            update(close_button_rect_.adjusted(-1, -1, 1, 1));
        }
        QWidget::leaveEvent(event);
    }

    void mousePressEvent(QMouseEvent *event) override
    {
        if (show_exit_button_ && event->button() == Qt::LeftButton &&
            close_button_rect_.contains(event->pos()))
        {
            requestStop();
            close();
            return;
        }
        QWidget::mousePressEvent(event);
    }

    void keyPressEvent(QKeyEvent *event) override
    {
        if (event->key() == Qt::Key_Escape || event->key() == Qt::Key_Q)
        {
            requestStop();
            close();
            return;
        }
        QWidget::keyPressEvent(event);
    }

    void closeEvent(QCloseEvent *event) override
    {
        requestStop();
        QWidget::closeEvent(event);
    }

private:
    void requestStop()
    {
        stop_requested_ = true;
    }

    void positionCloseButton()
    {
        const int margin = 14;
        constexpr int size = 30;
        close_button_rect_ = QRect(std::max(0, width() - size - margin), margin, size, size);
    }

    void drawCloseButton(QPainter &painter)
    {
        if (close_button_rect_.isNull())
        {
            positionCloseButton();
        }

        painter.save();
        painter.setRenderHint(QPainter::Antialiasing, true);
        painter.setPen(QPen(QColor(255, 255, 255, close_hover_ ? 115 : 55), 1));
        painter.setBrush(close_hover_ ? QColor(220, 45, 45, 210) : QColor(12, 16, 20, 130));
        painter.drawRoundedRect(close_button_rect_, 5, 5);

        const QPoint center = close_button_rect_.center();
        constexpr int arm = 6;
        painter.setRenderHint(QPainter::Antialiasing, true);
        painter.setPen(QPen(QColor(255, 255, 255, 220), 2));
        painter.drawLine(center.x() - arm, center.y() - arm, center.x() + arm, center.y() + arm);
        painter.drawLine(center.x() + arm, center.y() - arm, center.x() - arm, center.y() + arm);
        painter.restore();
    }

    void ensureFrameBuffer(int width, int height)
    {
#if QT_VERSION >= QT_VERSION_CHECK(5, 14, 0)
        constexpr QImage::Format format = QImage::Format_BGR888;
#else
        constexpr QImage::Format format = QImage::Format_RGB888;
#endif
        if (frame_image_.size() == QSize(width, height) && frame_image_.format() == format)
        {
            return;
        }

        frame_image_ = QImage(width, height, format);
        if (frame_image_.isNull())
        {
            throw std::bad_alloc();
        }
    }

    bool stop_requested_ = false;
    bool show_exit_button_ = false;
    bool close_hover_ = false;
    QRect close_button_rect_;
    QImage frame_image_;
};

void process_qt_events(QApplication *app, FullscreenVideoWidget *window, int delay_ms)
{
    if (app == nullptr || window == nullptr)
    {
        return;
    }

    app->processEvents(QEventLoop::AllEvents);
    if (delay_ms <= 0)
    {
        return;
    }

    QElapsedTimer timer;
    timer.start();
    while (!window->stopRequested() && timer.elapsed() < delay_ms)
    {
        app->processEvents(QEventLoop::AllEvents, 2);
        QThread::msleep(1);
    }
}

} // namespace

int main(int argc, char **argv)
{
    Options opt;
    if (!parse_args(argc, argv, &opt))
    {
        print_usage(argv[0]);
        return 1;
    }

    try
    {
        opt.model_path = resolve_file_path(opt.model_path);
        if (!opt.use_camera)
        {
            opt.source = resolve_file_path(opt.source);
        }

        if (!file_exists(opt.model_path))
        {
            std::cerr << "Error: model file not found: " << opt.model_path << std::endl;
            return 1;
        }
        if (!opt.use_camera && !file_exists(opt.source))
        {
            std::cerr << "Error: video file not found: " << opt.source << std::endl;
            return 1;
        }

        int qt_argc = 1;
        char *qt_argv[] = {argv[0], nullptr};
        std::unique_ptr<QApplication> app;
        std::unique_ptr<FullscreenVideoWidget> window;
        if (!opt.no_display)
        {
            app = std::make_unique<QApplication>(qt_argc, qt_argv);
        }

        dxrt::InferenceOption io;
        dxrt::InferenceEngine engine(opt.model_path, io);
        print_tensors("Inputs", engine.GetInputs());
        print_tensors("Declared outputs", engine.GetOutputs());

        const InputInfo input_info = analyze_input(engine, opt.img_size);
        std::cout << "Model input: " << input_info.net_w << "x" << input_info.net_h
                  << (input_info.nchw ? " NCHW " : " NHWC ") << input_info.type << std::endl;
        const RenderPalette &palette = kRenderPalettes[static_cast<std::size_t>(opt.palette_index)];
        std::cout << "Color palette: " << palette.name << std::endl;

        cv::VideoCapture cap;
        if (opt.use_camera)
        {
            cap.open(opt.camera_index);
        }
        else
        {
            cap.open(opt.source);
        }
        if (!cap.isOpened())
        {
            std::cerr << "Error: failed to open input" << std::endl;
            return 1;
        }

        double source_fps = cap.get(cv::CAP_PROP_FPS);
        if (!(source_fps > 1e-3))
        {
            source_fps = 30.0;
        }

        if (!opt.no_display)
        {
            window = std::make_unique<FullscreenVideoWidget>(opt.show_exit_button);
            window->showFullScreen();
            window->raise();
            window->activateWindow();
            process_qt_events(app.get(), window.get(), 0);
        }

        cv::VideoWriter writer;
        AverageMeter infer_meter;
        AverageMeter post_meter;

        const auto start_time = std::chrono::steady_clock::now();
        const auto inference_fps_start = start_time;
        std::size_t completed_inferences = 0;
        double inference_fps = 0.0;
        int next_frame_id = 0;
        int displayed_frames = 0;
        bool source_ended = false;
        bool stopped = !opt.no_display && window->stopRequested();
        std::deque<AsyncJob> in_flight;

        std::cout << "Inference mode: DXRT RunAsync, queue_max=" << opt.queue_max << std::endl;

        auto read_next_source_frame = [&]() -> cv::Mat {
            cv::Mat frame;
            if (cap.read(frame) && !frame.empty())
            {
                return frame;
            }

            if (!opt.loop || opt.use_camera)
            {
                return cv::Mat();
            }

            cap.set(cv::CAP_PROP_POS_FRAMES, 0);
            frame.release();
            if (cap.read(frame) && !frame.empty())
            {
                return frame;
            }

            cap.release();
            cap.open(opt.source);
            frame.release();
            if (cap.isOpened() && cap.read(frame) && !frame.empty())
            {
                return frame;
            }

            return cv::Mat();
        };

        auto submit_next_frame = [&]() -> bool {
            if (opt.max_frames > 0 && next_frame_id >= opt.max_frames)
            {
                source_ended = true;
                return false;
            }

            cv::Mat frame = read_next_source_frame();
            if (frame.empty())
            {
                source_ended = true;
                return false;
            }

            if (opt.resize_width > 0 && opt.resize_height > 0)
            {
                cv::resize(frame, frame, cv::Size(opt.resize_width, opt.resize_height), 0.0, 0.0, cv::INTER_LINEAR);
            }

            AsyncJob job;
            job.frame_id = next_frame_id;
            job.frame = std::move(frame);
            prepare_input(job.frame, input_info, &job.u8_input, &job.float_input, &job.input_ptr, &job.letterbox);

            job.submit_time = std::chrono::steady_clock::now();
            job.job_id = engine.RunAsync(job.input_ptr, nullptr, nullptr);
            in_flight.push_back(std::move(job));
            ++next_frame_id;
            return true;
        };

        while (!stopped)
        {
            while (!source_ended && static_cast<int>(in_flight.size()) < opt.queue_max)
            {
                submit_next_frame();
            }

            if (in_flight.empty())
            {
                break;
            }

            AsyncJob job = std::move(in_flight.front());
            in_flight.pop_front();
            dxrt::TensorPtrs outputs = engine.Wait(job.job_id);
            const auto t_infer1 = std::chrono::steady_clock::now();
            ++completed_inferences;
            const double fps_elapsed = std::chrono::duration<double>(t_infer1 - inference_fps_start).count();
            if (fps_elapsed > 1e-9)
            {
                inference_fps = static_cast<double>(completed_inferences) / fps_elapsed;
            }
            print_output_ptrs_once(outputs);

            std::vector<TensorView> det_layers;
            TensorView drive_view;
            TensorView lane_view;
            std::vector<Detection> detections;
            const auto t_post0 = std::chrono::steady_clock::now();
            try
            {
                collect_outputs(outputs, opt, input_info, &det_layers, &drive_view, &lane_view);

                std::vector<Detection> candidates;
                for (const TensorView &layer : det_layers)
                {
                    decode_detection_layer(layer, job.letterbox, opt, &candidates);
                }
                detections = nms(std::move(candidates), opt.iou_thres, opt.agnostic_nms);

                cv::Mat drive_mask;
                cv::Mat lane_mask;
                if (drive_view.data != nullptr)
                {
                    drive_mask = make_drive_mask(drive_view, job.letterbox);
                }
                if (lane_view.data != nullptr)
                {
                    lane_mask = make_lane_mask(lane_view, job.letterbox);
                }

                overlay_segmentation(job.frame, drive_mask, lane_mask, palette);
                draw_detections(job.frame, detections, palette);
            }
            catch (const std::exception &e)
            {
                throw std::runtime_error("post-processing failed: " + std::string(e.what()));
            }
            const auto t_post1 = std::chrono::steady_clock::now();

            const double infer_ms = std::chrono::duration<double, std::milli>(t_infer1 - job.submit_time).count();
            const double post_ms = std::chrono::duration<double, std::milli>(t_post1 - t_post0).count();
            infer_meter.update(infer_ms);
            post_meter.update(post_ms);

            draw_status(job.frame, inference_fps, palette);

            if (opt.print_detections)
            {
                print_detections(job.frame_id, detections);
            }

            if (opt.save_video && !writer.isOpened())
            {
                const std::string save_path = default_save_path(opt);
                writer.open(save_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                            source_fps, job.frame.size());
                if (!writer.isOpened())
                {
                    throw std::runtime_error("failed to open video writer: " + save_path);
                }
                std::cout << "Saving output to: " << save_path << std::endl;
            }
            if (writer.isOpened())
            {
                writer.write(job.frame);
            }

            if (!opt.no_display)
            {
                try
                {
                    window->setFrame(job.frame);
                }
                catch (const std::exception &e)
                {
                    throw std::runtime_error("Qt frame upload failed: " + std::string(e.what()));
                }

                try
                {
                    process_qt_events(app.get(), window.get(), opt.delay);
                    if (window->stopRequested())
                    {
                        stopped = true;
                    }
                }
                catch (const std::exception &e)
                {
                    throw std::runtime_error("Qt event processing failed: " + std::string(e.what()));
                }
            }

            ++displayed_frames;
            if (opt.log_interval > 0 && displayed_frames % opt.log_interval == 0)
            {
                std::cout << "frame " << displayed_frames
                          << ": infer " << infer_ms << " ms, post " << post_ms
                          << " ms, det " << detections.size()
                          << ", in_flight " << in_flight.size() << "/" << opt.queue_max
                          << ", infer_FPS " << inference_fps << std::endl;
            }
        }

        while (!in_flight.empty())
        {
            engine.Wait(in_flight.front().job_id);
            in_flight.pop_front();
        }

        if (writer.isOpened())
        {
            writer.release();
        }
        cap.release();
        if (!opt.no_display)
        {
            window->close();
            process_qt_events(app.get(), window.get(), 0);
            window.reset();
            app.reset();
        }

        const auto end_time = std::chrono::steady_clock::now();
        const double elapsed = std::chrono::duration<double>(end_time - start_time).count();
        std::cout << "Done. frames=" << displayed_frames
                  << " infer_avg=" << infer_meter.avg() << " ms"
                  << " post_avg=" << post_meter.avg() << " ms"
                  << " infer_fps_avg=" << inference_fps
                  << " elapsed=" << elapsed << " s" << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
