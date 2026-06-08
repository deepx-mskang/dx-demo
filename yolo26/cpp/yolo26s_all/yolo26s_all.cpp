/**
 * @file yolo26s_all.cpp
 * @brief Qt5 2x2 all-task YOLO26-S demo.
 */

#include <dxrt/dxrt_api.h>

#include <QApplication>
#include <QCloseEvent>
#include <QFrame>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QImage>
#include <QKeyEvent>
#include <QLabel>
#include <QMainWindow>
#include <QMetaObject>
#include <QPixmap>
#include <QSizePolicy>
#include <QTimer>
#include <QVBoxLayout>
#include <QWidget>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cctype>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "extern/cxxopts.hpp"

#include "common/base/i_processor.hpp"
#include "common/processors/simple_resize_preprocessor.hpp"
#include "common/utility/common_util.hpp"
#include "common/utility/labels.hpp"

#include "factory/yolo26s_factory.hpp"
#include "factory/yolo26s_pose_factory.hpp"
#include "factory/yolo26s_seg_factory.hpp"

namespace stdfs = std::filesystem;

namespace {

constexpr int kPanelCount = 4;
constexpr int kOdPanel = 0;
constexpr int kPosePanel = 1;
constexpr int kSegPanel = 2;
constexpr int kClsPanel = 3;

const char* kPanelTitles[kPanelCount] = {
    "YOLO26-S Object Detection",
    "YOLO26-S Pose Estimation",
    "YOLO26-S Instance Segmentation",
    "YOLO26-S Classification",
};

std::string projectPath(const std::string& rel) {
    stdfs::path root(PROJECT_ROOT_DIR);
    return (root / rel).lexically_normal().string();
}

std::string absolutePath(const std::string& path) {
    stdfs::path p(path);
    if (p.is_relative()) {
        p = stdfs::current_path() / p;
    }
    return stdfs::absolute(p).lexically_normal().string();
}

struct AppArgs {
    std::string model = projectPath("cpp/assets/yolo26s.dxnn");
    std::string model_pose = projectPath("cpp/assets/yolo26s-pose.dxnn");
    std::string model_seg = projectPath("cpp/assets/yolo26s-seg.dxnn");
    std::string model_cls = projectPath("cpp/assets/yolo26s-cls.dxnn");
    std::string video;
    bool no_loop_video = false;
    std::string device = "/dev/video0";
    int width = 1280;
    int height = 720;
    int fps = 30;
    std::string decoder = "mppjpegdec";
};

AppArgs parseArgs(int argc, char* argv[]) {
    AppArgs args;
    cxxopts::Options options(
        "yolo26s_all",
        "Qt5 2x2 YOLO26-S all-task demo: object detection, pose, instance segmentation, classification.");

    options.add_options()
        ("model", "YOLOv26 detection .dxnn",
         cxxopts::value<std::string>(args.model)->default_value(args.model))
        ("model-pose", "YOLOv26 pose .dxnn",
         cxxopts::value<std::string>(args.model_pose)->default_value(args.model_pose))
        ("model-seg", "YOLOv26 segmentation .dxnn",
         cxxopts::value<std::string>(args.model_seg)->default_value(args.model_seg))
        ("model-cls", "YOLOv26 classification .dxnn",
         cxxopts::value<std::string>(args.model_cls)->default_value(args.model_cls))
        ("video", "Video file path input. If omitted, USB webcam is used.",
         cxxopts::value<std::string>(args.video)->default_value(""))
        ("no-loop-video", "With --video, stop at EOF instead of looping.",
         cxxopts::value<bool>(args.no_loop_video)->default_value("false"))
        ("device", "V4L2 camera device, e.g. /dev/video0.",
         cxxopts::value<std::string>(args.device)->default_value(args.device))
        ("width", "Requested camera width.",
         cxxopts::value<int>(args.width)->default_value(std::to_string(args.width)))
        ("height", "Requested camera height.",
         cxxopts::value<int>(args.height)->default_value(std::to_string(args.height)))
        ("fps", "Requested camera FPS.",
         cxxopts::value<int>(args.fps)->default_value(std::to_string(args.fps)))
        ("decoder", "Accepted for CLI compatibility; unused without OpenCV GStreamer.",
         cxxopts::value<std::string>(args.decoder)->default_value(args.decoder))
        ("h,help", "Print usage");

    auto result = options.parse(argc, argv);
    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        std::exit(0);
    }
    return args;
}

void requireFile(const std::string& label, const std::string& path) {
    if (!stdfs::is_regular_file(path)) {
        throw std::runtime_error("[ERROR] " + label + " not found: " + path);
    }
}

void notifyLauncherReady() {
    const char* raw_path = std::getenv("DX_LAUNCHER_READY_FILE");
    if (raw_path == nullptr || *raw_path == '\0') {
        return;
    }

    try {
        stdfs::path path(raw_path);
        if (path.has_parent_path()) {
            stdfs::create_directories(path.parent_path());
        }
        std::ofstream out(path);
        out << "ready\n";
    } catch (const std::exception&) {
        // Launcher readiness is best-effort, matching the Python demo.
    }
}

int parseCameraIndex(const std::string& device) {
    if (device.empty()) {
        return 0;
    }
    if (std::all_of(device.begin(), device.end(), [](unsigned char ch) { return std::isdigit(ch); })) {
        return std::stoi(device);
    }
    const std::string prefix = "/dev/video";
    if (device.rfind(prefix, 0) == 0) {
        std::string suffix = device.substr(prefix.size());
        if (!suffix.empty() &&
            std::all_of(suffix.begin(), suffix.end(), [](unsigned char ch) { return std::isdigit(ch); })) {
            return std::stoi(suffix);
        }
    }
    return 0;
}

cv::Mat ensureBgr3(const cv::Mat& frame) {
    if (frame.empty()) {
        return {};
    }
    if (frame.channels() == 1) {
        cv::Mat bgr;
        cv::cvtColor(frame, bgr, cv::COLOR_GRAY2BGR);
        return bgr;
    }
    if (frame.channels() == 4) {
        cv::Mat bgr;
        cv::cvtColor(frame, bgr, cv::COLOR_BGRA2BGR);
        return bgr;
    }
    if (frame.channels() == 3) {
        return frame;
    }
    return {};
}

class LatestFrameQueue {
public:
    void push(const cv::Mat& frame) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            frame_ = frame.clone();
            has_frame_ = true;
        }
        cv_.notify_one();
    }

    bool waitPop(cv::Mat& frame, const std::atomic<bool>& running) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait_for(lock, std::chrono::milliseconds(50), [&] {
            return has_frame_ || !running.load(std::memory_order_relaxed);
        });
        if (!running.load(std::memory_order_relaxed)) {
            return false;
        }
        if (!has_frame_) {
            return false;
        }
        frame = std::move(frame_);
        has_frame_ = false;
        return true;
    }

    void wake() {
        cv_.notify_all();
    }

private:
    std::mutex mutex_;
    std::condition_variable cv_;
    cv::Mat frame_;
    bool has_frame_{false};
};

class IFrameConsumer {
public:
    virtual ~IFrameConsumer() = default;
    virtual void start() = 0;
    virtual void stop() = 0;
    virtual void join() = 0;
    virtual void enqueue(const cv::Mat& frame) = 0;
};

class QuadWindow;

dxrt::TensorPtrs runInference(dxrt::InferenceEngine& ie,
                              const cv::Mat& preprocessed,
                              bool is_float_input,
                              bool is_nhwc) {
    if (preprocessed.empty()) {
        return {};
    }
    if (is_float_input) {
        std::vector<float> float_buf = convertToFloatBuffer(preprocessed, is_nhwc);
        return ie.Run(float_buf.data(), nullptr, nullptr);
    }
    return ie.Run(preprocessed.data, nullptr, nullptr);
}

template <typename ResultT, typename FactoryT>
class ResultWorker final : public IFrameConsumer {
public:
    ResultWorker(int panel_index,
                 std::string name,
                 std::string model_path,
                 QuadWindow* window)
        : panel_index_(panel_index),
          name_(std::move(name)),
          model_path_(std::move(model_path)),
          window_(window) {}

    void start() override {
        running_.store(true, std::memory_order_relaxed);
        thread_ = std::thread(&ResultWorker::run, this);
    }

    void stop() override {
        running_.store(false, std::memory_order_relaxed);
        queue_.wake();
    }

    void join() override {
        if (thread_.joinable()) {
            thread_.join();
        }
    }

    void enqueue(const cv::Mat& frame) override {
        if (running_.load(std::memory_order_relaxed)) {
            queue_.push(frame);
        }
    }

private:
    void run();

    int panel_index_;
    std::string name_;
    std::string model_path_;
    QuadWindow* window_;
    LatestFrameQueue queue_;
    std::atomic<bool> running_{false};
    std::thread thread_;
};

class ClassificationWorker final : public IFrameConsumer {
public:
    ClassificationWorker(std::string model_path, QuadWindow* window)
        : model_path_(std::move(model_path)), window_(window) {}

    void start() override {
        running_.store(true, std::memory_order_relaxed);
        thread_ = std::thread(&ClassificationWorker::run, this);
    }

    void stop() override {
        running_.store(false, std::memory_order_relaxed);
        queue_.wake();
    }

    void join() override {
        if (thread_.joinable()) {
            thread_.join();
        }
    }

    void enqueue(const cv::Mat& frame) override {
        if (running_.load(std::memory_order_relaxed)) {
            queue_.push(frame);
        }
    }

private:
    struct TopEntry {
        std::string label;
        double probability{0.0};
    };

    void run();
    static std::vector<TopEntry> top3Softmax(const dxrt::TensorPtrs& outputs);
    static void drawTop3(cv::Mat& image, const std::vector<TopEntry>& entries);

    std::string model_path_;
    QuadWindow* window_;
    LatestFrameQueue queue_;
    std::atomic<bool> running_{false};
    std::thread thread_;
};

class CaptureThread final {
public:
    CaptureThread(AppArgs args, QuadWindow* window)
        : args_(std::move(args)), window_(window) {}

    void setConsumers(std::vector<IFrameConsumer*> consumers) {
        consumers_ = std::move(consumers);
    }

    void start() {
        running_.store(true, std::memory_order_relaxed);
        thread_ = std::thread(&CaptureThread::run, this);
    }

    void stop() {
        running_.store(false, std::memory_order_relaxed);
    }

    void join() {
        if (thread_.joinable()) {
            thread_.join();
        }
    }

private:
    void run();
    void runVideo();
    void runCamera();
    void publishFrame(const cv::Mat& frame);

    AppArgs args_;
    QuadWindow* window_;
    std::vector<IFrameConsumer*> consumers_;
    std::atomic<bool> running_{false};
    std::thread thread_;
};

QFrame* makePanel(const QString& title, QLabel** image_label, QLabel** fps_label) {
    auto* frame = new QFrame;
    frame->setObjectName("yolo_panel");
    frame->setStyleSheet(
        "QFrame#yolo_panel {"
        "background-color: #13151a;"
        "border: 1px solid #3a404c;"
        "border-radius: 8px;"
        "}");

    auto* vbox = new QVBoxLayout(frame);
    vbox->setContentsMargins(0, 0, 0, 0);
    vbox->setSpacing(0);

    auto* header = new QHBoxLayout;
    header->setContentsMargins(0, 0, 0, 0);
    header->setSpacing(0);

    auto* title_label = new QLabel(title);
    title_label->setObjectName("yolo_panel_title");
    title_label->setAlignment(Qt::AlignLeft | Qt::AlignVCenter);
    title_label->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    title_label->setFocusPolicy(Qt::NoFocus);
    title_label->setStyleSheet(
        "QLabel#yolo_panel_title {"
        "color: #eceef4;"
        "font-size: 13px;"
        "font-weight: 600;"
        "padding: 8px 12px 6px 12px;"
        "background-color: #1e222b;"
        "border-bottom: 1px solid #353b48;"
        "border-top-left-radius: 7px;"
        "}");

    auto* fps = new QLabel("-");
    fps->setObjectName("yolo_panel_fps");
    fps->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    fps->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Fixed);
    fps->setFocusPolicy(Qt::NoFocus);
    fps->setStyleSheet(
        "QLabel#yolo_panel_fps {"
        "color: #9aa3b2;"
        "font-size: 12px;"
        "font-weight: 500;"
        "font-family: monospace;"
        "padding: 8px 12px 6px 8px;"
        "background-color: #1e222b;"
        "border-bottom: 1px solid #353b48;"
        "border-top-right-radius: 7px;"
        "}");

    header->addWidget(title_label, 1);
    header->addWidget(fps, 0);
    vbox->addLayout(header);

    auto* image = new QLabel;
    image->setAlignment(Qt::AlignCenter);
    image->setScaledContents(true);
    image->setMinimumSize(280, 160);
    image->setFocusPolicy(Qt::NoFocus);
    image->setStyleSheet(
        "background-color: #0c0d10;"
        "border-bottom-left-radius: 7px;"
        "border-bottom-right-radius: 7px;");
    vbox->addWidget(image, 1);

    *image_label = image;
    *fps_label = fps;
    return frame;
}

class QuadWindow final : public QMainWindow {
public:
    explicit QuadWindow(const AppArgs& args)
        : args_(args), capture_(args, this) {
        setWindowTitle("yolo26s_all");

        auto* central = new QWidget;
        central->setFocusPolicy(Qt::StrongFocus);
        setCentralWidget(central);

        auto* grid = new QGridLayout(central);
        grid->setSpacing(8);
        grid->setContentsMargins(8, 8, 8, 8);

        const int cells[kPanelCount][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        for (int i = 0; i < kPanelCount; ++i) {
            QLabel* image = nullptr;
            QLabel* fps = nullptr;
            auto* panel = makePanel(kPanelTitles[i], &image, &fps);
            grid->addWidget(panel, cells[i][0], cells[i][1]);
            image_labels_.push_back(image);
            fps_labels_.push_back(fps);
            fps_counts_.push_back(0);
        }

        fps_timer_ = new QTimer(this);
        fps_timer_->setInterval(1000);
        QObject::connect(fps_timer_, &QTimer::timeout, this, [this] { tickFps(); });
        fps_timer_->start();

        workers_.push_back(std::make_unique<ResultWorker<dxapp::DetectionResult, dxapp::Yolo26sFactory>>(
            kOdPanel, "Object Detection", args_.model, this));
        workers_.push_back(std::make_unique<ResultWorker<dxapp::PoseResult, dxapp::Yolo26s_poseFactory>>(
            kPosePanel, "Pose Estimation", args_.model_pose, this));
        workers_.push_back(std::make_unique<ResultWorker<dxapp::InstanceSegmentationResult, dxapp::Yolo26s_segFactory>>(
            kSegPanel, "Instance Segmentation", args_.model_seg, this));
        workers_.push_back(std::make_unique<ClassificationWorker>(args_.model_cls, this));

        std::vector<IFrameConsumer*> consumers;
        consumers.reserve(workers_.size());
        for (auto& worker : workers_) {
            worker->start();
            consumers.push_back(worker.get());
        }
        capture_.setConsumers(std::move(consumers));
        capture_.start();
    }

    ~QuadWindow() override {
        shutdown();
    }

    void postFrame(int panel_index, const cv::Mat& frame) {
        if (closing_.load(std::memory_order_relaxed)) {
            return;
        }
        cv::Mat safe_frame = frame;
        QMetaObject::invokeMethod(this, [this, panel_index, safe_frame]() mutable {
            if (closing_.load(std::memory_order_relaxed)) {
                return;
            }
            setPanelFrame(panel_index, safe_frame);
        }, Qt::QueuedConnection);
    }

    void postError(const std::string& message) {
        QMetaObject::invokeMethod(this, [this, message]() {
            if (closing_.load(std::memory_order_relaxed)) {
                return;
            }
            showError(message);
        }, Qt::QueuedConnection);
    }

protected:
    void keyPressEvent(QKeyEvent* event) override {
        if (event->key() == Qt::Key_Escape || event->key() == Qt::Key_Q) {
            close();
            return;
        }
        QMainWindow::keyPressEvent(event);
    }

    void closeEvent(QCloseEvent* event) override {
        shutdown();
        event->accept();
    }

    void showEvent(QShowEvent* event) override {
        QMainWindow::showEvent(event);
        if (centralWidget() != nullptr) {
            centralWidget()->setFocus(Qt::OtherFocusReason);
        }
    }

private:
    void tickFps() {
        for (int i = 0; i < static_cast<int>(fps_labels_.size()); ++i) {
            int n = fps_counts_[i];
            fps_counts_[i] = 0;
            fps_labels_[i]->setText(n > 0 ? QString("%1 FPS").arg(n) : "-");
        }
    }

    void setPanelFrame(int panel_index, const cv::Mat& bgr) {
        if (panel_index < 0 || panel_index >= static_cast<int>(image_labels_.size()) || bgr.empty()) {
            return;
        }

        cv::Mat rgb;
        cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
        QImage image(rgb.data,
                     rgb.cols,
                     rgb.rows,
                     static_cast<int>(rgb.step),
                     QImage::Format_RGB888);
        image_labels_[panel_index]->setPixmap(QPixmap::fromImage(image.copy()));
        fps_counts_[panel_index] += 1;
    }

    void showError(const std::string& message) {
        std::cerr << "[ERROR] " << message << std::endl;
        for (auto* label : image_labels_) {
            label->clear();
            label->setText(QString::fromStdString(message));
            label->setStyleSheet(
                "background-color: #0c0d10;"
                "color: #eceef4;"
                "border-bottom-left-radius: 7px;"
                "border-bottom-right-radius: 7px;");
        }
    }

    void shutdown() {
        bool expected = false;
        if (!closing_.compare_exchange_strong(expected, true)) {
            return;
        }

        if (fps_timer_ != nullptr) {
            fps_timer_->stop();
        }

        capture_.stop();
        capture_.join();

        for (auto& worker : workers_) {
            worker->stop();
        }
        for (auto& worker : workers_) {
            worker->join();
        }
    }

    AppArgs args_;
    std::vector<QLabel*> image_labels_;
    std::vector<QLabel*> fps_labels_;
    std::vector<int> fps_counts_;
    QTimer* fps_timer_{nullptr};
    std::vector<std::unique_ptr<IFrameConsumer>> workers_;
    CaptureThread capture_;
    std::atomic<bool> closing_{false};
};

template <typename ResultT, typename FactoryT>
void ResultWorker<ResultT, FactoryT>::run() {
    try {
        dxrt::InferenceOption io;
        dxrt::InferenceEngine ie(model_path_, io);
        if (!dxapp::minversionforRTandCompiler(&ie)) {
            throw std::runtime_error(name_ + " model/runtime version mismatch: " + model_path_);
        }

        auto input_shape = ie.GetInputs().front().shape();
        int input_width = 0;
        int input_height = 0;
        parseInputShape(input_shape, input_width, input_height);
        bool is_float_input = (ie.GetInputs().front().type() == dxrt::DataType::FLOAT);
        bool is_nhwc = isInputNHWC(input_shape);

        FactoryT factory;
        auto preprocessor = factory.createPreprocessor(input_width, input_height);
        auto postprocessor = factory.createPostprocessor(input_width, input_height, ie.IsOrtConfigured());
        auto visualizer = factory.createVisualizer();

        std::cout << "[INFO] " << name_ << " model: " << model_path_ << std::endl;
        std::cout << "[INFO] " << name_ << " input size (WxH): "
                  << input_width << "x" << input_height << std::endl;

        while (running_.load(std::memory_order_relaxed)) {
            cv::Mat frame;
            if (!queue_.waitPop(frame, running_)) {
                continue;
            }
            if (frame.empty()) {
                continue;
            }

            dxapp::PreprocessContext ctx;
            cv::Mat preprocessed;
            preprocessor->process(frame, preprocessed, ctx);
            auto outputs = runInference(ie, preprocessed, is_float_input, is_nhwc);
            auto results = postprocessor->process(outputs, ctx);
            cv::Mat rendered = visualizer->draw(frame, results, ctx);
            window_->postFrame(panel_index_, rendered);
        }
    } catch (const std::exception& e) {
        running_.store(false, std::memory_order_relaxed);
        window_->postError(name_ + " worker failed: " + e.what());
    }
}

std::vector<ClassificationWorker::TopEntry> ClassificationWorker::top3Softmax(
    const dxrt::TensorPtrs& outputs) {
    std::vector<TopEntry> entries;
    if (outputs.empty()) {
        return entries;
    }

    const auto& tensor = outputs.front();
    if (tensor->type() != dxrt::DataType::FLOAT) {
        const uint16_t* data = static_cast<const uint16_t*>(tensor->data());
        int class_id = static_cast<int>(data[0]);
        entries.push_back({dxapp::getImageNetClassName(class_id), 1.0});
        return entries;
    }

    auto shape = tensor->shape();
    size_t n = 1;
    for (auto dim : shape) {
        if (dim > 0) {
            n *= static_cast<size_t>(dim);
        }
    }
    if (n == 0) {
        return entries;
    }

    const float* data = static_cast<const float*>(tensor->data());
    double max_logit = static_cast<double>(data[0]);
    for (size_t i = 1; i < n; ++i) {
        max_logit = std::max(max_logit, static_cast<double>(data[i]));
    }

    std::vector<double> probs(n);
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double v = std::exp(std::max(-80.0, std::min(80.0, static_cast<double>(data[i]) - max_logit)));
        probs[i] = v;
        sum += v;
    }
    if (sum <= 0.0) {
        return entries;
    }
    for (double& p : probs) {
        p /= sum;
    }

    int k = static_cast<int>(std::min<size_t>(3, probs.size()));
    std::vector<int> indices(probs.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
        [&probs](int a, int b) { return probs[a] > probs[b]; });

    entries.reserve(k);
    for (int i = 0; i < k; ++i) {
        int class_id = indices[i];
        entries.push_back({dxapp::getImageNetClassName(class_id), probs[class_id]});
    }
    return entries;
}

void ClassificationWorker::drawTop3(cv::Mat& image, const std::vector<TopEntry>& entries) {
    int y0 = 36;
    for (size_t i = 0; i < entries.size(); ++i) {
        std::string label = entries[i].label;
        auto comma = label.find(',');
        if (comma != std::string::npos) {
            label = label.substr(0, comma);
        }
        while (!label.empty() && label.front() == ' ') {
            label.erase(label.begin());
        }
        while (!label.empty() && label.back() == ' ') {
            label.pop_back();
        }
        if (label.size() > 48) {
            label = label.substr(0, 45) + "...";
        }

        std::ostringstream line;
        line << (i + 1) << ". " << label << "  ("
             << std::fixed << std::setprecision(3) << entries[i].probability << ")";

        int y = y0 + static_cast<int>(i) * 32;
        cv::putText(image, line.str(), cv::Point(12, y),
                    cv::FONT_HERSHEY_SIMPLEX, 0.65,
                    cv::Scalar(30, 30, 30), 4, cv::LINE_AA);
        cv::putText(image, line.str(), cv::Point(12, y),
                    cv::FONT_HERSHEY_SIMPLEX, 0.65,
                    cv::Scalar(180, 255, 180), 2, cv::LINE_AA);
    }
}

void ClassificationWorker::run() {
    try {
        dxrt::InferenceOption io;
        dxrt::InferenceEngine ie(model_path_, io);
        if (!dxapp::minversionforRTandCompiler(&ie)) {
            throw std::runtime_error("Classification model/runtime version mismatch: " + model_path_);
        }

        auto input_shape = ie.GetInputs().front().shape();
        int input_width = 0;
        int input_height = 0;
        parseInputShape(input_shape, input_width, input_height);
        bool is_float_input = (ie.GetInputs().front().type() == dxrt::DataType::FLOAT);
        bool is_nhwc = isInputNHWC(input_shape);
        dxapp::SimpleResizePreprocessor preprocessor(input_width, input_height, cv::COLOR_BGR2RGB);

        std::cout << "[INFO] Classification model: " << model_path_ << std::endl;
        std::cout << "[INFO] Classification input size (WxH): "
                  << input_width << "x" << input_height << std::endl;

        while (running_.load(std::memory_order_relaxed)) {
            cv::Mat frame;
            if (!queue_.waitPop(frame, running_)) {
                continue;
            }
            if (frame.empty()) {
                continue;
            }

            dxapp::PreprocessContext ctx;
            cv::Mat preprocessed;
            preprocessor.process(frame, preprocessed, ctx);
            auto outputs = runInference(ie, preprocessed, is_float_input, is_nhwc);
            auto top3 = top3Softmax(outputs);
            cv::Mat rendered = frame.clone();
            drawTop3(rendered, top3);
            window_->postFrame(kClsPanel, rendered);
        }
    } catch (const std::exception& e) {
        running_.store(false, std::memory_order_relaxed);
        window_->postError(std::string("Classification worker failed: ") + e.what());
    }
}

void CaptureThread::publishFrame(const cv::Mat& frame) {
    cv::Mat bgr = ensureBgr3(frame);
    if (bgr.empty()) {
        return;
    }
    for (auto* consumer : consumers_) {
        consumer->enqueue(bgr);
    }
}

void CaptureThread::runVideo() {
    cv::VideoCapture cap(args_.video);
    if (!cap.isOpened()) {
        window_->postError("Could not open video file: " + args_.video);
        return;
    }

    double source_fps = cap.get(cv::CAP_PROP_FPS);
    if (source_fps <= 1e-3) {
        source_fps = 30.0;
    }
    const auto frame_delay = std::chrono::duration<double>(1.0 / source_fps);

    while (running_.load(std::memory_order_relaxed)) {
        auto loop_start = std::chrono::steady_clock::now();
        cv::Mat frame;
        bool ok = cap.read(frame);
        if (!ok || frame.empty()) {
            if (args_.no_loop_video) {
                window_->postError("Video ended.");
                break;
            }

            cap.set(cv::CAP_PROP_POS_FRAMES, 0);
            ok = cap.read(frame);
            if (!ok || frame.empty()) {
                cap.release();
                cap.open(args_.video);
                if (!cap.isOpened()) {
                    window_->postError("Video loop failed to reopen file.");
                    break;
                }
                ok = cap.read(frame);
            }
            if (!ok || frame.empty()) {
                window_->postError("Video file has no readable frames.");
                break;
            }
        }

        publishFrame(frame);

        auto elapsed = std::chrono::steady_clock::now() - loop_start;
        if (elapsed < frame_delay && running_.load(std::memory_order_relaxed)) {
            std::this_thread::sleep_for(frame_delay - elapsed);
        }
    }
}

void CaptureThread::runCamera() {
    cv::VideoCapture cap;
    bool opened = false;

    if (!args_.device.empty() && args_.device[0] == '/') {
        opened = cap.open(args_.device, cv::CAP_V4L2);
    }
    if (!opened) {
        int index = parseCameraIndex(args_.device);
        opened = cap.open(index, cv::CAP_V4L2);
    }
    if (!opened) {
        window_->postError("VideoCapture failed for camera: " + args_.device);
        return;
    }

    if (args_.width > 0) {
        cap.set(cv::CAP_PROP_FRAME_WIDTH, args_.width);
    }
    if (args_.height > 0) {
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, args_.height);
    }
    if (args_.fps > 0) {
        cap.set(cv::CAP_PROP_FPS, args_.fps);
    }

    std::cout << "[INFO] Input: USB webcam " << args_.device << std::endl;
    std::cout << "[INFO] Camera resolution: "
              << cap.get(cv::CAP_PROP_FRAME_WIDTH) << "x"
              << cap.get(cv::CAP_PROP_FRAME_HEIGHT) << std::endl;

    while (running_.load(std::memory_order_relaxed)) {
        cv::Mat frame;
        if (!cap.read(frame) || frame.empty()) {
            window_->postError("cap.read() failed or stream ended.");
            break;
        }
        publishFrame(frame);
    }
}

void CaptureThread::run() {
    try {
        if (!args_.video.empty()) {
            std::cout << "[INFO] Input: video file " << args_.video << std::endl;
            if (args_.no_loop_video) {
                std::cout << "[INFO] Video will not loop (stop at EOF)" << std::endl;
            }
            runVideo();
        } else {
            std::cout << "[INFO] OpenCV GStreamer input is disabled for this demo build." << std::endl;
            std::cout << "[INFO] Ignoring --decoder=" << args_.decoder
                      << "; using V4L2 camera input." << std::endl;
            runCamera();
        }
    } catch (const std::exception& e) {
        window_->postError(std::string("Capture thread failed: ") + e.what());
    }
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        AppArgs args = parseArgs(argc, argv);
        args.model = absolutePath(args.model);
        args.model_pose = absolutePath(args.model_pose);
        args.model_seg = absolutePath(args.model_seg);
        args.model_cls = absolutePath(args.model_cls);
        if (!args.video.empty()) {
            args.video = absolutePath(args.video);
        }

        requireFile("OD model", args.model);
        requireFile("Pose model", args.model_pose);
        requireFile("Seg model", args.model_seg);
        requireFile("Cls model", args.model_cls);
        if (!args.video.empty()) {
            requireFile("Video file", args.video);
        }

        QApplication app(argc, argv);
        QuadWindow window(args);
        window.showFullScreen();
        notifyLauncherReady();
        return app.exec();
    } catch (const dxrt::Exception& e) {
        std::cerr << e.what() << " error-code=" << e.code() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
