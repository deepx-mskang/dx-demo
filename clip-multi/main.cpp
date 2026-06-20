#include "clip_tokenizer.hpp"

#include <dxrt/dxrt_api.h>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#include <QApplication>
#include <QCloseEvent>
#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QFontMetrics>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QImage>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QKeySequence>
#include <QLabel>
#include <QMainWindow>
#include <QPixmap>
#include <QPushButton>
#include <QShortcut>
#include <QSizePolicy>
#include <QThread>
#include <QTimer>
#include <QVBoxLayout>
#include <QWidget>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <filesystem>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

Q_DECLARE_METATYPE(cv::Mat)

namespace fs = std::filesystem;

namespace {

constexpr int kImageSize = 224;
constexpr int kContextLength = 77;
constexpr int kEmbeddingSize = 768;
constexpr int kMaxAsyncJobs = 24;

struct StreamConfig {
    QString name;
    std::string source;
    std::string pipeline;
    std::vector<QString> texts;
    float threshold = 0.25F;
};

struct AppConfig {
    std::string text_encoder;
    std::string image_encoder;
    std::string bpe_vocab;
    std::string model_name;
    int width = 640;
    int height = 360;
    int fps = 30;
    int skip_frames = 2;
    bool no_normalize = false;
    std::vector<StreamConfig> streams;
};

struct CliOptions {
    std::string config_path = "config.json";
    bool full_screen = false;
    bool exit_button = false;
};

void printUsage(const char* argv0)
{
    std::cout
        << "Usage: " << argv0 << " [OPTIONS]\n"
        << "  --config PATH      Multi-stream JSON configuration (default: config.json)\n"
        << "  --full_screen      Show the grid in fullscreen mode\n"
        << "  --exit-btn         Show an Exit button in the top header\n"
        << "  -h, --help         Show this help\n";
}

CliOptions parseArgs(int argc, char** argv)
{
    CliOptions options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            std::exit(0);
        } else if (arg == "--config") {
            if (++i >= argc) {
                throw std::runtime_error("--config requires a path");
            }
            options.config_path = argv[i];
        } else if (arg == "--full_screen") {
            options.full_screen = true;
        } else if (arg == "--exit-btn") {
            options.exit_button = true;
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    return options;
}

fs::path absoluteFrom(const fs::path& base, const QString& value)
{
    fs::path path(value.toStdString());
    if (path.is_relative()) {
        path = base / path;
    }
    return fs::absolute(path).lexically_normal();
}

QString requiredString(const QJsonObject& object, const char* key)
{
    const QJsonValue value = object.value(QLatin1String(key));
    if (!value.isString() || value.toString().trimmed().isEmpty()) {
        throw std::runtime_error(std::string("config field '") + key + "' must be a string");
    }
    return value.toString();
}

AppConfig loadConfig(const fs::path& path)
{
    QFile file(QString::fromStdString(path.string()));
    if (!file.open(QIODevice::ReadOnly)) {
        throw std::runtime_error("cannot open config: " + path.string());
    }
    QJsonParseError parse_error;
    const QJsonDocument document = QJsonDocument::fromJson(file.readAll(), &parse_error);
    if (parse_error.error != QJsonParseError::NoError || !document.isObject()) {
        throw std::runtime_error("invalid config JSON: " + parse_error.errorString().toStdString());
    }

    const fs::path base = fs::absolute(path).parent_path();
    const QJsonObject root = document.object();
    const QJsonObject models = root.value("models").toObject();
    const QJsonObject capture = root.value("capture").toObject();
    AppConfig config;
    config.text_encoder = absoluteFrom(base, requiredString(models, "text_encoder")).string();
    config.image_encoder = absoluteFrom(base, requiredString(models, "image_encoder")).string();
    config.bpe_vocab = absoluteFrom(
        base, models.value("bpe_vocab").toString("assets/bpe_simple_vocab_16e6.txt.gz")).string();
    config.model_name = models.value("model_name").toString("ViT-L-14-quickgelu").toStdString();
    config.width = capture.value("width").toInt(640);
    config.height = capture.value("height").toInt(360);
    config.fps = capture.value("fps").toInt(30);
    config.skip_frames = capture.value("skip_frames").toInt(2);
    config.no_normalize = root.value("no_normalize").toBool(false);
    if (config.width <= 0 || config.height <= 0 || config.fps <= 0 || config.skip_frames < 0) {
        throw std::runtime_error("capture width/height/fps must be positive and skip_frames non-negative");
    }

    const QJsonArray streams = root.value("streams").toArray();
    if (streams.size() != 4 && streams.size() != 9) {
        throw std::runtime_error("config 'streams' must contain exactly 4 or 9 entries");
    }
    for (int index = 0; index < streams.size(); ++index) {
        if (!streams[index].isObject()) {
            throw std::runtime_error("each stream entry must be an object");
        }
        const QJsonObject object = streams[index].toObject();
        StreamConfig stream;
        stream.name = object.value("name").toString(QString("Stream %1").arg(index + 1));
        stream.pipeline = object.value("pipeline").toString().trimmed().toStdString();
        const QString source = object.value("source").toString().trimmed();
        if (stream.pipeline.empty() && source.isEmpty()) {
            throw std::runtime_error("stream must define either 'source' or 'pipeline'");
        }
        if (!source.isEmpty()) {
            const bool device = source.startsWith("/dev/video") ||
                                std::all_of(source.begin(), source.end(), [](QChar ch) {
                                    return ch.isDigit();
                                });
            const bool uri = source.contains("://");
            stream.source = (device || uri) ? source.toStdString()
                                             : absoluteFrom(base, source).string();
        }
        stream.threshold = static_cast<float>(object.value("threshold").toDouble(0.25));
        if (!std::isfinite(stream.threshold)) {
            throw std::runtime_error("stream threshold must be finite");
        }
        const QJsonArray texts = object.value("texts").toArray();
        for (const QJsonValue& text : texts) {
            if (text.isString() && !text.toString().trimmed().isEmpty()) {
                stream.texts.push_back(text.toString().trimmed());
            }
        }
        if (stream.texts.empty()) {
            throw std::runtime_error("each stream must contain at least one text definition");
        }
        config.streams.push_back(std::move(stream));
    }

    for (const std::string* model : {&config.text_encoder, &config.image_encoder, &config.bpe_vocab}) {
        if (!fs::is_regular_file(*model)) {
            throw std::runtime_error("model/resource file not found: " + *model);
        }
    }
    return config;
}

std::string gstQuote(const std::string& value)
{
    std::string quoted = "\"";
    for (char ch : value) {
        if (ch == '\\' || ch == '"') {
            quoted.push_back('\\');
        }
        quoted.push_back(ch);
    }
    quoted.push_back('"');
    return quoted;
}

std::string makeGstreamerPipeline(const StreamConfig& stream, const AppConfig& config)
{
    if (!stream.pipeline.empty()) {
        return stream.pipeline;
    }
    std::string device = stream.source;
    if (!device.empty() && std::all_of(device.begin(), device.end(), [](unsigned char ch) {
            return std::isdigit(ch);
        })) {
        device = "/dev/video" + device;
    }
    const std::string live_sink =
        " ! videoconvert ! videoscale ! video/x-raw,format=BGR,width=" +
        std::to_string(config.width) + ",height=" + std::to_string(config.height) +
        ",pixel-aspect-ratio=1/1 ! appsink drop=true max-buffers=2 sync=false";
    if (device.rfind("/dev/video", 0) == 0) {
        return "v4l2src device=" + gstQuote(device) +
               " ! video/x-raw,framerate=" + std::to_string(config.fps) + "/1" + live_sink;
    }
    if (stream.source.find("://") != std::string::npos) {
        return "uridecodebin uri=" + gstQuote(stream.source) + live_sink;
    }
    return "filesrc location=" + gstQuote(stream.source) +
           " ! decodebin ! videoconvert ! videoscale ! videorate"
           " ! video/x-raw,format=BGR,width=" + std::to_string(config.width) +
           ",height=" + std::to_string(config.height) +
           ",framerate=" + std::to_string(config.fps) +
           "/1,pixel-aspect-ratio=1/1"
           " ! appsink drop=true max-buffers=2 sync=true";
}

QImage matToImage(const cv::Mat& bgr)
{
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    return QImage(rgb.data, rgb.cols, rgb.rows, static_cast<int>(rgb.step),
                  QImage::Format_RGB888).copy();
}

std::vector<float> preprocessFrame(const cv::Mat& frame)
{
    const double scale = static_cast<double>(kImageSize) /
                         static_cast<double>(std::min(frame.cols, frame.rows));
    const int width = std::max(kImageSize, static_cast<int>(frame.cols * scale));
    const int height = std::max(kImageSize, static_cast<int>(frame.rows * scale));
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(width, height), 0.0, 0.0, cv::INTER_CUBIC);
    const cv::Mat crop = resized(cv::Rect(
        (width - kImageSize) / 2, (height - kImageSize) / 2, kImageSize, kImageSize));
    constexpr std::array<float, 3> mean = {0.485F, 0.456F, 0.406F};
    constexpr std::array<float, 3> deviation = {0.229F, 0.224F, 0.225F};
    const size_t plane = kImageSize * kImageSize;
    std::vector<float> tensor(plane * 3);
    for (int y = 0; y < kImageSize; ++y) {
        const cv::Vec3b* pixels = crop.ptr<cv::Vec3b>(y);
        for (int x = 0; x < kImageSize; ++x) {
            const size_t offset = static_cast<size_t>(y * kImageSize + x);
            tensor[offset] = (pixels[x][2] / 255.0F - mean[0]) / deviation[0];
            tensor[plane + offset] = (pixels[x][1] / 255.0F - mean[1]) / deviation[1];
            tensor[plane * 2 + offset] = (pixels[x][0] / 255.0F - mean[2]) / deviation[2];
        }
    }
    return tensor;
}

void normalizeRows(std::vector<float>& values, size_t rows)
{
    if (values.size() != rows * kEmbeddingSize) {
        throw std::runtime_error("unexpected embedding matrix shape");
    }
    for (size_t row = 0; row < rows; ++row) {
        float* data = values.data() + row * kEmbeddingSize;
        double squared = 0.0;
        for (int column = 0; column < kEmbeddingSize; ++column) {
            squared += static_cast<double>(data[column]) * data[column];
        }
        const float norm = static_cast<float>(std::sqrt(squared));
        if (norm > 1e-12F) {
            for (int column = 0; column < kEmbeddingSize; ++column) {
                data[column] /= norm;
            }
        }
    }
}

class StreamThread : public QThread {
    Q_OBJECT

public:
    StreamThread(int stream_index, QString name, std::string pipeline, int fps, QObject* parent = nullptr)
        : QThread(parent), stream_index_(stream_index), name_(std::move(name)),
          pipeline_(std::move(pipeline)), fps_(fps)
    {
    }

    ~StreamThread() override { stop(); }

    void stop()
    {
        stop_requested_.store(true);
        if (isRunning()) {
            wait();
        }
    }

signals:
    void frameReady(int stream_index, const cv::Mat& frame);
    void streamError(int stream_index, const QString& message);

protected:
    void run() override
    {
        const double frame_period = 1.0 / std::max(1, fps_);
        bool reported_open_error = false;
        while (!stop_requested_.load()) {
            cv::VideoCapture capture(pipeline_, cv::CAP_GSTREAMER);
            if (!capture.isOpened()) {
                if (!reported_open_error) {
                    reported_open_error = true;
                    emit streamError(stream_index_, QString("%1: GStreamer open failed").arg(name_));
                }
                QThread::msleep(1000);
                continue;
            }
            reported_open_error = false;
            while (!stop_requested_.load()) {
                const auto started = std::chrono::steady_clock::now();
                cv::Mat frame;
                if (!capture.read(frame) || frame.empty()) {
                    break;
                }
                emit frameReady(stream_index_, frame.clone());
                const double elapsed = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - started).count();
                if (frame_period > elapsed) {
                    QThread::usleep(static_cast<unsigned long>((frame_period - elapsed) * 1'000'000.0));
                }
            }
            capture.release();
            if (!stop_requested_.load()) {
                QThread::msleep(20);
            }
        }
    }

private:
    int stream_index_ = 0;
    QString name_;
    std::string pipeline_;
    int fps_ = 30;
    std::atomic_bool stop_requested_{false};
};

class TextEncoder {
public:
    TextEncoder(const std::string& model_path, const std::string& bpe_path)
        : tokenizer_(bpe_path), environment_(ORT_LOGGING_LEVEL_WARNING, "clip_multi_text")
    {
        options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session_ = std::make_unique<Ort::Session>(environment_, model_path.c_str(), options_);
        Ort::AllocatorWithDefaultOptions allocator;
        input_name_ = session_->GetInputNameAllocated(0, allocator).get();
        output_name_ = session_->GetOutputNameAllocated(0, allocator).get();
    }

    std::vector<float> encode(const std::vector<QString>& texts)
    {
        std::vector<float> embeddings;
        embeddings.reserve(texts.size() * kEmbeddingSize);
        const std::array<int64_t, 2> shape = {1, kContextLength};
        Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(
            OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
        const char* input_names[] = {input_name_.c_str()};
        const char* output_names[] = {output_name_.c_str()};
        for (const QString& text : texts) {
            std::vector<int64_t> tokens = tokenizer_.tokenize(text, kContextLength);
            Ort::Value input = Ort::Value::CreateTensor<int64_t>(
                memory, tokens.data(), tokens.size(), shape.data(), shape.size());
            Ort::RunOptions run_options;
            auto outputs = session_->Run(run_options, input_names, &input, 1, output_names, 1);
            if (outputs.empty() || outputs[0].GetTensorTypeAndShapeInfo().GetElementCount() != kEmbeddingSize) {
                throw std::runtime_error("unexpected text encoder output shape");
            }
            const float* data = outputs[0].GetTensorData<float>();
            embeddings.insert(embeddings.end(), data, data + kEmbeddingSize);
        }
        return embeddings;
    }

private:
    ClipTokenizer tokenizer_;
    Ort::Env environment_;
    Ort::SessionOptions options_;
    std::unique_ptr<Ort::Session> session_;
    std::string input_name_;
    std::string output_name_;
};

class ImageEncoderAsync {
public:
    struct Result {
        int stream_index = 0;
        uint64_t frame_id = 0;
        std::vector<float> embedding;
        std::string error;
    };

    explicit ImageEncoderAsync(const std::string& model_path)
    {
        dxrt::InferenceOption option;
        option.bufferCount = kMaxAsyncJobs;
        engine_ = std::make_unique<dxrt::InferenceEngine>(model_path, option);
        const auto inputs = engine_->GetInputs();
        if (inputs.empty() || inputs[0].type() != dxrt::DataType::FLOAT ||
            inputs[0].shape() != std::vector<int64_t>({1, 3, 224, 224})) {
            throw std::runtime_error("image encoder input must be FLOAT [1,3,224,224]");
        }
        engine_->RegisterCallback([this](dxrt::TensorPtrs& outputs, void* user_arg) {
            return onComplete(outputs, user_arg);
        });
    }

    ~ImageEncoderAsync() { shutdown(); }

    bool trySubmit(int stream_index, uint64_t frame_id, std::vector<float> input)
    {
        if (closing_.load()) {
            return false;
        }
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            if (in_flight_ >= kMaxAsyncJobs) {
                return false;
            }
            ++in_flight_;
        }
        auto* job = new Job{stream_index, frame_id, std::move(input)};
        try {
            const int request = engine_->RunAsync(job->input.data(), job);
            std::lock_guard<std::mutex> lock(state_mutex_);
            last_request_ = request;
            return true;
        } catch (...) {
            delete job;
            finishOne();
            throw;
        }
    }

    std::vector<Result> drain()
    {
        std::lock_guard<std::mutex> lock(results_mutex_);
        std::vector<Result> output;
        while (!results_.empty()) {
            output.push_back(std::move(results_.front()));
            results_.pop_front();
        }
        return output;
    }

    void shutdown()
    {
        if (closing_.exchange(true) || !engine_) {
            return;
        }
        int last = -1;
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            last = last_request_;
        }
        if (last >= 0) {
            try {
                engine_->Wait(last);
            } catch (...) {
            }
        }
        std::unique_lock<std::mutex> lock(state_mutex_);
        state_cv_.wait(lock, [this] { return in_flight_ == 0; });
        lock.unlock();
        engine_->RegisterCallback({});
        engine_.reset();
    }

private:
    struct Job {
        int stream_index;
        uint64_t frame_id;
        std::vector<float> input;
    };

    int onComplete(dxrt::TensorPtrs& outputs, void* user_arg)
    {
        std::unique_ptr<Job> job(static_cast<Job*>(user_arg));
        Result result;
        if (job) {
            result.stream_index = job->stream_index;
            result.frame_id = job->frame_id;
        }
        try {
            if (!job || outputs.empty() || !outputs[0] || outputs[0]->type() != dxrt::DataType::FLOAT) {
                throw std::runtime_error("invalid image encoder callback output");
            }
            const size_t count = static_cast<size_t>(outputs[0]->size_in_bytes()) / sizeof(float);
            if (count != kEmbeddingSize) {
                throw std::runtime_error("unexpected image encoder output shape");
            }
            const float* data = static_cast<const float*>(outputs[0]->data());
            result.embedding.assign(data, data + count);
        } catch (const std::exception& error) {
            result.error = error.what();
        }
        if (!closing_.load()) {
            std::lock_guard<std::mutex> lock(results_mutex_);
            results_.push_back(std::move(result));
        }
        finishOne();
        return 0;
    }

    void finishOne()
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        --in_flight_;
        state_cv_.notify_all();
    }

    std::unique_ptr<dxrt::InferenceEngine> engine_;
    std::atomic_bool closing_{false};
    std::mutex state_mutex_;
    std::condition_variable state_cv_;
    int in_flight_ = 0;
    int last_request_ = -1;
    std::mutex results_mutex_;
    std::deque<Result> results_;
};

class StreamTile : public QWidget {
public:
    explicit StreamTile(const StreamConfig& config, QWidget* parent = nullptr) : QWidget(parent)
    {
        setStyleSheet("QWidget#streamTile{background:#171c26;border:1px solid #394150;border-radius:8px;}");
        setObjectName("streamTile");
        auto* layout = new QVBoxLayout(this);
        layout->setContentsMargins(6, 6, 6, 6);
        layout->setSpacing(5);
        auto* title = new QLabel(config.name);
        title->setFixedHeight(22);
        title->setStyleSheet("color:#dbe4f0;font-size:14px;font-weight:700;border:none;");
        layout->addWidget(title);
        preview_ = new QLabel("Waiting for GStreamer input");
        preview_->setAlignment(Qt::AlignCenter);
        preview_->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
        preview_->setMinimumSize(160, 90);
        preview_->setStyleSheet("background:#05070a;color:#738096;border-radius:5px;");
        layout->addWidget(preview_, 1);
        output_ = new QLabel(" \n ");
        output_->setFixedHeight(48);
        output_->setAlignment(Qt::AlignLeft | Qt::AlignVCenter);
        output_->setStyleSheet(
            "color:#d1d8e4;background:#0d1118;border:1px solid #303846;"
            "border-radius:5px;padding:3px 8px;font-size:12px;font-weight:600;");
        layout->addWidget(output_);
    }

    void setFrame(const cv::Mat& frame)
    {
        latest_image_ = matToImage(frame);
        refreshPixmap();
    }

    void setMatches(const std::vector<std::pair<QString, float>>& matches)
    {
        QStringList lines;
        for (size_t i = 0; i < 2; ++i) {
            if (i < matches.size()) {
                const QString score = QString::number(matches[i].second, 'f', 3);
                const int available = std::max(40, output_->width() - 90);
                const QString text = output_->fontMetrics().elidedText(
                    matches[i].first, Qt::ElideRight, available);
                lines << QString("%1  (%2)").arg(text, score);
            } else {
                lines << " ";
            }
        }
        output_->setText(lines.join('\n'));
        const bool matched = !matches.empty();
        const int new_state = matched ? 1 : 0;
        if (output_state_ != new_state) {
            output_state_ = new_state;
            output_->setStyleSheet(QString(
                "color:%1;background:%2;border:1px solid %3;border-radius:5px;"
                "padding:3px 8px;font-size:12px;font-weight:600;")
                .arg(matched ? "#ecfdf5" : "#d1d8e4",
                     matched ? "#12352b" : "#0d1118",
                     matched ? "#22c55e" : "#303846"));
        }
    }

    void setError(const QString& message)
    {
        output_->setText(message + "\n ");
        if (output_state_ != 2) {
            output_state_ = 2;
            output_->setStyleSheet(
                "color:#fee2e2;background:#3b1519;border:1px solid #ef4444;"
                "border-radius:5px;padding:3px 8px;font-size:12px;font-weight:600;");
        }
    }

protected:
    void resizeEvent(QResizeEvent* event) override
    {
        QWidget::resizeEvent(event);
        refreshPixmap();
    }

private:
    void refreshPixmap()
    {
        if (!latest_image_.isNull()) {
            preview_->setPixmap(QPixmap::fromImage(latest_image_).scaled(
                preview_->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
        }
    }

    QLabel* preview_ = nullptr;
    QLabel* output_ = nullptr;
    QImage latest_image_;
    int output_state_ = 0;
};

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    MainWindow(AppConfig config, bool exit_button, QWidget* parent = nullptr)
        : QMainWindow(parent), config_(std::move(config))
    {
        setWindowTitle("DEEPX CLIP Multi-Stream");
        resize(1920, 1080);
        setupUi(exit_button);
        setupModels();
        setupStreams();
        result_timer_ = new QTimer(this);
        connect(result_timer_, &QTimer::timeout, this, &MainWindow::pollResults);
        result_timer_->start(20);

        auto* escape = new QShortcut(QKeySequence(Qt::Key_Escape), this);
        escape->setContext(Qt::ApplicationShortcut);
        connect(escape, &QShortcut::activated, this, &MainWindow::close);
        auto* quit = new QShortcut(QKeySequence(Qt::Key_Q), this);
        quit->setContext(Qt::ApplicationShortcut);
        connect(quit, &QShortcut::activated, this, &MainWindow::close);
    }

protected:
    void closeEvent(QCloseEvent* event) override
    {
        closing_ = true;
        if (result_timer_) {
            result_timer_->stop();
        }
        for (StreamThread* thread : stream_threads_) {
            disconnect(thread, nullptr, this, nullptr);
            thread->stop();
        }
        stream_threads_.clear();
        if (image_encoder_) {
            image_encoder_->shutdown();
            image_encoder_.reset();
        }
        event->accept();
    }

private:
    void setupUi(bool exit_button)
    {
        auto* root = new QWidget();
        root->setStyleSheet("background:#0b0f16;");
        setCentralWidget(root);
        auto* root_layout = new QVBoxLayout(root);
        root_layout->setContentsMargins(10, 8, 10, 10);
        root_layout->setSpacing(8);

        auto* header = new QHBoxLayout();
        auto* title = new QLabel(QString("CLIP Multi-Stream  ·  %1 Channels").arg(config_.streams.size()));
        title->setStyleSheet("color:#f8fafc;font-size:20px;font-weight:700;");
        header->addWidget(title);
        header->addStretch(1);
        if (exit_button) {
            auto* button = new QPushButton("Exit");
            button->setFixedSize(68, 34);
            button->setFocusPolicy(Qt::NoFocus);
            button->setStyleSheet(R"(
                QPushButton{color:#f4f4f5;background:#252a34;border:1px solid #525866;
                            border-radius:6px;font-size:13px;font-weight:700;}
                QPushButton:hover{color:#fff;background:#dc2626;border-color:#ef4444;}
                QPushButton:pressed{background:#991b1b;}
            )");
            connect(button, &QPushButton::clicked, this, &MainWindow::close);
            header->addWidget(button);
        }
        root_layout->addLayout(header);

        auto* grid = new QGridLayout();
        grid->setContentsMargins(0, 0, 0, 0);
        grid->setSpacing(8);
        const int dimension = config_.streams.size() == 4 ? 2 : 3;
        frame_counters_.resize(config_.streams.size(), 0);
        next_frame_ids_.resize(config_.streams.size(), 0);
        last_result_ids_.resize(config_.streams.size(), 0);
        has_result_.resize(config_.streams.size(), false);
        for (size_t index = 0; index < config_.streams.size(); ++index) {
            auto* tile = new StreamTile(config_.streams[index]);
            tiles_.push_back(tile);
            grid->addWidget(tile, static_cast<int>(index) / dimension,
                            static_cast<int>(index) % dimension);
        }
        for (int i = 0; i < dimension; ++i) {
            grid->setRowStretch(i, 1);
            grid->setColumnStretch(i, 1);
        }
        root_layout->addLayout(grid, 1);
    }

    void setupModels()
    {
        TextEncoder text_encoder(config_.text_encoder, config_.bpe_vocab);
        stream_features_.resize(config_.streams.size());
        for (size_t index = 0; index < config_.streams.size(); ++index) {
            stream_features_[index] = text_encoder.encode(config_.streams[index].texts);
            if (!config_.no_normalize) {
                normalizeRows(stream_features_[index], config_.streams[index].texts.size());
            }
        }
        image_encoder_ = std::make_unique<ImageEncoderAsync>(config_.image_encoder);
    }

    void setupStreams()
    {
        for (size_t index = 0; index < config_.streams.size(); ++index) {
            const std::string pipeline = makeGstreamerPipeline(config_.streams[index], config_);
            std::cout << "[Stream " << (index + 1) << "] " << pipeline << std::endl;
            auto* thread = new StreamThread(static_cast<int>(index), config_.streams[index].name,
                                            pipeline, config_.fps, this);
            connect(thread, &StreamThread::frameReady,
                    this, &MainWindow::onFrame, Qt::QueuedConnection);
            connect(thread, &StreamThread::streamError,
                    this, &MainWindow::onStreamError, Qt::QueuedConnection);
            stream_threads_.push_back(thread);
            thread->start();
        }
    }

    void onFrame(int stream_index, const cv::Mat& frame)
    {
        if (closing_ || stream_index < 0 ||
            stream_index >= static_cast<int>(config_.streams.size())) {
            return;
        }
        const size_t index = static_cast<size_t>(stream_index);
        tiles_[index]->setFrame(frame);
        const uint64_t frame_number = frame_counters_[index]++;
        if (frame_number % static_cast<uint64_t>(config_.skip_frames + 1) != 0) {
            return;
        }
        try {
            image_encoder_->trySubmit(stream_index, next_frame_ids_[index]++, preprocessFrame(frame));
        } catch (const std::exception& error) {
            tiles_[index]->setError(QString::fromStdString(error.what()));
        }
    }

    void onStreamError(int stream_index, const QString& message)
    {
        if (stream_index >= 0 && stream_index < static_cast<int>(tiles_.size())) {
            tiles_[static_cast<size_t>(stream_index)]->setError(message);
        }
    }

    void pollResults()
    {
        if (closing_ || !image_encoder_) {
            return;
        }
        for (auto& result : image_encoder_->drain()) {
            if (result.stream_index < 0 ||
                result.stream_index >= static_cast<int>(config_.streams.size())) {
                continue;
            }
            const size_t index = static_cast<size_t>(result.stream_index);
            if (!result.error.empty()) {
                tiles_[index]->setError(QString::fromStdString(result.error));
                continue;
            }
            if (has_result_[index] && result.frame_id <= last_result_ids_[index]) {
                continue;
            }
            has_result_[index] = true;
            last_result_ids_[index] = result.frame_id;
            if (!config_.no_normalize) {
                normalizeRows(result.embedding, 1);
            }
            updateMatches(index, result.embedding);
        }
    }

    void updateMatches(size_t stream_index, const std::vector<float>& image_embedding)
    {
        const StreamConfig& stream = config_.streams[stream_index];
        const std::vector<float>& text_features = stream_features_[stream_index];
        std::vector<std::pair<QString, float>> matches;
        for (size_t row = 0; row < stream.texts.size(); ++row) {
            const float score = std::inner_product(
                text_features.begin() + static_cast<std::ptrdiff_t>(row * kEmbeddingSize),
                text_features.begin() + static_cast<std::ptrdiff_t>((row + 1) * kEmbeddingSize),
                image_embedding.begin(), 0.0F);
            if (score >= stream.threshold) {
                matches.emplace_back(stream.texts[row], score);
            }
        }
        std::stable_sort(matches.begin(), matches.end(), [](const auto& left, const auto& right) {
            return left.second > right.second;
        });
        if (matches.size() > 2) {
            matches.resize(2);
        }
        tiles_[stream_index]->setMatches(matches);
    }

    AppConfig config_;
    bool closing_ = false;
    QTimer* result_timer_ = nullptr;
    std::vector<StreamTile*> tiles_;
    std::vector<StreamThread*> stream_threads_;
    std::unique_ptr<ImageEncoderAsync> image_encoder_;
    std::vector<std::vector<float>> stream_features_;
    std::vector<uint64_t> frame_counters_;
    std::vector<uint64_t> next_frame_ids_;
    std::vector<uint64_t> last_result_ids_;
    std::vector<bool> has_result_;
};

void notifyLauncherReady()
{
    const QByteArray ready_path = qgetenv("DX_LAUNCHER_READY_FILE");
    if (ready_path.isEmpty()) {
        return;
    }
    const QFileInfo info(QString::fromUtf8(ready_path));
    QDir().mkpath(info.absolutePath());
    QFile file(info.absoluteFilePath());
    if (file.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
        file.write("ready\n");
    }
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        const CliOptions cli = parseArgs(argc, argv);
        fs::path config_path(cli.config_path);
        if (!fs::exists(config_path)) {
#ifdef CLIP_MULTI_ROOT_DIR
            const fs::path rooted = fs::path(CLIP_MULTI_ROOT_DIR) / config_path;
            if (fs::exists(rooted)) {
                config_path = rooted;
            }
#endif
        }
        config_path = fs::absolute(config_path);
        AppConfig config = loadConfig(config_path);
        qRegisterMetaType<cv::Mat>("cv::Mat");
        QApplication app(argc, argv);
        MainWindow window(std::move(config), cli.exit_button);
        if (cli.full_screen) {
            window.showFullScreen();
        } else {
            window.showMaximized();
        }
        notifyLauncherReady();
        return app.exec();
    } catch (const Ort::Exception& error) {
        std::cerr << "ONNX Runtime error: " << error.what() << std::endl;
    } catch (const dxrt::Exception& error) {
        std::cerr << "DXRT error: " << error.what() << std::endl;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << std::endl;
    }
    printUsage(argv[0]);
    return 1;
}

#include "main.moc"
