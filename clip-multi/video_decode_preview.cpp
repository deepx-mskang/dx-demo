#include <gst/gst.h>
#include <opencv2/opencv.hpp>

#include <QApplication>
#include <QCloseEvent>
#include <QElapsedTimer>
#include <QFile>
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
#include <QResizeEvent>
#include <QShortcut>
#include <QThread>
#include <QTimer>
#include <QVBoxLayout>
#include <QWidget>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct StreamConfig {
    QString name;
    std::string source;
    std::string pipeline;
};

struct AppConfig {
    int width = 640;
    int height = 360;
    int fps = 30;
    std::vector<StreamConfig> streams;
};

struct CliOptions {
    std::string config_path = "config.9.json";
    bool full_screen = false;
    bool exit_button = false;
    bool gst_probe = true;
    int display_fps = 15;
    int workers = 0;
};

void printUsage(const char* argv0)
{
    std::cout
        << "Usage: " << argv0 << " [OPTIONS]\n"
        << "  --config PATH      Stream JSON configuration (default: config.9.json)\n"
        << "  --full_screen      Show the preview grid in fullscreen mode\n"
        << "  --exit-btn         Show an Exit button in the header\n"
        << "  --no-gst-probe     Skip startup GStreamer decoder probe logs\n"
        << "  --display-fps N    Limit UI refresh rate (default: 15)\n"
        << "  --workers N        Frame conversion workers (default: auto)\n"
        << "  -h, --help         Show this help\n";
}

int parsePositiveInt(const char* option, const char* value)
{
    try {
        size_t consumed = 0;
        const int result = std::stoi(value, &consumed);
        if (value[consumed] != '\0' || result <= 0) {
            throw std::runtime_error("invalid value");
        }
        return result;
    } catch (const std::exception&) {
        throw std::runtime_error(std::string(option) + " requires a positive integer");
    }
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
        } else if (arg == "--no-gst-probe") {
            options.gst_probe = false;
        } else if (arg == "--display-fps") {
            if (++i >= argc) {
                throw std::runtime_error("--display-fps requires a value");
            }
            options.display_fps = parsePositiveInt("--display-fps", argv[i]);
            if (options.display_fps > 60) {
                throw std::runtime_error("--display-fps must not exceed 60");
            }
        } else if (arg == "--workers") {
            if (++i >= argc) {
                throw std::runtime_error("--workers requires a value");
            }
            options.workers = parsePositiveInt("--workers", argv[i]);
            if (options.workers > 32) {
                throw std::runtime_error("--workers must not exceed 32");
            }
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
    const QJsonObject capture = root.value("capture").toObject();
    AppConfig config;
    config.width = capture.value("width").toInt(640);
    config.height = capture.value("height").toInt(360);
    config.fps = capture.value("fps").toInt(30);
    if (config.width <= 0 || config.height <= 0 || config.fps <= 0) {
        throw std::runtime_error("capture width, height, and fps must be positive");
    }

    const QJsonArray streams = root.value("streams").toArray();
    if (streams.isEmpty() || streams.size() > 16) {
        throw std::runtime_error("config 'streams' must contain between 1 and 16 entries");
    }
    for (int index = 0; index < streams.size(); ++index) {
        if (!streams[index].isObject()) {
            throw std::runtime_error("each stream entry must be an object");
        }
        const QJsonObject object = streams[index].toObject();
        StreamConfig stream;
        stream.name = object.value("name").toString(QString("Channel %1").arg(index + 1));
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
        config.streams.push_back(std::move(stream));
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
           " ! decodebin ! videoconvert ! videoscale"
           " ! video/x-raw,format=BGR,width=" + std::to_string(config.width) +
           ",height=" + std::to_string(config.height) +
           ",pixel-aspect-ratio=1/1"
           " ! appsink drop=true max-buffers=2 sync=false";
}

void initGstreamer()
{
    static std::once_flag once;
    std::call_once(once, []() {
        int argc = 0;
        char** argv = nullptr;
        gst_init(&argc, &argv);
        std::cout << "[GStreamer] version " << gst_version_string() << std::endl;
    });
}

bool containsCaseInsensitive(const std::string& text, const std::string& needle)
{
    return std::search(text.begin(), text.end(), needle.begin(), needle.end(),
                       [](unsigned char left, unsigned char right) {
                           return std::tolower(left) == std::tolower(right);
                       }) != text.end();
}

bool looksLikeHardwareCodecFactory(const std::string& factory_name, const std::string& klass)
{
    static const std::array<const char*, 13> keywords = {
        "vaapi", "nvdec", "nvv4l2", "v4l2", "mfx", "qsv", "d3d11", "amf",
        "videotoolbox", "cuda", "mpp", "nvh", "nvcodec"};
    return containsCaseInsensitive(klass, "Hardware") ||
           std::any_of(keywords.begin(), keywords.end(), [&](const char* keyword) {
               return containsCaseInsensitive(factory_name, keyword) ||
                      containsCaseInsensitive(klass, keyword);
           });
}

void logGstreamerDecoderProbe(int stream_index, const QString& stream_name,
                              const std::string& pipeline)
{
    initGstreamer();
    GError* error = nullptr;
    GstElement* root = gst_parse_launch(pipeline.c_str(), &error);
    if (!root) {
        std::cout << "[Channel " << (stream_index + 1) << "] probe parse failed: "
                  << (error && error->message ? error->message : "unknown error") << std::endl;
        if (error) {
            g_error_free(error);
        }
        return;
    }
    if (error) {
        std::cout << "[Channel " << (stream_index + 1) << "] probe parse warning: "
                  << error->message << std::endl;
        g_error_free(error);
    }

    gst_element_set_state(root, GST_STATE_PAUSED);
    GstState current = GST_STATE_NULL;
    GstState pending = GST_STATE_NULL;
    gst_element_get_state(root, &current, &pending, 3 * GST_SECOND);

    bool decoder_found = false;
    bool hardware_decoder_found = false;
    GstIterator* iterator = gst_bin_iterate_recurse(GST_BIN(root));
    GValue item = G_VALUE_INIT;
    while (gst_iterator_next(iterator, &item) == GST_ITERATOR_OK) {
        GstElement* element = GST_ELEMENT(g_value_get_object(&item));
        GstElementFactory* factory = gst_element_get_factory(element);
        if (factory) {
            const gchar* factory_name = gst_plugin_feature_get_name(GST_PLUGIN_FEATURE(factory));
            const gchar* klass = gst_element_factory_get_metadata(factory, GST_ELEMENT_METADATA_KLASS);
            const std::string klass_text = klass ? klass : "";
            if (containsCaseInsensitive(klass_text, "Decoder") &&
                containsCaseInsensitive(klass_text, "Video")) {
                decoder_found = true;
                const std::string factory_text = factory_name ? factory_name : "";
                const bool hardware = looksLikeHardwareCodecFactory(factory_text, klass_text);
                hardware_decoder_found = hardware_decoder_found || hardware;
                std::cout << "[Channel " << (stream_index + 1) << "] decoder: name=\""
                          << stream_name.toStdString() << "\" factory=\"" << factory_text
                          << "\" hw=" << (hardware ? "yes" : "no") << std::endl;
            }
        }
        g_value_reset(&item);
    }
    g_value_unset(&item);
    gst_iterator_free(iterator);

    std::cout << "[Channel " << (stream_index + 1) << "] decoder probe: "
              << (!decoder_found ? "no decoder discovered"
                  : hardware_decoder_found ? "hardware-looking decoder detected"
                                           : "software-looking decoder detected")
              << std::endl;
    gst_element_set_state(root, GST_STATE_NULL);
    gst_object_unref(root);
}

struct ChannelState {
    std::mutex image_mutex;
    QImage latest_image;
    uint64_t image_sequence = 0;
    std::atomic<uint64_t> decoded_total{0};
    std::atomic<uint64_t> decoded_interval{0};
    std::atomic<int> source_width{0};
    std::atomic<int> source_height{0};
    std::atomic<int> target_width{640};
    std::atomic<int> target_height{360};
};

class FrameProcessorPool {
public:
    FrameProcessorPool(std::vector<std::shared_ptr<ChannelState>> states, int worker_count)
        : states_(std::move(states)), slots_(states_.size())
    {
        workers_.reserve(static_cast<size_t>(worker_count));
        for (int index = 0; index < worker_count; ++index) {
            workers_.emplace_back(&FrameProcessorPool::workerLoop, this);
        }
        std::cout << "[Preview] frame conversion workers=" << worker_count << std::endl;
    }

    ~FrameProcessorPool() { shutdown(); }

    FrameProcessorPool(const FrameProcessorPool&) = delete;
    FrameProcessorPool& operator=(const FrameProcessorPool&) = delete;

    void publish(size_t channel, cv::Mat frame)
    {
        if (channel >= slots_.size()) {
            return;
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stopping_) {
                return;
            }
            WorkSlot& slot = slots_[channel];
            slot.frame = std::move(frame);
            ++slot.sequence;
            if (!slot.scheduled) {
                slot.scheduled = true;
                pending_channels_.push_back(channel);
            }
        }
        condition_.notify_one();
    }

    void shutdown()
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stopping_) {
                return;
            }
            stopping_ = true;
            pending_channels_.clear();
        }
        condition_.notify_all();
        for (std::thread& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
        workers_.clear();
    }

private:
    struct WorkSlot {
        cv::Mat frame;
        uint64_t sequence = 0;
        bool scheduled = false;
    };

    void workerLoop()
    {
        while (true) {
            size_t channel = 0;
            uint64_t sequence = 0;
            cv::Mat frame;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                condition_.wait(lock, [this]() {
                    return stopping_ || !pending_channels_.empty();
                });
                if (stopping_) {
                    return;
                }
                channel = pending_channels_.front();
                pending_channels_.pop_front();
                WorkSlot& slot = slots_[channel];
                frame = slot.frame;
                sequence = slot.sequence;
            }

            processFrame(channel, sequence, frame);

            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (stopping_) {
                    return;
                }
                WorkSlot& slot = slots_[channel];
                if (slot.sequence != sequence) {
                    pending_channels_.push_back(channel);
                    condition_.notify_one();
                } else {
                    slot.scheduled = false;
                }
            }
        }
    }

    void processFrame(size_t channel, uint64_t sequence, const cv::Mat& frame)
    {
        if (frame.empty()) {
            return;
        }
        const std::shared_ptr<ChannelState>& state = states_[channel];
        const int target_width = std::max(1, state->target_width.load(std::memory_order_relaxed));
        const int target_height = std::max(1, state->target_height.load(std::memory_order_relaxed));
        const double scale = std::min(static_cast<double>(target_width) / frame.cols,
                                      static_cast<double>(target_height) / frame.rows);
        const int output_width = std::max(1, static_cast<int>(std::round(frame.cols * scale)));
        const int output_height = std::max(1, static_cast<int>(std::round(frame.rows * scale)));

        cv::Mat resized;
        if (output_width == frame.cols && output_height == frame.rows) {
            resized = frame;
        } else {
            cv::resize(frame, resized, cv::Size(output_width, output_height), 0.0, 0.0,
                       scale < 1.0 ? cv::INTER_AREA : cv::INTER_LINEAR);
        }
        cv::Mat rgb;
        cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
        QImage image(rgb.data, rgb.cols, rgb.rows, static_cast<int>(rgb.step),
                     QImage::Format_RGB888);
        image = image.copy();

        std::lock_guard<std::mutex> lock(state->image_mutex);
        state->latest_image = std::move(image);
        state->image_sequence = sequence;
    }

    std::vector<std::shared_ptr<ChannelState>> states_;
    std::vector<WorkSlot> slots_;
    std::deque<size_t> pending_channels_;
    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable condition_;
    bool stopping_ = false;
};

class StreamThread : public QThread {
    Q_OBJECT

public:
    StreamThread(int index, QString name, std::string pipeline, int fps, bool gst_probe,
                 std::shared_ptr<ChannelState> state, FrameProcessorPool* processor_pool,
                 QObject* parent = nullptr)
        : QThread(parent), index_(index), name_(std::move(name)), pipeline_(std::move(pipeline)),
          fps_(fps), gst_probe_(gst_probe), state_(std::move(state)),
          processor_pool_(processor_pool)
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
    void statusChanged(int index, const QString& status, bool error);

protected:
    void run() override
    {
        const double frame_period = 1.0 / std::max(1, fps_);
        bool reported_open_error = false;
        uint64_t loop_count = 0;
        while (!stop_requested_.load()) {
            if (!probe_logged_ && gst_probe_) {
                probe_logged_ = true;
                logGstreamerDecoderProbe(index_, name_, pipeline_);
            }

            emit statusChanged(index_, loop_count == 0 ? "Opening decoder..." : "Restarting video...", false);
            cv::VideoCapture capture(pipeline_, cv::CAP_GSTREAMER);
            if (!capture.isOpened()) {
                if (!reported_open_error) {
                    reported_open_error = true;
                    emit statusChanged(index_, "GStreamer open failed; retrying", true);
                }
                QThread::msleep(1000);
                continue;
            }

            reported_open_error = false;
            std::cout << "[Channel " << (index_ + 1) << "] capture backend="
                      << capture.getBackendName() << " name=\"" << name_.toStdString() << "\""
                      << std::endl;
            emit statusChanged(index_, "Decoder opened; waiting for frames", false);

            bool received_frame = false;
            while (!stop_requested_.load()) {
                const auto started = std::chrono::steady_clock::now();
                cv::Mat frame;
                if (!capture.read(frame) || frame.empty()) {
                    break;
                }
                received_frame = true;
                state_->decoded_total.fetch_add(1, std::memory_order_relaxed);
                state_->decoded_interval.fetch_add(1, std::memory_order_relaxed);
                state_->source_width.store(frame.cols, std::memory_order_relaxed);
                state_->source_height.store(frame.rows, std::memory_order_relaxed);
                processor_pool_->publish(static_cast<size_t>(index_), std::move(frame));
                const double elapsed = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - started).count();
                if (frame_period > elapsed) {
                    QThread::usleep(
                        static_cast<unsigned long>((frame_period - elapsed) * 1'000'000.0));
                }
            }
            capture.release();
            if (!stop_requested_.load()) {
                if (!received_frame) {
                    emit statusChanged(index_, "No frame decoded; retrying", true);
                    QThread::msleep(500);
                } else {
                    ++loop_count;
                }
            }
        }
    }

private:
    int index_ = 0;
    QString name_;
    std::string pipeline_;
    int fps_ = 30;
    bool gst_probe_ = true;
    bool probe_logged_ = false;
    std::shared_ptr<ChannelState> state_;
    FrameProcessorPool* processor_pool_ = nullptr;
    std::atomic_bool stop_requested_{false};
};

class PreviewTile : public QWidget {
    Q_OBJECT

public:
    PreviewTile(const QString& name, std::shared_ptr<ChannelState> state,
                QWidget* parent = nullptr)
        : QWidget(parent), state_(std::move(state))
    {
        setStyleSheet("background:#111827;border:1px solid #334155;border-radius:6px;");
        auto* layout = new QVBoxLayout(this);
        layout->setContentsMargins(6, 6, 6, 6);
        layout->setSpacing(4);

        name_ = new QLabel(name);
        name_->setStyleSheet("color:#f8fafc;font-size:14px;font-weight:700;border:none;");
        layout->addWidget(name_);

        preview_ = new QLabel("Waiting for GStreamer input");
        preview_->setAlignment(Qt::AlignCenter);
        preview_->setMinimumSize(160, 90);
        preview_->setStyleSheet("color:#64748b;background:#05080d;border:none;");
        layout->addWidget(preview_, 1);

        status_ = new QLabel("STARTING");
        status_->setStyleSheet("color:#fbbf24;font-size:12px;font-weight:600;border:none;");
        layout->addWidget(status_);
    }

    void setImage(const QImage& image)
    {
        preview_->setPixmap(QPixmap::fromImage(image));
        ++displayed_total_;
        ++displayed_interval_;
    }

    void setStatus(const QString& text, bool error)
    {
        detail_ = text;
        if (error) {
            healthy_ = false;
            status_->setText("ERROR  ·  " + text);
            status_->setStyleSheet("color:#f87171;font-size:12px;font-weight:700;border:none;");
        } else if (displayed_total_ == 0) {
            status_->setText(text);
            status_->setStyleSheet("color:#fbbf24;font-size:12px;font-weight:600;border:none;");
        }
    }

    bool updateStats(double seconds, uint64_t decoded_frames, uint64_t decoded_total,
                     int source_width, int source_height, uint64_t& displayed_frames)
    {
        displayed_frames = displayed_interval_;
        const double decode_fps = seconds > 0.0
                                      ? static_cast<double>(decoded_frames) / seconds
                                      : 0.0;
        const double display_fps = seconds > 0.0
                                       ? static_cast<double>(displayed_interval_) / seconds
                                       : 0.0;
        const bool active = decoded_frames > 0;
        if (active) {
            healthy_ = true;
            status_->setText(QString("DEC %1 FPS  ·  DISP %2 FPS  ·  %3 frames  ·  %4x%5")
                                 .arg(decode_fps, 0, 'f', 1)
                                 .arg(display_fps, 0, 'f', 1)
                                 .arg(decoded_total)
                                 .arg(source_width)
                                 .arg(source_height));
            status_->setStyleSheet("color:#4ade80;font-size:12px;font-weight:700;border:none;");
        } else if (decoded_total > 0 && healthy_) {
            status_->setText("STALLED  ·  " + detail_);
            status_->setStyleSheet("color:#fbbf24;font-size:12px;font-weight:700;border:none;");
            healthy_ = false;
        }
        displayed_interval_ = 0;
        return active;
    }

protected:
    void resizeEvent(QResizeEvent* event) override
    {
        QWidget::resizeEvent(event);
        state_->target_width.store(std::max(1, preview_->width()), std::memory_order_relaxed);
        state_->target_height.store(std::max(1, preview_->height()), std::memory_order_relaxed);
    }

private:
    QLabel* name_ = nullptr;
    QLabel* preview_ = nullptr;
    QLabel* status_ = nullptr;
    std::shared_ptr<ChannelState> state_;
    QString detail_ = "No frames received";
    uint64_t displayed_total_ = 0;
    uint64_t displayed_interval_ = 0;
    bool healthy_ = false;
};

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    MainWindow(AppConfig config, bool exit_button, bool gst_probe, int display_fps,
               int requested_workers, QWidget* parent = nullptr)
        : QMainWindow(parent), config_(std::move(config)), gst_probe_(gst_probe),
          display_fps_(display_fps)
    {
        setWindowTitle("9-Channel Video Decode Preview");
        resize(1920, 1080);
        setupUi(exit_button);
        const unsigned int hardware_threads = std::max(1U, std::thread::hardware_concurrency());
        const int automatic_workers = std::max(
            1, std::min(static_cast<int>(config_.streams.size()),
                        static_cast<int>(std::max(1U, hardware_threads / 2))));
        worker_count_ = requested_workers > 0
                            ? std::min(requested_workers, static_cast<int>(config_.streams.size()))
                            : automatic_workers;
        processor_pool_ = std::make_unique<FrameProcessorPool>(channel_states_, worker_count_);
        setupStreams();

        display_timer_ = new QTimer(this);
        display_timer_->setTimerType(Qt::PreciseTimer);
        connect(display_timer_, &QTimer::timeout, this, &MainWindow::refreshFrames);
        display_timer_->start(
            std::max(1, static_cast<int>(std::round(1000.0 / display_fps_))));

        stats_clock_.start();
        stats_timer_ = new QTimer(this);
        connect(stats_timer_, &QTimer::timeout, this, &MainWindow::updateStats);
        stats_timer_->start(1000);

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
        display_timer_->stop();
        stats_timer_->stop();
        for (StreamThread* thread : stream_threads_) {
            disconnect(thread, nullptr, this, nullptr);
            thread->stop();
        }
        stream_threads_.clear();
        if (processor_pool_) {
            processor_pool_->shutdown();
            processor_pool_.reset();
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
        title_ = new QLabel(QString("Video Decode Preview  ·  %1 Channels  ·  inference disabled")
                                .arg(config_.streams.size()));
        title_->setStyleSheet("color:#f8fafc;font-size:20px;font-weight:700;");
        header->addWidget(title_);
        header->addStretch(1);
        if (exit_button) {
            auto* button = new QPushButton("Exit");
            button->setFixedSize(68, 34);
            button->setFocusPolicy(Qt::NoFocus);
            connect(button, &QPushButton::clicked, this, &MainWindow::close);
            header->addWidget(button);
        }
        root_layout->addLayout(header);

        auto* grid = new QGridLayout();
        grid->setContentsMargins(0, 0, 0, 0);
        grid->setSpacing(8);
        const int columns = static_cast<int>(std::ceil(std::sqrt(config_.streams.size())));
        for (size_t index = 0; index < config_.streams.size(); ++index) {
            auto state = std::make_shared<ChannelState>();
            state->target_width.store(config_.width, std::memory_order_relaxed);
            state->target_height.store(config_.height, std::memory_order_relaxed);
            channel_states_.push_back(state);
            auto* tile = new PreviewTile(config_.streams[index].name, state);
            tiles_.push_back(tile);
            grid->addWidget(tile, static_cast<int>(index) / columns,
                            static_cast<int>(index) % columns);
        }
        for (int i = 0; i < columns; ++i) {
            grid->setRowStretch(i, 1);
            grid->setColumnStretch(i, 1);
        }
        root_layout->addLayout(grid, 1);
    }

    void setupStreams()
    {
        initGstreamer();
        for (size_t index = 0; index < config_.streams.size(); ++index) {
            const std::string pipeline = makeGstreamerPipeline(config_.streams[index], config_);
            std::cout << "[Channel " << (index + 1) << "] pipeline: " << pipeline << std::endl;
            auto* thread = new StreamThread(static_cast<int>(index), config_.streams[index].name,
                                            pipeline, config_.fps, gst_probe_,
                                            channel_states_[index], processor_pool_.get(), this);
            connect(thread, &StreamThread::statusChanged,
                    this, &MainWindow::onStatus, Qt::QueuedConnection);
            stream_threads_.push_back(thread);
            thread->start();
        }
    }

    void refreshFrames()
    {
        if (closing_) {
            return;
        }
        for (size_t index = 0; index < channel_states_.size(); ++index) {
            QImage image;
            uint64_t sequence = 0;
            {
                const std::shared_ptr<ChannelState>& state = channel_states_[index];
                std::lock_guard<std::mutex> lock(state->image_mutex);
                sequence = state->image_sequence;
                if (sequence == 0 || sequence == displayed_sequences_[index]) {
                    continue;
                }
                image = state->latest_image;
            }
            displayed_sequences_[index] = sequence;
            tiles_[index]->setImage(image);
        }
    }

    void onStatus(int index, const QString& status, bool error)
    {
        if (!closing_ && index >= 0 && index < static_cast<int>(tiles_.size())) {
            tiles_[static_cast<size_t>(index)]->setStatus(status, error);
        }
    }

    void updateStats()
    {
        const double seconds = static_cast<double>(stats_clock_.restart()) / 1000.0;
        int active = 0;
        uint64_t decoded_frames = 0;
        uint64_t displayed_frames = 0;
        for (size_t index = 0; index < tiles_.size(); ++index) {
            const std::shared_ptr<ChannelState>& state = channel_states_[index];
            const uint64_t interval = state->decoded_interval.exchange(0, std::memory_order_relaxed);
            uint64_t displayed_interval = 0;
            decoded_frames += interval;
            active += tiles_[index]->updateStats(
                          seconds, interval,
                          state->decoded_total.load(std::memory_order_relaxed),
                          state->source_width.load(std::memory_order_relaxed),
                          state->source_height.load(std::memory_order_relaxed),
                          displayed_interval)
                          ? 1
                          : 0;
            displayed_frames += displayed_interval;
        }
        title_->setText(QString("Video Decode Preview  ·  %1/%2 Active  ·  %3 workers  ·  UI %4 FPS")
                            .arg(active)
                            .arg(tiles_.size())
                            .arg(worker_count_)
                            .arg(display_fps_));
        title_->setStyleSheet(active == static_cast<int>(tiles_.size())
                                  ? "color:#4ade80;font-size:20px;font-weight:700;"
                                  : "color:#fbbf24;font-size:20px;font-weight:700;");
        if (stats_intervals_++ % 5 == 0 || active != last_active_count_) {
            std::cout << "[Summary] active channels=" << active << "/" << tiles_.size()
                      << " aggregate decode FPS="
                      << (seconds > 0.0 ? static_cast<double>(decoded_frames) / seconds : 0.0)
                      << " display FPS="
                      << (seconds > 0.0 ? static_cast<double>(displayed_frames) / seconds : 0.0)
                      << std::endl;
        }
        last_active_count_ = active;
    }

    AppConfig config_;
    bool gst_probe_ = true;
    bool closing_ = false;
    int display_fps_ = 15;
    int worker_count_ = 1;
    QLabel* title_ = nullptr;
    QTimer* display_timer_ = nullptr;
    QTimer* stats_timer_ = nullptr;
    QElapsedTimer stats_clock_;
    uint64_t stats_intervals_ = 0;
    int last_active_count_ = -1;
    std::vector<std::shared_ptr<ChannelState>> channel_states_;
    std::vector<uint64_t> displayed_sequences_ =
        std::vector<uint64_t>(config_.streams.size(), 0);
    std::vector<PreviewTile*> tiles_;
    std::vector<StreamThread*> stream_threads_;
    std::unique_ptr<FrameProcessorPool> processor_pool_;
};

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

        QApplication app(argc, argv);
        MainWindow window(std::move(config), cli.exit_button, cli.gst_probe,
                          cli.display_fps, cli.workers);
        if (cli.full_screen) {
            window.showFullScreen();
        } else {
            window.showMaximized();
        }
        return app.exec();
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << std::endl;
    }
    printUsage(argv[0]);
    return 1;
}

#include "video_decode_preview.moc"
