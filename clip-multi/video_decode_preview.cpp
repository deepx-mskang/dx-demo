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
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

Q_DECLARE_METATYPE(cv::Mat)

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
};

void printUsage(const char* argv0)
{
    std::cout
        << "Usage: " << argv0 << " [OPTIONS]\n"
        << "  --config PATH      Stream JSON configuration (default: config.9.json)\n"
        << "  --full_screen      Show the preview grid in fullscreen mode\n"
        << "  --exit-btn         Show an Exit button in the header\n"
        << "  --no-gst-probe     Skip startup GStreamer decoder probe logs\n"
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
        } else if (arg == "--no-gst-probe") {
            options.gst_probe = false;
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

QImage matToImage(const cv::Mat& bgr)
{
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    return QImage(rgb.data, rgb.cols, rgb.rows, static_cast<int>(rgb.step),
                  QImage::Format_RGB888).copy();
}

class StreamThread : public QThread {
    Q_OBJECT

public:
    StreamThread(int index, QString name, std::string pipeline, int fps, bool gst_probe,
                 QObject* parent = nullptr)
        : QThread(parent), index_(index), name_(std::move(name)), pipeline_(std::move(pipeline)),
          fps_(fps), gst_probe_(gst_probe)
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
    void frameReady(int index, const cv::Mat& frame);
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
                emit frameReady(index_, frame.clone());
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
    std::atomic_bool stop_requested_{false};
};

class PreviewTile : public QWidget {
    Q_OBJECT

public:
    explicit PreviewTile(const QString& name, QWidget* parent = nullptr) : QWidget(parent)
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

    void setFrame(const cv::Mat& frame)
    {
        latest_image_ = matToImage(frame);
        refreshPixmap();
        ++total_frames_;
        ++interval_frames_;
        frame_width_ = frame.cols;
        frame_height_ = frame.rows;
        healthy_ = true;
    }

    void setStatus(const QString& text, bool error)
    {
        detail_ = text;
        if (error) {
            healthy_ = false;
            status_->setText("ERROR  ·  " + text);
            status_->setStyleSheet("color:#f87171;font-size:12px;font-weight:700;border:none;");
        } else if (total_frames_ == 0) {
            status_->setText(text);
            status_->setStyleSheet("color:#fbbf24;font-size:12px;font-weight:600;border:none;");
        }
    }

    bool updateStats(double seconds)
    {
        const double fps = seconds > 0.0 ? static_cast<double>(interval_frames_) / seconds : 0.0;
        const bool active = interval_frames_ > 0;
        if (active) {
            healthy_ = true;
            status_->setText(QString("DECODING  ·  %1 FPS  ·  %2 frames  ·  %3x%4")
                                 .arg(fps, 0, 'f', 1)
                                 .arg(total_frames_)
                                 .arg(frame_width_)
                                 .arg(frame_height_));
            status_->setStyleSheet("color:#4ade80;font-size:12px;font-weight:700;border:none;");
        } else if (total_frames_ > 0 && healthy_) {
            status_->setText("STALLED  ·  " + detail_);
            status_->setStyleSheet("color:#fbbf24;font-size:12px;font-weight:700;border:none;");
            healthy_ = false;
        }
        interval_frames_ = 0;
        return active;
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
                preview_->size(), Qt::KeepAspectRatio, Qt::FastTransformation));
        }
    }

    QLabel* name_ = nullptr;
    QLabel* preview_ = nullptr;
    QLabel* status_ = nullptr;
    QImage latest_image_;
    QString detail_ = "No frames received";
    uint64_t total_frames_ = 0;
    uint64_t interval_frames_ = 0;
    int frame_width_ = 0;
    int frame_height_ = 0;
    bool healthy_ = false;
};

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    MainWindow(AppConfig config, bool exit_button, bool gst_probe, QWidget* parent = nullptr)
        : QMainWindow(parent), config_(std::move(config)), gst_probe_(gst_probe)
    {
        setWindowTitle("9-Channel Video Decode Preview");
        resize(1920, 1080);
        setupUi(exit_button);
        setupStreams();

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
        stats_timer_->stop();
        for (StreamThread* thread : stream_threads_) {
            disconnect(thread, nullptr, this, nullptr);
            thread->stop();
        }
        stream_threads_.clear();
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
            auto* tile = new PreviewTile(config_.streams[index].name);
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
                                            pipeline, config_.fps, gst_probe_, this);
            connect(thread, &StreamThread::frameReady,
                    this, &MainWindow::onFrame, Qt::QueuedConnection);
            connect(thread, &StreamThread::statusChanged,
                    this, &MainWindow::onStatus, Qt::QueuedConnection);
            stream_threads_.push_back(thread);
            thread->start();
        }
    }

    void onFrame(int index, const cv::Mat& frame)
    {
        if (!closing_ && index >= 0 && index < static_cast<int>(tiles_.size())) {
            tiles_[static_cast<size_t>(index)]->setFrame(frame);
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
        for (PreviewTile* tile : tiles_) {
            active += tile->updateStats(seconds) ? 1 : 0;
        }
        title_->setText(QString("Video Decode Preview  ·  %1/%2 Channels Active  ·  inference disabled")
                            .arg(active)
                            .arg(tiles_.size()));
        title_->setStyleSheet(active == static_cast<int>(tiles_.size())
                                  ? "color:#4ade80;font-size:20px;font-weight:700;"
                                  : "color:#fbbf24;font-size:20px;font-weight:700;");
        if (stats_intervals_++ % 5 == 0 || active != last_active_count_) {
            std::cout << "[Summary] active channels=" << active << "/" << tiles_.size()
                      << std::endl;
        }
        last_active_count_ = active;
    }

    AppConfig config_;
    bool gst_probe_ = true;
    bool closing_ = false;
    QLabel* title_ = nullptr;
    QTimer* stats_timer_ = nullptr;
    QElapsedTimer stats_clock_;
    uint64_t stats_intervals_ = 0;
    int last_active_count_ = -1;
    std::vector<PreviewTile*> tiles_;
    std::vector<StreamThread*> stream_threads_;
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

        qRegisterMetaType<cv::Mat>("cv::Mat");
        QApplication app(argc, argv);
        MainWindow window(std::move(config), cli.exit_button, cli.gst_probe);
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
