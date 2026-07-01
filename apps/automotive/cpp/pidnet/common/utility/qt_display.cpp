#include "qt_display.hpp"

#include <QApplication>
#include <QCloseEvent>
#include <QGuiApplication>
#include <QImage>
#include <QKeyEvent>
#include <QMouseEvent>
#include <QPaintEvent>
#include <QPainter>
#include <QScreen>
#include <QWidget>

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>

#include "common/utility/run_dir.hpp"

namespace dxapp {
namespace {

constexpr int kExitButtonWidth = 32;
constexpr int kExitButtonHeight = 28;
constexpr int kExitButtonMargin = 14;

struct DisplayFpsState {
    std::mutex mutex;
    double fps = 0.0;
    bool external = false;
    std::chrono::steady_clock::time_point last_ts = std::chrono::steady_clock::now();
};

DisplayFpsState& fpsState() {
    static DisplayFpsState instance;
    return instance;
}

void resetDisplayFpsState() {
    auto& s = fpsState();
    std::lock_guard<std::mutex> lock(s.mutex);
    s.fps = 0.0;
    s.external = false;
    s.last_ts = std::chrono::steady_clock::now();
}

void updateFallbackDisplayFps() {
    auto& s = fpsState();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (s.external) {
        return;
    }

    const auto now = std::chrono::steady_clock::now();
    const double dt = std::chrono::duration<double>(now - s.last_ts).count();
    if (dt > 1e-6) {
        const double instant = 1.0 / dt;
        s.fps = (s.fps <= 0.0) ? instant : (s.fps * 0.85 + instant * 0.15);
    }
    s.last_ts = now;
}

double currentDisplayFps() {
    auto& s = fpsState();
    std::lock_guard<std::mutex> lock(s.mutex);
    return s.fps;
}

class QtOutputWidget : public QWidget {
public:
    explicit QtOutputWidget(QWidget* parent = nullptr) : QWidget(parent) {
        setAttribute(Qt::WA_OpaquePaintEvent, true);
        setWindowTitle("Output");
    }

    void setShowExitButton(bool enabled) { show_exit_button_ = enabled; }

    void setQuitHandler(std::function<void()> handler) { quit_handler_ = std::move(handler); }

    void updateFrame(const cv::Mat& bgr, cv::Mat& rgb_cache) {
        if (bgr.empty()) {
            return;
        }
        if (rgb_cache.rows != bgr.rows || rgb_cache.cols != bgr.cols || rgb_cache.type() != CV_8UC3) {
            rgb_cache.create(bgr.rows, bgr.cols, CV_8UC3);
        }
        cv::cvtColor(bgr, rgb_cache, cv::COLOR_BGR2RGB);
        frame_image_ = QImage(
            rgb_cache.data,
            rgb_cache.cols,
            rgb_cache.rows,
            static_cast<int>(rgb_cache.step),
            QImage::Format_RGB888);
        updateFallbackDisplayFps();
        update();
    }

protected:
    void paintEvent(QPaintEvent*) override {
        QPainter painter(this);
        painter.fillRect(rect(), QColor(0, 0, 0));

        if (!frame_image_.isNull()) {
            const QRectF image_rect = imageDrawRect();
            painter.setRenderHint(QPainter::SmoothPixmapTransform, true);
            painter.drawImage(image_rect, frame_image_);
        }

        painter.setRenderHint(QPainter::Antialiasing, true);
        drawFpsOverlay(painter);
        drawExitButton(painter);
    }

    void keyPressEvent(QKeyEvent* event) override {
        if (event->key() == Qt::Key_Escape || event->key() == Qt::Key_Q) {
            requestQuit();
            return;
        }
        QWidget::keyPressEvent(event);
    }

    void mousePressEvent(QMouseEvent* event) override {
        if (show_exit_button_ &&
            event->button() == Qt::LeftButton &&
            exitButtonRect().contains(event->pos())) {
            requestQuit();
            return;
        }
        QWidget::mousePressEvent(event);
    }

    void closeEvent(QCloseEvent* event) override {
        requestQuit();
        QWidget::closeEvent(event);
    }

private:
    QRectF imageDrawRect() const {
        if (frame_image_.isNull()) {
            return QRectF();
        }
        QSize target = frame_image_.size();
        target.scale(size(), Qt::KeepAspectRatio);
        const QPointF top_left(
            (width() - target.width()) * 0.5,
            (height() - target.height()) * 0.5);
        return QRectF(top_left, QSizeF(target));
    }

    QRectF exitButtonRect() const {
        const qreal x = std::max<qreal>(0.0, width() - kExitButtonWidth - kExitButtonMargin);
        return QRectF(x, kExitButtonMargin, kExitButtonWidth, kExitButtonHeight);
    }

    void drawExitButton(QPainter& painter) const {
        if (!show_exit_button_) {
            return;
        }

        const QRectF button_rect = exitButtonRect();
        painter.save();
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

    void drawFpsOverlay(QPainter& painter) const {
        const QString text = QString("%1 FPS").arg(currentDisplayFps(), 0, 'f', 1);

        QFont font = painter.font();
        font.setFamily("DejaVu Sans");
        font.setPixelSize(23);
        font.setBold(true);
        painter.setFont(font);

        const QFontMetrics fm(font);
        const int pad_x = 16;
        const int pad_y = 10;
        const int text_w = fm.horizontalAdvance(text);
        const int text_h = fm.height();
        const int box_w = text_w + pad_x * 2;
        const int box_h = text_h + pad_y * 2;

        const qreal x = kExitButtonMargin;
        const qreal y = kExitButtonMargin;
        const QRectF box(x, y, box_w, box_h);

        painter.save();
        painter.setPen(QPen(QColor(96, 102, 110), 1));
        painter.setBrush(QColor(38, 40, 44, 210));
        painter.drawRoundedRect(box, 8, 8);
        painter.setPen(QColor(236, 240, 245));
        painter.drawText(
            QRectF(box.left() + pad_x, box.top() + pad_y, text_w, text_h),
            Qt::AlignLeft | Qt::AlignVCenter,
            text);
        painter.restore();
    }

    void requestQuit() {
        if (quit_handler_) {
            quit_handler_();
        }
    }

    QImage frame_image_;
    bool show_exit_button_ = false;
    std::function<void()> quit_handler_;
};

struct QtDisplayState {
    std::vector<std::string> argv_storage;
    std::vector<char*> argv_ptrs;
    int qt_argc = 0;
    std::unique_ptr<QApplication> app;
    std::unique_ptr<QtOutputWidget> window;
    cv::Mat rgb_buffer;
    bool full_screen = false;
    bool show_exit_button = false;
    bool window_sized = false;
    bool closed = false;
    bool launcher_ready_notified = false;
    std::atomic<bool> exit_clicked{false};
};

QtDisplayState& state() {
    static QtDisplayState instance;
    return instance;
}

void storeArgv(int argc, char** argv) {
    auto& s = state();
    s.argv_storage.clear();
    s.argv_ptrs.clear();
    if (argc <= 0 || argv == nullptr) {
        s.argv_storage.emplace_back("pidnet");
        s.argv_ptrs.push_back(s.argv_storage.back().data());
        return;
    }
    for (int i = 0; i < argc; ++i) {
        if (argv[i] != nullptr) {
            s.argv_storage.emplace_back(argv[i]);
        }
    }
    if (s.argv_storage.empty()) {
        s.argv_storage.emplace_back("pidnet");
    }
    for (auto& arg : s.argv_storage) {
        s.argv_ptrs.push_back(arg.data());
    }
    s.qt_argc = static_cast<int>(s.argv_ptrs.size());
}

void ensureApplication(int argc, char** argv) {
    auto& s = state();
    if (s.app) {
        return;
    }
    storeArgv(argc, argv);
    s.app = std::make_unique<QApplication>(s.qt_argc, s.argv_ptrs.data());
    s.app->setQuitOnLastWindowClosed(true);
}

void requestQuit() {
    auto& s = state();
    if (s.closed) {
        return;
    }
    s.closed = true;
    s.exit_clicked.store(true);
    g_interrupted().store(true);
    if (s.window) {
        s.window->hide();
    }
}

void notifyLauncherReadyOnce() {
    auto& s = state();
    if (s.launcher_ready_notified) {
        return;
    }
    s.launcher_ready_notified = true;
    const char* path = std::getenv("DX_LAUNCHER_READY_FILE");
    if (path == nullptr || *path == '\0') {
        return;
    }
    std::ofstream ready(path, std::ios::trunc);
    if (ready) {
        ready << "ready\n";
    }
}

void ensureWindow() {
    auto& s = state();
    if (!s.app) {
        return;
    }
    if (!s.window) {
        s.window = std::make_unique<QtOutputWidget>();
        s.window->setQuitHandler(requestQuit);
        s.window->setShowExitButton(s.show_exit_button);
        s.window->show();
    }
    if (!s.window_sized && s.window) {
        if (s.full_screen) {
            s.window->showFullScreen();
        } else {
            const QScreen* screen = QGuiApplication::primaryScreen();
            const QSize screen_size = screen ? screen->availableSize() : QSize(1920, 1080);
            const int win_w = std::max(320, screen_size.width() / 2);
            const int win_h = std::max(240, screen_size.height() / 2);
            s.window->resize(win_w, win_h);
            s.window->move(
                (screen_size.width() - win_w) / 2,
                (screen_size.height() - win_h) / 2);
        }
        s.window_sized = true;
    }
}

void pumpEvents() {
    auto& s = state();
    if (s.app) {
        s.app->processEvents(QEventLoop::AllEvents, 1);
    }
}

}  // namespace

void configureDisplay(bool full_screen, bool show_exit_button, int argc, char** argv) {
    auto& s = state();
    s.full_screen = full_screen;
    s.show_exit_button = show_exit_button;
    ensureApplication(argc, argv);
}

void resetDisplayState() {
    auto& s = state();
    s.closed = false;
    s.exit_clicked.store(false);
    s.window_sized = false;
    s.launcher_ready_notified = false;
    s.rgb_buffer.release();
    resetDisplayFpsState();
    if (s.window) {
        s.window->close();
        s.window.reset();
    }
}

void shutdownDisplay() {
    resetDisplayState();
}

bool consumeExitButtonClick() {
    return state().exit_clicked.exchange(false);
}

bool windowShouldClose(const std::string& /*winname*/) {
    auto& s = state();
    if (s.closed) {
        return true;
    }
    pumpEvents();
    return s.closed;
}

void setDisplayFps(double fps) {
    auto& s = fpsState();
    std::lock_guard<std::mutex> lock(s.mutex);
    s.fps = std::max(0.0, fps);
    s.external = true;
}

void showOutput(const cv::Mat& frame) {
    auto& s = state();
    if (s.closed) {
        return;
    }
    if (!s.app) {
        return;
    }
    ensureWindow();
    if (!s.window) {
        return;
    }

    s.window->setShowExitButton(s.show_exit_button);
    if (!frame.empty()) {
        s.window->updateFrame(frame, s.rgb_buffer);
    }
    pumpEvents();
    if (!frame.empty()) {
        notifyLauncherReadyOnce();
    }
}

}  // namespace dxapp
