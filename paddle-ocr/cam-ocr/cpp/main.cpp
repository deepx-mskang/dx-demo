#include "ocr_engine.hpp"

#include <opencv2/opencv.hpp>

#include <QApplication>
#include <QCheckBox>
#include <QCloseEvent>
#include <QFile>
#include <QFont>
#include <QFontDatabase>
#include <QFrame>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QImage>
#include <QKeyEvent>
#include <QLabel>
#include <QPainter>
#include <QPixmap>
#include <QPushButton>
#include <QSlider>
#include <QTextEdit>
#include <QThread>
#include <QTimer>
#include <QVBoxLayout>
#include <QWidget>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <cmath>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

namespace fs = std::filesystem;
using OcrResultPtr = std::shared_ptr<camocr::OcrResult>;

Q_DECLARE_METATYPE(cv::Mat)
Q_DECLARE_METATYPE(OcrResultPtr)

namespace {

constexpr double kConfidenceThreshold = 0.2;

const char* kBg = "#1e1e1e";
const char* kPanel = "#252526";
const char* kCard = "#2d2d30";
const char* kAccent = "#007acc";
const char* kAccentDark = "#005a9e";
const char* kText = "#cccccc";
const char* kTextDim = "#858585";
const char* kBorder = "#3c3c3c";

struct CameraConfig {
    int deviceIndex = 0;
    int width = 1280;
    int height = 720;
    double fps = 15.0;
};

enum class SharpnessMode {
    Off,
    Soft,
    Medium,
    Strong,
};

const char* sharpnessModeName(SharpnessMode mode)
{
    switch (mode) {
    case SharpnessMode::Off:
        return "off";
    case SharpnessMode::Soft:
        return "soft";
    case SharpnessMode::Medium:
        return "medium";
    case SharpnessMode::Strong:
        return "strong";
    }
    return "off";
}

cv::Mat applyOcrSharpness(const cv::Mat& frame, SharpnessMode mode)
{
    if (frame.empty() || mode == SharpnessMode::Off) {
        return frame;
    }

    struct UnsharpParams {
        double srcWeight;
        double blurWeight;
        double sigma;
    };

    UnsharpParams params{};
    switch (mode) {
    case SharpnessMode::Soft:
        params = {1.25, -0.25, 0.8};
        break;
    case SharpnessMode::Medium:
        params = {1.5, -0.5, 1.0};
        break;
    case SharpnessMode::Strong:
        params = {1.8, -0.8, 1.2};
        break;
    case SharpnessMode::Off:
        return frame;
    }

    cv::Mat lab;
    cv::cvtColor(frame, lab, cv::COLOR_BGR2Lab);

    std::vector<cv::Mat> channels;
    cv::split(lab, channels);

    cv::Mat blurredL;
    cv::GaussianBlur(channels[0], blurredL, cv::Size(0, 0), params.sigma);
    cv::addWeighted(channels[0], params.srcWeight, blurredL, params.blurWeight, 0.0, channels[0]);

    cv::merge(channels, lab);

    cv::Mat sharpened;
    cv::cvtColor(lab, sharpened, cv::COLOR_Lab2BGR);
    return sharpened;
}

cv::Mat centerCropToSize(const cv::Mat& frame, int cropSize)
{
    if (frame.empty()) {
        return {};
    }

    if (frame.cols >= cropSize && frame.rows >= cropSize) {
        const int x = (frame.cols - cropSize) / 2;
        const int y = (frame.rows - cropSize) / 2;
        return frame(cv::Rect(x, y, cropSize, cropSize)).clone();
    }

    const int squareSize = std::min(frame.cols, frame.rows);
    const int x = std::max(0, (frame.cols - squareSize) / 2);
    const int y = std::max(0, (frame.rows - squareSize) / 2);
    cv::Mat cropped = frame(cv::Rect(x, y, squareSize, squareSize));
    cv::Mat resized;
    cv::resize(cropped, resized, cv::Size(cropSize, cropSize), 0.0, 0.0, cv::INTER_LINEAR);
    return resized;
}

fs::path defaultRoot()
{
#ifdef CAM_OCR_ROOT_DIR
    return CAM_OCR_ROOT_DIR;
#else
    return fs::current_path();
#endif
}

fs::path defaultModelsBaseDir(const fs::path& root)
{
    return root / "cpp" / "assets";
}

fs::path defaultModelsDir(const fs::path& root, const std::string& modelProfile)
{
    return defaultModelsBaseDir(root) / modelProfile;
}

void notifyLauncherReady()
{
    const char* readyPath = std::getenv("DX_LAUNCHER_READY_FILE");
    if (!readyPath || std::string(readyPath).empty()) {
        return;
    }
    try {
        fs::path path(readyPath);
        fs::create_directories(path.parent_path());
        std::ofstream out(path);
        out << "ready\n";
    } catch (...) {
    }
}

QImage matToQImageRgb(const cv::Mat& bgr)
{
    if (bgr.empty()) {
        return {};
    }
    cv::Mat rgb;
    if (bgr.channels() == 3) {
        cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    } else if (bgr.channels() == 1) {
        cv::cvtColor(bgr, rgb, cv::COLOR_GRAY2RGB);
    } else {
        cv::cvtColor(bgr, rgb, cv::COLOR_BGRA2RGB);
    }
    return QImage(rgb.data, rgb.cols, rgb.rows, static_cast<int>(rgb.step), QImage::Format_RGB888).copy();
}

cv::Mat qImageToBgr(const QImage& image)
{
    QImage rgb = image.convertToFormat(QImage::Format_RGB888);
    cv::Mat rgbMat(rgb.height(), rgb.width(), CV_8UC3, const_cast<uchar*>(rgb.bits()), rgb.bytesPerLine());
    cv::Mat bgr;
    cv::cvtColor(rgbMat, bgr, cv::COLOR_RGB2BGR);
    return bgr.clone();
}

void setScaledPixmap(QLabel* label, const cv::Mat& bgr)
{
    if (!label || bgr.empty()) {
        return;
    }
    QImage image = matToQImageRgb(bgr);
    const QSize target = label->contentsRect().size().isValid() ? label->contentsRect().size() : label->size();
    label->setPixmap(QPixmap::fromImage(image).scaled(
        target, Qt::KeepAspectRatio, Qt::SmoothTransformation));
}

std::vector<std::vector<cv::Point>> toIntPolys(const std::vector<std::vector<cv::Point2f>>& boxes)
{
    std::vector<std::vector<cv::Point>> polys;
    polys.reserve(boxes.size());
    for (const auto& box : boxes) {
        std::vector<cv::Point> poly;
        poly.reserve(box.size());
        for (const auto& p : box) {
            poly.emplace_back(static_cast<int>(std::round(p.x)), static_cast<int>(std::round(p.y)));
        }
        polys.push_back(poly);
    }
    return polys;
}

void drawDashedLine(
    cv::Mat& image,
    cv::Point p0,
    cv::Point p1,
    const cv::Scalar& color,
    int thickness,
    int dashLength,
    int gapLength)
{
    const double dx = static_cast<double>(p1.x - p0.x);
    const double dy = static_cast<double>(p1.y - p0.y);
    const double length = std::hypot(dx, dy);
    if (length < 1.0) {
        return;
    }

    const double ux = dx / length;
    const double uy = dy / length;
    double traveled = 0.0;
    bool drawSegment = true;

    while (traveled < length) {
        const double segmentLength = drawSegment ? static_cast<double>(dashLength) : static_cast<double>(gapLength);
        const double next = std::min(traveled + segmentLength, length);
        if (drawSegment) {
            const cv::Point start(
                static_cast<int>(std::round(p0.x + ux * traveled)),
                static_cast<int>(std::round(p0.y + uy * traveled)));
            const cv::Point end(
                static_cast<int>(std::round(p0.x + ux * next)),
                static_cast<int>(std::round(p0.y + uy * next)));
            cv::line(image, start, end, color, thickness, cv::LINE_AA);
        }
        traveled = next;
        drawSegment = !drawSegment;
    }
}

void drawDashedPolyline(
    cv::Mat& image,
    const std::vector<cv::Point>& poly,
    const cv::Scalar& color,
    int thickness,
    int dashLength = 6,
    int gapLength = 4)
{
    if (poly.size() < 2) {
        return;
    }

    for (std::size_t i = 0; i < poly.size(); ++i) {
        drawDashedLine(image, poly[i], poly[(i + 1) % poly.size()], color, thickness, dashLength, gapLength);
    }
}

std::unordered_set<int> recognizedBoxIndices(const std::vector<camocr::OcrText>& texts)
{
    std::unordered_set<int> indices;
    indices.reserve(texts.size());
    for (const auto& item : texts) {
        if (item.bboxIndex >= 0 && !item.text.empty() && item.score > kConfidenceThreshold) {
            indices.insert(item.bboxIndex);
        }
    }
    return indices;
}

struct RightTextLayout {
    QString text;
    int originX = 0;
    int originY = 0;
    int boxW = 0;
    int boxH = 0;
    int padding = 0;
    int fontSize = 0;
    int pendingFontSize = 0;
    int pendingFontDirection = 0;
    int pendingFontCount = 0;
};

using RightTextLayoutCache = std::unordered_map<int, RightTextLayout>;

bool isSimilarLayoutText(const QString& previous, const QString& current)
{
    if (previous == current) {
        return true;
    }
    if (previous.isEmpty() || current.isEmpty()) {
        return false;
    }

    const int previousLen = previous.size();
    const int currentLen = current.size();
    const int maxLen = std::max(previousLen, currentLen);
    const int allowedDistance = std::max(1, maxLen / 5);
    if (std::abs(previousLen - currentLen) > allowedDistance) {
        return false;
    }

    std::vector<int> prevRow(currentLen + 1);
    std::vector<int> curRow(currentLen + 1);
    for (int j = 0; j <= currentLen; ++j) {
        prevRow[j] = j;
    }
    for (int i = 1; i <= previousLen; ++i) {
        curRow[0] = i;
        int rowMin = curRow[0];
        for (int j = 1; j <= currentLen; ++j) {
            const int substitutionCost = previous[i - 1] == current[j - 1] ? 0 : 1;
            curRow[j] = std::min({
                prevRow[j] + 1,
                curRow[j - 1] + 1,
                prevRow[j - 1] + substitutionCost,
            });
            rowMin = std::min(rowMin, curRow[j]);
        }
        if (rowMin > allowedDistance) {
            return false;
        }
        std::swap(prevRow, curRow);
    }
    return prevRow[currentLen] <= allowedDistance;
}

cv::Mat makeLeftDisplayImage(
    const cv::Mat& bgr,
    const std::vector<std::vector<cv::Point2f>>& boxes,
    const std::vector<camocr::OcrText>& texts)
{
    if (bgr.empty()) {
        return {};
    }

    cv::Mat display = bgr.clone();
    const auto polys = toIntPolys(boxes);
    const auto recognized = recognizedBoxIndices(texts);

    constexpr double kFillAlpha = 0.14;
    const cv::Scalar accentFill(204, 122, 0);       // BGR #007acc
    const cv::Scalar accentBorder(255, 180, 60);
    const cv::Scalar emptyBorder(133, 133, 133);    // BGR #858585 (kTextDim)

    for (std::size_t i = 0; i < polys.size(); ++i) {
        const auto& poly = polys[i];
        if (poly.size() < 3) {
            continue;
        }

        if (recognized.count(static_cast<int>(i)) > 0) {
            cv::Mat tinted = display.clone();
            cv::fillPoly(tinted, std::vector<std::vector<cv::Point>>{poly}, accentFill);
            cv::addWeighted(display, 1.0 - kFillAlpha, tinted, kFillAlpha, 0.0, display);
            cv::polylines(display, poly, true, accentBorder, 1, cv::LINE_AA);
        } else {
            drawDashedPolyline(display, poly, emptyBorder, 1);
        }
    }

    return display;
}

QString fontPathForLanguage(const fs::path& root, const std::string& language)
{
    const fs::path defaultFontPath = root / "cpp" / "assets" / "NotoSansJP-VariableFont_wght.ttf";
    if (fs::exists(defaultFontPath)) {
        return QString::fromStdString(defaultFontPath.string());
    }

    fs::path fontPath = root / "engine" / "fonts" / "NotoSansJP-VariableFont_wght.ttf";
    if (fs::exists(fontPath)) {
        return QString::fromStdString(fontPath.string());
    }

    if (language == "korean") {
        fontPath = root / "engine" / "fonts" / "korean_font.ttf";
    } else if (language == "german") {
        fontPath = root / "engine" / "fonts" / "Roboto-Regular.ttf";
    } else {
        fontPath = root / "engine" / "fonts" / "simfang.ttf";
    }
    return QString::fromStdString(fontPath.string());
}

QString fontFamilyFromPath(const QString& fontPath)
{
    const int fontId = QFontDatabase::addApplicationFont(fontPath);
    if (fontId >= 0) {
        const QStringList families = QFontDatabase::applicationFontFamilies(fontId);
        if (!families.isEmpty()) {
            return families.front();
        }
    }
    return "Sans Serif";
}

cv::Mat makeRightDisplayImage(
    const cv::Mat& bgr,
    const std::vector<camocr::OcrText>& texts,
    const QString& fontPath,
    RightTextLayoutCache* layoutCache)
{
    if (bgr.empty()) {
        return {};
    }

    QImage canvas(bgr.cols, bgr.rows, QImage::Format_RGB888);
    canvas.fill(Qt::white);

    const QString family = fontFamilyFromPath(fontPath);

    QPainter painter(&canvas);
    painter.setRenderHint(QPainter::TextAntialiasing, true);
    painter.setPen(Qt::black);

    constexpr int kPositionSnap = 4;
    constexpr int kFontSizeSnap = 2;
    constexpr double kSizeHysteresisRatio = 0.15;
    constexpr int kFontChangeConfirmFrames = 3;
    const auto snapToGrid = [](float value, int grid) {
        return static_cast<int>(std::round(value / static_cast<float>(grid))) * grid;
    };
    const auto withinRatio = [](int previous, int current, double ratio) {
        if (previous <= 0 || current <= 0) {
            return false;
        }
        return std::abs(previous - current) <= static_cast<int>(std::round(previous * ratio));
    };
    const auto boxIoU = [](int ax, int ay, int aw, int ah, int bx, int by, int bw, int bh) {
        const int left = std::max(ax, bx);
        const int top = std::max(ay, by);
        const int right = std::min(ax + aw, bx + bw);
        const int bottom = std::min(ay + ah, by + bh);
        const int intersectionW = std::max(0, right - left);
        const int intersectionH = std::max(0, bottom - top);
        const int intersection = intersectionW * intersectionH;
        const int areaA = std::max(0, aw) * std::max(0, ah);
        const int areaB = std::max(0, bw) * std::max(0, bh);
        const int united = areaA + areaB - intersection;
        return united > 0 ? static_cast<double>(intersection) / static_cast<double>(united) : 0.0;
    };
    const auto fitFontSize = [&](const QString& text, int targetBoxW, int targetBoxH, int targetPadding) {
        int size = std::max(8, snapToGrid(static_cast<float>(targetBoxH) * 0.75f, kFontSizeSnap));
        QFont font(family, size);
        for (; size > 7; size -= kFontSizeSnap) {
            font.setPointSize(size);
            QFontMetrics metrics(font);
            if (metrics.horizontalAdvance(text) <= std::max(8, targetBoxW - 2 * targetPadding) &&
                metrics.height() <= std::max(8, targetBoxH - 2 * targetPadding)) {
                break;
            }
        }
        return size;
    };

    std::unordered_set<int> visibleKeys;
    int fallbackKey = -1;
    for (const auto& item : texts) {
        if (item.text.empty() || item.bbox.size() != 4) {
            continue;
        }

        float minX = item.bbox[0].x;
        float maxX = item.bbox[0].x;
        float minY = item.bbox[0].y;
        float maxY = item.bbox[0].y;
        for (const auto& p : item.bbox) {
            minX = std::min(minX, p.x);
            maxX = std::max(maxX, p.x);
            minY = std::min(minY, p.y);
            maxY = std::max(maxY, p.y);
        }

        const int originX = std::clamp(snapToGrid(minX, kPositionSnap), 0, bgr.cols - 1);
        const int originY = std::clamp(snapToGrid(minY, kPositionSnap), 0, bgr.rows - 1);
        const int boxW = std::max(10, snapToGrid(maxX - minX, kPositionSnap));
        const int boxH = std::max(10, snapToGrid(maxY - minY, kPositionSnap));
        const int padding = std::max(2, std::min(boxW, boxH) / 20);
        const QString qText = QString::fromUtf8(item.text.c_str());

        int drawOriginX = originX;
        int drawOriginY = originY;
        int drawBoxW = boxW;
        int drawBoxH = boxH;
        int drawPadding = padding;
        int pendingFontSize = 0;
        int pendingFontDirection = 0;
        int pendingFontCount = 0;

        const int cacheKey = item.bboxIndex >= 0 ? item.bboxIndex : fallbackKey--;
        visibleKeys.insert(cacheKey);
        bool hasReusableCache = false;
        bool hasSimilarText = false;
        if (layoutCache) {
            auto cached = layoutCache->find(cacheKey);
            if (cached != layoutCache->end()) {
                pendingFontSize = cached->second.pendingFontSize;
                pendingFontDirection = cached->second.pendingFontDirection;
                pendingFontCount = cached->second.pendingFontCount;

                hasSimilarText = isSimilarLayoutText(cached->second.text, qText);
                const bool stableMove =
                    std::abs(cached->second.originX - originX) <= kPositionSnap &&
                    std::abs(cached->second.originY - originY) <= kPositionSnap;
                const bool stableSize =
                    withinRatio(cached->second.boxW, boxW, kSizeHysteresisRatio) &&
                    withinRatio(cached->second.boxH, boxH, kSizeHysteresisRatio);
                const double overlap = boxIoU(
                    cached->second.originX,
                    cached->second.originY,
                    cached->second.boxW,
                    cached->second.boxH,
                    originX,
                    originY,
                    boxW,
                    boxH);
                const bool stableOverlap = overlap >= 0.70;

                if (hasSimilarText && ((stableMove && stableSize) || stableOverlap)) {
                    drawOriginX = cached->second.originX;
                    drawOriginY = cached->second.originY;
                    drawBoxW = cached->second.boxW;
                    drawBoxH = cached->second.boxH;
                    drawPadding = cached->second.padding;
                    hasReusableCache = true;
                }
            }
        }

        const int desiredFontSize = fitFontSize(qText, drawBoxW, drawBoxH, drawPadding);
        int fontSize = desiredFontSize;
        if (layoutCache) {
            auto cached = layoutCache->find(cacheKey);
            if (cached != layoutCache->end() && cached->second.fontSize > 0 &&
                hasSimilarText && desiredFontSize != cached->second.fontSize) {
                const int direction = desiredFontSize > cached->second.fontSize ? 1 : -1;
                if (cached->second.pendingFontDirection == direction &&
                    cached->second.pendingFontSize == desiredFontSize) {
                    pendingFontCount = cached->second.pendingFontCount + 1;
                } else {
                    pendingFontCount = 1;
                }
                pendingFontDirection = direction;
                pendingFontSize = desiredFontSize;
                if (pendingFontCount < kFontChangeConfirmFrames) {
                    fontSize = cached->second.fontSize;
                } else {
                    pendingFontSize = 0;
                    pendingFontDirection = 0;
                    pendingFontCount = 0;
                }
            } else if (cached != layoutCache->end() && hasReusableCache) {
                fontSize = cached->second.fontSize;
                pendingFontSize = 0;
                pendingFontDirection = 0;
                pendingFontCount = 0;
            }
        }

        QFont font(family, fontSize);
        if (layoutCache) {
            (*layoutCache)[cacheKey] = RightTextLayout{
                qText,
                drawOriginX,
                drawOriginY,
                drawBoxW,
                drawBoxH,
                drawPadding,
                fontSize,
                pendingFontSize,
                pendingFontDirection,
                pendingFontCount,
            };
        }
        painter.setFont(font);
        const QFontMetrics metrics(font);
        painter.drawText(QPointF(drawOriginX + drawPadding, drawOriginY + drawPadding + metrics.ascent()), qText);
    }
    if (layoutCache) {
        for (auto it = layoutCache->begin(); it != layoutCache->end();) {
            if (visibleKeys.count(it->first) == 0) {
                it = layoutCache->erase(it);
            } else {
                ++it;
            }
        }
    }
    painter.end();
    return qImageToBgr(canvas);
}

QString appStyleSheet()
{
    return QString(R"(
        QWidget {
            background-color: %1;
            color: %2;
        }
        QLabel {
            color: %2;
        }
        QScrollArea {
            border: 1px solid %3;
            border-radius: 8px;
            background-color: %4;
        }
    )").arg(kBg, kText, kBorder, kCard);
}

QString titleStyle(int size = 16)
{
    return QString("font-size:%1px; font-weight:bold; color:%2; padding:4px 0;").arg(size).arg(kAccent);
}

}  // namespace

class CameraThread : public QThread {
    Q_OBJECT

public:
    explicit CameraThread(
        CameraConfig cameraConfig,
        QString videoPath = {},
        SharpnessMode sharpnessMode = SharpnessMode::Off,
        bool sharpnessEnabled = false,
        QObject* parent = nullptr)
        : QThread(parent),
          videoPath_(std::move(videoPath)),
          cameraConfig_(cameraConfig),
          sharpnessMode_(sharpnessMode),
          sharpnessEnabled_(sharpnessEnabled)
    {
    }

    void stop()
    {
        running_.store(false);
        wait();
    }

    void enqueueFocusUi(int value)
    {
        std::lock_guard<std::mutex> lock(focusMutex_);
        pendingFocus_ = std::max(0, std::min(100, value));
        hasPendingFocus_ = true;
    }

    void setSharpnessEnabled(bool enabled)
    {
        sharpnessEnabled_.store(enabled);
    }

    void setSharpnessMode(SharpnessMode mode)
    {
        sharpnessMode_.store(mode);
    }

signals:
    void frameReady(quint64 frameId, const cv::Mat& frame);
    void ocrFrameReady(quint64 frameId, const cv::Mat& frame);
    void cameraFocusState(int ui, bool focusOk);

protected:
    void run() override
    {
        cv::VideoCapture cap;
        const bool useVideo = !videoPath_.isEmpty();
        if (useVideo) {
            const std::string path = videoPath_.toStdString();
            cap.open(path);
            if (!cap.isOpened()) {
                std::cerr << "Failed to open video: " << path << std::endl;
                return;
            }
            cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
            std::cout << "Video input: " << path << std::endl;
        } else {
            cap.open(cameraConfig_.deviceIndex, cv::CAP_V4L2);
            if (!cap.isOpened()) {
                std::cerr << "Failed to open camera " << cameraConfig_.deviceIndex << std::endl;
                return;
            }
            cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
            cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
            cap.set(cv::CAP_PROP_FRAME_WIDTH, cameraConfig_.width);
            cap.set(cv::CAP_PROP_FRAME_HEIGHT, cameraConfig_.height);
            cap.set(cv::CAP_PROP_FPS, cameraConfig_.fps);
            std::cout << "Camera input: /dev/video" << cameraConfig_.deviceIndex
                      << " requested " << cameraConfig_.width << "x" << cameraConfig_.height
                      << " @ " << cameraConfig_.fps << " FPS"
                      << " format MJPG" << std::endl;
            setAutoFocus(cap, false);
        }

        std::cout << "w: " << cap.get(cv::CAP_PROP_FRAME_WIDTH)
                  << ", h: " << cap.get(cv::CAP_PROP_FRAME_HEIGHT)
                  << ", FPS: " << cap.get(cv::CAP_PROP_FPS)
                  << ", CC: " << cap.get(cv::CAP_PROP_FOURCC) << std::endl;

        if (!useVideo) {
            double rawFocus = cap.get(cv::CAP_PROP_FOCUS);
            int uiFocus = static_cast<int>(rawFocus / 10.0);
            bool ok = uiFocus >= 0;
            uiFocus = std::max(0, std::min(100, uiFocus));
            emit cameraFocusState(uiFocus, ok);
        }

        running_.store(true);
        quint64 nextFrameId = 1;

        while (running_.load()) {
            if (!useVideo) {
                drainFocus(cap);
            }

            cv::Mat frame;
            if (!cap.read(frame) || frame.empty()) {
                if (useVideo) {
                    cap.set(cv::CAP_PROP_POS_FRAMES, 0);
                }
                continue;
            }

            if (!useVideo) {
                frame = centerCropToSize(frame, 640);
            }

            if (sharpnessEnabled_.load()) {
                frame = applyOcrSharpness(frame, sharpnessMode_.load());
            }
            const quint64 frameId = nextFrameId++;
            emit frameReady(frameId, frame.clone());
            emit ocrFrameReady(frameId, frame.clone());

            if (useVideo) {
                msleep(100);
            }
        }
        if (!useVideo) {
            setAutoFocus(cap, true);
        }
        cap.release();
    }

private:
    static void setAutoFocus(cv::VideoCapture& cap, bool enabled)
    {
        const bool ok = cap.set(cv::CAP_PROP_AUTOFOCUS, enabled ? 1.0 : 0.0);
        std::cout << "Camera autofocus " << (enabled ? "enable" : "disable")
                  << (ok ? " requested" : " request failed") << std::endl;
    }

    void drainFocus(cv::VideoCapture& cap)
    {
        int value = 0;
        {
            std::lock_guard<std::mutex> lock(focusMutex_);
            if (!hasPendingFocus_) {
                return;
            }
            value = pendingFocus_;
            hasPendingFocus_ = false;
        }
        cap.set(cv::CAP_PROP_FOCUS, value * 10.0);
    }

    QString videoPath_;
    CameraConfig cameraConfig_;
    std::atomic_bool running_{false};
    std::atomic<SharpnessMode> sharpnessMode_{SharpnessMode::Off};
    std::atomic_bool sharpnessEnabled_{false};
    std::mutex focusMutex_;
    int pendingFocus_ = 0;
    bool hasPendingFocus_ = false;
};

class OcrProcessThread : public QThread {
    Q_OBJECT

public:
    explicit OcrProcessThread(std::shared_ptr<camocr::PaddleOcrEngine> engine, QObject* parent = nullptr)
        : QThread(parent), engine_(std::move(engine))
    {
    }

    void processFrame(quint64 frameId, const cv::Mat& frame)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.size() >= 10) {
            queue_.pop_front();
        }
        queue_.push_back({frameId, frame.clone()});
        cv_.notify_one();
    }

    void stop()
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            running_ = false;
        }
        cv_.notify_one();
        wait();
    }

signals:
    void resultReady(quint64 frameId, OcrResultPtr result);

protected:
    void run() override
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            running_ = true;
        }

        while (true) {
            QueuedFrame queuedFrame;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait_for(lock, std::chrono::milliseconds(50), [&] {
                    return !running_ || !queue_.empty();
                });
                if (!running_) {
                    break;
                }
                if (queue_.empty()) {
                    continue;
                }
                queuedFrame = queue_.back();
                queue_.clear();
            }

            try {
                auto result = std::make_shared<camocr::OcrResult>(engine_->run(queuedFrame.frame));
                emit resultReady(queuedFrame.frameId, result);
            } catch (const std::exception& e) {
                std::cerr << "OCR processing error: " << e.what() << std::endl;
            }
        }
    }

private:
    struct QueuedFrame {
        quint64 frameId = 0;
        cv::Mat frame;
    };

    std::shared_ptr<camocr::PaddleOcrEngine> engine_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<QueuedFrame> queue_;
    bool running_ = false;
};

class PerformanceInfoWidget : public QWidget {
    Q_OBJECT

public:
    explicit PerformanceInfoWidget(QWidget* parent = nullptr)
        : QWidget(parent)
    {
        auto* layout = new QVBoxLayout(this);
        layout->setContentsMargins(0, 0, 0, 0);
        layout->setSpacing(10);
        setFixedHeight(184);
        setStyleSheet("background:transparent; border:none;");

        auto* title = new QLabel("Performance Information (FPS)");
        title->setStyleSheet(QString("font-size:18px; font-weight:bold; color:%1; border:none; background:transparent;").arg(kAccent));
        layout->addWidget(title);

        auto* grid = new QGridLayout();
        grid->setContentsMargins(0, 0, 0, 0);
        grid->setHorizontalSpacing(8);
        grid->setVerticalSpacing(8);
        grid->addWidget(createMetricCard("DET (Detection)", &detValue_), 0, 0);
        grid->addWidget(createMetricCard("CLS (Classification)", &clsValue_), 0, 1);
        grid->addWidget(createMetricCard("REC (Recognition)", &recValue_), 0, 2);
        grid->addWidget(createMetricCard("E2E (End-to-End)", &e2eValue_), 1, 0);
        grid->addWidget(createMetricCard("CPS (Characters Per Sec)", &cpsValue_), 1, 1);
        grid->addWidget(createMetricCard("CHARS / BOXES", &charsValue_), 1, 2);
        layout->addLayout(grid);
        updateData({});
    }

    void updateData(const camocr::PerfStats& stats)
    {
        const bool hasText = stats.totalChars > 0;
        detValue_->setText(fpsText(stats.detTimeMs));
        clsValue_->setText(hasText ? fpsText(stats.clsTimeMs) : "--");
        recValue_->setText(hasText ? fpsText(stats.recTimeMs) : "--");
        e2eValue_->setText(fpsText(stats.e2eTimeMs));
        cpsValue_->setText(QString::number(stats.cps, 'f', 1));
        charsValue_->setText(QString("%1 / %2").arg(stats.totalChars).arg(stats.numBoxes));
    }

private:
    static QString fpsText(double ms)
    {
        if (ms <= 0.0) {
            return "--";
        }
        return QString::number(1000.0 / (ms + 1e-10), 'f', 2);
    }

    QFrame* createMetricCard(const QString& label, QLabel** valueLabel)
    {
        auto* frame = new QFrame();
        frame->setFixedHeight(70);
        frame->setStyleSheet(QString("QFrame { background-color:%1; border:1px solid %2; border-radius:8px; }").arg(kCard, kBorder));

        auto* box = new QVBoxLayout(frame);
        box->setContentsMargins(10, 8, 10, 8);
        box->setSpacing(4);

        auto* name = new QLabel(label);
        name->setStyleSheet(QString("font-size:11px; font-weight:bold; color:%1; border:none; background:transparent;").arg(kTextDim));
        box->addWidget(name);

        auto* value = new QLabel("--");
        value->setStyleSheet(QString("font-family:'Consolas','Courier New',monospace; font-size:20px; font-weight:bold; color:%1; border:none; background:transparent;").arg(kText));
        value->setAlignment(Qt::AlignLeft | Qt::AlignVCenter);
        box->addWidget(value);

        *valueLabel = value;
        return frame;
    }

    QLabel* detValue_ = nullptr;
    QLabel* clsValue_ = nullptr;
    QLabel* recValue_ = nullptr;
    QLabel* e2eValue_ = nullptr;
    QLabel* cpsValue_ = nullptr;
    QLabel* charsValue_ = nullptr;
};

class ImageViewerApp : public QWidget {
    Q_OBJECT

public:
    ImageViewerApp(
        std::shared_ptr<camocr::PaddleOcrEngine> engine,
        fs::path rootDir,
        std::string language,
        CameraConfig cameraConfig,
        QString videoPath,
        SharpnessMode sharpnessMode,
        bool sharpnessEnabled,
        QWidget* parent = nullptr)
        : QWidget(parent),
          engine_(std::move(engine)),
          rootDir_(std::move(rootDir)),
          language_(std::move(language)),
          cameraConfig_(cameraConfig),
          videoPath_(std::move(videoPath)),
          sharpnessMode_(sharpnessMode),
          sharpnessEnabled_(sharpnessEnabled),
          fontPath_(fontPathForLanguage(rootDir_, language_))
    {
        setWindowTitle("DEEPX M1 Live OCR Demo");
        initUi();
    }

    void startCamera()
    {
        if (cameraActive_) {
            return;
        }
        hasShownResultFrameId_ = false;
        lastShownResultFrameId_ = 0;
        lastResult_.reset();
        lastRightImage_.release();
        rightTextLayoutCache_.clear();
        cameraThread_ = new CameraThread(cameraConfig_, videoPath_, sharpnessMode_, sharpnessEnabled_, this);
        ocrThread_ = new OcrProcessThread(engine_, this);

        connect(cameraThread_, &CameraThread::frameReady, this, &ImageViewerApp::onCameraFrameDisplay);
        connect(cameraThread_, &CameraThread::ocrFrameReady, ocrThread_, &OcrProcessThread::processFrame);
        connect(cameraThread_, &CameraThread::cameraFocusState, this, &ImageViewerApp::onCameraFocusState);
        connect(ocrThread_, &OcrProcessThread::resultReady, this, &ImageViewerApp::onOcrResult);

        ocrThread_->start();
        cameraThread_->start();
        cameraActive_ = true;
    }

    void stopCamera()
    {
        if (cameraThread_) {
            cameraThread_->stop();
            cameraThread_->deleteLater();
            cameraThread_ = nullptr;
        }
        if (ocrThread_) {
            ocrThread_->stop();
            ocrThread_->deleteLater();
            ocrThread_ = nullptr;
        }
        cameraActive_ = false;
        if (videoPath_.isEmpty()) {
            focusSlider_->setEnabled(false);
        }
    }

protected:
    void closeEvent(QCloseEvent* event) override
    {
        stopCamera();
        QWidget::closeEvent(event);
    }

    void keyPressEvent(QKeyEvent* event) override
    {
        if (event->key() == Qt::Key_Escape || event->key() == Qt::Key_Q) {
            close();
            return;
        }
        QWidget::keyPressEvent(event);
    }

    void resizeEvent(QResizeEvent* event) override
    {
        if (!lastFrame_.empty()) {
            updateLeftPreview(lastFrame_);
        }
        if (!lastRightImage_.empty()) {
            setScaledPixmap(rightImageLabel_, lastRightImage_);
        }
        QWidget::resizeEvent(event);
    }

private slots:
    void onCameraFrameDisplay(quint64 frameId, const cv::Mat& frame)
    {
        lastCameraFrameId_ = frameId;
        lastFrame_ = frame.clone();
        updateLeftPreview(lastFrame_);
    }

    void onOcrResult(quint64 frameId, OcrResultPtr result)
    {
        if (!result) {
            return;
        }
        if (hasShownResultFrameId_ && frameId <= lastShownResultFrameId_) {
            std::cout << "[OCR] Drop stale result frame_id=" << frameId
                      << " last_shown_result_frame_id=" << lastShownResultFrameId_ << std::endl;
            return;
        }
        hasShownResultFrameId_ = true;
        lastShownResultFrameId_ = frameId;
        lastResult_ = result;
        updatePerformance(*result);
        updateTextInfo(*result);
        updateRightOverlay(*result);
        if (!lastFrame_.empty()) {
            updateLeftPreview(lastFrame_);
        }
    }

    void onCameraFocusState(int ui, bool focusOk)
    {
        focusSlider_->blockSignals(true);
        focusSlider_->setValue(ui);
        focusSlider_->blockSignals(false);
        focusSlider_->setEnabled(focusOk && videoPath_.isEmpty());
    }

    void onFocusChanged(int value)
    {
        if (cameraThread_ && cameraActive_ && videoPath_.isEmpty()) {
            cameraThread_->enqueueFocusUi(value);
        }
    }

    void onSharpnessToggled(bool enabled)
    {
        sharpnessEnabled_ = enabled;
        if (cameraThread_) {
            cameraThread_->setSharpnessEnabled(enabled);
        }
    }

private:
    void initUi()
    {
        auto* rootLayout = new QVBoxLayout(this);
        rootLayout->setContentsMargins(16, 16, 16, 16);
        rootLayout->setSpacing(12);

        auto* titleBar = new QFrame();
        titleBar->setFixedHeight(68);
        titleBar->setStyleSheet(QString(R"(
            QFrame {
                background-color: %1;
                border: 1px solid %2;
                border-radius: 8px;
            }
        )").arg(kPanel, kBorder));
        auto* titleLayout = new QHBoxLayout(titleBar);
        titleLayout->setContentsMargins(16, 0, 16, 0);
        titleLayout->setSpacing(8);

        auto* titleLeftPad = new QWidget();
        titleLeftPad->setFixedSize(32, 28);
        titleLeftPad->setStyleSheet("border:none; background:transparent;");
        titleLayout->addWidget(titleLeftPad);

        auto* appTitle = new QLabel("DEEPX M1 Live OCR Demo");
        appTitle->setAlignment(Qt::AlignCenter);
        appTitle->setStyleSheet(QString(R"(
            QLabel {
                font-size: 34px;
                font-weight: 800;
                letter-spacing: 0px;
                color: #f4f4f5;
                background: transparent;
                border: none;
                padding-bottom: 2px;
            }
        )"));
        titleLayout->addWidget(appTitle, 1);

        auto* exitButton = new QPushButton("X");
        exitButton->setFixedSize(32, 28);
        exitButton->setFocusPolicy(Qt::NoFocus);
        exitButton->setToolTip("Exit");
        exitButton->setStyleSheet(QString(R"(
            QPushButton {
                color: %1;
                background-color: %2;
                border: 1px solid %3;
                border-radius: 6px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #3a3a3d;
                color: #ffffff;
            }
            QPushButton:pressed {
                background-color: %4;
            }
        )").arg(kText, kCard, kBorder, kAccentDark));
        connect(exitButton, &QPushButton::clicked, this, [this] {
            close();
        });
        titleLayout->addWidget(exitButton);

        rootLayout->addWidget(titleBar);

        auto* contentLayout = new QHBoxLayout();
        contentLayout->setContentsMargins(0, 0, 0, 0);
        contentLayout->setSpacing(108);

        auto* leftColumn = new QVBoxLayout();
        leftColumn->setContentsMargins(0, 0, 0, 0);
        leftColumn->setSpacing(10);

        auto* leftTitle = new QLabel("Input Image + BBox");
        leftTitle->setFixedWidth(640);
        leftTitle->setFixedHeight(30);
        leftTitle->setAlignment(Qt::AlignCenter);
        leftTitle->setStyleSheet(titleStyle(18));
        leftColumn->addWidget(leftTitle, 0, Qt::AlignHCenter);

        leftImageLabel_ = new QLabel("Waiting for camera");
        leftImageLabel_->setAlignment(Qt::AlignCenter);
        leftImageLabel_->setFixedSize(640, 640);
        leftImageLabel_->setStyleSheet(QString("border:2px solid %1; border-radius:8px; background-color:%2; color:%3;").arg(kBorder, kPanel, kTextDim));
        leftColumn->addWidget(leftImageLabel_, 0, Qt::AlignHCenter);

        auto* leftInfoPanel = new QFrame();
        leftInfoPanel->setFixedSize(640, 264);
        leftInfoPanel->setStyleSheet(QString("QFrame { background-color:%1; border:1px solid %2; border-radius:8px; }").arg(kPanel, kBorder));
        auto* leftInfoLayout = new QVBoxLayout(leftInfoPanel);
        leftInfoLayout->setContentsMargins(14, 12, 14, 12);
        leftInfoLayout->setSpacing(10);

        performanceWidget_ = new PerformanceInfoWidget();
        leftInfoLayout->addWidget(performanceWidget_);

        auto* focusRow = new QWidget();
        focusRow->setFixedHeight(34);
        focusRow->setStyleSheet("background:transparent; border:none;");
        auto* focusLayout = new QHBoxLayout(focusRow);
        focusLayout->setContentsMargins(0, 0, 0, 0);
        auto* focusLabel = new QLabel("Camera Focus Tuning");
        focusLabel->setStyleSheet(QString("color:%1; font-size:12px; border:none; background:transparent;").arg(kTextDim));
        sharpnessCheckBox_ = new QCheckBox("Sharpness");
        sharpnessCheckBox_->setChecked(sharpnessEnabled_);
        sharpnessCheckBox_->setToolTip(
            QString("Enable OCR sharpening for preview and OCR input (mode: %1)")
                .arg(sharpnessModeName(sharpnessMode_)));
        sharpnessCheckBox_->setStyleSheet(QString(R"(
            QCheckBox {
                color: %1;
                font-size: 12px;
                border: none;
                background: transparent;
                spacing: 6px;
            }
            QCheckBox::indicator {
                width: 14px;
                height: 14px;
            }
            QCheckBox::indicator:unchecked {
                border: 1px solid %2;
                background: %3;
                border-radius: 3px;
            }
            QCheckBox::indicator:checked {
                border: 1px solid %4;
                background: %5;
                border-radius: 3px;
            }
        )").arg(kTextDim, kBorder, kCard, kAccent, kAccent));
        connect(sharpnessCheckBox_, &QCheckBox::toggled, this, &ImageViewerApp::onSharpnessToggled);
        focusSlider_ = new QSlider(Qt::Horizontal);
        focusSlider_->setRange(0, 100);
        focusSlider_->setFixedWidth(330);
        focusSlider_->setEnabled(false);
        focusSlider_->setStyleSheet(QString(R"(
            QSlider::groove:horizontal { height: 6px; background: %1; border-radius: 3px; }
            QSlider::handle:horizontal { background: %2; width: 14px; margin: -5px 0; border-radius: 7px; }
            QSlider::sub-page:horizontal { background: %3; border-radius: 3px; }
        )").arg(kBorder, kAccent, kAccentDark));
        connect(focusSlider_, &QSlider::valueChanged, this, &ImageViewerApp::onFocusChanged);
        focusLayout->addWidget(focusLabel);
        focusLayout->addWidget(sharpnessCheckBox_);
        focusLayout->addStretch(1);
        focusLayout->addWidget(focusSlider_);
        leftInfoLayout->addWidget(focusRow);
        leftColumn->addWidget(leftInfoPanel, 0, Qt::AlignHCenter);

        auto* leftPane = new QWidget();
        leftPane->setFixedWidth(640);
        leftPane->setLayout(leftColumn);
        contentLayout->addWidget(leftPane, 0, Qt::AlignTop);

        auto* rightColumn = new QVBoxLayout();
        rightColumn->setContentsMargins(0, 0, 0, 0);
        rightColumn->setSpacing(10);

        auto* rightTitle = new QLabel("Result Text Overlay");
        rightTitle->setFixedWidth(640);
        rightTitle->setFixedHeight(30);
        rightTitle->setAlignment(Qt::AlignCenter);
        rightTitle->setStyleSheet(titleStyle(18));
        rightColumn->addWidget(rightTitle, 0, Qt::AlignHCenter);

        rightImageLabel_ = new QLabel("Waiting for OCR");
        rightImageLabel_->setAlignment(Qt::AlignCenter);
        rightImageLabel_->setFixedSize(640, 640);
        rightImageLabel_->setStyleSheet(QString("border:2px solid %1; border-radius:8px; background-color:%2; color:%3;").arg(kBorder, kPanel, kTextDim));
        rightColumn->addWidget(rightImageLabel_, 0, Qt::AlignHCenter);

        auto* textPanel = new QFrame();
        textPanel->setFixedSize(640, 264);
        textPanel->setStyleSheet(QString("QFrame { background-color:%1; border:1px solid %2; border-radius:8px; }").arg(kPanel, kBorder));
        auto* textPanelLayout = new QVBoxLayout(textPanel);
        textPanelLayout->setContentsMargins(14, 12, 14, 12);
        textPanelLayout->setSpacing(8);

        auto* infoTitle = new QLabel("Inference Result Text");
        infoTitle->setStyleSheet(QString("font-size:18px; font-weight:bold; color:%1; border:none; background:transparent;").arg(kAccent));
        textPanelLayout->addWidget(infoTitle);

        infoText_ = new QTextEdit();
        infoText_->setReadOnly(true);
        infoText_->setFont(QFont(fontFamilyFromPath(fontPath_), 14));
        infoText_->setTextInteractionFlags(Qt::TextSelectableByMouse | Qt::TextSelectableByKeyboard);
        infoText_->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
        infoText_->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
        infoText_->setStyleSheet(QString(R"(
            QTextEdit {
                line-height: 1.2;
                padding: 8px;
                font-size: 14px;
                background-color: %1;
                color: %2;
                border: 1px solid %3;
                border-radius: 8px;
            }
        )").arg(kCard, kText, kBorder));
        textPanelLayout->addWidget(infoText_, 1);
        rightColumn->addWidget(textPanel, 0, Qt::AlignHCenter);

        auto* rightPane = new QWidget();
        rightPane->setFixedWidth(640);
        rightPane->setLayout(rightColumn);
        contentLayout->addWidget(rightPane, 0, Qt::AlignTop);
        rootLayout->addLayout(contentLayout, 1);
        contentLayout->setAlignment(Qt::AlignHCenter | Qt::AlignTop);
    }

    void updateLeftPreview(const cv::Mat& frame)
    {
        if (frame.empty()) {
            return;
        }

        if (lastResult_) {
            setScaledPixmap(leftImageLabel_, makeLeftDisplayImage(frame, lastResult_->boxes, lastResult_->texts));
        } else {
            setScaledPixmap(leftImageLabel_, frame);
        }
    }

    void updateRightOverlay(const camocr::OcrResult& result)
    {
        lastRightImage_ = makeRightDisplayImage(
            result.preprocessedImage,
            result.texts,
            fontPath_,
            &rightTextLayoutCache_);
        if (!lastRightImage_.empty()) {
            setScaledPixmap(rightImageLabel_, lastRightImage_);
        }
    }

    void updatePerformance(const camocr::OcrResult& result)
    {
        performanceWidget_->updateData(result.perf);
    }

    void updateTextInfo(const camocr::OcrResult& result)
    {
        QString text;
        text += "  recognized text : confidence score\n\n";
        int index = 1;
        for (const auto& item : result.texts) {
            if (item.score > kConfidenceThreshold) {
                text += QString("%1. %2 : %3\n")
                            .arg(index++)
                            .arg(QString::fromUtf8(item.text.c_str()))
                            .arg(item.score, 0, 'f', 2);
            }
        }
        infoText_->setText(text);
    }

    std::shared_ptr<camocr::PaddleOcrEngine> engine_;
    fs::path rootDir_;
    std::string language_;
    CameraConfig cameraConfig_;
    QString videoPath_;
    SharpnessMode sharpnessMode_ = SharpnessMode::Off;
    bool sharpnessEnabled_ = false;
    QString fontPath_;

    QLabel* leftImageLabel_ = nullptr;
    QLabel* rightImageLabel_ = nullptr;
    QTextEdit* infoText_ = nullptr;
    PerformanceInfoWidget* performanceWidget_ = nullptr;
    QCheckBox* sharpnessCheckBox_ = nullptr;
    QSlider* focusSlider_ = nullptr;

    CameraThread* cameraThread_ = nullptr;
    OcrProcessThread* ocrThread_ = nullptr;
    bool cameraActive_ = false;

    quint64 lastCameraFrameId_ = 0;
    quint64 lastShownResultFrameId_ = 0;
    bool hasShownResultFrameId_ = false;
    cv::Mat lastFrame_;
    cv::Mat lastRightImage_;
    OcrResultPtr lastResult_;
    RightTextLayoutCache rightTextLayoutCache_;
};

struct Args {
    QString videoPath;
    std::string language = "ch";
    std::string modelProfile = "mobile";
    CameraConfig camera;
    bool enableUvdoc = false;
    SharpnessMode sharpnessMode = SharpnessMode::Medium;
    bool enableSharpness = false;
    int recAsyncQueueSize = camocr::kRecAsyncQueueDefault;
};

int parsePositiveInt(const QString& value, const char* optionName)
{
    bool ok = false;
    const int parsed = value.toInt(&ok);
    if (!ok || parsed <= 0) {
        std::cerr << "Invalid " << optionName << ": " << value.toStdString() << std::endl;
        std::exit(2);
    }
    return parsed;
}

int parseNonNegativeInt(const QString& value, const char* optionName)
{
    bool ok = false;
    const int parsed = value.toInt(&ok);
    if (!ok || parsed < 0) {
        std::cerr << "Invalid " << optionName << ": " << value.toStdString() << std::endl;
        std::exit(2);
    }
    return parsed;
}

double parsePositiveDouble(const QString& value, const char* optionName)
{
    bool ok = false;
    const double parsed = value.toDouble(&ok);
    if (!ok || parsed <= 0.0) {
        std::cerr << "Invalid " << optionName << ": " << value.toStdString() << std::endl;
        std::exit(2);
    }
    return parsed;
}

SharpnessMode parseSharpnessMode(const QString& value, const char* optionName)
{
    const QString normalized = value.trimmed().toLower();
    if (normalized == "off" || normalized == "none" || normalized == "0") {
        return SharpnessMode::Off;
    }
    if (normalized == "soft" || normalized == "low" || normalized == "1") {
        return SharpnessMode::Soft;
    }
    if (normalized == "medium" || normalized == "med" || normalized == "2") {
        return SharpnessMode::Medium;
    }
    if (normalized == "strong" || normalized == "high" || normalized == "3") {
        return SharpnessMode::Strong;
    }

    std::cerr << "Invalid " << optionName << ": " << value.toStdString()
              << " (expected off|soft|medium|strong)" << std::endl;
    std::exit(2);
}

void parseResolution(const QString& value, CameraConfig* camera)
{
    const QString normalized = value.toLower();
    const QStringList parts = normalized.split('x', Qt::SkipEmptyParts);
    if (parts.size() != 2) {
        std::cerr << "Invalid --resolution: " << value.toStdString() << " (expected WIDTHxHEIGHT)" << std::endl;
        std::exit(2);
    }
    camera->width = parsePositiveInt(parts[0], "--resolution width");
    camera->height = parsePositiveInt(parts[1], "--resolution height");
}

void printUsage()
{
    std::cout
        << "Usage: cam_ocr_demo [--video PATH] [--camera N] [--resolution WIDTHxHEIGHT]\n"
        << "                    [--width W --height H] [--fps FPS]\n"
        << "                    [--model mobile|server|hybrid] [--language ch|korean|german]\n"
        << "                    [--rec-queue-size N] [--enable-uvdoc]\n"
        << "                    [--enable-sharpness] [--sharpness off|soft|medium|strong]\n"
        << "Defaults: sharpness is off. GUI checkbox toggles the selected sharpness mode,\n"
        << "and `--enable-sharpness` uses the default `medium` mode.\n";
}

Args parseArgs(const QStringList& args)
{
    Args parsed;
    for (int i = 1; i < args.size(); ++i) {
        const QString arg = args[i];
        if ((arg == "-v" || arg == "--video") && i + 1 < args.size()) {
            parsed.videoPath = args[++i];
        } else if ((arg == "-c" || arg == "--camera" || arg == "--camera-id" || arg == "--device") && i + 1 < args.size()) {
            parsed.camera.deviceIndex = parseNonNegativeInt(args[++i], "--camera");
        } else if ((arg == "-r" || arg == "--resolution") && i + 1 < args.size()) {
            parseResolution(args[++i], &parsed.camera);
        } else if (arg == "--width" && i + 1 < args.size()) {
            parsed.camera.width = parsePositiveInt(args[++i], "--width");
        } else if (arg == "--height" && i + 1 < args.size()) {
            parsed.camera.height = parsePositiveInt(args[++i], "--height");
        } else if (arg == "--fps" && i + 1 < args.size()) {
            parsed.camera.fps = parsePositiveDouble(args[++i], "--fps");
        } else if ((arg == "--model" || arg == "--model-profile" || arg == "--profile") && i + 1 < args.size()) {
            parsed.modelProfile = args[++i].toStdString();
        } else if (arg == "--language" && i + 1 < args.size()) {
            parsed.language = args[++i].toStdString();
        } else if ((arg == "--rec-queue-size" || arg == "--rec-async-queue") && i + 1 < args.size()) {
            parsed.recAsyncQueueSize = parsePositiveInt(args[++i], "--rec-queue-size");
            if (parsed.recAsyncQueueSize > camocr::kRecAsyncQueueMax) {
                std::cerr << "Invalid --rec-queue-size: max is " << camocr::kRecAsyncQueueMax << std::endl;
                std::exit(2);
            }
        } else if (arg == "--enable-uvdoc") {
            parsed.enableUvdoc = true;
        } else if (arg == "--enable-sharpness") {
            parsed.enableSharpness = true;
        } else if (arg == "--sharpness") {
            if (i + 1 < args.size() && !args[i + 1].startsWith('-')) {
                parsed.sharpnessMode = parseSharpnessMode(args[++i], "--sharpness");
                parsed.enableSharpness = parsed.sharpnessMode != SharpnessMode::Off;
            } else {
                parsed.enableSharpness = true;
            }
        } else if (arg == "--hide-preview") {
            continue;
        } else if (arg == "-h" || arg == "--help") {
            printUsage();
            std::exit(0);
        } else {
            std::cerr << "Unknown or incomplete option: " << arg.toStdString() << std::endl;
            printUsage();
            std::exit(2);
        }
    }

    if (parsed.language != "ch" && parsed.language != "korean" && parsed.language != "german") {
        std::cerr << "Unsupported language: " << parsed.language << ". Falling back to ch.\n";
        parsed.language = "ch";
    }
    if (parsed.modelProfile != "mobile" && parsed.modelProfile != "server" && parsed.modelProfile != "hybrid") {
        std::cerr << "Unsupported model profile: " << parsed.modelProfile << ". Use mobile, server, or hybrid.\n";
        std::exit(2);
    }
    return parsed;
}

int main(int argc, char** argv)
{
    QApplication app(argc, argv);
    qRegisterMetaType<cv::Mat>("cv::Mat");
    qRegisterMetaType<OcrResultPtr>("OcrResultPtr");
    app.setStyleSheet(appStyleSheet());

    const Args args = parseArgs(app.arguments());
    if (!args.videoPath.isEmpty() && !QFile::exists(args.videoPath)) {
        std::cerr << "video file not found: " << args.videoPath.toStdString() << std::endl;
        return 2;
    }

    std::cout << "v5 selected. Will operate with orientation and distortion correction models and algorithms." << std::endl;
    std::cout << "v5 detection model: Modified to support 640 and 960 resolutions" << std::endl;

    const fs::path root = defaultRoot();
    const fs::path modelsBaseDir = defaultModelsBaseDir(root);
    camocr::EngineOptions options;
    options.rootDir = root;
    options.modelsBaseDir = modelsBaseDir;
    options.language = args.language;
    options.modelProfile = args.modelProfile;
    std::cout << "Model directory: " << defaultModelsDir(root, args.modelProfile).string() << std::endl;
    options.enableUvdoc = args.enableUvdoc;
    options.recAsyncQueueSize = args.recAsyncQueueSize;
    std::cout << "Sharpness mode: " << sharpnessModeName(args.sharpnessMode)
              << ", enabled: " << (args.enableSharpness ? "yes" : "no") << std::endl;

    std::shared_ptr<camocr::PaddleOcrEngine> engine;
    try {
        engine = std::make_shared<camocr::PaddleOcrEngine>(options);
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize OCR engine: " << e.what() << std::endl;
        return 1;
    }

    ImageViewerApp viewer(
        engine, root, args.language, args.camera, args.videoPath, args.sharpnessMode, args.enableSharpness);
    notifyLauncherReady();
    viewer.showFullScreen();
    QTimer::singleShot(0, &viewer, &ImageViewerApp::startCamera);
    return app.exec();
}

#include "main.moc"
