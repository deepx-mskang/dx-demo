#include <algorithm>
#include <chrono>
#include <atomic>
#include <iostream>
#include <memory>
#include <cstdio>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>
#include <dxrt/dxrt_api.h>

#include <QApplication>
#include <QCloseEvent>
#include <QFrame>
#include <QFontMetrics>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QImage>
#include <QKeyEvent>
#include <QLabel>
#include <QMainWindow>
#include <QPainter>
#include <QPainterPath>
#include <QPen>
#include <QPushButton>
#include <QShowEvent>
#include <QSizePolicy>
#include <QTimer>
#include <QVBoxLayout>
#include <QWidget>

#include "face_recognition.h"
#include "encrypted_model_engine.h"
#include "inference_fps.h"
#include "mobed_detector.h"
#include "pose_seg.h"

#include <cstdlib>
#include <fstream>

namespace
{


bool ready_notified = false;

inline void notify_launcher_ready() {
    const char* path = std::getenv("DX_LAUNCHER_READY_FILE");
    if (!path || !*path)
        return;
    // append: 기존 파일이 있어도 변경만 되면 됨
    std::ofstream f(path, std::ios::app);
    if (f)
        f << "ready\n";
}

constexpr float kDefaultFrThreshold = 0.5f;
constexpr int kVideoPlaybackFps = 30;
constexpr int kCameraCaptureWidth = 960;
constexpr int kCameraCaptureHeight = 530;
constexpr int kCameraCaptureFps = 30;
const char *kFaceRecognitionWindowTitle = "Face Recognition";
const char *kPoseSegmentationWindowTitle = "Pose Estimation & Segmentation";
const char *kMobedDetectionWindowTitle = "DAL-e Object Detection";

constexpr int kMobedStaticPictureW = 960;
constexpr int kMobedStaticPictureH = 530;
constexpr int kExitButtonW = 32;
constexpr int kExitButtonH = 28;
constexpr int kExitButtonMargin = 8;

const char *kUsage =
    "Face Recognition Demo\n"
    "  -mps0, --pose_modelpath   (* required) define yolo pose model path\n"
    "  -mps1, --seg_modelpath    (* required) define segmentation model path\n"
    "  -m0, --fd_modelpath       (* required) face detection model dxnn file path\n"
    "  -m1, --lm_modelpath       (* required) face align model dxnn file path\n"
    "  -m2, --fr_modelpath       (* required) face recognition model dxnn file path\n"
    "  -md, --det_modelpath      (* required) define mobED detection model path\n"
    "  -m3, --age_sex_modelpath  age and sex estimation model dxnn file path\n"
    "  -th, --threshold          Similarity Threshold\n"
    "  -p,  --dbpath             face database directory\n"
    "  -v,  --video              use video file input\n"
    "  -c,  --camera             use camera input\n"
    "       --exit-btn           show a small in-app exit button\n"
    "  -h,  --help               show help\n";

struct AppOptions
{
    std::string dbPath;
    std::string videoFile;
    std::string mps0ModelPath;
    std::string mps1ModelPath;
    std::string fdModelPath;
    std::string lmModelPath;
    std::string frModelPath;
    std::string detModelPath;
    std::string genderModelPath;
    bool cameraInput = false;
    bool classifierGender = false;
    bool showExitButton = false;
    float frThreshold = kDefaultFrThreshold;
};

void help()
{
    std::cout << kUsage << std::endl;
}

bool has_next_arg(int index, int argc)
{
    return index < argc;
}

bool parse_args(int argc, char *argv[], AppOptions *options)
{
    if (argc == 1)
    {
        std::cout << "Error: no arguments." << std::endl;
        help();
        return false;
    }

    int i = 1;
    while (i < argc)
    {
        std::string arg(argv[i++]);
        if (arg == "-th" || arg == "--threshold")
        {
            if (!has_next_arg(i, argc))
            {
                std::cout << "Error: missing value for " << arg << std::endl;
                return false;
            }
            options->frThreshold = std::stof(argv[i++]);
        }
        else if (arg == "-mps0" || arg == "--pose_modelpath")
        {
            if (!has_next_arg(i, argc))
            {
                std::cout << "Error: missing value for " << arg << std::endl;
                return false;
            }
            options->mps0ModelPath = argv[i++];
        }
        else if (arg == "-mps1" || arg == "--seg_modelpath")
        {
            if (!has_next_arg(i, argc))
            {
                std::cout << "Error: missing value for " << arg << std::endl;
                return false;
            }
            options->mps1ModelPath = argv[i++];
        }
        else if (arg == "-m0" || arg == "--fd_modelpath")
        {
            if (!has_next_arg(i, argc))
            {
                std::cout << "Error: missing value for " << arg << std::endl;
                return false;
            }
            options->fdModelPath = argv[i++];
        }
        else if (arg == "-m1" || arg == "--lm_modelpath")
        {
            if (!has_next_arg(i, argc))
            {
                std::cout << "Error: missing value for " << arg << std::endl;
                return false;
            }
            options->lmModelPath = argv[i++];
        }
        else if (arg == "-m2" || arg == "--fr_modelpath")
        {
            if (!has_next_arg(i, argc))
            {
                std::cout << "Error: missing value for " << arg << std::endl;
                return false;
            }
            options->frModelPath = argv[i++];
        }
        else if (arg == "-md" || arg == "--det_modelpath")
        {
            if (!has_next_arg(i, argc))
            {
                std::cout << "Error: missing value for " << arg << std::endl;
                return false;
            }
            options->detModelPath = argv[i++];
        }
        else if (arg == "-m3" || arg == "--age_sex_modelpath")
        {
            if (!has_next_arg(i, argc))
            {
                std::cout << "Error: missing value for " << arg << std::endl;
                return false;
            }
            options->genderModelPath = argv[i++];
            options->classifierGender = true;
        }
        else if (arg == "-p" || arg == "--dbpath")
        {
            if (!has_next_arg(i, argc))
            {
                std::cout << "Error: missing value for " << arg << std::endl;
                return false;
            }
            options->dbPath = argv[i++];
        }
        else if (arg == "-v" || arg == "--video")
        {
            if (!has_next_arg(i, argc))
            {
                std::cout << "Error: missing value for " << arg << std::endl;
                return false;
            }
            options->videoFile = argv[i++];
        }
        else if (arg == "-c" || arg == "--camera")
        {
            options->cameraInput = true;
        }
        else if (arg == "--exit-btn")
        {
            options->showExitButton = true;
        }
        else if (arg == "-h" || arg == "--help")
        {
            help();
            std::exit(0);
        }
        else
        {
            std::cout << "Error: unknown argument " << arg << std::endl;
            help();
            return false;
        }
    }

    return true;
}

SsdParam make_fd_config()
{
    return {
        512,
        true,
        0.25,
        0.25,
        4,
        2,
        {"BACKGROUND", "person", "no_mask", "mask"},
        {"1275", "1305", "1335", "1365", "1392", "1416", "1440"},
        {"1290", "1320", "1350", "1380", "1404", "1428", "1452"},
        {
            7,
            0.2,
            0.95,
            0.1,
            0.2,
            {
                {64, 64, 6},
                {32, 32, 6},
                {16, 16, 6},
                {8, 8, 6},
                {4, 4, 6},
                {2, 2, 4},
                {1, 1, 4},
            },
            "./sample/face_prior_boxes.bin"
        },
    };
}

bool open_video_capture(const AppOptions &options, cv::VideoCapture *cap)
{
    if (options.cameraInput)
    {
        if (!cap->open(0, cv::CAP_V4L2))
        {
            std::cout << "Error: failed to open camera input." << std::endl;
            return false;
        }
        cap->set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
        cap->set(cv::CAP_PROP_FRAME_WIDTH, static_cast<double>(kCameraCaptureWidth));
        cap->set(cv::CAP_PROP_FRAME_HEIGHT, static_cast<double>(kCameraCaptureHeight));
        cap->set(cv::CAP_PROP_FPS, static_cast<double>(kCameraCaptureFps));
        return true;
    }

    if (!cap->open(options.videoFile))
    {
        std::cout << "Error: failed to open video file: " << options.videoFile << std::endl;
        return false;
    }

    cap->set(cv::CAP_PROP_FPS, static_cast<double>(kVideoPlaybackFps));
    return true;
}

bool is_exit_key(int key)
{
    return key == 27 || key == 'q' || key == 'Q';
}

cv::Mat load_mobed_static_picture_resized()
{
    const std::string path = std::string(PROJECT_ROOT_DIR) + "/../../../assets/robotics/mobed.jpg";
    cv::Mat src = cv::imread(path);
    cv::Mat out;
    if (src.empty())
    {
        std::cout << "Warning: could not load " << path << std::endl;
        return out;
    }
    cv::resize(src, out, cv::Size(kMobedStaticPictureW, kMobedStaticPictureH));
    return out;
}

/** When using camera, some drivers ignore CAP_PROP size; normalize display to 960x530. */
void resize_camera_display_if_needed(bool camera_input, cv::Mat *view)
{
    if (!camera_input || view == nullptr || view->empty())
    {
        return;
    }
    if (view->cols == kCameraCaptureWidth && view->rows == kCameraCaptureHeight)
    {
        return;
    }
    cv::Mat resized;
    cv::resize(*view, resized, cv::Size(kCameraCaptureWidth, kCameraCaptureHeight));
    *view = std::move(resized);
}

QRect fitted_rect(const QRect &area, const QSize &image_size)
{
    if (!area.isValid() || image_size.isEmpty())
    {
        return {};
    }

    QSize scaled = image_size;
    scaled.scale(area.size(), Qt::KeepAspectRatio);
    const int x = area.x() + (area.width() - scaled.width()) / 2;
    const int y = area.y() + (area.height() - scaled.height()) / 2;
    return QRect(QPoint(x, y), scaled);
}

void draw_bgr_image(QPainter *painter, const cv::Mat &bgr, const QRect &target)
{
    if (painter == nullptr || bgr.empty() || bgr.type() != CV_8UC3 || !target.isValid())
    {
        return;
    }

    QImage image(bgr.data, bgr.cols, bgr.rows, static_cast<int>(bgr.step), QImage::Format_BGR888);
    painter->drawImage(target, image);
}

void draw_key_badge(QPainter *painter, const QRect &rect, const QString &key, const QString &label)
{
    if (painter == nullptr || !rect.isValid())
    {
        return;
    }

    const QRect key_rect(rect.x(), rect.y(), 28, rect.height());
    painter->setPen(QPen(QColor("#5a6372"), 1));
    painter->setBrush(QColor(28, 32, 40, 235));
    painter->drawRoundedRect(key_rect, 5, 5);

    QFont key_font = painter->font();
    key_font.setPointSize(11);
    key_font.setBold(true);
    painter->setFont(key_font);
    painter->setPen(QColor("#f4f7fb"));
    painter->drawText(key_rect, Qt::AlignCenter, key);

    QFont label_font = painter->font();
    label_font.setPointSize(10);
    label_font.setBold(false);
    painter->setFont(label_font);
    painter->setPen(QColor("#b8c0cc"));
    painter->drawText(rect.adjusted(36, 0, 0, 0), Qt::AlignLeft | Qt::AlignVCenter, label);
}

void draw_empty_face_icon(QPainter *painter, const QRect &rect)
{
    if (painter == nullptr || !rect.isValid())
    {
        return;
    }

    painter->save();
    painter->setRenderHint(QPainter::Antialiasing, true);
    painter->setPen(QPen(QColor("#7dd3fc"), 2));
    painter->setBrush(QColor(125, 211, 252, 24));

    const int head_size = std::min(rect.width(), rect.height()) / 2;
    const QRect head(rect.center().x() - head_size / 2,
                     rect.y() + rect.height() / 6,
                     head_size,
                     head_size);
    painter->drawEllipse(head);

    const QRect shoulders(rect.x() + rect.width() / 5,
                          rect.y() + rect.height() / 2,
                          rect.width() * 3 / 5,
                          rect.height() / 3);
    painter->drawArc(shoulders, 20 * 16, 140 * 16);
    painter->restore();
}

QString panel_rate_text(const std::string &mode_label, const std::string &fps_text)
{
    return QString("%1  %2").arg(QString::fromStdString(mode_label), QString::fromStdString(fps_text));
}

void set_label_text_if_changed(QLabel *label, const QString &text)
{
    if (label != nullptr && label->text() != text)
    {
        label->setText(text);
    }
}

class BgrImageView : public QWidget
{
public:
    explicit BgrImageView(QWidget *parent = nullptr)
        : QWidget(parent)
    {
        setMinimumSize(280, 160);
        setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
        setFocusPolicy(Qt::NoFocus);
        setAttribute(Qt::WA_OpaquePaintEvent, true);
    }

    void SetFrame(cv::Mat frame)
    {
        frame_ = std::move(frame);
        update();
    }

protected:
    void paintEvent(QPaintEvent *) override
    {
        QPainter painter(this);
        painter.fillRect(rect(), QColor("#0c0d10"));
        painter.setRenderHint(QPainter::SmoothPixmapTransform, false);
        const QRect target = fitted_rect(rect(), QSize(frame_.cols, frame_.rows));
        draw_bgr_image(&painter, frame_, target);
    }

private:
    cv::Mat frame_;
};

class GalleryOverlayView : public QWidget
{
public:
    explicit GalleryOverlayView(QWidget *parent = nullptr)
        : QWidget(parent)
    {
        setMinimumSize(280, 160);
        setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
        setFocusPolicy(Qt::NoFocus);
        setAttribute(Qt::WA_OpaquePaintEvent, true);
    }

    void SetBaseFrame(cv::Mat frame)
    {
        base_frame_ = std::move(frame);
        update();
    }

    void SetGalleryFrame(cv::Mat frame)
    {
        gallery_frame_ = std::move(frame);
        gallery_count_ = EstimateGalleryCount();
        update();
    }

protected:
    void paintEvent(QPaintEvent *) override
    {
        QPainter painter(this);
        painter.fillRect(rect(), QColor("#0c0d10"));
        painter.setRenderHint(QPainter::SmoothPixmapTransform, false);

        const QRect base_target = fitted_rect(rect(), QSize(base_frame_.cols, base_frame_.rows));
        draw_bgr_image(&painter, base_frame_, base_target.isValid() ? base_target : rect());

        const QRect overlay_area = base_target.isValid() ? base_target : rect();
        if (gallery_frame_.empty() || gallery_frame_.type() != CV_8UC3 || gallery_count_ <= 0)
        {
            DrawEmptyState(&painter, overlay_area);
            return;
        }

        DrawRegisteredGallery(&painter, overlay_area);
    }

private:
    int EstimateGalleryCount() const
    {
        if (gallery_frame_.empty() || gallery_frame_.rows <= 0)
        {
            return 0;
        }
        return std::max(1, gallery_frame_.cols / gallery_frame_.rows);
    }

    QRect OverlayCardRect(const QRect &area, int preferred_width, int preferred_height) const
    {
        constexpr int kMargin = 10;
        const int available_w = std::max(1, area.width() - 2 * kMargin);
        const int available_h = std::max(1, area.height() - 2 * kMargin);
        const int card_w = std::min(preferred_width, available_w);
        const int card_h = std::min(preferred_height, available_h);
        return QRect(area.x() + kMargin, area.y() + kMargin, card_w, card_h);
    }

    void DrawCardBackground(QPainter *painter, const QRect &card) const
    {
        painter->save();
        painter->setRenderHint(QPainter::Antialiasing, true);
        painter->setPen(QPen(QColor(82, 91, 108, 180), 1));
        painter->setBrush(QColor(12, 14, 18, 215));
        painter->drawRoundedRect(card, 10, 10);
        painter->restore();
    }

    void DrawEmptyState(QPainter *painter, const QRect &area) const
    {
        const QRect card = OverlayCardRect(area, 380, 178);
        DrawCardBackground(painter, card);

        const QRect icon_rect(card.x() + 18, card.y() + 26, 76, 76);
        draw_empty_face_icon(painter, icon_rect);

        QFont title_font = painter->font();
        title_font.setPointSize(18);
        title_font.setBold(true);
        painter->setFont(title_font);
        painter->setPen(QColor("#f4f7fb"));
        painter->drawText(QRect(card.x() + 108, card.y() + 34, card.width() - 130, 28),
                          Qt::AlignLeft | Qt::AlignVCenter,
                          "No Face Registered");

        QFont body_font = painter->font();
        body_font.setPointSize(11);
        body_font.setBold(false);
        painter->setFont(body_font);
        painter->setPen(QColor("#aeb7c4"));
        painter->drawText(QRect(card.x() + 108, card.y() + 68, card.width() - 130, 24),
                          Qt::AlignLeft | Qt::AlignVCenter,
                          "Capture a face from the live view");

        draw_key_badge(painter,
                       QRect(card.x() + 108, card.y() + 112, 112, 24),
                       "A",
                       "Add");
    }

    void DrawRegisteredGallery(QPainter *painter, const QRect &area) const
    {
        const QRect card = OverlayCardRect(area, 520, 202);
        DrawCardBackground(painter, card);

        QFont title_font = painter->font();
        title_font.setPointSize(14);
        title_font.setBold(true);
        painter->setFont(title_font);
        painter->setPen(QColor("#f4f7fb"));
        painter->drawText(QRect(card.x() + 16, card.y() + 12, card.width() - 140, 24),
                          Qt::AlignLeft | Qt::AlignVCenter,
                          "Registered Faces");

        const QString count_text = QString("%1 saved").arg(gallery_count_);
        QFont count_font = painter->font();
        count_font.setPointSize(10);
        count_font.setBold(true);
        painter->setFont(count_font);
        const int pill_w = std::max(72, QFontMetrics(count_font).horizontalAdvance(count_text) + 22);
        const QRect pill(card.right() - pill_w - 16, card.y() + 13, pill_w, 22);
        painter->setRenderHint(QPainter::Antialiasing, true);
        painter->setPen(Qt::NoPen);
        painter->setBrush(QColor(0, 122, 204, 185));
        painter->drawRoundedRect(pill, 11, 11);
        painter->setPen(QColor("#ffffff"));
        painter->drawText(pill, Qt::AlignCenter, count_text);

        const int thumb_size = 72;
        const int gap = 12;
        const int strip_x = card.x() + 16;
        const int strip_y = card.y() + 54;
        const int strip_w = std::max(1, card.width() - 32);
        const int max_slots = std::max(1, (strip_w + gap) / (thumb_size + gap));
        const bool show_more = gallery_count_ > max_slots && max_slots > 1;
        const int visible_count = show_more ? max_slots - 1 : std::min(gallery_count_, max_slots);
        const int source_w = std::max(1, gallery_frame_.rows);
        QImage strip(gallery_frame_.data,
                     gallery_frame_.cols,
                     gallery_frame_.rows,
                     static_cast<int>(gallery_frame_.step),
                     QImage::Format_BGR888);

        for (int i = 0; i < visible_count; ++i)
        {
            const QRect thumb_rect(strip_x + i * (thumb_size + gap), strip_y, thumb_size, thumb_size);
            painter->setPen(QPen(QColor(102, 116, 135, 210), 1));
            painter->setBrush(QColor(22, 26, 34, 230));
            painter->drawRoundedRect(thumb_rect.adjusted(-3, -3, 3, 3), 10, 10);

            QPainterPath clip_path;
            clip_path.addRoundedRect(thumb_rect, 8, 8);
            painter->save();
            painter->setClipPath(clip_path);
            painter->drawImage(thumb_rect, strip, QRect(i * source_w, 0, source_w, gallery_frame_.rows));
            painter->restore();

            QFont label_font = painter->font();
            label_font.setPointSize(10);
            label_font.setBold(true);
            painter->setFont(label_font);
            painter->setPen(QColor("#dce3ec"));
            painter->drawText(QRect(thumb_rect.x(), thumb_rect.bottom() + 6, thumb_size, 18),
                              Qt::AlignCenter,
                              QString("ID %1").arg(i + 1, 2, 10, QChar('0')));
        }

        if (show_more)
        {
            const QRect more_rect(strip_x + visible_count * (thumb_size + gap), strip_y, thumb_size, thumb_size);
            painter->setPen(QPen(QColor(102, 116, 135, 210), 1));
            painter->setBrush(QColor(22, 26, 34, 230));
            painter->drawRoundedRect(more_rect, 10, 10);
            painter->setPen(QColor("#f4f7fb"));
            QFont more_font = painter->font();
            more_font.setPointSize(14);
            more_font.setBold(true);
            painter->setFont(more_font);
            painter->drawText(more_rect, Qt::AlignCenter, QString("+%1").arg(gallery_count_ - visible_count));
        }

        const int keys_y = card.bottom() - 36;
        draw_key_badge(painter, QRect(card.x() + 16, keys_y, 96, 24), "A", "Add");
        draw_key_badge(painter, QRect(card.x() + 120, keys_y, 120, 24), "D", "Remove");
    }

    cv::Mat base_frame_;
    cv::Mat gallery_frame_;
    int gallery_count_ = 0;
};

QFrame *make_panel(const QString &title, BgrImageView **image_view, QLabel **fps_label,
                   bool show_exit_button = false)
{
    auto *frame = new QFrame;
    frame->setObjectName("robotics_panel");
    frame->setStyleSheet(
        "QFrame#robotics_panel {"
        "background-color: #13151a;"
        "border: 1px solid #3a404c;"
        "border-radius: 8px;"
        "}");

    auto *vbox = new QVBoxLayout(frame);
    vbox->setContentsMargins(0, 0, 0, 0);
    vbox->setSpacing(0);

    auto *header = new QHBoxLayout;
    header->setContentsMargins(0, 0, 0, 0);
    header->setSpacing(0);

    auto *title_label = new QLabel(title);
    title_label->setObjectName("robotics_panel_title");
    title_label->setAlignment(Qt::AlignLeft | Qt::AlignVCenter);
    title_label->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    title_label->setFocusPolicy(Qt::NoFocus);
    title_label->setStyleSheet(
        "QLabel#robotics_panel_title {"
        "color: #eceef4;"
        "font-size: 13px;"
        "font-weight: 600;"
        "padding: 8px 12px 6px 12px;"
        "background-color: #1e222b;"
        "border-bottom: 1px solid #353b48;"
        "border-top-left-radius: 7px;"
        "}");

    auto *fps = new QLabel("-");
    fps->setObjectName("robotics_panel_fps");
    fps->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    fps->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Fixed);
    fps->setFocusPolicy(Qt::NoFocus);
    fps->setStyleSheet(QString(
        "QLabel#robotics_panel_fps {"
        "color: #9aa3b2;"
        "font-size: 12px;"
        "font-weight: 500;"
        "font-family: monospace;"
        "padding: 8px 12px 6px 8px;"
        "background-color: #1e222b;"
        "border-bottom: 1px solid #353b48;"
        "%1"
        "}").arg(show_exit_button ? "" : "border-top-right-radius: 7px;"));

    header->addWidget(title_label, 1);
    header->addWidget(fps, 0);

    if (show_exit_button)
    {
        auto *exit_area = new QWidget;
        exit_area->setObjectName("robotics_panel_exit_area");
        exit_area->setFixedWidth(kExitButtonW + 2 * kExitButtonMargin);
        exit_area->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Expanding);
        exit_area->setStyleSheet(
            "QWidget#robotics_panel_exit_area {"
            "background-color: #1e222b;"
            "border-bottom: 1px solid #353b48;"
            "border-top-right-radius: 7px;"
            "}");

        auto *exit_layout = new QHBoxLayout(exit_area);
        exit_layout->setContentsMargins(kExitButtonMargin, 0, kExitButtonMargin, 0);
        exit_layout->setSpacing(0);

        auto *exit_button = new QPushButton("X");
        exit_button->setFixedSize(kExitButtonW, kExitButtonH);
        exit_button->setFocusPolicy(Qt::NoFocus);
        exit_button->setToolTip("Exit");
        exit_button->setStyleSheet(
            "QPushButton {"
            "color: #cccccc;"
            "background-color: #2d2d30;"
            "border: 1px solid #3c3c3c;"
            "border-radius: 6px;"
            "font-size: 13px;"
            "font-weight: bold;"
            "}"
            "QPushButton:hover {"
            "background-color: #3a3a3d;"
            "color: #ffffff;"
            "}"
            "QPushButton:pressed {"
            "background-color: #005a9e;"
            "}");
        QObject::connect(exit_button, &QPushButton::clicked, frame, [frame]() {
            frame->window()->close();
        });
        exit_layout->addWidget(exit_button, 0, Qt::AlignCenter);
        header->addWidget(exit_area);
    }
    vbox->addLayout(header);

    auto *image = new BgrImageView;
    image->setStyleSheet(
        "background-color: #0c0d10;"
        "border-bottom-left-radius: 7px;"
        "border-bottom-right-radius: 7px;");
    vbox->addWidget(image, 1);

    *image_view = image;
    *fps_label = fps;
    return frame;
}

QFrame *make_gallery_panel(GalleryOverlayView **gallery_view, QLabel **fps_label)
{
    auto *frame = new QFrame;
    frame->setObjectName("robotics_panel");
    frame->setStyleSheet(
        "QFrame#robotics_panel {"
        "background-color: #13151a;"
        "border: 1px solid #3a404c;"
        "border-radius: 8px;"
        "}");

    auto *vbox = new QVBoxLayout(frame);
    vbox->setContentsMargins(0, 0, 0, 0);
    vbox->setSpacing(0);

    auto *header = new QHBoxLayout;
    header->setContentsMargins(0, 0, 0, 0);
    header->setSpacing(0);

    auto *title_label = new QLabel("Face ID Gallery");
    title_label->setObjectName("robotics_panel_title");
    title_label->setAlignment(Qt::AlignLeft | Qt::AlignVCenter);
    title_label->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    title_label->setFocusPolicy(Qt::NoFocus);
    title_label->setStyleSheet(
        "QLabel#robotics_panel_title {"
        "color: #eceef4;"
        "font-size: 13px;"
        "font-weight: 600;"
        "padding: 8px 12px 6px 12px;"
        "background-color: #1e222b;"
        "border-bottom: 1px solid #353b48;"
        "border-top-left-radius: 7px;"
        "}");

    auto *fps = new QLabel("");
    fps->setObjectName("robotics_panel_fps");
    fps->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    fps->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Fixed);
    fps->setFocusPolicy(Qt::NoFocus);
    fps->setStyleSheet(
        "QLabel#robotics_panel_fps {"
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

    auto *image = new GalleryOverlayView;
    image->setStyleSheet(
        "background-color: #0c0d10;"
        "border-bottom-left-radius: 7px;"
        "border-bottom-right-radius: 7px;");
    vbox->addWidget(image, 1);

    *gallery_view = image;
    *fps_label = fps;
    return frame;
}

class RoboticsQtWindow final : public QMainWindow
{
public:
    RoboticsQtWindow(const AppOptions &options, cv::VideoCapture *cap,
                     FaceRecognitionVideoSession *face_session, PoseSegVideoSession *pose_session,
                     MobedVideoSession *mobed_session, SlidingWindowInferenceFps *face_fps,
                     SlidingWindowInferenceFps *pose_fps, SlidingWindowInferenceFps *mobed_fps,
                     cv::Mat mobed_static_picture)
        : options_(options),
          cap_(cap),
          face_session_(face_session),
          pose_session_(pose_session),
          mobed_session_(mobed_session),
          face_fps_(face_fps),
          pose_fps_(pose_fps),
          mobed_fps_(mobed_fps)
    {
        setWindowTitle("robotics_demo");

        auto *central = new QWidget;
        central->setFocusPolicy(Qt::StrongFocus);
        central->setStyleSheet("background-color: #0c0d10;");
        setCentralWidget(central);

        auto *grid = new QGridLayout(central);
        grid->setSpacing(0);
        grid->setContentsMargins(0, 0, 0, 0);
        grid->setColumnStretch(0, 1);
        grid->setColumnStretch(1, 1);
        grid->setRowStretch(0, 1);
        grid->setRowStretch(1, 1);

        grid->addWidget(make_panel(kPoseSegmentationWindowTitle, &pose_view_, &pose_fps_label_), 0, 0);
        grid->addWidget(make_panel(kFaceRecognitionWindowTitle, &face_view_, &face_fps_label_,
                                   options_.showExitButton), 0, 1);
        grid->addWidget(make_panel(kMobedDetectionWindowTitle, &mobed_view_, &mobed_fps_label_), 1, 0);
        grid->addWidget(make_gallery_panel(&gallery_view_, &gallery_fps_label_), 1, 1);

        if (gallery_view_ != nullptr && !mobed_static_picture.empty())
        {
            gallery_view_->SetBaseFrame(std::move(mobed_static_picture));
        }

        frame_timer_ = new QTimer(this);
        frame_timer_->setTimerType(Qt::PreciseTimer);
        connect(frame_timer_, &QTimer::timeout, this, [this]() {
            ProcessFrameTick();
        });
        frame_timer_->start(options_.cameraInput ? 1 : std::max(1, 1000 / kVideoPlaybackFps));
    }

    ~RoboticsQtWindow() override
    {
        Shutdown();
    }

protected:
    void keyPressEvent(QKeyEvent *event) override
    {
        int key = -1;
        if (event->key() == Qt::Key_Escape)
        {
            key = 27;
        }
        else if (!event->text().isEmpty())
        {
            key = event->text().at(0).toLatin1();
        }

        if (key > 0 && face_session_ != nullptr)
        {
            face_session_->SubmitKey(key);
        }

        if (is_exit_key(key))
        {
            close();
            return;
        }

        QMainWindow::keyPressEvent(event);
    }

    void closeEvent(QCloseEvent *event) override
    {
        Shutdown();
        event->accept();
    }

    void showEvent(QShowEvent *event) override
    {
        QMainWindow::showEvent(event);
        if (centralWidget() != nullptr)
        {
            centralWidget()->setFocus(Qt::OtherFocusReason);
        }
        if (!ready_notified)
        {
            ready_notified = true;
            notify_launcher_ready();
        }
    }

private:
    void ProcessFrameTick()
    {
        if (closing_.load(std::memory_order_relaxed))
        {
            return;
        }
        if (face_session_ != nullptr && face_session_->IsStopRequested())
        {
            close();
            return;
        }

        cv::Mat frame;
        (*cap_) >> frame;
        if (frame.empty())
        {
            if (options_.cameraInput)
            {
                RequestStopAndClose();
                return;
            }

            cap_->set(cv::CAP_PROP_POS_FRAMES, 0);
            (*cap_) >> frame;
            if (frame.empty())
            {
                if (!cap_->open(options_.videoFile))
                {
                    std::cout << "Error: failed to reopen video file for loop." << std::endl;
                    RequestStopAndClose();
                    return;
                }
                cap_->set(cv::CAP_PROP_FPS, static_cast<double>(kVideoPlaybackFps));
                (*cap_) >> frame;
            }
            if (frame.empty())
            {
                std::cout << "Error: video file has no decodable frames." << std::endl;
                RequestStopAndClose();
                return;
            }
        }

        const std::string mode_label = options_.cameraInput ? "LIVE" : "VIDEO";
        const std::string pose_inf_text = pose_fps_ != nullptr ? pose_fps_->TickUi() : "0.0 inf/s";
        const std::string face_inf_text = face_fps_ != nullptr ? face_fps_->TickUi() : "0.0 inf/s";
        const std::string mobed_inf_text = mobed_fps_ != nullptr ? mobed_fps_->TickUi() : "0.0 inf/s";

        if (pose_fps_label_ != nullptr)
        {
            set_label_text_if_changed(pose_fps_label_, panel_rate_text(mode_label, pose_inf_text));
        }
        if (face_fps_label_ != nullptr)
        {
            set_label_text_if_changed(face_fps_label_, panel_rate_text(mode_label, face_inf_text));
        }
        if (mobed_fps_label_ != nullptr)
        {
            set_label_text_if_changed(mobed_fps_label_, panel_rate_text(mode_label, mobed_inf_text));
        }

        if (pose_session_ != nullptr)
        {
            pose_session_->SubmitFrame(frame.clone());
            ConsumePoseDisplay();
        }
        ConsumeFaceDisplay();

        if (mobed_session_ != nullptr)
        {
            mobed_session_->SubmitFrame(frame.clone());
            ConsumeMobedDisplay();
        }

        if (face_session_ != nullptr)
        {
            face_session_->SubmitFrame(std::move(frame));
        }
    }

    void ConsumeFaceDisplay()
    {
        if (face_session_ == nullptr)
        {
            return;
        }

        cv::Mat view;
        cv::Mat gallery_view;
        if (!face_session_->ConsumeDisplay(&view, &gallery_view))
        {
            return;
        }

        if (!view.empty() && face_view_ != nullptr)
        {
            resize_camera_display_if_needed(options_.cameraInput, &view);
            face_view_->SetFrame(std::move(view));
        }
        if (gallery_view_ != nullptr)
        {
            gallery_view_->SetGalleryFrame(std::move(gallery_view));
        }
    }

    void ConsumePoseDisplay()
    {
        if (pose_session_ == nullptr || pose_view_ == nullptr)
        {
            return;
        }

        cv::Mat view;
        if (!pose_session_->ConsumeDisplay(&view) || view.empty())
        {
            return;
        }

        resize_camera_display_if_needed(options_.cameraInput, &view);
        pose_view_->SetFrame(std::move(view));
    }

    void ConsumeMobedDisplay()
    {
        if (mobed_session_ == nullptr || mobed_view_ == nullptr)
        {
            return;
        }

        cv::Mat view;
        if (!mobed_session_->ConsumeDisplay(&view) || view.empty())
        {
            return;
        }

        resize_camera_display_if_needed(options_.cameraInput, &view);
        mobed_view_->SetFrame(std::move(view));
    }

    void RequestStopAndClose()
    {
        if (face_session_ != nullptr)
        {
            face_session_->RequestStop();
        }
        close();
    }

    void Shutdown()
    {
        bool expected = false;
        if (!closing_.compare_exchange_strong(expected, true))
        {
            return;
        }

        if (frame_timer_ != nullptr)
        {
            frame_timer_->stop();
        }

        if (face_session_ != nullptr)
        {
            face_session_->RequestStop();
        }
        if (pose_session_ != nullptr)
        {
            pose_session_->RequestStop();
        }
        if (mobed_session_ != nullptr)
        {
            mobed_session_->RequestStop();
        }

        if (face_session_ != nullptr)
        {
            face_session_->Join();
        }
        if (pose_session_ != nullptr)
        {
            pose_session_->Join();
        }
        if (mobed_session_ != nullptr)
        {
            mobed_session_->Join();
        }
        if (cap_ != nullptr)
        {
            cap_->release();
        }
    }

    AppOptions options_;
    cv::VideoCapture *cap_ = nullptr;
    FaceRecognitionVideoSession *face_session_ = nullptr;
    PoseSegVideoSession *pose_session_ = nullptr;
    MobedVideoSession *mobed_session_ = nullptr;
    SlidingWindowInferenceFps *face_fps_ = nullptr;
    SlidingWindowInferenceFps *pose_fps_ = nullptr;
    SlidingWindowInferenceFps *mobed_fps_ = nullptr;
    BgrImageView *pose_view_ = nullptr;
    BgrImageView *face_view_ = nullptr;
    BgrImageView *mobed_view_ = nullptr;
    GalleryOverlayView *gallery_view_ = nullptr;
    QLabel *pose_fps_label_ = nullptr;
    QLabel *face_fps_label_ = nullptr;
    QLabel *mobed_fps_label_ = nullptr;
    QLabel *gallery_fps_label_ = nullptr;
    QTimer *frame_timer_ = nullptr;
    std::atomic<bool> closing_{false};
};

}  // namespace

int main(int argc, char *argv[])
{
    AppOptions options;
    if (!parse_args(argc, argv, &options))
    {
        return -1;
    }

    if (options.mps0ModelPath.empty() || options.mps1ModelPath.empty() ||
        options.fdModelPath.empty() || options.lmModelPath.empty() || options.frModelPath.empty() ||
        options.detModelPath.empty())
    {
        std::cout << "[NOTICE] required model paths are missing" << std::endl;
        help();
        return -1;
    }

    EncryptedModelEngine ie_mps0(options.mps0ModelPath);
    EncryptedModelEngine ie_mps1(options.mps1ModelPath);
    EncryptedModelEngine ie_fd(options.fdModelPath);
    EncryptedModelEngine ie_lm(options.lmModelPath);
    EncryptedModelEngine ie_fr(options.frModelPath);
    EncryptedModelEngine ie_mobed(options.detModelPath);
    MobedVideoSession mobed_session(ie_mobed.get());
    if (!mobed_session.ok())
    {
        std::cout << "Error: MobED detector failed to initialize." << std::endl;
        return -1;
    }
    PoseSegVideoSession pose_session(ie_mps0.get(), ie_mps1.get());

    std::shared_ptr<EncryptedModelEngine> ie_gender;
    if (options.classifierGender)
    {
        ie_gender = std::make_shared<EncryptedModelEngine>(options.genderModelPath);
    }

    dxrt::InferenceEngine *gender_engine = ie_gender ? ie_gender->get() : nullptr;
    SsdParam fd_config = make_fd_config();

    if (!options.videoFile.empty() || options.cameraInput)
    {
        cv::VideoCapture cap;
        if (!open_video_capture(options, &cap))
        {
            return -1;
        }

        cv::Mat mobed_static_picture = load_mobed_static_picture_resized();

        SlidingWindowInferenceFps face_inference_fps;
        SlidingWindowInferenceFps pose_inference_fps;
        SlidingWindowInferenceFps mobed_inference_fps;

        FaceRecognitionVideoSession face_session(ie_fd.get(), ie_lm.get(), ie_fr.get(), gender_engine, fd_config,
                                                 options.dbPath, options.frThreshold);
        face_session.SetOnInferenceDone([&face_inference_fps]() { face_inference_fps.RecordInferenceComplete(); });
        pose_session.SetOnInferenceDone([&pose_inference_fps]() { pose_inference_fps.RecordInferenceComplete(); });
        mobed_session.SetOnInferenceDone([&mobed_inference_fps]() { mobed_inference_fps.RecordInferenceComplete(); });
        face_session.Start();
        pose_session.Start();
        mobed_session.Start();

        QApplication app(argc, argv);
        RoboticsQtWindow window(options, &cap, &face_session, &pose_session, &mobed_session,
                                &face_inference_fps, &pose_inference_fps, &mobed_inference_fps,
                                std::move(mobed_static_picture));
        window.showFullScreen();
        return app.exec();
    }
    else
    {
        std::cout << "Error: either --video or --camera must be provided." << std::endl;
        help();
        return -1;
    }

    return 0;
}
