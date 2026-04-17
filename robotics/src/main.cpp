#include <chrono>
#include <iostream>
#include <memory>
#include <cstdio>
#include <string>
#include <thread>

#include <opencv2/opencv.hpp>
#include <dxrt/dxrt_api.h>

#include "face_recognition.h"
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
const char *kMobedDetectionWindowTitle = "MobED Object Detection";
const char *kMobedStaticPictureWindowTitle = "MobED picture";
const char *kGalleryWindowTitle = "gallary";

/** No toolbar/status strip under title (default is WINDOW_GUI_EXPANDED). */
constexpr int kImshowWindowFlags = cv::WINDOW_GUI_NORMAL | cv::WINDOW_AUTOSIZE;

constexpr int kMobedStaticPictureW = 960;
constexpr int kMobedStaticPictureH = 530;
constexpr int kMobedStaticPictureScreenRefW = 1920;
constexpr int kMobedStaticPictureScreenRefH = 1080;
constexpr int kMobedStaticPictureMargin = 16;

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
    const std::string path = std::string(PROJECT_ROOT_DIR) + "/assets/mobed.jpg";
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

/** When using camera, some drivers ignore CAP_PROP size; normalize display to 960x540 for imshow. */
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

void draw_fps_overlay(cv::Mat *image, const std::string &mode_label, const std::string &fps_text)
{
    if (image == nullptr || image->empty())
    {
        return;
    }

    constexpr int kMargin = 18;
    constexpr int kPanelHeight = 56;
    constexpr int kPanelPaddingX = 18;
    constexpr int kAccentWidth = 6;
    constexpr int kShadowOffset = 4;
    constexpr double kTitleScale = 0.45;
    constexpr double kValueScale = 0.72;
    const int panel_width = 182;
    const int x = std::max(0, image->cols - panel_width - kMargin);
    const int y = kMargin;

    if (x + panel_width > image->cols || y + kPanelHeight > image->rows)
    {
        return;
    }

    const cv::Rect shadow_rect(x + kShadowOffset, y + kShadowOffset, panel_width, kPanelHeight);
    cv::rectangle(*image, shadow_rect, cv::Scalar(20, 20, 20), cv::FILLED);

    cv::Mat overlay = image->clone();
    const cv::Rect panel_rect(x, y, panel_width, kPanelHeight);
    cv::rectangle(overlay, panel_rect, cv::Scalar(30, 34, 42), cv::FILLED);
    cv::rectangle(overlay, cv::Rect(x, y, kAccentWidth, kPanelHeight), cv::Scalar(70, 190, 255),
                  cv::FILLED);
    cv::addWeighted(overlay, 0.78, *image, 0.22, 0.0, *image);

    cv::putText(*image, mode_label, cv::Point(x + kPanelPaddingX, y + 18),
                cv::FONT_HERSHEY_DUPLEX, kTitleScale, cv::Scalar(215, 220, 228), 1, cv::LINE_AA);
    cv::putText(*image, fps_text, cv::Point(x + kPanelPaddingX, y + 41),
                cv::FONT_HERSHEY_DUPLEX, kValueScale, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
}

void consume_face_display(FaceRecognitionVideoSession *face_session, bool camera_input,
                          const std::string &mode_label, const std::string &fps_text)
{
    cv::Mat view;
    cv::Mat gallery_view;
    if (!face_session->ConsumeDisplay(&view, &gallery_view))
    {
        return;
    }

    if (!view.empty())
    {
        resize_camera_display_if_needed(camera_input, &view);
        draw_fps_overlay(&view, mode_label, fps_text);
        cv::imshow(kFaceRecognitionWindowTitle, view);
    }
    if (!gallery_view.empty())
    {
        //resize_camera_display_if_needed(camera_input, &gallery_view);
        cv::imshow(kGalleryWindowTitle, gallery_view);
    }
}

void consume_pose_seg_display(PoseSegVideoSession *pose_session, bool camera_input,
                              const std::string &mode_label, const std::string &fps_text)
{
    if (pose_session == nullptr)
    {
        return;
    }

    cv::Mat view;
    if (!pose_session->ConsumeDisplay(&view) || view.empty())
    {
        return;
    }

    resize_camera_display_if_needed(camera_input, &view);
    draw_fps_overlay(&view, mode_label, fps_text);
    cv::imshow(kPoseSegmentationWindowTitle, view);
}

void consume_mobed_display(MobedVideoSession *mobed_session, bool camera_input,
                           const std::string &mode_label, const std::string &inf_text)
{
    if (mobed_session == nullptr)
    {
        return;
    }

    cv::Mat view;
    if (!mobed_session->ConsumeDisplay(&view) || view.empty())
    {
        return;
    }

    resize_camera_display_if_needed(camera_input, &view);
    draw_fps_overlay(&view, mode_label, inf_text);
    cv::imshow(kMobedDetectionWindowTitle, view);
}

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

    dxrt::InferenceEngine ie_mps0(options.mps0ModelPath);
    dxrt::InferenceEngine ie_mps1(options.mps1ModelPath);
    dxrt::InferenceEngine ie_fd(options.fdModelPath);
    dxrt::InferenceEngine ie_lm(options.lmModelPath);
    dxrt::InferenceEngine ie_fr(options.frModelPath);
    dxrt::InferenceEngine ie_mobed(options.detModelPath);
    MobedVideoSession mobed_session(&ie_mobed);
    if (!mobed_session.ok())
    {
        std::cout << "Error: MobED detector failed to initialize." << std::endl;
        return -1;
    }
    PoseSegVideoSession pose_session(&ie_mps0, &ie_mps1);

    std::shared_ptr<dxrt::InferenceEngine> ie_gender;
    if (options.classifierGender)
    {
        ie_gender = std::make_shared<dxrt::InferenceEngine>(options.genderModelPath);
    }

    dxrt::InferenceEngine *gender_engine = ie_gender.get();
    SsdParam fd_config = make_fd_config();

    if (!options.videoFile.empty() || options.cameraInput)
    {
        cv::VideoCapture cap;
        if (!open_video_capture(options, &cap))
        {
            return -1;
        }



        cv::namedWindow(kFaceRecognitionWindowTitle, kImshowWindowFlags);
        cv::namedWindow(kGalleryWindowTitle, kImshowWindowFlags);
        cv::namedWindow(kPoseSegmentationWindowTitle, kImshowWindowFlags);
        cv::namedWindow(kMobedDetectionWindowTitle, kImshowWindowFlags);

        cv::moveWindow(kFaceRecognitionWindowTitle, 0, 0);
        cv::moveWindow(kGalleryWindowTitle, 0, 400);
        cv::setWindowProperty(kGalleryWindowTitle, cv::WND_PROP_TOPMOST, 1);
        cv::moveWindow(kPoseSegmentationWindowTitle, 960, 0);
        cv::moveWindow(kMobedDetectionWindowTitle, 0, 550);

        cv::Mat mobed_static_picture = load_mobed_static_picture_resized();
        const bool mobed_static_ok = !mobed_static_picture.empty();
        if (mobed_static_ok)
        {
            cv::namedWindow(kMobedStaticPictureWindowTitle, kImshowWindowFlags);
            cv::moveWindow(kMobedStaticPictureWindowTitle, 960, 550);
        }

        SlidingWindowInferenceFps face_inference_fps;
        SlidingWindowInferenceFps pose_inference_fps;
        SlidingWindowInferenceFps mobed_inference_fps;

        FaceRecognitionVideoSession face_session(&ie_fd, &ie_lm, &ie_fr, gender_engine, fd_config,
                                                 options.dbPath, options.frThreshold);
        face_session.SetOnInferenceDone([&face_inference_fps]() { face_inference_fps.RecordInferenceComplete(); });
        pose_session.SetOnInferenceDone([&pose_inference_fps]() { pose_inference_fps.RecordInferenceComplete(); });
        mobed_session.SetOnInferenceDone([&mobed_inference_fps]() { mobed_inference_fps.RecordInferenceComplete(); });
        face_session.Start();
        pose_session.Start();
        mobed_session.Start();

        while (true)
        {
            if (face_session.IsStopRequested())
            {
                break;
            }

            const auto frame_tick_start = std::chrono::steady_clock::now();

            cv::Mat frame;
            cap >> frame;
            if (frame.empty())
            {
                if (options.cameraInput)
                {
                    face_session.RequestStop();
                    break;
                }

                cap.set(cv::CAP_PROP_POS_FRAMES, 0);
                cap >> frame;
                if (frame.empty())
                {
                    if (!cap.open(options.videoFile))
                    {
                        std::cout << "Error: failed to reopen video file for loop." << std::endl;
                        face_session.RequestStop();
                        break;
                    }
                    cap.set(cv::CAP_PROP_FPS, static_cast<double>(kVideoPlaybackFps));
                    cap >> frame;
                }
                if (frame.empty())
                {
                    std::cout << "Error: video file has no decodable frames." << std::endl;
                    face_session.RequestStop();
                    break;
                }
            }

            const std::string mode_label = options.cameraInput ? "LIVE" : "VIDEO";
            const std::string &pose_inf_text = pose_inference_fps.TickUi();
            const std::string &face_inf_text = face_inference_fps.TickUi();

            pose_session.SubmitFrame(frame.clone());
            consume_pose_seg_display(&pose_session, options.cameraInput, mode_label, pose_inf_text);
            consume_face_display(&face_session, options.cameraInput, mode_label, face_inf_text);

            mobed_session.SubmitFrame(frame.clone());
            consume_mobed_display(&mobed_session, options.cameraInput, mode_label,
                                  mobed_inference_fps.TickUi());

            if (mobed_static_ok)
            {
                cv::imshow(kMobedStaticPictureWindowTitle, mobed_static_picture);
            }

            cv::setWindowProperty(kGalleryWindowTitle, cv::WND_PROP_TOPMOST, 1);

            if (!ready_notified) {
                ready_notified = true;
                notify_launcher_ready();
            }

            int key = cv::waitKey(1);
            if (key > 0)
            {
                face_session.SubmitKey(key);
                if (is_exit_key(key))
                {
                    face_session.RequestStop();
                }
            }

            if (face_session.IsStopRequested())
            {
                break;
            }

            face_session.SubmitFrame(std::move(frame));

            if (!options.cameraInput)
            {
                const auto elapsed = std::chrono::steady_clock::now() - frame_tick_start;
                const auto budget = std::chrono::microseconds(1000000 / kVideoPlaybackFps);
                if (elapsed < budget)
                {
                    std::this_thread::sleep_for(budget - elapsed);
                }
            }
        }

        face_session.RequestStop();
        pose_session.RequestStop();
        mobed_session.RequestStop();
        face_session.Join();
        pose_session.Join();
        mobed_session.Join();
        consume_face_display(&face_session, options.cameraInput,
                             options.cameraInput ? "LIVE" : "PLAYBACK", face_inference_fps.TickUi());
        consume_pose_seg_display(&pose_session, options.cameraInput,
                                 options.cameraInput ? "LIVE" : "PLAYBACK", pose_inference_fps.TickUi());
        consume_mobed_display(&mobed_session, options.cameraInput,
                              options.cameraInput ? "LIVE" : "PLAYBACK", mobed_inference_fps.TickUi());

        if (options.cameraInput)
        {
            cv::destroyWindow(kPoseSegmentationWindowTitle);
        }
        cv::destroyWindow(kFaceRecognitionWindowTitle);
        cv::destroyWindow(kGalleryWindowTitle);
        cv::destroyWindow(kMobedDetectionWindowTitle);
        if (mobed_static_ok)
        {
            cv::destroyWindow(kMobedStaticPictureWindowTitle);
        }
    }
    else
    {
        std::cout << "Error: either --video or --camera must be provided." << std::endl;
        help();
        return -1;
    }

    return 0;
}
