#include <algorithm>
#include <array>
#include <cmath>
#include <chrono>
#include <cstring>
#include <functional>
#include <future>
#include <iostream>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <unordered_map>
#include <unordered_set>

#include <dxrt/dxrt_api.h>
#include "face_recognition.h"
#include "ssd.h"
#include "utils.h"

#ifndef UNUSEDVAR
#define UNUSEDVAR(x) (void)(x);
#endif

#define FD_INPUT_WIDTH 512
#define FD_INPUT_HEIGHT 512
#define FR_INPUT_WIDTH 112
#define FR_INPUT_HEIGHT 112

#define FR_THRESHOLD 0.5

namespace
{
constexpr size_t kMaxFacesFullPipeline = 8;
constexpr int kFaceRecogMinFrameInterval = 2;
}  // namespace

static std::vector<std::string> age_classes = {
    "0-3", "14-24", "25-36", "37-48", "4-6", "49-58", "59-100", "7-13"
};

static std::vector<std::string> gender_classes = {"F", "M"};

struct TrackerWorkerState
{
    std::mutex mutex;
    std::condition_variable cv;
    cv::Mat frame;
    cv::Mat view;
    cv::Mat gallery_view;
    bool has_frame = false;
    bool has_rendered_frame = false;
    bool stop_requested = false;
    int pending_key = -1;
};

std::vector<int> age_gender_post_processing(dxrt::TensorPtrs outputs)
{
    std::string age_name = "onnx::Concat_568", gender_name = "onnx::Concat_572";
    std::vector<int> result(2);
    for (auto &output: outputs)
    {
        int max_idx = 0;
        float* data = (float*)output->data();
        if(output->name() == age_name)
        {
            for(int i=0;i<8;i++){
                if(data[i] > data[max_idx])
                    max_idx = i;
            }
            result[0] = max_idx;
        }
        else if (output->name() == gender_name)
        {
            for(int i=0;i<2;i++){
                if(data[i] > data[max_idx])
                    max_idx = i;
            }
            result[1] = max_idx;
        }
    }
    return result;
}

std::vector<cv::Point2f> run_landmark(dxrt::InferenceEngine *ie, cv::Mat image, cv::Rect crop)
{
    cv::Mat fl_cropped = image(crop);
    cv::Mat fl_input = preprocess(fl_cropped, cv::Size(192, 192));

    auto fl_tensors = ie->Run(fl_input.data);
    float *fl_data = (float *)fl_tensors[0]->data();

    auto landmark = get_landmark(fl_data, fl_cropped.cols, fl_cropped.rows, crop.x, crop.y);
    return landmark;
}

FaceData run_recognition(dxrt::InferenceEngine *ie, cv::Mat image, int id)
{
    cv::Mat input = preprocess(image, cv::Size(FR_INPUT_WIDTH, FR_INPUT_HEIGHT));
    auto tensors = ie->Run(input.data);
    float *feature_vector = (float *)tensors[0]->data();
    return FaceData(id, image, feature_vector);
}

std::vector<FaceData> get_gallary(std::string dir, Ssd *detector, dxrt::InferenceEngine *ie_fd, dxrt::InferenceEngine *ie_fl, dxrt::InferenceEngine *ie_fr)
{
    std::vector<FaceData> gallary;
    if (dir == "")
    {
        cv::Mat face_image = cv::Mat::zeros(FR_INPUT_HEIGHT, FR_INPUT_WIDTH, CV_8UC3);
        auto face_data = run_recognition(ie_fr, face_image, 0);
        gallary.emplace_back(face_data);
    }
    else
    {
        auto files = dxrt::GetFileList(dir);
        for (auto &file_name : files)
        {
            std::string file_path = dir + "/" + file_name;
            cv::Mat image = cv::imread(file_path);

            cv::Mat fd_input = preprocess(image, cv::Size(FD_INPUT_WIDTH, FD_INPUT_HEIGHT));

            auto fd_tensor = ie_fd->Run(fd_input.data);
            auto fd_result = detector->PostProc(fd_tensor);
            if (fd_result.size() == 0)
            {
                continue;
            }
            else
            {
                auto detected = fd_result[0];
                cv::Rect rect = get_rect(detected.box, image.cols, image.rows);
                auto landmark = run_landmark(ie_fl, image, rect);
                cv::Mat fr_warped = warp(image, landmark);
                auto face_data = run_recognition(ie_fr, fr_warped, 0);
                gallary.emplace_back(face_data);
            }
        }
    }
    return gallary;
}

cv::Mat make_gallary_view(const std::vector<FaceData> &gallary, bool empty_db)
{
    std::vector<cv::Mat> gallary_images;
    const size_t first_visible_index = empty_db ? 1U : 0U;
    if (gallary.size() <= first_visible_index)
    {
        return {};
    }

    gallary_images.reserve(gallary.size() - first_visible_index);
    for (size_t i = first_visible_index; i < gallary.size(); i++)
    {
        auto face_image = gallary[i].image.clone();
        if (face_image.empty())
        {
            continue;
        }
        gallary_images.emplace_back(std::move(face_image));
    }

    cv::Mat gallary_view;
    if (gallary_images.empty())
    {
        return gallary_view;
    }
    cv::hconcat(gallary_images, gallary_view);
    return gallary_view;
}

void run_tracker_video_sync(dxrt::InferenceEngine *ie_fd, dxrt::InferenceEngine *ie_fl,
                            dxrt::InferenceEngine *ie_fr, dxrt::InferenceEngine *ie_gender,
                            TrackerWorkerState *worker_state, SsdParam fdCfg, std::string dbPath,
                            float frThreshold, std::function<void()> on_inference_done)
{
    auto fdDataInfo = ie_fd->GetOutputs();
    Ssd detector = Ssd(fdCfg, fdDataInfo);
    Tracker tracker(0.25);

    int face_image_idx = 0;
    int selected = 0;
    std::vector<int> face_ids;
    std::vector<cv::Mat> face_images;
    for (int i = 0; i < 20; i++)
    {
        face_ids.emplace_back(-1);
        face_images.emplace_back(cv::Mat::zeros(FR_INPUT_HEIGHT, FR_INPUT_WIDTH, CV_8UC3));
    }

    auto gallary = get_gallary(dbPath, &detector, ie_fd, ie_fl, ie_fr);

    int frame_seq = 0;
    std::unordered_map<int, std::pair<std::array<float, 512>, int>> fr_feature_cache;

    bool running = true;
    while (running)
    {
        ++frame_seq;

        cv::Mat frame;
        {
            std::unique_lock<std::mutex> lock(worker_state->mutex);
            worker_state->cv.wait(lock, [worker_state]() {
                return worker_state->has_frame || worker_state->stop_requested;
            });

            if (worker_state->stop_requested && !worker_state->has_frame)
            {
                break;
            }

            frame = std::move(worker_state->frame);
            worker_state->has_frame = false;
        }

        if (frame.empty())
        {
            break;
        }

        cv::Mat view = frame.clone();

        cv::Mat fd_input = preprocess(frame, cv::Size(FD_INPUT_WIDTH, FD_INPUT_HEIGHT));
        auto fd_tensors = ie_fd->Run(fd_input.data);
        auto fd_result = detector.PostProc(fd_tensors);

        std::vector<cv::Rect> D;
        for (size_t i = 0; i < fd_result.size(); i++)
        {
            auto detected = fd_result[i];
            cv::Rect rect = get_rect(detected.box, frame.cols, frame.rows);

            D.emplace_back(rect);

            // Visualization
            cv::Scalar color(0, 255, 0);
            cv::rectangle(view, rect, color, 2);
            cv::putText(view, detected.labelname, cv::Point(rect.x + 8, rect.y + 16), 0, 0.5, color, 1);
        }
        std::vector<bool> visited_indices;
        for (size_t i = 0; i < gallary.size(); i++)
        {
            visited_indices.emplace_back(false);
        }

        // Tracking
        tracker.run(D);

        std::vector<size_t> track_order(tracker.T.size());
        std::iota(track_order.begin(), track_order.end(), 0U);
        std::sort(track_order.begin(), track_order.end(), [&](size_t a, size_t b) {
            return tracker.T[a].box.area() > tracker.T[b].box.area();
        });

        std::unordered_set<size_t> full_pipeline_indices;
        const size_t heavy_limit = std::min(kMaxFacesFullPipeline, track_order.size());
        for (size_t r = 0; r < heavy_limit; ++r)
        {
            full_pipeline_indices.insert(track_order[r]);
        }

        face_image_idx = 0;
        for (size_t i = 0; i < tracker.T.size(); i++)
        {
            auto tracked = tracker.T[i];

            if (full_pipeline_indices.find(i) == full_pipeline_indices.end())
            {
                const cv::Scalar light(0, 165, 255);
                cv::rectangle(view, tracked.box, light, 2);
                cv::putText(view, "...", cv::Point(tracked.box.x + 4, tracked.box.y + 16), cv::FONT_HERSHEY_SIMPLEX,
                            0.5, light, 1);
                continue;
            }

            auto landmark = run_landmark(ie_fl, frame, tracker.T[i].box);

            visualize_landmark(view, landmark);

            cv::Mat fr_warped = warp(frame, landmark);

            face_image_idx = static_cast<int>(i);
            face_image_idx %= static_cast<int>(face_images.size());
            face_images[face_image_idx] = fr_warped;
            face_ids[face_image_idx] = tracker.T[i].id;

            std::array<float, 512> feat_buf{};
            int8_t age_idx = -1;
            int8_t gender_idx = -1;

            const auto cache_it = fr_feature_cache.find(tracked.id);
            const bool use_cached_fr =
                cache_it != fr_feature_cache.end() && (frame_seq - cache_it->second.second) < kFaceRecogMinFrameInterval;

            if (!use_cached_fr)
            {
                auto face_data = run_recognition(ie_fr, face_images[face_image_idx], face_ids[face_image_idx]);
                std::memcpy(feat_buf.data(), face_data.feature_vector, sizeof(feat_buf));
                auto &slot = fr_feature_cache[tracked.id];
                slot.first = feat_buf;
                slot.second = frame_seq;
            }
            else
            {
                feat_buf = cache_it->second.first;
            }

            float *feature_vector = feat_buf.data();
            float similarity_max = 0;
            int similarity_max_index = -1;
            for (size_t j = 0; j < gallary.size(); j++)
            {
                float similarity = cos_sim(gallary[j].feature_vector, feature_vector, 512);
                if (similarity > similarity_max && visited_indices[j] == false)
                {
                    similarity_max = similarity;
                    similarity_max_index = j;
                }
            }
	    
            cv::Scalar color;
            if(face_ids[selected] != tracked.id) 
            {
                cv::Scalar temp(0, 255, 0);
                color = temp;
            }
            else 
            {
                cv::Scalar temp(0, 0, 255);
                color = temp;
            }

            std::string id_str = "id ";
            if (similarity_max > frThreshold)
            {
                id_str += std::to_string(similarity_max_index);
                visited_indices[similarity_max_index] = true;
            }
            else
            {
                id_str += "?";
            }
            std::string caption_str = id_str + " (" + std::to_string(similarity_max) + ")";

            if (age_idx > 0)
            {
                caption_str += ", age " + age_classes[static_cast<size_t>(age_idx)] + ", (" +
                               gender_classes[static_cast<size_t>(gender_idx)] + ")";
            }
            int txtBaseline = 0;
            auto textSize = cv::getTextSize(caption_str, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &txtBaseline);
            cv::rectangle(view, cv::Point(tracked.box.x, tracked.box.y - textSize.height -5), cv::Point(tracked.box.x + textSize.width + 10, tracked.box.y + 5), color, -1);
            cv::putText(view, caption_str, cv::Point(tracked.box.x + 5, tracked.box.y), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2);
	    }

        //std::cout << "detector result: " << fd_result.size() << " / tracker : " << tracker.T.size() << " / gallary: " << gallary.size() << std::endl;
        //std::cout << "tracker.T.size(): " << tracker.T.size() << " / faceidx: " << face_image_idx << " / selected: " << selected << std::endl;

        int key = -1;
        {
            std::lock_guard<std::mutex> lock(worker_state->mutex);
            worker_state->view = std::move(view);
            worker_state->gallery_view = make_gallary_view(gallary, dbPath.empty());
            worker_state->has_rendered_frame = true;
            key = worker_state->pending_key;
            worker_state->pending_key = -1;
        }

        if (on_inference_done)
        {
            on_inference_done();
        }

        if (key > 0)
        {
            std::cout << "Key: " << key << std::endl;
        }
        switch (key)
        {
        case 27:
        case 'Q':
        case 'q':
            running = false;
            break;
        case 'A':
        case 'a':
        {
            auto face_data = run_recognition(ie_fr, face_images[selected], 0);
            gallary.emplace_back(face_data);
            break;
        }
        case 'D':
        case 'd':
        {
            if (gallary.size() > 1)
            {
                gallary.pop_back();
            }
            break;
        }
        case 'S':
        case 's':
	    {
	        ++selected %= tracker.T.size();
            //if (tracker.T.size())
            //{
            //    tracker.T.erase(tracker.T.begin());
            //}
        }
        }

        if (!running)
        {
            std::lock_guard<std::mutex> lock(worker_state->mutex);
            worker_state->stop_requested = true;
        }
    }

    worker_state->cv.notify_all();
}

FaceRecognitionVideoSession::FaceRecognitionVideoSession(dxrt::InferenceEngine *ie_fd,
                                                         dxrt::InferenceEngine *ie_fl,
                                                         dxrt::InferenceEngine *ie_fr,
                                                         dxrt::InferenceEngine *ie_gender,
                                                         SsdParam fdCfg, std::string db_path,
                                                         float fr_threshold)
    : ie_fd_(ie_fd),
      ie_fl_(ie_fl),
      ie_fr_(ie_fr),
      ie_gender_(ie_gender),
      fdCfg_(std::move(fdCfg)),
      db_path_(std::move(db_path)),
      fr_threshold_(fr_threshold),
      worker_state_(std::make_unique<TrackerWorkerState>())
{
}

FaceRecognitionVideoSession::~FaceRecognitionVideoSession()
{
    RequestStop();
    Join();
}

void FaceRecognitionVideoSession::SetOnInferenceDone(std::function<void()> cb)
{
    on_inference_done_ = std::move(cb);
}

void FaceRecognitionVideoSession::Start()
{
    if (worker_thread_.joinable())
    {
        return;
    }

    std::function<void()> done_cb = on_inference_done_;
    worker_thread_ = std::thread([this, done_cb]() {
        run_tracker_video_sync(ie_fd_, ie_fl_, ie_fr_, ie_gender_, worker_state_.get(), fdCfg_,
                               db_path_, fr_threshold_, done_cb);
    });
}

void FaceRecognitionVideoSession::SubmitFrame(cv::Mat frame)
{
    {
        std::lock_guard<std::mutex> lock(worker_state_->mutex);
        if (worker_state_->stop_requested)
        {
            return;
        }
        worker_state_->frame = std::move(frame);
        worker_state_->has_frame = !worker_state_->frame.empty();
    }
    worker_state_->cv.notify_one();
}

void FaceRecognitionVideoSession::SubmitKey(int key)
{
    std::lock_guard<std::mutex> lock(worker_state_->mutex);
    worker_state_->pending_key = key;
}

bool FaceRecognitionVideoSession::ConsumeDisplay(cv::Mat *view, cv::Mat *gallery_view)
{
    std::lock_guard<std::mutex> lock(worker_state_->mutex);
    if (!worker_state_->has_rendered_frame)
    {
        return false;
    }

    if (view != nullptr)
    {
        *view = std::move(worker_state_->view);
    }
    if (gallery_view != nullptr)
    {
        *gallery_view = std::move(worker_state_->gallery_view);
    }
    worker_state_->has_rendered_frame = false;
    return true;
}

bool FaceRecognitionVideoSession::IsStopRequested() const
{
    std::lock_guard<std::mutex> lock(worker_state_->mutex);
    return worker_state_->stop_requested;
}

void FaceRecognitionVideoSession::RequestStop()
{
    {
        std::lock_guard<std::mutex> lock(worker_state_->mutex);
        worker_state_->stop_requested = true;
    }
    worker_state_->cv.notify_all();
}

void FaceRecognitionVideoSession::Join()
{
    if (worker_thread_.joinable())
    {
        worker_thread_.join();
    }
}
