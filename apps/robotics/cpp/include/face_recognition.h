#pragma once

#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include <opencv2/opencv.hpp>
#include <dxrt/dxrt_api.h>

#include "ssd.h"

void run_tracker_video_sync(dxrt::InferenceEngine *ie_fd, dxrt::InferenceEngine *ie_fl,
                            dxrt::InferenceEngine *ie_fr, dxrt::InferenceEngine *ie_gender,
                            struct TrackerWorkerState *worker_state, SsdParam fdCfg, std::string dbPath,
                            float frThreshold, std::function<void()> on_inference_done = {});

class FaceRecognitionVideoSession
{
public:
    FaceRecognitionVideoSession(dxrt::InferenceEngine *ie_fd, dxrt::InferenceEngine *ie_fl,
                                dxrt::InferenceEngine *ie_fr, dxrt::InferenceEngine *ie_gender,
                                SsdParam fdCfg, std::string db_path, float fr_threshold);
    ~FaceRecognitionVideoSession();

    void Start();
    void SetOnInferenceDone(std::function<void()> cb);
    void SubmitFrame(cv::Mat frame);
    void SubmitKey(int key);
    bool ConsumeDisplay(cv::Mat *view, cv::Mat *gallery_view);
    bool IsStopRequested() const;
    void RequestStop();
    void Join();

private:
    dxrt::InferenceEngine *ie_fd_;
    dxrt::InferenceEngine *ie_fl_;
    dxrt::InferenceEngine *ie_fr_;
    dxrt::InferenceEngine *ie_gender_;
    SsdParam fdCfg_;
    std::string db_path_;
    float fr_threshold_;
    std::unique_ptr<TrackerWorkerState> worker_state_;
    std::thread worker_thread_;
    std::function<void()> on_inference_done_;
};
