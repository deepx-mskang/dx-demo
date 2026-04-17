#pragma once

#include <functional>
#include <memory>
#include <thread>

#include <opencv2/opencv.hpp>

namespace dxrt
{
class InferenceEngine;
}

struct PoseSegWorkerState;

class PoseSegReferenceRenderer
{
public:
    PoseSegReferenceRenderer(dxrt::InferenceEngine *pose_engine, dxrt::InferenceEngine *seg_engine);
    ~PoseSegReferenceRenderer();

    bool RenderPreview(const cv::Mat &frame, cv::Mat *output);
    bool IsInferenceReady() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class PoseSegVideoSession
{
public:
    PoseSegVideoSession(dxrt::InferenceEngine *pose_engine, dxrt::InferenceEngine *seg_engine);
    ~PoseSegVideoSession();

    void Start();
    void SetOnInferenceDone(std::function<void()> cb);
    void SubmitFrame(cv::Mat frame);
    bool ConsumeDisplay(cv::Mat *view);
    bool IsStopRequested() const;
    void RequestStop();
    void Join();

private:
    dxrt::InferenceEngine *pose_engine_;
    dxrt::InferenceEngine *seg_engine_;
    std::unique_ptr<PoseSegWorkerState> worker_state_;
    std::thread worker_thread_;
    std::function<void()> on_inference_done_;
};
