#pragma once

#include <functional>
#include <memory>
#include <opencv2/core.hpp>
#include <string>
#include <thread>

#include <dxrt/dxrt_api.h>

struct MobedWorkerState;

class MobedDetector
{
public:
    MobedDetector(const std::string &model_path, int num_classes = 80, float nms_conf_thre = 0.5f,
                  float nms_iou_thre = 0.45f, int infer_w = 416, int infer_h = 416);
    MobedDetector(dxrt::InferenceEngine *engine, int num_classes = 80, float nms_conf_thre = 0.5f,
                  float nms_iou_thre = 0.45f, int infer_w = 416, int infer_h = 416);

    bool ok() const { return valid_; }

    void SetOnInferenceDone(std::function<void()> cb);
    void DetectAndDraw(cv::Mat &frame_bgr);

private:
    bool AnalyzeOutputLayout(const dxrt::Tensors &formats);
    static int64_t NumBoxesFromShape(const std::vector<int64_t> &sh, int last_dim);

    std::unique_ptr<dxrt::InferenceEngine> owned_engine_;
    dxrt::InferenceEngine *engine_;
    int num_classes_ = 80;
    float nms_conf_thre_ = 0.5f;
    float nms_iou_thre_ = 0.45f;
    int infer_w_ = 416;
    int infer_h_ = 416;
    bool valid_ = false;
    size_t cls_out_idx_ = 0;
    size_t box_out_idx_ = 1;
    int64_t num_boxes_ = 0;
    std::function<void()> on_inference_done_;
};

class MobedVideoSession
{
public:
    explicit MobedVideoSession(dxrt::InferenceEngine *engine, int num_classes = 80, float nms_conf_thre = 0.5f,
                               float nms_iou_thre = 0.45f, int infer_w = 416, int infer_h = 416);
    ~MobedVideoSession();

    bool ok() const { return init_ok_; }

    void Start();
    void SetOnInferenceDone(std::function<void()> cb);
    void SubmitFrame(cv::Mat frame);
    bool ConsumeDisplay(cv::Mat *view);
    bool IsStopRequested() const;
    void RequestStop();
    void Join();

private:
    dxrt::InferenceEngine *engine_;
    bool init_ok_ = false;
    int num_classes_;
    float nms_conf_thre_;
    float nms_iou_thre_;
    int infer_w_;
    int infer_h_;
    std::unique_ptr<MobedWorkerState> worker_state_;
    std::thread worker_thread_;
    std::function<void()> on_inference_done_;
};
