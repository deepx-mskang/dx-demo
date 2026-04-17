#include "mobed_detector.h"

#include "nms.h"

#include <condition_variable>
#include <functional>
#include <mutex>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace
{

static const char *const kCoco80[] = {
    "person",        "bicycle",      "car",           "motorcycle",    "airplane",     "bus",           "train",         "truck",
    "boat",          "traffic light", "fire hydrant", "stop sign",     "parking meter", "bench",        "bird",          "cat",
    "dog",           "horse",        "sheep",         "cow",           "elephant",     "bear",          "zebra",         "giraffe",
    "backpack",      "umbrella",     "handbag",       "tie",           "suitcase",     "frisbee",       "skis",          "snowboard",
    "sports ball",   "kite",         "baseball bat",  "baseball glove", "skateboard",  "surfboard",     "tennis racket", "bottle",
    "wine glass",    "cup",          "fork",          "knife",         "spoon",        "bowl",          "banana",        "apple",
    "sandwich",      "orange",       "broccoli",      "carrot",        "hot dog",      "pizza",         "donut",         "cake",
    "chair",         "couch",        "potted plant",  "bed",           "dining table", "toilet",        "tv",            "laptop",
    "mouse",         "remote",       "keyboard",      "cell phone",    "microwave",    "oven",          "toaster",       "sink",
    "refrigerator",  "book",         "clock",         "vase",          "scissors",     "teddy bear",    "hair drier",    "toothbrush"};

constexpr int kMaxDetections = 500;

cv::Scalar ClassBgr(int cls_id)
{
    const int hue = (cls_id * 37) % 180;
    cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(hue, 200, 255));
    cv::Mat bgr;
    cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
    const cv::Vec3b &p = bgr.at<cv::Vec3b>(0, 0);
    return cv::Scalar(p[0], p[1], p[2]);
}

void RawBoxToXyxy(const float *raw, float *xyxy)
{
    const float x0 = raw[0];
    const float y0 = raw[1];
    const float a = raw[2];
    const float b = raw[3];
    if (a > x0 && b > y0)
    {
        xyxy[0] = x0;
        xyxy[1] = y0;
        xyxy[2] = a;
        xyxy[3] = b;
    }
    else
    {
        const float cx = x0;
        const float cy = y0;
        const float w = a;
        const float h = b;
        xyxy[0] = cx - w * 0.5f;
        xyxy[1] = cy - h * 0.5f;
        xyxy[2] = cx + w * 0.5f;
        xyxy[3] = cy + h * 0.5f;
    }
}

const float *BoxPtr(const float *bbox_data, int /*num_boxes*/, int num_classes, int row, int cls_id, int bbox_stride)
{
    if (bbox_stride == 4)
    {
        return bbox_data + row * 4;
    }
    return bbox_data + row * (num_classes * 4) + cls_id * 4;
}

struct DetCand
{
    float score = 0.0f;
    int cls = 0;
    float xyxy[4] = {0.0f, 0.0f, 0.0f, 0.0f};
};

void GreedyNmsXyxy(std::vector<DetCand> &cands, float iou_thr)
{
    std::sort(cands.begin(), cands.end(),
              [](const DetCand &a, const DetCand &b) { return a.score > b.score; });
    std::vector<DetCand> kept;
    kept.reserve(cands.size());
    for (const auto &c : cands)
    {
        bool ok = true;
        for (const auto &k : kept)
        {
            float abox[4] = {c.xyxy[0], c.xyxy[1], c.xyxy[2], c.xyxy[3]};
            float kbox[4] = {k.xyxy[0], k.xyxy[1], k.xyxy[2], k.xyxy[3]};
            if (CalcIOU(abox, kbox) > iou_thr)
            {
                ok = false;
                break;
            }
        }
        if (ok)
        {
            kept.push_back(c);
        }
    }
    cands.swap(kept);
}

void MulticlassNms(const float *bbox_data, int num_boxes, int num_classes, int bbox_stride, const float *cls_scores,
                   float score_thr, float iou_thr, int max_num, std::vector<DetCand> *out)
{
    out->clear();
    std::vector<std::vector<DetCand>> by_class(static_cast<size_t>(num_classes));

    for (int i = 0; i < num_boxes; ++i)
    {
        for (int c = 0; c < num_classes; ++c)
        {
            const float s = cls_scores[i * num_classes + c];
            if (s <= score_thr)
            {
                continue;
            }
            const float *braw = BoxPtr(bbox_data, num_boxes, num_classes, i, c, bbox_stride);
            DetCand d;
            d.score = s;
            d.cls = c;
            RawBoxToXyxy(braw, d.xyxy);
            by_class[static_cast<size_t>(c)].push_back(d);
        }
    }

    std::vector<DetCand> merged;
    for (int c = 0; c < num_classes; ++c)
    {
        if (by_class[static_cast<size_t>(c)].empty())
        {
            continue;
        }
        GreedyNmsXyxy(by_class[static_cast<size_t>(c)], iou_thr);
        merged.insert(merged.end(), by_class[static_cast<size_t>(c)].begin(),
                       by_class[static_cast<size_t>(c)].end());
    }

    std::sort(merged.begin(), merged.end(),
              [](const DetCand &a, const DetCand &b) { return a.score > b.score; });
    if (max_num > 0 && static_cast<int>(merged.size()) > max_num)
    {
        merged.resize(static_cast<size_t>(max_num));
    }
    *out = std::move(merged);
}

}  // namespace

int64_t MobedDetector::NumBoxesFromShape(const std::vector<int64_t> &sh, int last_dim)
{
    if (sh.size() < 2)
    {
        return 0;
    }
    int64_t n = 1;
    for (size_t i = 0; i + 1 < sh.size(); ++i)
    {
        const int64_t d = sh[i];
        if (d <= 0)
        {
            return 0;
        }
        n *= d;
    }
    if (sh.back() != last_dim)
    {
        return 0;
    }
    return n;
}

bool MobedDetector::AnalyzeOutputLayout(const dxrt::Tensors &formats)
{
    if (formats.size() < 2)
    {
        std::cerr << "MobedDetector: expected at least 2 output tensors, got " << formats.size() << std::endl;
        return false;
    }

    size_t idx_cls = SIZE_MAX;
    size_t idx_box = SIZE_MAX;
    int cls_last = -1;

    for (size_t i = 0; i < formats.size(); ++i)
    {
        const auto &sh = formats[i].shape();
        if (sh.empty())
        {
            continue;
        }
        const int last = static_cast<int>(sh.back());
        if (last == 4 && idx_box == SIZE_MAX)
        {
            idx_box = i;
        }
    }

    for (size_t i = 0; i < formats.size(); ++i)
    {
        if (i == idx_box)
        {
            continue;
        }
        const auto &sh = formats[i].shape();
        if (sh.empty())
        {
            continue;
        }
        const int last = static_cast<int>(sh.back());
        if (last <= 1)
        {
            continue;
        }
        if (last == num_classes_)
        {
            idx_cls = i;
            cls_last = last;
            break;
        }
        if (idx_cls == SIZE_MAX || cls_last < last)
        {
            idx_cls = i;
            cls_last = last;
        }
    }

    if (idx_cls == SIZE_MAX || idx_box == SIZE_MAX)
    {
        const int l0 = static_cast<int>(formats[0].shape().back());
        const int l1 = static_cast<int>(formats[1].shape().back());
        if (l0 == num_classes_ && l1 == 4)
        {
            idx_cls = 0;
            idx_box = 1;
            cls_last = l0;
        }
        else if (l1 == num_classes_ && l0 == 4)
        {
            idx_cls = 1;
            idx_box = 0;
            cls_last = l1;
        }
    }

    if (idx_cls == SIZE_MAX || idx_box == SIZE_MAX)
    {
        std::cerr << "MobedDetector: could not match cls / bbox outputs (expected score + box tensors)." << std::endl;
        return false;
    }

    if (cls_last > 0 && cls_last != num_classes_)
    {
        std::cout << "MobedDetector: num_classes " << num_classes_ << " -> " << cls_last << " (from model output)"
                  << std::endl;
        num_classes_ = cls_last;
    }

    const int64_t n_cls = NumBoxesFromShape(formats[idx_cls].shape(), num_classes_);
    const int64_t n_box = NumBoxesFromShape(formats[idx_box].shape(), 4);
    if (n_cls <= 0 || n_box <= 0 || n_cls != n_box)
    {
        std::cerr << "MobedDetector: box count mismatch cls=" << n_cls << " box=" << n_box << std::endl;
        return false;
    }
    if (num_classes_ > 4096 || n_cls > 1000000)
    {
        std::cerr << "MobedDetector: unreasonable layout num_classes=" << num_classes_ << " num_boxes=" << n_cls
                  << std::endl;
        return false;
    }

    cls_out_idx_ = idx_cls;
    box_out_idx_ = idx_box;
    num_boxes_ = n_cls;
    return true;
}

void MobedDetector::SetOnInferenceDone(std::function<void()> cb)
{
    on_inference_done_ = std::move(cb);
}

MobedDetector::MobedDetector(const std::string &model_path, int num_classes, float nms_conf_thre, float nms_iou_thre,
                             int infer_w, int infer_h)
    : owned_engine_(std::make_unique<dxrt::InferenceEngine>(model_path)),
      engine_(owned_engine_.get()),
      num_classes_(num_classes),
      nms_conf_thre_(nms_conf_thre),
      nms_iou_thre_(nms_iou_thre),
      infer_w_(infer_w),
      infer_h_(infer_h)
{
    dxrt::Tensors formats = engine_->GetOutputs();
    valid_ = AnalyzeOutputLayout(formats);
    if (!valid_)
    {
        std::cerr << "MobedDetector: init failed for " << model_path << std::endl;
    }
}

MobedDetector::MobedDetector(dxrt::InferenceEngine *engine, int num_classes, float nms_conf_thre, float nms_iou_thre,
                             int infer_w, int infer_h)
    : owned_engine_(),
      engine_(engine),
      num_classes_(num_classes),
      nms_conf_thre_(nms_conf_thre),
      nms_iou_thre_(nms_iou_thre),
      infer_w_(infer_w),
      infer_h_(infer_h)
{
    if (engine_ == nullptr)
    {
        valid_ = false;
        return;
    }
    dxrt::Tensors formats = engine_->GetOutputs();
    valid_ = AnalyzeOutputLayout(formats);
}

void MobedDetector::DetectAndDraw(cv::Mat &frame_bgr)
{
    if (!valid_ || frame_bgr.empty())
    {
        return;
    }

    cv::Mat rgb;
    cv::cvtColor(frame_bgr, rgb, cv::COLOR_BGR2RGB);
    cv::Mat resized;
    cv::resize(rgb, resized, cv::Size(infer_w_, infer_h_), 0.0, 0.0, cv::INTER_LINEAR);
    if (!resized.isContinuous())
    {
        resized = resized.clone();
    }

    auto outputs = engine_->Run(resized.data);
    if (outputs.size() <= std::max(cls_out_idx_, box_out_idx_))
    {
        return;
    }

    auto *t_cls = outputs[cls_out_idx_].get();
    auto *t_box = outputs[box_out_idx_].get();
    if (t_cls == nullptr || t_box == nullptr)
    {
        return;
    }

    if (t_cls->type() != dxrt::DataType::FLOAT || t_box->type() != dxrt::DataType::FLOAT)
    {
        std::cerr << "MobedDetector: expected FLOAT outputs" << std::endl;
        return;
    }

    if (on_inference_done_)
    {
        on_inference_done_();
    }

    const float *cls_scores = static_cast<const float *>(t_cls->data());
    const float *bbox_data = static_cast<const float *>(t_box->data());
    const int bbox_stride = (static_cast<int>(t_box->shape().back()) == 4) ? 4 : num_classes_ * 4;

    std::vector<DetCand> dets;
    MulticlassNms(bbox_data, static_cast<int>(num_boxes_), num_classes_, bbox_stride, cls_scores, nms_conf_thre_,
                  nms_iou_thre_, kMaxDetections, &dets);

    const float sx = static_cast<float>(frame_bgr.cols) / static_cast<float>(infer_w_);
    const float sy = static_cast<float>(frame_bgr.rows) / static_cast<float>(infer_h_);

    for (const auto &d : dets)
    {
        const float x1 = d.xyxy[0] * sx;
        const float y1 = d.xyxy[1] * sy;
        const float x2 = d.xyxy[2] * sx;
        const float y2 = d.xyxy[3] * sy;

        const int ix1 = std::max(0, static_cast<int>(std::floor(x1)));
        const int iy1 = std::max(0, static_cast<int>(std::floor(y1)));
        const int ix2 = std::min(frame_bgr.cols - 1, static_cast<int>(std::ceil(x2)));
        const int iy2 = std::min(frame_bgr.rows - 1, static_cast<int>(std::ceil(y2)));
        if (ix2 <= ix1 || iy2 <= iy1)
        {
            continue;
        }

        const cv::Scalar color = ClassBgr(d.cls);
        cv::rectangle(frame_bgr, cv::Point(ix1, iy1), cv::Point(ix2, iy2), color, 2);

        const char *name = "?";
        if (d.cls >= 0 && d.cls < 80 && static_cast<size_t>(d.cls) < sizeof(kCoco80) / sizeof(kCoco80[0]))
        {
            name = kCoco80[d.cls];
        }

        char text[128];
        std::snprintf(text, sizeof(text), "%s:%.1f%%", name, d.score * 100.0f);

        int baseline = 0;
        const double font_scale = 0.4;
        const int thickness = 1;
        const cv::Size ts = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, font_scale, thickness, &baseline);
        const cv::Scalar bg_color(color[0] * 0.7, color[1] * 0.7, color[2] * 0.7);
        cv::rectangle(frame_bgr, cv::Point(ix1, iy1 + 1), cv::Point(ix1 + ts.width + 1, iy1 + static_cast<int>(1.5 * ts.height)),
                      bg_color, -1);
        const cv::Scalar txt_color =
            (color[0] + color[1] + color[2]) / 3.0 > 127.0 ? cv::Scalar(0, 0, 0) : cv::Scalar(255, 255, 255);
        cv::putText(frame_bgr, text, cv::Point(ix1, iy1 + ts.height), cv::FONT_HERSHEY_SIMPLEX, font_scale, txt_color,
                    thickness, cv::LINE_AA);
    }
}

struct MobedWorkerState
{
    std::mutex mutex;
    std::condition_variable cv;
    cv::Mat frame;
    cv::Mat display_frame;
    bool has_frame = false;
    bool has_rendered_frame = false;
    bool stop_requested = false;
};

namespace
{

void run_mobed_worker(dxrt::InferenceEngine *engine, int num_classes, float nms_conf_thre, float nms_iou_thre,
                      int infer_w, int infer_h, MobedWorkerState *worker_state, std::function<void()> on_inference_done)
{
    MobedDetector detector(engine, num_classes, nms_conf_thre, nms_iou_thre, infer_w, infer_h);
    if (!detector.ok())
    {
        return;
    }
    detector.SetOnInferenceDone(std::move(on_inference_done));

    while (true)
    {
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

        cv::Mat display = frame.clone();
        detector.DetectAndDraw(display);

        {
            std::lock_guard<std::mutex> lock(worker_state->mutex);
            worker_state->display_frame = std::move(display);
            worker_state->has_rendered_frame = true;
        }
    }
}

}  // namespace

MobedVideoSession::MobedVideoSession(dxrt::InferenceEngine *engine, int num_classes, float nms_conf_thre,
                                     float nms_iou_thre, int infer_w, int infer_h)
    : engine_(engine),
      num_classes_(num_classes),
      nms_conf_thre_(nms_conf_thre),
      nms_iou_thre_(nms_iou_thre),
      infer_w_(infer_w),
      infer_h_(infer_h),
      worker_state_(std::make_unique<MobedWorkerState>())
{
    if (engine_ != nullptr)
    {
        MobedDetector probe(engine_, num_classes_, nms_conf_thre_, nms_iou_thre_, infer_w_, infer_h_);
        init_ok_ = probe.ok();
    }
}

MobedVideoSession::~MobedVideoSession()
{
    RequestStop();
    Join();
}

void MobedVideoSession::SetOnInferenceDone(std::function<void()> cb)
{
    on_inference_done_ = std::move(cb);
}

void MobedVideoSession::Start()
{
    if (worker_thread_.joinable())
    {
        return;
    }

    std::function<void()> done_cb = on_inference_done_;
    worker_thread_ = std::thread([this, done_cb]() {
        run_mobed_worker(engine_, num_classes_, nms_conf_thre_, nms_iou_thre_, infer_w_, infer_h_, worker_state_.get(),
                         done_cb);
    });
}

void MobedVideoSession::SubmitFrame(cv::Mat frame)
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

bool MobedVideoSession::ConsumeDisplay(cv::Mat *view)
{
    if (view == nullptr)
    {
        return false;
    }
    std::lock_guard<std::mutex> lock(worker_state_->mutex);
    if (!worker_state_->has_rendered_frame)
    {
        return false;
    }
    *view = std::move(worker_state_->display_frame);
    worker_state_->has_rendered_frame = false;
    return true;
}

bool MobedVideoSession::IsStopRequested() const
{
    std::lock_guard<std::mutex> lock(worker_state_->mutex);
    return worker_state_->stop_requested;
}

void MobedVideoSession::RequestStop()
{
    {
        std::lock_guard<std::mutex> lock(worker_state_->mutex);
        worker_state_->stop_requested = true;
    }
    worker_state_->cv.notify_all();
}

void MobedVideoSession::Join()
{
    if (worker_thread_.joinable())
    {
        worker_thread_.join();
    }
}
