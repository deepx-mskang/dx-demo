#include "pose_seg.h"

#include <functional>
#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include <dxrt/dxrt_api.h>

namespace
{

constexpr int kPoseInputWidth = 640;
constexpr int kPoseInputHeight = 640;
constexpr int kSegInputWidth = 768;
constexpr int kSegInputHeight = 384;
constexpr float kPoseScoreThreshold = 0.3f;
constexpr float kPoseIouThreshold = 0.4f;

struct PoseLayerParam
{
    int numGridX = 0;
    int numGridY = 0;
    std::vector<float> anchorWidth;
    std::vector<float> anchorHeight;
};

struct PoseBoundingBox
{
    unsigned int label = 0;
    float score = 0.0f;
    std::string labelname = "person";
    float box[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float kpt[51] = {0.0f};

    PoseBoundingBox() = default;

    PoseBoundingBox(unsigned int label_in, const std::string &labelname_in, float score_in,
                    float x1, float y1, float x2, float y2, const float *keypoints)
        : label(label_in), score(score_in), labelname(labelname_in)
    {
        box[0] = x1;
        box[1] = y1;
        box[2] = x2;
        box[3] = y2;
        if (keypoints != nullptr)
        {
            for (int i = 0; i < 51; ++i)
            {
                kpt[i] = keypoints[i];
            }
        }
    }
};

struct SegmentationParam
{
    int classIndex;
    std::string className;
    uint8_t colorB;
    uint8_t colorG;
    uint8_t colorR;
};

const std::array<std::array<int, 2>, 19> kSkeletonNodes = {{
    {{15, 13}}, {{13, 11}}, {{16, 14}}, {{14, 12}}, {{11, 12}}, {{5, 11}}, {{6, 12}},
    {{5, 6}}, {{5, 7}}, {{6, 8}}, {{7, 9}}, {{8, 10}}, {{1, 2}}, {{0, 1}},
    {{0, 2}}, {{1, 3}}, {{2, 4}}, {{3, 5}}, {{4, 6}},
}};

const std::array<cv::Scalar, 19> kPoseLimbColors = {{
    cv::Scalar(51, 153, 255), cv::Scalar(51, 153, 255), cv::Scalar(51, 153, 255),
    cv::Scalar(51, 153, 255), cv::Scalar(255, 51, 255), cv::Scalar(255, 51, 255),
    cv::Scalar(255, 51, 255), cv::Scalar(255, 128, 0), cv::Scalar(255, 128, 0),
    cv::Scalar(255, 128, 0), cv::Scalar(255, 128, 0), cv::Scalar(255, 128, 0),
    cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 0),
    cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 0),
}};

const std::array<cv::Scalar, 17> kPoseKptColors = {{
    cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 0),
    cv::Scalar(0, 255, 0), cv::Scalar(255, 128, 0), cv::Scalar(255, 128, 0), cv::Scalar(255, 128, 0),
    cv::Scalar(255, 128, 0), cv::Scalar(255, 128, 0), cv::Scalar(255, 128, 0), cv::Scalar(51, 153, 255),
    cv::Scalar(51, 153, 255), cv::Scalar(51, 153, 255), cv::Scalar(51, 153, 255), cv::Scalar(51, 153, 255),
    cv::Scalar(51, 153, 255),
}};

const std::array<cv::Scalar, 4> kObjectColors = {{
    cv::Scalar(113, 129, 39),
    cv::Scalar(133, 80, 164),
    cv::Scalar(114, 122, 83),
    cv::Scalar(172, 81, 99),
}};

const std::array<SegmentationParam, 3> kSegmentationConfig = {{
    {0, "background", 0, 0, 0},
    {1, "foot", 0, 128, 0},
    {2, "body", 0, 0, 128},
}};

float CalcIou(const float *box, const float *truth)
{
    const float left = std::max(box[0], truth[0]);
    const float right = std::min(box[2], truth[2]);
    const float top = std::max(box[1], truth[1]);
    const float bottom = std::min(box[3], truth[3]);
    const float width = right - left;
    const float height = bottom - top;
    if (width < 0.0f || height < 0.0f)
    {
        return 0.0f;
    }

    const float overlap_area = width * height;
    const float union_area =
        (box[2] - box[0]) * (box[3] - box[1]) +
        (truth[2] - truth[0]) * (truth[3] - truth[1]) -
        overlap_area;
    return overlap_area / union_area;
}

void RunNms(std::vector<std::pair<float, int>> *score_indices, float *boxes, float *keypoints,
            std::vector<PoseBoundingBox> *result)
{
    auto &scores = *score_indices;
    std::sort(scores.begin(), scores.end(), [](const auto &lhs, const auto &rhs) {
        return lhs.first > rhs.first;
    });

    std::vector<bool> valid(scores.size(), true);
    for (size_t i = 0; i < scores.size(); ++i)
    {
        if (!valid[i])
        {
            continue;
        }

        const int box_index = scores[i].second;
        result->emplace_back(
            0, "person", scores[i].first,
            boxes[4 * box_index + 0],
            boxes[4 * box_index + 1],
            boxes[4 * box_index + 2],
            boxes[4 * box_index + 3],
            &keypoints[51 * box_index]
        );

        for (size_t j = i + 1; j < scores.size(); ++j)
        {
            if (!valid[j])
            {
                continue;
            }

            const float iou = CalcIou(&boxes[4 * scores[j].second], &boxes[4 * box_index]);
            if (iou > kPoseIouThreshold)
            {
                valid[j] = false;
            }
        }
    }
}

void Preprocess(const cv::Mat &src, cv::Mat *dest, bool keep_ratio, bool bgr_to_rgb, uint8_t pad_value)
{
    if (keep_ratio)
    {
        const float dest_ratio = static_cast<float>(dest->cols) / dest->rows;
        const float src_ratio = static_cast<float>(src.cols) / src.rows;
        int new_width = 0;
        int new_height = 0;
        if (src_ratio < dest_ratio)
        {
            new_height = dest->rows;
            new_width = static_cast<int>(new_height * src_ratio);
        }
        else
        {
            new_width = dest->cols;
            new_height = static_cast<int>(new_width / src_ratio);
        }

        cv::Mat resized;
        cv::resize(src, resized, cv::Size(new_width, new_height), 0, 0, cv::INTER_LINEAR);
        const float dw = (dest->cols - resized.cols) / 2.0f;
        const float dh = (dest->rows - resized.rows) / 2.0f;
        const uint16_t top = static_cast<uint16_t>(std::round(dh - 0.1f));
        const uint16_t bottom = static_cast<uint16_t>(std::round(dh + 0.1f));
        const uint16_t left = static_cast<uint16_t>(std::round(dw - 0.1f));
        const uint16_t right = static_cast<uint16_t>(std::round(dw + 0.1f));
        cv::copyMakeBorder(resized, *dest, top, bottom, left, right, cv::BORDER_CONSTANT,
                           cv::Scalar(pad_value, pad_value, pad_value));
    }
    else
    {
        cv::resize(src, *dest, dest->size(), 0, 0, cv::INTER_LINEAR);
    }

    if (bgr_to_rgb)
    {
        cv::cvtColor(*dest, *dest, cv::COLOR_BGR2RGB);
    }
}

void DecodeSegmentationUint16(uint16_t *input, uint8_t *output, int rows, int cols)
{
    for (int h = 0; h < rows; ++h)
    {
        for (int w = 0; w < cols; ++w)
        {
            const int cls = input[cols * h + w];
            if (cls >= 0 && cls < static_cast<int>(kSegmentationConfig.size()))
            {
                output[3 * cols * h + 3 * w + 0] = kSegmentationConfig[cls].colorB;
                output[3 * cols * h + 3 * w + 1] = kSegmentationConfig[cls].colorG;
                output[3 * cols * h + 3 * w + 2] = kSegmentationConfig[cls].colorR;
            }
        }
    }
}

void DecodeSegmentationFloat(float *input, uint8_t *output, int rows, int cols,
                             const std::vector<int64_t> &shape)
{
    const int num_classes = static_cast<int>(kSegmentationConfig.size());
    const bool need_transpose = static_cast<int>(shape[1]) == num_classes;
    const int pitch = static_cast<int>(shape[3]);

    for (int h = 0; h < rows; ++h)
    {
        for (int w = 0; w < cols; ++w)
        {
            int max_index = 0;
            for (int c = 0; c < num_classes; ++c)
            {
                int compare_max_index = 0;
                int compare_channel_index = 0;
                if (need_transpose)
                {
                    compare_max_index = w + (cols * h) + (max_index * rows * cols);
                    compare_channel_index = w + (cols * h) + (c * rows * cols);
                }
                else
                {
                    compare_max_index = max_index + ((cols * h) + w) * pitch;
                    compare_channel_index = c + ((cols * h) + w) * pitch;
                }

                if (input[compare_max_index] < input[compare_channel_index])
                {
                    max_index = c;
                }
            }

            output[3 * cols * h + 3 * w + 0] = kSegmentationConfig[max_index].colorB;
            output[3 * cols * h + 3 * w + 1] = kSegmentationConfig[max_index].colorG;
            output[3 * cols * h + 3 * w + 2] = kSegmentationConfig[max_index].colorR;
        }
    }
}

void DrawPoseBoundingBoxes(cv::Mat *frame, std::vector<PoseBoundingBox> *result,
                           float origin_height, float origin_width)
{
    const float w = static_cast<float>(frame->cols);
    const float h = static_cast<float>(frame->rows);
    const float r = std::min(origin_width / w, origin_height / h);
    const float w_pad = (origin_width - w * r) / 2.0f;
    const float h_pad = (origin_height - h * r) / 2.0f;
    int text_baseline = 0;

    for (auto &bbox : *result)
    {
        float x1 = (bbox.box[0] - w_pad) / r;
        float x2 = (bbox.box[2] - w_pad) / r;
        float y1 = (bbox.box[1] - h_pad) / r;
        float y2 = (bbox.box[3] - h_pad) / r;

        x1 = std::min(w, std::max(0.0f, x1));
        x2 = std::min(w, std::max(0.0f, x2));
        y1 = std::min(h, std::max(0.0f, y1));
        y2 = std::min(h, std::max(0.0f, y2));

        char score_text[16];
        std::snprintf(score_text, sizeof(score_text), "%.2f", bbox.score);
        const std::string label = bbox.labelname + "=" + score_text;
        const auto text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &text_baseline);

        cv::rectangle(*frame, cv::Point(static_cast<int>(x1), static_cast<int>(y1)),
                      cv::Point(static_cast<int>(x2), static_cast<int>(y2)), kObjectColors[0], 2);
        cv::rectangle(*frame, cv::Point(static_cast<int>(x1), static_cast<int>(y1) - text_size.height),
                      cv::Point(static_cast<int>(x1) + text_size.width, static_cast<int>(y1)),
                      kObjectColors[0], cv::FILLED);
        cv::putText(*frame, label, cv::Point(static_cast<int>(x1), static_cast<int>(y1)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255));

        std::array<cv::Point, 17> points;
        for (int k = 0; k < 17; ++k)
        {
            const float kx = (bbox.kpt[k * 3 + 0] - w_pad) / r;
            const float ky = (bbox.kpt[k * 3 + 1] - h_pad) / r;
            const float ks = bbox.kpt[k * 3 + 2];
            points[k] = ks > 0.5f ? cv::Point(static_cast<int>(kx), static_cast<int>(ky))
                                  : cv::Point(-1, -1);
        }

        for (size_t index = 0; index < kSkeletonNodes.size(); ++index)
        {
            const auto &pair = kSkeletonNodes[index];
            if (points[pair[0]].x >= 0 && points[pair[1]].x >= 0)
            {
                cv::line(*frame, points[pair[0]], points[pair[1]], kPoseLimbColors[index], 2, cv::LINE_AA);
            }
        }

        for (size_t index = 0; index < points.size(); ++index)
        {
            cv::circle(*frame, points[index], 3, kPoseKptColors[index], -1, cv::LINE_AA);
        }
    }
}

class FixedPosePostProcessor
{
public:
    FixedPosePostProcessor()
        : layers_({
              PoseLayerParam{80, 80, {19.0f, 44.0f, 38.0f}, {27.0f, 40.0f, 94.0f}},
              PoseLayerParam{40, 40, {96.0f, 86.0f, 180.0f}, {68.0f, 152.0f, 137.0f}},
              PoseLayerParam{20, 20, {140.0f, 303.0f, 238.0f}, {301.0f, 264.0f, 542.0f}},
              PoseLayerParam{10, 10, {436.0f, 739.0f, 925.0f}, {615.0f, 380.0f, 792.0f}},
          }),
          boxes_(2000 * 4),
          keypoints_(2000 * 51)
    {
    }

    bool Initialize(dxrt::Tensors output_info)
    {
        if (output_info.empty())
        {
            std::cout << "Error: pose model has no outputs." << std::endl;
            return false;
        }

        if (static_cast<int>(output_info.front().type()) < static_cast<int>(dxrt::DataType::POSE))
        {
            std::cout << "Error: pose model is not PPU pose output. -p0 is fixed to 1 in this path." << std::endl;
            return false;
        }

        return true;
    }

    std::vector<PoseBoundingBox> PostProcess(dxrt::TensorPtrs &outputs)
    {
        std::vector<PoseBoundingBox> result;
        std::vector<std::pair<float, int>> score_indices;
        int box_index = 0;
        auto *data = static_cast<dxrt::DevicePose_t *>(outputs.front()->data());
        const int num_elements = static_cast<int>(outputs.front()->shape()[1]);

        for (int i = 0; i < num_elements; ++i)
        {
            if (data[i].score <= kPoseScoreThreshold)
            {
                continue;
            }

            const auto &layer = layers_[data[i].layer_idx];
            const int stride_x = kPoseInputWidth / layer.numGridX;
            const int stride_y = kPoseInputHeight / layer.numGridY;
            const int g_x = data[i].grid_x;
            const int g_y = data[i].grid_y;

            const float center_x = (data[i].x * 2.0f - 0.5f + g_x) * stride_x;
            const float center_y = (data[i].y * 2.0f - 0.5f + g_y) * stride_y;
            const float width = std::pow(data[i].w * 2.0f, 2.0f) * layer.anchorWidth[data[i].box_idx];
            const float height = std::pow(data[i].h * 2.0f, 2.0f) * layer.anchorHeight[data[i].box_idx];

            boxes_[box_index * 4 + 0] = center_x - width / 2.0f;
            boxes_[box_index * 4 + 1] = center_y - height / 2.0f;
            boxes_[box_index * 4 + 2] = center_x + width / 2.0f;
            boxes_[box_index * 4 + 3] = center_y + height / 2.0f;

            for (int k = 0; k < 17; ++k)
            {
                keypoints_[box_index * 51 + k * 3 + 0] = (data[i].kpts[k][0] * 2.0f - 0.5f + g_x) * stride_x;
                keypoints_[box_index * 51 + k * 3 + 1] = (data[i].kpts[k][1] * 2.0f - 0.5f + g_y) * stride_y;
                keypoints_[box_index * 51 + k * 3 + 2] = data[i].kpts[k][2];
            }

            score_indices.emplace_back(data[i].score, box_index);
            ++box_index;
        }

        RunNms(&score_indices, boxes_.data(), keypoints_.data(), &result);
        return result;
    }

private:
    std::vector<PoseLayerParam> layers_;
    std::vector<float> boxes_;
    std::vector<float> keypoints_;
};

}  // namespace

struct PoseSegReferenceRenderer::Impl
{
    Impl(dxrt::InferenceEngine *pose_engine_in, dxrt::InferenceEngine *seg_engine_in)
        : pose_engine(pose_engine_in),
          seg_engine(seg_engine_in),
          pose_input(kPoseInputHeight, kPoseInputWidth, CV_8UC3),
          seg_input(kSegInputHeight, kSegInputWidth, CV_8UC3),
          seg_overlay(kSegInputHeight, kSegInputWidth, CV_8UC3, cv::Scalar(0, 0, 0))
    {
        initialized = post_processor.Initialize(pose_engine->GetOutputs());
    }

    dxrt::InferenceEngine *pose_engine;
    dxrt::InferenceEngine *seg_engine;
    FixedPosePostProcessor post_processor;
    cv::Mat pose_input;
    cv::Mat seg_input;
    cv::Mat seg_overlay;
    cv::Mat resized_overlay;
    bool initialized = false;
};

PoseSegReferenceRenderer::PoseSegReferenceRenderer(dxrt::InferenceEngine *pose_engine,
                                                   dxrt::InferenceEngine *seg_engine)
    : impl_(std::make_unique<Impl>(pose_engine, seg_engine))
{
}

PoseSegReferenceRenderer::~PoseSegReferenceRenderer() = default;

bool PoseSegReferenceRenderer::IsInferenceReady() const
{
    return impl_ != nullptr && impl_->initialized;
}

bool PoseSegReferenceRenderer::RenderPreview(const cv::Mat &frame, cv::Mat *output)
{
    if (output == nullptr || frame.empty() || !impl_->initialized)
    {
        return false;
    }

    Preprocess(frame, &impl_->pose_input, true, true, 114);
    Preprocess(frame, &impl_->seg_input, false, true, 0);

    auto seg_outputs = impl_->seg_engine->Run(impl_->seg_input.data);
    auto pose_outputs = impl_->pose_engine->Run(impl_->pose_input.data);

    impl_->seg_overlay.setTo(cv::Scalar(0, 0, 0));
    if (!seg_outputs.empty())
    {
        if (seg_outputs.front()->type() == dxrt::DataType::UINT16)
        {
            DecodeSegmentationUint16(static_cast<uint16_t *>(seg_outputs.front()->data()),
                                     impl_->seg_overlay.data, impl_->seg_overlay.rows,
                                     impl_->seg_overlay.cols);
        }
        else if (seg_outputs.front()->type() == dxrt::DataType::FLOAT)
        {
            DecodeSegmentationFloat(static_cast<float *>(seg_outputs.front()->data()),
                                    impl_->seg_overlay.data, impl_->seg_overlay.rows,
                                    impl_->seg_overlay.cols, seg_outputs.front()->shape());
        }
    }

    cv::resize(impl_->seg_overlay, impl_->resized_overlay, frame.size());

    // addWeighted on the whole image darkens background too (0.6*frame + 0.4*black), which looks like a gray
    // wash. Blend only where segmentation is non-background (overlay not all zeros).
    cv::Mat blended;
    cv::addWeighted(frame, 0.6, impl_->resized_overlay, 0.4, 0.0, blended);
    std::vector<cv::Mat> seg_ch;
    cv::split(impl_->resized_overlay, seg_ch);
    cv::Mat max_bg;
    cv::max(seg_ch[0], seg_ch[1], max_bg);
    cv::Mat max_bgr;
    cv::max(max_bg, seg_ch[2], max_bgr);
    cv::Mat fg_mask;
    cv::compare(max_bgr, 0, fg_mask, cv::CMP_GT);
    frame.copyTo(*output);
    blended.copyTo(*output, fg_mask);

    auto pose_result = impl_->post_processor.PostProcess(pose_outputs);
    DrawPoseBoundingBoxes(output, &pose_result, static_cast<float>(kPoseInputHeight),
                          static_cast<float>(kPoseInputWidth));
    return true;
}

struct PoseSegWorkerState
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

void run_pose_seg_worker(dxrt::InferenceEngine *pose_engine, dxrt::InferenceEngine *seg_engine,
                         PoseSegWorkerState *worker_state, std::function<void()> on_inference_done)
{
    PoseSegReferenceRenderer renderer(pose_engine, seg_engine);
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

        cv::Mat pose_seg_view;
        cv::Mat display_frame = frame.clone();
        if (renderer.IsInferenceReady())
        {
            if (renderer.RenderPreview(frame, &pose_seg_view) && !pose_seg_view.empty())
            {
                display_frame = std::move(pose_seg_view);
            }
            if (on_inference_done)
            {
                on_inference_done();
            }
        }

        {
            std::lock_guard<std::mutex> lock(worker_state->mutex);
            worker_state->display_frame = std::move(display_frame);
            worker_state->has_rendered_frame = true;
        }
    }
}

}  // namespace

PoseSegVideoSession::PoseSegVideoSession(dxrt::InferenceEngine *pose_engine,
                                         dxrt::InferenceEngine *seg_engine)
    : pose_engine_(pose_engine),
      seg_engine_(seg_engine),
      worker_state_(std::make_unique<PoseSegWorkerState>())
{
}

PoseSegVideoSession::~PoseSegVideoSession()
{
    RequestStop();
    Join();
}

void PoseSegVideoSession::SetOnInferenceDone(std::function<void()> cb)
{
    on_inference_done_ = std::move(cb);
}

void PoseSegVideoSession::Start()
{
    if (worker_thread_.joinable())
    {
        return;
    }

    std::function<void()> done_cb = on_inference_done_;
    worker_thread_ = std::thread([this, done_cb]() {
        run_pose_seg_worker(pose_engine_, seg_engine_, worker_state_.get(), done_cb);
    });
}

void PoseSegVideoSession::SubmitFrame(cv::Mat frame)
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

bool PoseSegVideoSession::ConsumeDisplay(cv::Mat *view)
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

bool PoseSegVideoSession::IsStopRequested() const
{
    std::lock_guard<std::mutex> lock(worker_state_->mutex);
    return worker_state_->stop_requested;
}

void PoseSegVideoSession::RequestStop()
{
    {
        std::lock_guard<std::mutex> lock(worker_state_->mutex);
        worker_state_->stop_requested = true;
    }
    worker_state_->cv.notify_all();
}

void PoseSegVideoSession::Join()
{
    if (worker_thread_.joinable())
    {
        worker_thread_.join();
    }
}
