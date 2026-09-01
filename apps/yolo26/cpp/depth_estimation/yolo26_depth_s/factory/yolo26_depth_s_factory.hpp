/**
 * @file yolo26_depth_s_factory.hpp
 * @brief Yolo26DepthSFactory Abstract Factory implementation
 *
 * YOLO26-Depth-S monocular depth estimation. The compiled model
 * `yolo26-depth-s_768x768.dxnn` takes a uint8 NHWC input
 * `images` [1,768,768,3] and produces a dense float32 depth map
 * `depth` [1,1,768,768].
 */

#ifndef YOLO26_DEPTH_S_FACTORY_HPP
#define YOLO26_DEPTH_S_FACTORY_HPP

#include "common/base/i_factory.hpp"
#include "common/processors/letterbox_preprocessor.hpp"
#include "common/processors/depth_postprocessor.hpp"
#include "common/visualizers/depth_visualizer.hpp"
#include "common/config/model_config.hpp"

namespace dxapp {

class Yolo26DepthSFactory : public IDepthEstimationFactory {
public:
    Yolo26DepthSFactory() = default;

    // Upstream dx_app uses SimpleResizePreprocessor here. We letterbox instead:
    // YOLO26-Depth-S is a YOLO-family model trained on padded input, and a plain
    // resize would squash 16:9 to 1:1 on its way into the model. This also matches
    // the four sibling yolo26 factories, all of which use DetectionPreprocessor.
    PreprocessorPtr createPreprocessor(int input_width, int input_height) override {
        return std::make_unique<DetectionPreprocessor>(input_width, input_height);
    }

    // IDepthEstimationFactory takes no is_ort_configured flag, unlike the
    // detection / pose / instance-segmentation interfaces.
    PostprocessorPtr<DepthResult> createPostprocessor(
        int input_width, int input_height) override {
        return std::make_unique<FastDepthPostprocessor>(input_width, input_height);
    }

    VisualizerPtr<DepthResult> createVisualizer() override {
        return std::make_unique<DepthVisualizer>();
    }

    std::string getModelName() const override { return "YOLO26-Depth-S"; }
    std::string getTaskType() const override { return "depth_estimation"; }

    // getInputNormalization() is intentionally NOT overridden. The compiled model
    // consumes uint8 NHWC directly (the /255 normalization is folded in at
    // compile time), so the runner stays on the raw-uint8 path. Declaring a float
    // spec would hand the engine a 4x-oversized float32 buffer -> out-of-bounds
    // read / segfault.
};

}  // namespace dxapp

#endif  // YOLO26_DEPTH_S_FACTORY_HPP
