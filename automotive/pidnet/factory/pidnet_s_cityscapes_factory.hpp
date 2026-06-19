/**
 * @file pidnet_s_cityscapes_factory.hpp
 * @brief PIDNet-S Cityscapes factory for semantic segmentation
 */

#ifndef PIDNET_S_CITYSCAPES_FACTORY_HPP
#define PIDNET_S_CITYSCAPES_FACTORY_HPP

#include "common/base/i_factory.hpp"
#include "common/config/model_config.hpp"
#include "common/processors/simple_resize_preprocessor.hpp"
#include "common/processors/segmentation_postprocessor.hpp"
#include "common/visualizers/segmentation_visualizer.hpp"

namespace dxapp {

class Pidnet_s_cityscapesFactory : public ISegmentationFactory {
public:
    Pidnet_s_cityscapesFactory() = default;

    void loadConfig(const ModelConfig& config) override {
        pidnet_argmax_scale_ = config.get<double>("pidnet_argmax_scale", pidnet_argmax_scale_);
    }

    PreprocessorPtr createPreprocessor(int input_width, int input_height) override {
        return std::make_unique<SimpleResizePreprocessor>(input_width, input_height);
    }

    PostprocessorPtr<SegmentationResult> createPostprocessor(
        int input_width, int input_height) override {
        return std::make_unique<PidNetPostprocessor>(input_width, input_height, pidnet_argmax_scale_);
    }

    VisualizerPtr<SegmentationResult> createVisualizer() override {
        return std::make_unique<SemanticSegmentationVisualizer>();
    }

    std::string getModelName() const override { return "pidnet_s_cityscapes"; }
    std::string getTaskType() const override { return "semantic_segmentation"; }

private:
    double pidnet_argmax_scale_ = 0.0;
};

}  // namespace dxapp

#endif  // PIDNET_S_CITYSCAPES_FACTORY_HPP
