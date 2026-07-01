/**
 * @file pidnet_s_cityscapes_async.cpp
 * @brief PIDNet-S Cityscapes asynchronous semantic segmentation example
 */

#include "factory/pidnet_s_cityscapes_factory.hpp"
#include "common/runner/async_semantic_seg_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Pidnet_s_cityscapesFactory>();
    dxapp::AsyncSemanticSegRunner<dxapp::Pidnet_s_cityscapesFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
