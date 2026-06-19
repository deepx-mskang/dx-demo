/**
 * @file pidnet_s_cityscapes_sync.cpp
 * @brief PIDNet-S Cityscapes synchronous semantic segmentation example
 */

#include "factory/pidnet_s_cityscapes_factory.hpp"
#include "common/runner/sync_semantic_seg_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::Pidnet_s_cityscapesFactory>();
    dxapp::SyncSemanticSegRunner<dxapp::Pidnet_s_cityscapesFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
