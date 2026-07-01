#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <dxrt/dxrt_api.h>
#include "nms.h"
#define sigmoid(x) (1 / (1 + std::exp(-x)))

struct PriorBoxDim
{
    int num_grid_x;
    int num_grid_y;
    int num_boxes;
};
struct PriorBoxParam
{
    int num_layers;
    float min_scale;
    float max_scale;
    float center_variance;
    float size_variance;
    std::vector<PriorBoxDim> dim;
    std::string data_file;
    void Show();
};
struct SsdParam
{
    int image_size;
    bool use_softmax;
    float score_threshold;
    float iou_threshold;
    int num_classes;
    uint32_t start_class;
    std::vector<std::string> class_names;
    std::vector<std::string> score_names;
    std::vector<std::string> loc_names;
    PriorBoxParam priorBoxes;
    void Show();
};

class Ssd
{
private:
    SsdParam cfg;
    std::vector<dxrt::Tensor> datainfo;
    std::vector<float> PriorBoxes;
    std::vector<float> Boxes;
    std::vector<float> Scores;
    std::map<std::string, int> layerMap;
    std::vector< std::vector<std::pair<float, int>> > ScoreIndices;
    std::vector< BoundingBox > Result;
    std::vector< std::string > ClassNames;
    // PriorBoxGenerator pb;
    uint32_t numClasses;
    uint32_t numBoxes;
    uint32_t numLayers;
    int location_front = 1;
    bool use_ort = false;
    struct OutputLayer
    {
        unsigned int locAlign;
        unsigned int locOffset;
        unsigned int scoreOffset;
        unsigned int scoreAlign;
        int     boxes;
        int     gridX;
        int     gridY;
        void Show() {
            std::cout << "OutputLayer: " << gridX << "x" << gridY << ", "
                << boxes << " boxes, loc at " << std::hex << locOffset << 
                ", score at " << scoreOffset << ", loc align " << std::hex << locAlign << 
                ", score align " << scoreAlign;
            std::cout << std::dec << std::endl;
        }
    };
    std::vector<OutputLayer> OutputLayers;
    std::vector<std::shared_ptr<dxrt::Tensor>> outputs;
public:
    ~Ssd();
    Ssd();
    Ssd(SsdParam &_cfg, std::vector<dxrt::Tensor> &_datainfo);
    void SetOutputLayer();
    void CreatePriorBoxes(const std::string &f); /* Import Eyenix-specific prior box bin. data */
    // void CreatePriorBoxes(PriorBoxGenerator &pb);
    std::vector< BoundingBox > PostProc(float *data);
    std::vector< BoundingBox > PostProc(std::vector<std::shared_ptr<dxrt::Tensor>> outputs_);
    void FilterWithSoftmax(float *data, std::vector<std::shared_ptr<dxrt::Tensor>> outputs_);
    void FilterWithSigmoid(float *data, std::vector<std::shared_ptr<dxrt::Tensor>> outputs_);
    void ShowResult(void){
        std::cout << "  Detected " << std::dec << Result.size() << " boxes." << std::endl;
        for(int i=0;i<(int)Result.size();i++)
        {
            Result[i].Show();
        }
    }
};
