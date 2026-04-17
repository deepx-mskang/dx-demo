#include <algorithm>
#include "ssd.h"
#if _WIN32
#include <io.h>
#define access _access
#define F_OK 0
#endif

void PriorBoxParam::Show()
{
    std::cout << "  PriorBoxParam: " << std::endl
         << "    layers: " << num_layers << ", "
         << "min_scale: " << min_scale << ", "
         << "max_scale: " << max_scale << ", "
         << "center_variance: " << center_variance << ", "
         << "size_variance: " << size_variance << std::endl;
    for (auto &d : dim)
    {
        std::cout << "    " << d.num_grid_x << " x " << d.num_grid_y << " x " << d.num_boxes << std::endl;
    }
}
void SsdParam::Show()
{
    std::cout << "  SsdParam: " << std::endl
         << "    score_threshold: " << score_threshold << ", "
         << "iou_threshold: " << iou_threshold << ", "
         << "num_classes: " << num_classes
         << ", "
         << "use_softmax:" << use_softmax << std::endl;
    std::cout << "    - classes: [";
    for (auto &c : class_names)
        std::cout << c << ", ";
    std::cout << "]" << std::endl;
    priorBoxes.Show();
}
Ssd::Ssd(SsdParam &_cfg, std::vector<dxrt::Tensor> &_datainfo)
    : cfg(_cfg), datainfo(_datainfo)
{
    // setup number of boxes, classes, layers
    numClasses = cfg.num_classes;
    numBoxes = 0;
    for (auto &dim : cfg.priorBoxes.dim)
    {
        numBoxes += dim.num_grid_x * dim.num_grid_y * dim.num_boxes;
    }
    numLayers = cfg.priorBoxes.num_layers;
    ClassNames = cfg.class_names;
    // memory allocate
#if 0
    Boxes = new float[numBoxes * 4];
    Scores = new float[numBoxes * numClasses];
    PriorBoxes = new float[numBoxes * 4];
#else
    Boxes = std::vector<float>(numBoxes * 4);
    Scores = std::vector<float>(numBoxes * numClasses);
    PriorBoxes = std::vector<float>(numBoxes * 4);
#endif

    std::cout << "Ssd created : " << numBoxes << " boxes, " << numClasses << " classes, "
         << numLayers << " layers." << std::endl;
    cfg.Show();
    // create prior boxes
    CreatePriorBoxes(cfg.priorBoxes.data_file);
    // prepare score indices
    for (uint32_t i = 0; i < numClasses; i++)
    {
        std::vector<std::pair<float, int>> v;
        ScoreIndices.emplace_back(v);
    }
    // setup output layers
    auto find_index = std::find(cfg.loc_names.begin(), cfg.loc_names.end(), datainfo[0].name());
    if(find_index == cfg.loc_names.end())
        location_front = 0;
    if(datainfo.size() < 3)
        use_ort = true;
    for (uint32_t layer = 0; layer < numLayers; layer++)
    {
        OutputLayer outputLayer = {};
        auto priorBox = cfg.priorBoxes.dim[layer];
        outputLayer.boxes = priorBox.num_boxes;
        outputLayer.gridX = priorBox.num_grid_x;
        outputLayer.gridY = priorBox.num_grid_y;
        for (size_t i = 0; i < datainfo.size(); i++)
        {
            auto d = datainfo[i];
            if (d.name() == cfg.score_names[layer])
            {
                outputLayer.scoreAlign = d.shape()[3];
                // outputLayer.scoreOffset = d.mem_offset;
                layerMap[d.name()] = i / 2;
            }
            else if (d.name() == cfg.loc_names[layer])
            {
                outputLayer.locAlign = d.shape()[3];
                // outputLayer.locOffset = d.mem_offset;
                layerMap[d.name()] = i / 2;
            }
        }
        OutputLayers.emplace_back(outputLayer);
        outputLayer.Show();
    }
}

Ssd::Ssd() {}
Ssd::~Ssd() {}

void Ssd::CreatePriorBoxes(const std::string &file)
{
    std::cout << "Create Prior Boxes from file: " << file << std::endl;
    int ret = access(file.c_str(), F_OK); /* Not OK : -1 */
    if (ret != -1)
    {
        std::ifstream fin(file, std::ios_base::binary);
        fin.read((char *)&PriorBoxes[0], sizeof(float) * numBoxes * 4);
        fin.close();
    }
    else
    {
        std::cout << __func__ << ": " << file << " doesn't exist." << std::endl;
        exit(-1);
    }
}

static bool scoreComapre(const std::pair<float, int> &a, const std::pair<float, int> &b)
{
    if(a.first > b.first)
        return true;
    else
        return false;
};

void Ssd::FilterWithSoftmax(float *org, std::vector<std::shared_ptr<dxrt::Tensor>> outputs_)
{
    int boxIdx = 0;
    int incLoc = 4;
    int incScore = numClasses;
#if 0
    int x = 1, y = 0, w = 3, h = 2;
#else
    int x = 0, y = 1, w = 2, h = 3;
#endif
    float scoreThreshold = cfg.score_threshold;
    float *boxLocation, *classScore;
    float centerVariance = cfg.priorBoxes.center_variance;
    float sizeVariance = cfg.priorBoxes.size_variance;
    float center_x, center_y, width, height;
    if(use_ort)
    {
        auto outputs = outputs_.front(); // front : location, back : scores
        auto resultNum = outputs->shape()[1];
        auto strideNum = outputs->shape()[2];
        for(int i=0;i<resultNum;i++)
        {
            float *cls_data = (float*)outputs_[0]->data() + (strideNum * i);
            float *loc_data = (float*)outputs_[1]->data() + (strideNum * i);
            float maxScore = scoreThreshold;
            int maxClsIndex = -1;
            for(int cls=0;cls<incScore;cls++)
            {
                if (cls_data[cls] >= maxScore)
                {
                    maxScore = cls_data[cls];
                    maxClsIndex = cls;
                }
            }
            if(maxClsIndex > 0)
            {
                ScoreIndices[maxClsIndex - 1].emplace_back(maxScore, boxIdx);
                float *boxOut = &Boxes[boxIdx * 4];

                boxOut[0] = loc_data[0];
                boxOut[1] = loc_data[1];
                boxOut[2] = loc_data[2];
                boxOut[3] = loc_data[3];
                boxIdx++;
            }
        }
    }
    else
    {
        for (uint32_t layer = 0; layer < numLayers; layer++)
        {
            auto outputLayer = OutputLayers[layer];
            int numGridX = outputLayer.gridX;
            int numGridY = outputLayer.gridY;
            int tensorIdx = layerMap[cfg.score_names[layer]];
            int _numBoxes = outputLayer.boxes;
            int inc1 = 1;
            for (int gY = 0; gY < numGridY; gY++)
            {
                for (int gX = 0; gX < numGridX; gX++)
                {
                    if (org == nullptr)
                    {
                        classScore = (float *)(static_cast<uint8_t*>(outputs_[2 * tensorIdx + location_front]->data()) + sizeof(float) * outputLayer.scoreAlign * (gY * numGridX + gX));  
                        boxLocation = (float *)(static_cast<uint8_t*>(outputs_[2 * tensorIdx + !location_front]->data()) + sizeof(float) * outputLayer.locAlign * (gY * numGridX + gX));
                    }
                    else
                    {
                        classScore = org + outputLayer.scoreOffset / 4 + outputLayer.scoreAlign * (gY * numGridX + gX);
                        boxLocation = org + outputLayer.locOffset / 4 + outputLayer.locAlign * (gY * numGridX + gX);
                    }
                    for (int box = 0; box < _numBoxes; box++)
                    {
                        bool boxDecoded = false;
                        float sum = 0;
                        for (uint32_t cls = 0; cls < numClasses; cls++)
                        {
                            sum += std::exp(classScore[cls * inc1]);
                        }
                        for (uint32_t cls = cfg.start_class; cls < numClasses; cls++)
                        {
                            float score = std::exp(classScore[cls * inc1]) * (1 / sum);
                            if (score > scoreThreshold)
                            {
                                ScoreIndices[cls].emplace_back(score, boxIdx);
                                if (!boxDecoded)
                                {
                                    float *boxOut = &Boxes[boxIdx * 4];
                                    float *prior = &PriorBoxes[boxIdx * 4];
                                    center_x = prior[0] + boxLocation[x * inc1] * centerVariance * prior[2];
                                    center_y = prior[1] + boxLocation[y * inc1] * centerVariance * prior[3];
                                    width = std::exp(boxLocation[w * inc1] * sizeVariance) * prior[2];
                                    height = std::exp(boxLocation[h * inc1] * sizeVariance) * prior[3];
                                    boxOut[0] = center_x - width / 2;
                                    boxOut[1] = center_y - height / 2;
                                    boxOut[2] = center_x + width / 2;
                                    boxOut[3] = center_y + height / 2;
                                    boxDecoded = true;
                                }
                            }
                        }
                        boxLocation += incLoc;
                        classScore += incScore;
                        boxIdx++;
                    }
                }
            }
        }
    }
    for (uint32_t cls = 1; cls < numClasses; cls++)
    {
        std::sort(ScoreIndices[cls].begin(), ScoreIndices[cls].end(), scoreComapre);
    }
}
void Ssd::FilterWithSigmoid(float *org, std::vector<std::shared_ptr<dxrt::Tensor>> outputs_)
{
    int boxIdx = 0;
    int incLoc = 4;
    int incScore = numClasses;
    int x = 1, y = 0, w = 3, h = 2;
    float scoreThreshold = cfg.score_threshold;
    float rawThreshold = std::log(scoreThreshold / (1 - scoreThreshold));
    float *boxLocation, *classScore;
    float centerVariance = cfg.priorBoxes.center_variance;
    float sizeVariance = cfg.priorBoxes.size_variance;
    float center_x, center_y, width, height;
    for (uint32_t layer = 0; layer < numLayers; layer++)
    {
        auto outputLayer = OutputLayers[layer];
        int numGridX = outputLayer.gridX;
        int numGridY = outputLayer.gridY;
        int tensorIdx = layerMap[cfg.score_names[layer]];
        // outputs_[2*layer]->Show();
        // outputs_[2*layer+1]->Show();
        // outputLayer.Show();
        int _numBoxes = outputLayer.boxes;
        int inc1 = 1;
        for (int gY = 0; gY < numGridY; gY++)
        {
            for (int gX = 0; gX < numGridX; gX++)
            {
                if (org == nullptr)
                {
                    classScore = (float *)(static_cast<uint8_t*>(outputs_[2 * tensorIdx + location_front]->data()) + sizeof(float) * outputLayer.scoreAlign * (gY * numGridX + gX));  
                    boxLocation = (float *)(static_cast<uint8_t*>(outputs_[2 * tensorIdx + !location_front]->data()) + sizeof(float) * outputLayer.locAlign * (gY * numGridX + gX));
                }
                else
                {
                    classScore = org + outputLayer.scoreOffset / 4 + outputLayer.scoreAlign * (gY * numGridX + gX);
                    boxLocation = org + outputLayer.locOffset / 4 + outputLayer.locAlign * (gY * numGridX + gX);
                }
                for (int box = 0; box < _numBoxes; box++)
                {
                    bool boxDecoded = false;
                    for (uint32_t cls = 1; cls < numClasses; cls++)
                    {
                        float score = classScore[cls * inc1];
                        if (score > rawThreshold)
                        {
                            ScoreIndices[cls].emplace_back(sigmoid(score), boxIdx);
                            if (!boxDecoded)
                            {
                                float *boxOut = &Boxes[boxIdx * 4];
                                float *prior = &PriorBoxes[boxIdx * 4];
                                center_x = prior[0] + boxLocation[x * inc1] * centerVariance * prior[2];
                                center_y = prior[1] + boxLocation[y * inc1] * centerVariance * prior[3];
                                width = std::exp(boxLocation[w * inc1] * sizeVariance) * prior[2];
                                height = std::exp(boxLocation[h * inc1] * sizeVariance) * prior[3];
                                boxOut[0] = center_x - width / 2;
                                boxOut[1] = center_y - height / 2;
                                boxOut[2] = center_x + width / 2;
                                boxOut[3] = center_y + height / 2;
                                boxDecoded = true;
                            }
                        }
                    }
                    boxLocation += incLoc;
                    classScore += incScore;
                    boxIdx++;
                }
            }
        }
    }
    for (uint32_t cls = 1; cls < numClasses; cls++)
    {
        std::sort(ScoreIndices[cls].begin(), ScoreIndices[cls].end(), scoreComapre);
    }
}

std::vector<BoundingBox> Ssd::PostProc(std::vector<std::shared_ptr<dxrt::Tensor>> outputs_)
{
    outputs = outputs_;
    for (uint32_t cls = 1; cls < numClasses; cls++)
    {
        ScoreIndices[cls].clear();
    }
    Result.clear();
    if (cfg.use_softmax)
        FilterWithSoftmax(nullptr, outputs);
    else
        FilterWithSigmoid(nullptr, outputs);
    Nms(
        numClasses,
        0,
        ClassNames,
        ScoreIndices, &Boxes[0], cfg.iou_threshold,
        Result,
        1);
    return Result;
}
