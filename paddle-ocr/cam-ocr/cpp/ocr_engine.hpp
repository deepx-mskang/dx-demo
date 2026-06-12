#pragma once

#include <dxrt/dxrt_api.h>
#include <opencv2/opencv.hpp>

#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace camocr {

struct OcrText {
    int bboxIndex = -1;
    std::vector<cv::Point2f> bbox;
    std::string text;
    double score = 0.0;
};

struct PerfStats {
    double detTimeMs = 0.0;
    double clsTimeMs = 0.0;
    double recTimeMs = 0.0;
    double e2eTimeMs = 0.0;
    double cps = 0.0;
    int totalChars = 0;
    int numBoxes = 0;
    int numCrops = 0;
};

struct OcrResult {
    cv::Mat preprocessedImage;
    std::vector<std::vector<cv::Point2f>> boxes;
    std::vector<OcrText> texts;
    PerfStats perf;
};

struct EngineOptions {
    std::filesystem::path rootDir;
    std::string language = "ch";
    std::string modelProfile = "mobile";
    bool enableUvdoc = false;
};

class PaddleOcrEngine {
public:
    explicit PaddleOcrEngine(const EngineOptions& options);
    OcrResult run(const cv::Mat& bgrImage);
    cv::Mat preprocessDocument(const cv::Mat& bgrImage);

private:
    struct PaddingInfo {
        int origWidth = 0;
        int origHeight = 0;
        int paddedWidth = 0;
        int paddedHeight = 0;
    };

    std::filesystem::path rootDir_;
    std::filesystem::path modelDir_;
    std::filesystem::path fontDir_;
    std::string language_;
    std::string modelProfile_ = "mobile";
    bool enableUvdoc_ = false;

    std::map<int, std::filesystem::path> detModelPaths_;
    std::map<int, std::filesystem::path> recModelPaths_;
    std::unique_ptr<dxrt::InferenceEngine> activeDetModel_;
    int activeDetRes_ = 0;
    int lastLoggedDetRes_ = 0;
    std::unique_ptr<dxrt::InferenceEngine> clsModel_;
    std::map<int, std::unique_ptr<dxrt::InferenceEngine>> activeRecModels_;
    std::unique_ptr<dxrt::InferenceEngine> docOriModel_;
    std::unique_ptr<dxrt::InferenceEngine> uvdocModel_;
    std::vector<std::string> characters_;

    static std::filesystem::path resolveRoot(const std::filesystem::path& requested);
    std::filesystem::path resolveModelDir() const;
    std::filesystem::path resolveDictPath() const;
    std::string detModelFilename(int res) const;
    std::string recModelFilename(int ratio, const std::string& suffix = "") const;
    std::filesystem::path resolveRecModelPath(int ratio) const;
    std::unique_ptr<dxrt::InferenceEngine> loadEngine(const std::filesystem::path& path) const;
    void loadModels();
    void loadCharacters();
    dxrt::InferenceEngine& detEngine(int res);
    dxrt::InferenceEngine& clsEngine();
    dxrt::InferenceEngine& recEngine(int ratio);
    dxrt::InferenceEngine& docOriEngine();
    dxrt::InferenceEngine& uvdocEngine();
    void releaseDetEngine();
    void releaseRecEngine();
    void releaseDocEngines();

    static int routeDetection(int width, int height);
    static int routeRecognition(int width, int height);
    static cv::Mat resizePpocr(const cv::Mat& image, int targetHeight, int targetWidth, PaddingInfo* paddingInfo);
    static cv::Mat resizeDefault(const cv::Mat& image, int targetHeight, int targetWidth);
    static cv::Mat resizeShortCenterCrop(const cv::Mat& image, int shortSide, int cropHeight, int cropWidth);
    static cv::Mat rotateIfVertical(const cv::Mat& crop);
    static cv::Mat getRotateCropImage(const cv::Mat& image, const std::vector<cv::Point2f>& points);

    std::vector<std::vector<cv::Point2f>> detect(const cv::Mat& image, double* latencyMs);
    std::vector<std::pair<std::string, double>> classify(const std::vector<cv::Mat>& crops, double* latencyMs);
    std::vector<OcrText> recognize(
        const std::vector<std::vector<cv::Point2f>>& boxes,
        const std::vector<cv::Mat>& crops,
        double* latencyMs);

    cv::Mat runDocOrientation(const cv::Mat& image);
    cv::Mat runDocUnwarp(const cv::Mat& image);

    static cv::Mat tensorToFloatMat(const dxrt::TensorPtr& tensor);
    static std::vector<float> tensorToFloatVector(const dxrt::TensorPtr& tensor);
    std::pair<std::string, double> decodeRecognition(const dxrt::TensorPtr& tensor) const;

    static std::vector<cv::Point2f> getMiniBox(const std::vector<cv::Point>& contour, float* minSide);
    static std::vector<cv::Point2f> getMiniBox(const std::vector<cv::Point2f>& points, float* minSide);
    static std::vector<cv::Point2f> unclipApprox(const std::vector<cv::Point>& contour, double unclipRatio);
    static double boxScoreFast(const cv::Mat& bitmap, const std::vector<cv::Point2f>& box);
    static std::vector<cv::Point2f> orderPointsClockwise(const std::vector<cv::Point2f>& points);
    static bool clipAndValidate(std::vector<cv::Point2f>* box, int imageWidth, int imageHeight);
    static void sortBoxes(std::vector<std::vector<cv::Point2f>>* boxes);
};

}  // namespace camocr
