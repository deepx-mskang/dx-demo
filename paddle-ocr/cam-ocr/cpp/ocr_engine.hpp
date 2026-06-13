#pragma once

#include <dxrt/dxrt_api.h>
#include <opencv2/opencv.hpp>

#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
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
    std::filesystem::path modelsBaseDir;
    std::string language = "ch";
    std::string modelProfile = "mobile";
    bool enableUvdoc = false;
    int recAsyncQueueSize = 6;
};

inline constexpr int kRecAsyncQueueMax = 10;
inline constexpr int kRecAsyncQueueDefault = 6;

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
    std::filesystem::path modelsBaseDir_;
    std::filesystem::path detModelDir_;
    std::filesystem::path recModelDir_;
    std::filesystem::path fontDir_;
    std::string language_;
    std::string modelProfile_ = "mobile";
    std::string detModelProfile_ = "mobile";
    std::string recModelProfile_ = "mobile";
    bool enableUvdoc_ = false;
    int recAsyncQueueSize_ = kRecAsyncQueueDefault;

    std::filesystem::path detModelPath_;
    std::map<int, std::filesystem::path> recModelPaths_;
    std::unique_ptr<dxrt::InferenceEngine> detModel_;
    std::unique_ptr<dxrt::InferenceEngine> clsModel_;
    std::map<int, std::unique_ptr<dxrt::InferenceEngine>> recModels_;
    std::vector<std::string> characters_;
    std::mutex recInflightMutex_;
    std::condition_variable recInflightCv_;
    int recInflight_{0};
    bool recCallbacksRegistered_{false};

    static std::filesystem::path resolveRoot(const std::filesystem::path& requested);
    std::filesystem::path modelsAssetsDir(const std::string& profile) const;
    std::filesystem::path resolveModelDir(const std::string& profile) const;
    std::filesystem::path resolveDictPath() const;
    static std::string detModelFilenameForProfile(int res, const std::string& profile);
    static std::string recModelFilenameForProfile(int ratio, const std::string& profile, const std::string& suffix = "");
    std::string detModelFilename() const;
    std::string recModelFilename(int ratio, const std::string& suffix = "") const;
    std::filesystem::path resolveRecModelPath(int ratio) const;
    std::unique_ptr<dxrt::InferenceEngine> loadEngine(const std::filesystem::path& path, int bufferCount = 0) const;
    void loadModels();
    void loadCharacters();
    void setupRecAsyncCallbacks();
    bool acquireRecInflightSlot();
    void releaseRecInflightSlot();
    void waitForRecInflightEmpty();
    int onRecAsyncComplete(dxrt::TensorPtrs& outputs, void* userData);
    dxrt::InferenceEngine& detEngine();
    dxrt::InferenceEngine& clsEngine();
    dxrt::InferenceEngine& recEngine(int ratio);

    struct ModelInputShape {
        int height = 0;
        int width = 0;
        int channels = 3;
        std::uint64_t expectedBytes = 0;
    };

    static ModelInputShape getModelInputShape(dxrt::InferenceEngine& engine);
    static void validateModelInput(
        const cv::Mat& input,
        const ModelInputShape& shape,
        const char* stage,
        const std::string& modelName);
    static dxrt::TensorPtrs runInference(dxrt::InferenceEngine& engine, const cv::Mat& input, const char* stage);

    static int routeRecognition(int width, int height);
    static cv::Mat resizePpocr(const cv::Mat& image, int targetHeight, int targetWidth, PaddingInfo* paddingInfo);
    static cv::Mat resizeDefault(const cv::Mat& image, int targetHeight, int targetWidth);
    static cv::Mat rotateIfVertical(const cv::Mat& crop);
    static cv::Mat getRotateCropImage(const cv::Mat& image, const std::vector<cv::Point2f>& points);

    std::vector<std::vector<cv::Point2f>> detect(const cv::Mat& image, double* latencyMs);
    std::vector<std::pair<std::string, double>> classify(const std::vector<cv::Mat>& crops, double* latencyMs);
    std::vector<OcrText> recognize(
        const std::vector<std::vector<cv::Point2f>>& boxes,
        const std::vector<cv::Mat>& crops,
        double* latencyMs);

    static cv::Mat tensorToFloatMat(const dxrt::TensorPtr& tensor);
    static std::vector<float> tensorToFloatVector(const dxrt::TensorPtr& tensor);
    std::pair<std::string, double> decodeRecognition(const dxrt::TensorPtr& tensor) const;

    static std::vector<cv::Point2f> getMiniBox(const std::vector<cv::Point>& contour, float* minSide);
    static std::vector<cv::Point2f> getMiniBox(const std::vector<cv::Point2f>& points, float* minSide);
    static std::vector<cv::Point2f> unclipApprox(const std::vector<cv::Point>& contour, double unclipRatio);
    static double boxScoreFast(const cv::Mat& bitmap, const std::vector<cv::Point2f>& box);
    static std::vector<cv::Point2f> orderPointsClockwise(const std::vector<cv::Point2f>& points);
    static bool clipAndValidate(std::vector<cv::Point2f>* box, int imageWidth, int imageHeight);
    static std::vector<std::vector<cv::Point2f>> clipAndFilterBoxes(
        std::vector<std::vector<cv::Point2f>> boxes,
        int imageWidth,
        int imageHeight,
        int minSide);
    static void filterValidCrops(
        std::vector<std::vector<cv::Point2f>>* boxes,
        std::vector<cv::Mat>* crops,
        int minSide);
    static bool isValidCropForRecognition(const cv::Mat& crop);
    static void sortBoxes(std::vector<std::vector<cv::Point2f>>* boxes);
};

}  // namespace camocr
