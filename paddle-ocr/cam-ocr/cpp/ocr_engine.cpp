#include "ocr_engine.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

namespace camocr {
namespace {

constexpr double kDetThresh = 0.3;
constexpr double kDetBoxThresh = 0.6;
constexpr double kDetUnclipRatio = 1.5;
constexpr int kDetMaxCandidates = 1500;
constexpr double kClsThresh = 0.9;
constexpr double kRecDropScore = 0.3;
constexpr int kMaxOcrBoxes = 50;
constexpr int kMinBoxSide = 4;
constexpr int kMinCropSide = 4;
constexpr int kMaxCropSide = 5000;

struct RecAsyncJob {
    cv::Mat input;
    std::size_t cropIndex = 0;
    std::vector<std::pair<std::string, double>>* decodedResults = nullptr;
    PaddleOcrEngine* engine = nullptr;
    std::mutex* inflightMutex = nullptr;
    std::condition_variable* inflightCv = nullptr;
    int* inflightCount = nullptr;
};

double elapsedMs(const std::chrono::steady_clock::time_point& start)
{
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

std::int64_t product(const std::vector<int64_t>& shape)
{
    std::int64_t result = 1;
    for (const auto dim : shape) {
        if (dim > 0) {
            result *= dim;
        }
    }
    return result;
}

std::size_t matInputBytes(const cv::Mat& mat)
{
    return mat.total() * mat.elemSize();
}

}  // namespace

PaddleOcrEngine::ModelInputShape PaddleOcrEngine::getModelInputShape(dxrt::InferenceEngine& engine)
{
    ModelInputShape out;
    out.expectedBytes = engine.GetInputSize();

    const dxrt::Tensors inputs = engine.GetInputs();
    if (inputs.empty()) {
        throw std::runtime_error("Model has no input tensors: " + engine.GetModelName());
    }

    const auto& shape = inputs.front().shape();
    if (shape.size() == 4) {
        if (shape[3] == 1 || shape[3] == 3) {
            out.height = static_cast<int>(shape[1]);
            out.width = static_cast<int>(shape[2]);
            out.channels = static_cast<int>(shape[3]);
        } else if (shape[1] == 1 || shape[1] == 3) {
            out.channels = static_cast<int>(shape[1]);
            out.height = static_cast<int>(shape[2]);
            out.width = static_cast<int>(shape[3]);
        } else {
            out.height = static_cast<int>(shape[1]);
            out.width = static_cast<int>(shape[2]);
            out.channels = static_cast<int>(shape[3]);
        }
    } else if (shape.size() == 3) {
        out.height = static_cast<int>(shape[0]);
        out.width = static_cast<int>(shape[1]);
        out.channels = static_cast<int>(shape[2]);
    } else {
        throw std::runtime_error("Unsupported input tensor rank for model: " + engine.GetModelName());
    }

    if (out.expectedBytes == 0) {
        out.expectedBytes = static_cast<std::uint64_t>(out.height) * out.width * out.channels;
    }

    return out;
}

void PaddleOcrEngine::validateModelInput(
    const cv::Mat& input,
    const ModelInputShape& shape,
    const char* stage,
    const std::string& modelName)
{
    const std::size_t actualBytes = matInputBytes(input);
    const std::size_t shapeBytes =
        static_cast<std::size_t>(shape.height) * shape.width * shape.channels;

    std::cout << "[OCR] " << stage << " input check (" << modelName << "): "
              << "mat=" << input.cols << "x" << input.rows << "x" << input.channels()
              << " (" << actualBytes << " bytes), "
              << "model=" << shape.width << "x" << shape.height << "x" << shape.channels
              << " (" << shape.expectedBytes << " bytes expected)" << std::endl;

    assert(shape.height > 0);
    assert(shape.width > 0);
    assert(shape.channels > 0);
    assert(input.rows == shape.height);
    assert(input.cols == shape.width);
    assert(input.channels() == shape.channels);
    assert(actualBytes == shapeBytes);
    assert(actualBytes == shape.expectedBytes);

    if (input.rows != shape.height || input.cols != shape.width || input.channels() != shape.channels) {
        throw std::runtime_error(
            std::string(stage) + " input dimension mismatch for model " + modelName);
    }
    if (actualBytes != shape.expectedBytes) {
        throw std::runtime_error(
            std::string(stage) + " input byte size mismatch for model " + modelName +
            ": got " + std::to_string(actualBytes) + ", expected " + std::to_string(shape.expectedBytes));
    }
    if (!input.isContinuous()) {
        throw std::runtime_error(std::string(stage) + " input must be continuous for model " + modelName);
    }
}

dxrt::TensorPtrs PaddleOcrEngine::runInference(
    dxrt::InferenceEngine& engine,
    const cv::Mat& input,
    const char* stage)
{
    const ModelInputShape shape = getModelInputShape(engine);
    validateModelInput(input, shape, stage, engine.GetModelName());
    return engine.Run(input.data);
}

namespace {

std::string trimLine(std::string line)
{
    while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) {
        line.pop_back();
    }
    return line;
}

}  // namespace

PaddleOcrEngine::PaddleOcrEngine(const EngineOptions& options)
    : rootDir_(resolveRoot(options.rootDir)),
      modelsBaseDir_(options.modelsBaseDir.empty() ? resolveRoot(options.rootDir) / "cpp" / "assets"
                                                   : fs::absolute(options.modelsBaseDir)),
      language_(options.language),
      modelProfile_(options.modelProfile),
      enableUvdoc_(options.enableUvdoc),
      recAsyncQueueSize_(std::clamp(options.recAsyncQueueSize, 1, kRecAsyncQueueMax))
{
    if (modelProfile_ != "mobile" && modelProfile_ != "server" && modelProfile_ != "hybrid") {
        throw std::runtime_error("Unsupported OCR model profile: " + modelProfile_);
    }
    if (modelProfile_ == "hybrid") {
        detModelProfile_ = "mobile";
        recModelProfile_ = "server";
    } else {
        detModelProfile_ = modelProfile_;
        recModelProfile_ = modelProfile_;
    }
    detModelDir_ = resolveModelDir(detModelProfile_);
    recModelDir_ = resolveModelDir(recModelProfile_);
    fontDir_ = rootDir_ / "engine" / "fonts";
    if (enableUvdoc_) {
        std::cerr << "[OCR] Warning: --enable-uvdoc is ignored (only det/cls/rec models are loaded)."
                  << std::endl;
    }
    loadCharacters();
    loadModels();
}

fs::path PaddleOcrEngine::resolveRoot(const fs::path& requested)
{
    if (!requested.empty() && fs::exists(requested)) {
        return fs::absolute(requested);
    }
#ifdef CAM_OCR_ROOT_DIR
    fs::path compiledRoot = CAM_OCR_ROOT_DIR;
    if (fs::exists(compiledRoot)) {
        return fs::absolute(compiledRoot);
    }
#endif
    return fs::current_path();
}

fs::path PaddleOcrEngine::modelsAssetsDir(const std::string& profile) const
{
    return modelsBaseDir_ / profile;
}

fs::path PaddleOcrEngine::resolveModelDir(const std::string& profile) const
{
    const std::vector<fs::path> candidates = {
        modelsAssetsDir(profile),
        rootDir_ / "engine" / "model_files" / profile,
        rootDir_ / "engine" / "model_files",
    };

    for (const auto& candidate : candidates) {
        if (fs::exists(candidate / detModelFilenameForProfile(640, profile)) &&
            fs::exists(candidate / recModelFilenameForProfile(3, profile))) {
            return candidate;
        }
    }

    throw std::runtime_error(
        "OCR model files were not found for profile '" + profile +
        "'. Expected models under " + modelsAssetsDir(profile).string() +
        " or engine/model_files/" + profile + ".");
}

fs::path PaddleOcrEngine::resolveDictPath() const
{
    fs::path langDict;
    if (language_ == "korean") {
        langDict = recModelDir_ / "ppocrv5_korean_dict.txt";
    } else if (language_ == "german") {
        langDict = recModelDir_ / "ppocrv5_latin_dict.txt";
    }

    if (!langDict.empty() && fs::exists(langDict)) {
        return langDict;
    }
    if (fs::exists(recModelDir_ / "ppocrv5_dict.txt")) {
        return recModelDir_ / "ppocrv5_dict.txt";
    }
    return modelsBaseDir_ / "ppocrv5_dict.txt";
}

std::string PaddleOcrEngine::detModelFilenameForProfile(int res, const std::string& profile)
{
    if (profile == "mobile") {
        return "det_mobile_" + std::to_string(res) + ".dxnn";
    }
    return "det_v5_" + std::to_string(res) + ".dxnn";
}

std::string PaddleOcrEngine::recModelFilenameForProfile(int ratio, const std::string& profile, const std::string& suffix)
{
    const std::string base = (profile == "mobile") ? "rec_mobile_ratio_" : "rec_v5_ratio_";
    return base + std::to_string(ratio) + suffix + ".dxnn";
}

std::string PaddleOcrEngine::detModelFilename() const
{
    return detModelFilenameForProfile(640, detModelProfile_);
}

std::string PaddleOcrEngine::recModelFilename(int ratio, const std::string& suffix) const
{
    return recModelFilenameForProfile(ratio, recModelProfile_, suffix);
}

fs::path PaddleOcrEngine::resolveRecModelPath(int ratio) const
{
    std::string suffix;
    if (language_ == "korean") {
        suffix = "_korean";
    } else if (language_ == "german") {
        suffix = "_latin";
    }

    if (!suffix.empty()) {
        fs::path langPath = recModelDir_ / recModelFilename(ratio, suffix);
        if (fs::exists(langPath)) {
            return langPath;
        }
    }
    return recModelDir_ / recModelFilename(ratio);
}

std::unique_ptr<dxrt::InferenceEngine> PaddleOcrEngine::loadEngine(const fs::path& path, int bufferCount) const
{
    if (!fs::exists(path)) {
        throw std::runtime_error("Missing model: " + path.string());
    }
    dxrt::InferenceOption option;
    option.useORT = true;
    if (bufferCount > 0) {
        option.bufferCount = bufferCount;
    }
    return std::make_unique<dxrt::InferenceEngine>(path.string(), option);
}

void PaddleOcrEngine::loadModels()
{
    detModelPath_ = detModelDir_ / detModelFilename();

    for (int ratio : {3, 5, 10, 15, 25, 35}) {
        recModelPaths_[ratio] = resolveRecModelPath(ratio);
    }

    if (!fs::exists(detModelPath_)) {
        throw std::runtime_error("Missing model: " + detModelPath_.string());
    }
    if (!fs::exists(detModelDir_ / "textline_ori.dxnn")) {
        throw std::runtime_error("Missing model: " + (detModelDir_ / "textline_ori.dxnn").string());
    }
    for (const auto& [_, path] : recModelPaths_) {
        if (!fs::exists(path)) {
            throw std::runtime_error("Missing model: " + path.string());
        }
    }

    std::cout << "[OCR] Model profile: " << modelProfile_ << std::endl;
    std::cout << "[OCR] Detection/classification: " << detModelProfile_
              << " (" << detModelDir_ << ")" << std::endl;
    std::cout << "[OCR] Recognition: " << recModelProfile_
              << " (" << recModelDir_ << ")" << std::endl;
    std::cout << "[OCR] Recognition async queue size: " << recAsyncQueueSize_ << std::endl;
    std::cout << "[OCR] Loading det/cls/rec models at startup..." << std::endl;

    std::cout << "[OCR]   det: " << detModelPath_.filename().string() << std::endl;
    detModel_ = loadEngine(detModelPath_);

    clsModel_ = loadEngine(detModelDir_ / "textline_ori.dxnn");
    std::cout << "[OCR]   cls: textline_ori.dxnn" << std::endl;

    for (const auto& [ratio, path] : recModelPaths_) {
        std::cout << "[OCR]   rec: " << path.filename().string() << std::endl;
        recModels_[ratio] = loadEngine(path, recAsyncQueueSize_);
    }

    setupRecAsyncCallbacks();
    std::cout << "[OCR] All det/cls/rec models loaded and kept resident." << std::endl;
}

void PaddleOcrEngine::setupRecAsyncCallbacks()
{
    if (recCallbacksRegistered_) {
        return;
    }

    for (auto& [ratio, model] : recModels_) {
        (void)ratio;
        model->RegisterCallback([this](dxrt::TensorPtrs& outputs, void* userData) -> int {
            return onRecAsyncComplete(outputs, userData);
        });
    }
    recCallbacksRegistered_ = true;
}

bool PaddleOcrEngine::acquireRecInflightSlot()
{
    std::unique_lock<std::mutex> lock(recInflightMutex_);
    recInflightCv_.wait(lock, [this] { return recInflight_ < recAsyncQueueSize_; });
    ++recInflight_;
    return true;
}

void PaddleOcrEngine::releaseRecInflightSlot()
{
    {
        std::lock_guard<std::mutex> lock(recInflightMutex_);
        if (recInflight_ > 0) {
            --recInflight_;
        }
    }
    recInflightCv_.notify_all();
}

void PaddleOcrEngine::waitForRecInflightEmpty()
{
    std::unique_lock<std::mutex> lock(recInflightMutex_);
    recInflightCv_.wait(lock, [this] { return recInflight_ == 0; });
}

int PaddleOcrEngine::onRecAsyncComplete(dxrt::TensorPtrs& outputs, void* userData)
{
    auto job = std::unique_ptr<RecAsyncJob>(static_cast<RecAsyncJob*>(userData));
    if (!job) {
        releaseRecInflightSlot();
        return -1;
    }

    try {
        if (job->engine != nullptr && job->decodedResults != nullptr && !outputs.empty()) {
            (*job->decodedResults)[job->cropIndex] = job->engine->decodeRecognition(outputs.front());
        }
    } catch (const std::exception& e) {
        std::cerr << "[OCR] Recognition async callback error: " << e.what() << std::endl;
    }

    if (job->inflightMutex != nullptr && job->inflightCv != nullptr && job->inflightCount != nullptr) {
        {
            std::lock_guard<std::mutex> lock(*job->inflightMutex);
            if (*job->inflightCount > 0) {
                --(*job->inflightCount);
            }
        }
        job->inflightCv->notify_all();
    } else {
        releaseRecInflightSlot();
    }

    return 0;
}

dxrt::InferenceEngine& PaddleOcrEngine::detEngine()
{
    return *detModel_;
}

dxrt::InferenceEngine& PaddleOcrEngine::clsEngine()
{
    return *clsModel_;
}

dxrt::InferenceEngine& PaddleOcrEngine::recEngine(int ratio)
{
    return *recModels_.at(ratio);
}

void PaddleOcrEngine::loadCharacters()
{
    const fs::path dictPath = resolveDictPath();
    std::ifstream in(dictPath, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Failed to open recognition dictionary: " + dictPath.string());
    }

    characters_.clear();
    characters_.push_back("blank");

    std::string line;
    while (std::getline(in, line)) {
        characters_.push_back(trimLine(line));
    }
    characters_.push_back(" ");

    std::cout << "[OCR] Dictionary: " << dictPath << " (" << characters_.size() << " tokens)" << std::endl;
}

OcrResult PaddleOcrEngine::run(const cv::Mat& bgrImage)
{
    if (bgrImage.empty()) {
        return {};
    }

    const auto start = std::chrono::steady_clock::now();
    OcrResult result;
    result.preprocessedImage = preprocessDocument(bgrImage);

    double detMs = 0.0;
    double clsMs = 0.0;
    double recMs = 0.0;

    result.boxes = detect(result.preprocessedImage, &detMs);
    result.boxes = clipAndFilterBoxes(
        std::move(result.boxes),
        result.preprocessedImage.cols,
        result.preprocessedImage.rows,
        kMinBoxSide);
    if (result.boxes.size() > static_cast<std::size_t>(kMaxOcrBoxes)) {
        std::cout << "[OCR] Limiting " << result.boxes.size() << " boxes to " << kMaxOcrBoxes
                  << " for stability" << std::endl;
        result.boxes.resize(kMaxOcrBoxes);
    }

    std::vector<cv::Mat> crops;
    crops.reserve(result.boxes.size());
    for (const auto& box : result.boxes) {
        crops.push_back(rotateIfVertical(getRotateCropImage(result.preprocessedImage, box)));
    }
    filterValidCrops(&result.boxes, &crops, kMinCropSide);

    const auto clsResults = classify(crops, &clsMs);
    for (std::size_t i = 0; i < clsResults.size() && i < crops.size(); ++i) {
        if (clsResults[i].first == "180" && clsResults[i].second > kClsThresh) {
            cv::rotate(crops[i], crops[i], cv::ROTATE_180);
            if (!crops[i].isContinuous()) {
                crops[i] = crops[i].clone();
            }
        }
    }

    result.texts = recognize(result.boxes, crops, &recMs);

    int totalChars = 0;
    for (const auto& text : result.texts) {
        totalChars += static_cast<int>(text.text.size());
    }

    result.perf.detTimeMs = detMs;
    result.perf.clsTimeMs = crops.empty() ? 0.0 : clsMs / static_cast<double>(crops.size());
    result.perf.recTimeMs = crops.empty() ? 0.0 : recMs / static_cast<double>(crops.size());
    result.perf.e2eTimeMs = elapsedMs(start);
    result.perf.totalChars = totalChars;
    result.perf.cps = result.perf.e2eTimeMs > 0.0 ? totalChars / (result.perf.e2eTimeMs / 1000.0) : 0.0;
    result.perf.numBoxes = static_cast<int>(result.boxes.size());
    result.perf.numCrops = static_cast<int>(crops.size());
    return result;
}

cv::Mat PaddleOcrEngine::preprocessDocument(const cv::Mat& bgrImage)
{
    return bgrImage.clone();
}

int PaddleOcrEngine::routeRecognition(int width, int height)
{
    const double ratio = height > 0 ? static_cast<double>(width) / static_cast<double>(height) : 1000.0;
    if (ratio <= 3.0) return 3;
    if (ratio <= 5.0) return 5;
    if (ratio <= 10.0) return 10;
    if (ratio <= 15.0) return 15;
    if (ratio <= 25.0) return 25;
    return 35;
}

cv::Mat PaddleOcrEngine::resizePpocr(const cv::Mat& image, int targetHeight, int targetWidth, PaddingInfo* paddingInfo)
{
    const int origHeight = image.rows;
    const int origWidth = image.cols;
    const double targetRatio = static_cast<double>(targetWidth) / static_cast<double>(targetHeight);
    const double origRatio = static_cast<double>(origWidth) / static_cast<double>(origHeight);

    cv::Mat padded = image;
    int paddedWidth = origWidth;
    int paddedHeight = origHeight;
    if (origRatio < targetRatio) {
        paddedWidth = static_cast<int>(origHeight * targetRatio);
        const int padRight = std::max(0, paddedWidth - origWidth);
        cv::copyMakeBorder(image, padded, 0, 0, 0, padRight, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    }

    if (paddingInfo) {
        paddingInfo->origWidth = origWidth;
        paddingInfo->origHeight = origHeight;
        paddingInfo->paddedWidth = paddedWidth;
        paddingInfo->paddedHeight = paddedHeight;
    }

    cv::Mat resized;
    cv::resize(padded, resized, cv::Size(targetWidth, targetHeight), 0.0, 0.0, cv::INTER_LINEAR);
    return resized.isContinuous() ? resized : resized.clone();
}

cv::Mat PaddleOcrEngine::resizeDefault(const cv::Mat& image, int targetHeight, int targetWidth)
{
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(targetWidth, targetHeight), 0.0, 0.0, cv::INTER_LINEAR);
    return resized.isContinuous() ? resized : resized.clone();
}

cv::Mat PaddleOcrEngine::rotateIfVertical(const cv::Mat& crop)
{
    if (crop.rows > crop.cols * 2) {
        cv::Mat rotated;
        cv::rotate(crop, rotated, cv::ROTATE_90_COUNTERCLOCKWISE);
        return rotated;
    }
    return crop;
}

cv::Mat PaddleOcrEngine::getRotateCropImage(const cv::Mat& image, const std::vector<cv::Point2f>& points)
{
    if (points.size() != 4) {
        return {};
    }
    const int cropWidth = static_cast<int>(std::max(
        cv::norm(points[0] - points[1]),
        cv::norm(points[2] - points[3])));
    const int cropHeight = static_cast<int>(std::max(
        cv::norm(points[0] - points[3]),
        cv::norm(points[1] - points[2])));
    if (cropWidth <= 0 || cropHeight <= 0) {
        return {};
    }

    std::vector<cv::Point2f> dst = {
        {0.0f, 0.0f},
        {static_cast<float>(cropWidth), 0.0f},
        {static_cast<float>(cropWidth), static_cast<float>(cropHeight)},
        {0.0f, static_cast<float>(cropHeight)},
    };
    const cv::Mat transform = cv::getPerspectiveTransform(points, dst);
    cv::Mat crop;
    cv::warpPerspective(image, crop, transform, cv::Size(cropWidth, cropHeight), cv::INTER_CUBIC, cv::BORDER_REPLICATE);
    return crop;
}

std::vector<std::vector<cv::Point2f>> PaddleOcrEngine::detect(const cv::Mat& image, double* latencyMs)
{
    PaddingInfo padding;
    padding.origWidth = image.cols;
    padding.origHeight = image.rows;
    padding.paddedWidth = image.cols;
    padding.paddedHeight = image.rows;

    auto& engine = detEngine();
    const ModelInputShape modelShape = getModelInputShape(engine);

    cv::Mat input;
    if (image.cols == modelShape.width && image.rows == modelShape.height) {
        input = image.isContinuous() ? image : image.clone();
    } else {
        input = resizePpocr(image, modelShape.height, modelShape.width, &padding);
    }

    const auto start = std::chrono::steady_clock::now();
    auto outputs = runInference(engine, input, "detection");
    if (latencyMs) {
        *latencyMs = elapsedMs(start);
    }

    cv::Mat pred = tensorToFloatMat(outputs.front());
    cv::Mat mask;
    cv::threshold(pred, mask, kDetThresh, 1.0, cv::THRESH_BINARY);
    mask.convertTo(mask, CV_8U, 255.0);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    std::vector<std::vector<cv::Point2f>> boxes;
    const int contourCount = std::min(static_cast<int>(contours.size()), kDetMaxCandidates);
    for (int i = 0; i < contourCount; ++i) {
        float minSide = 0.0f;
        auto miniBox = getMiniBox(contours[i], &minSide);
        if (minSide < 3.0f) {
            continue;
        }

        const double score = boxScoreFast(pred, miniBox);
        if (score < kDetBoxThresh) {
            continue;
        }

        auto expanded = unclipApprox(contours[i], kDetUnclipRatio);
        if (expanded.empty()) {
            continue;
        }

        float expandedSide = 0.0f;
        auto box = getMiniBox(expanded, &expandedSide);
        if (expandedSide < 5.0f) {
            continue;
        }

        const double scaleX = static_cast<double>(padding.paddedWidth) / pred.cols;
        const double scaleY = static_cast<double>(padding.paddedHeight) / pred.rows;
        for (auto& p : box) {
            p.x = static_cast<float>(p.x * scaleX);
            p.y = static_cast<float>(p.y * scaleY);
        }

        box = orderPointsClockwise(box);
        if (clipAndValidate(&box, image.cols, image.rows)) {
            boxes.push_back(box);
        }
    }

    sortBoxes(&boxes);
    return boxes;
}

std::vector<std::pair<std::string, double>> PaddleOcrEngine::classify(const std::vector<cv::Mat>& crops, double* latencyMs)
{
    const auto start = std::chrono::steady_clock::now();
    std::vector<std::pair<std::string, double>> results;
    results.reserve(crops.size());

    auto& engine = clsEngine();
    const ModelInputShape modelShape = getModelInputShape(engine);

    for (const auto& crop : crops) {
        if (crop.empty()) {
            results.emplace_back("0", 0.0);
            continue;
        }
        const cv::Mat input = resizeDefault(crop, modelShape.height, modelShape.width);
        auto outputs = runInference(engine, input, "classification");
        const auto logits = tensorToFloatVector(outputs.front());
        if (logits.size() >= 2 && logits[1] > logits[0]) {
            results.emplace_back("180", logits[1]);
        } else {
            results.emplace_back("0", logits.empty() ? 0.0 : logits[0]);
        }
    }

    if (latencyMs) {
        *latencyMs = elapsedMs(start);
    }
    return results;
}

std::vector<OcrText> PaddleOcrEngine::recognize(
    const std::vector<std::vector<cv::Point2f>>& boxes,
    const std::vector<cv::Mat>& crops,
    double* latencyMs)
{
    const auto start = std::chrono::steady_clock::now();
    std::vector<OcrText> results;
    std::vector<std::pair<std::string, double>> decodedResults(crops.size(), {"", 0.0});
    std::map<int, int> lastJobIdByRatio;

    setupRecAsyncCallbacks();

    for (std::size_t i = 0; i < crops.size(); ++i) {
        if (!isValidCropForRecognition(crops[i])) {
            if (!crops[i].empty()) {
                std::cout << "[OCR] Warning: Invalid crop size " << crops[i].cols << "x" << crops[i].rows
                          << " at index " << i << ", skipping" << std::endl;
            }
            continue;
        }

        const int ratio = routeRecognition(crops[i].cols, crops[i].rows);
        auto& engine = recEngine(ratio);
        const ModelInputShape modelShape = getModelInputShape(engine);
        cv::Mat input = resizePpocr(crops[i], modelShape.height, modelShape.width, nullptr);
        validateModelInput(input, modelShape, "recognition", engine.GetModelName());

        std::cout << "[OCR] Recognition async submit: ratio " << ratio
                  << " (" << recModelPaths_.at(ratio).filename().string()
                  << ") crop " << input.cols << "x" << input.rows << std::endl;

        if (!acquireRecInflightSlot()) {
            throw std::runtime_error("Failed to acquire recognition async queue slot");
        }

        auto* job = new RecAsyncJob();
        job->input = input.clone();
        job->cropIndex = i;
        job->decodedResults = &decodedResults;
        job->engine = this;
        job->inflightMutex = &recInflightMutex_;
        job->inflightCv = &recInflightCv_;
        job->inflightCount = &recInflight_;

        try {
            const int jobId = engine.RunAsync(job->input.data, static_cast<void*>(job), nullptr);
            lastJobIdByRatio[ratio] = jobId;
            job = nullptr;
        } catch (...) {
            delete job;
            releaseRecInflightSlot();
            throw;
        }
    }

    for (const auto& [ratio, jobId] : lastJobIdByRatio) {
        if (jobId >= 0) {
            recModels_.at(ratio)->Wait(jobId);
        }
    }
    waitForRecInflightEmpty();

    for (std::size_t i = 0; i < decodedResults.size() && i < boxes.size(); ++i) {
        const auto& decoded = decodedResults[i];
        if (decoded.second > kRecDropScore) {
            OcrText text;
            text.bboxIndex = static_cast<int>(i);
            text.bbox = boxes[i];
            text.text = decoded.first;
            text.score = decoded.second;
            results.push_back(text);
        }
    }

    if (latencyMs) {
        *latencyMs = elapsedMs(start);
    }
    return results;
}

cv::Mat PaddleOcrEngine::tensorToFloatMat(const dxrt::TensorPtr& tensor)
{
    const auto& shape = tensor->shape();
    if (shape.size() != 4) {
        throw std::runtime_error("Expected 4D tensor for detection output");
    }
    const int h = static_cast<int>(shape[2]);
    const int w = static_cast<int>(shape[3]);
    cv::Mat mat(h, w, CV_32F, tensor->data());
    return mat.clone();
}

std::vector<float> PaddleOcrEngine::tensorToFloatVector(const dxrt::TensorPtr& tensor)
{
    const std::int64_t count = product(tensor->shape());
    const float* data = static_cast<const float*>(tensor->data());
    return std::vector<float>(data, data + count);
}

std::pair<std::string, double> PaddleOcrEngine::decodeRecognition(const dxrt::TensorPtr& tensor) const
{
    const auto& shape = tensor->shape();
    if (shape.size() != 3) {
        return {"", 0.0};
    }

    const int steps = static_cast<int>(shape[1]);
    const int classes = static_cast<int>(shape[2]);
    const float* data = static_cast<const float*>(tensor->data());
    std::string text;
    double confSum = 0.0;
    int confCount = 0;
    int prevIdx = -1;

    for (int t = 0; t < steps; ++t) {
        const float* row = data + t * classes;
        int bestIdx = 0;
        float bestScore = row[0];
        for (int c = 1; c < classes; ++c) {
            if (row[c] > bestScore) {
                bestScore = row[c];
                bestIdx = c;
            }
        }

        const bool duplicate = (t > 0 && bestIdx == prevIdx);
        prevIdx = bestIdx;
        if (bestIdx == 0 || duplicate) {
            continue;
        }
        if (bestIdx >= 0 && bestIdx < static_cast<int>(characters_.size())) {
            text += characters_[bestIdx];
            confSum += bestScore;
            ++confCount;
        }
    }

    if (confCount == 0) {
        return {"", 0.0};
    }
    return {text, confSum / confCount};
}

std::vector<cv::Point2f> PaddleOcrEngine::getMiniBox(const std::vector<cv::Point>& contour, float* minSide)
{
    std::vector<cv::Point2f> points;
    points.reserve(contour.size());
    for (const auto& p : contour) {
        points.emplace_back(static_cast<float>(p.x), static_cast<float>(p.y));
    }
    return getMiniBox(points, minSide);
}

std::vector<cv::Point2f> PaddleOcrEngine::getMiniBox(const std::vector<cv::Point2f>& points, float* minSide)
{
    if (points.empty()) {
        if (minSide) *minSide = 0.0f;
        return {};
    }

    cv::RotatedRect rect = cv::minAreaRect(points);
    cv::Point2f raw[4];
    rect.points(raw);
    std::vector<cv::Point2f> sorted(raw, raw + 4);
    std::sort(sorted.begin(), sorted.end(), [](const cv::Point2f& a, const cv::Point2f& b) {
        if (std::abs(a.x - b.x) > 1e-5f) {
            return a.x < b.x;
        }
        return a.y < b.y;
    });

    int i1 = 0;
    int i2 = 2;
    int i3 = 3;
    int i4 = 1;
    if (sorted[1].y <= sorted[0].y) {
        i1 = 1;
        i4 = 0;
    }
    if (sorted[3].y <= sorted[2].y) {
        i2 = 3;
        i3 = 2;
    }

    if (minSide) {
        *minSide = std::min(rect.size.width, rect.size.height);
    }
    return {sorted[i1], sorted[i2], sorted[i3], sorted[i4]};
}

std::vector<cv::Point2f> PaddleOcrEngine::unclipApprox(const std::vector<cv::Point>& contour, double unclipRatio)
{
    if (contour.size() < 3) {
        return {};
    }
    const double area = std::abs(cv::contourArea(contour));
    const double perimeter = cv::arcLength(contour, true);
    if (area <= 1e-6 || perimeter <= 1e-6) {
        return {};
    }

    const double distance = area * unclipRatio / perimeter;
    cv::RotatedRect rect = cv::minAreaRect(contour);
    rect.size.width = static_cast<float>(rect.size.width + 2.0 * distance);
    rect.size.height = static_cast<float>(rect.size.height + 2.0 * distance);
    if (rect.size.width <= 0.0f || rect.size.height <= 0.0f) {
        return {};
    }

    cv::Point2f raw[4];
    rect.points(raw);
    return std::vector<cv::Point2f>(raw, raw + 4);
}

double PaddleOcrEngine::boxScoreFast(const cv::Mat& bitmap, const std::vector<cv::Point2f>& box)
{
    if (box.empty()) {
        return 0.0;
    }

    float xminF = box[0].x;
    float xmaxF = box[0].x;
    float yminF = box[0].y;
    float ymaxF = box[0].y;
    for (const auto& p : box) {
        xminF = std::min(xminF, p.x);
        xmaxF = std::max(xmaxF, p.x);
        yminF = std::min(yminF, p.y);
        ymaxF = std::max(ymaxF, p.y);
    }

    const int xmin = std::max(0, std::min(bitmap.cols - 1, static_cast<int>(std::floor(xminF))));
    const int xmax = std::max(0, std::min(bitmap.cols - 1, static_cast<int>(std::ceil(xmaxF))));
    const int ymin = std::max(0, std::min(bitmap.rows - 1, static_cast<int>(std::floor(yminF))));
    const int ymax = std::max(0, std::min(bitmap.rows - 1, static_cast<int>(std::ceil(ymaxF))));
    if (xmax < xmin || ymax < ymin) {
        return 0.0;
    }

    std::vector<cv::Point> local;
    local.reserve(box.size());
    for (const auto& p : box) {
        local.emplace_back(static_cast<int>(std::round(p.x - xmin)), static_cast<int>(std::round(p.y - ymin)));
    }

    cv::Mat mask = cv::Mat::zeros(ymax - ymin + 1, xmax - xmin + 1, CV_8U);
    cv::fillPoly(mask, std::vector<std::vector<cv::Point>>{local}, cv::Scalar(1));
    return cv::mean(bitmap(cv::Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1)), mask)[0];
}

std::vector<cv::Point2f> PaddleOcrEngine::orderPointsClockwise(const std::vector<cv::Point2f>& points)
{
    if (points.size() != 4) {
        return points;
    }

    std::vector<cv::Point2f> rect(4);
    auto sum = [](const cv::Point2f& p) { return p.x + p.y; };
    auto diff = [](const cv::Point2f& p) { return p.y - p.x; };

    rect[0] = *std::min_element(points.begin(), points.end(), [&](const auto& a, const auto& b) {
        return sum(a) < sum(b);
    });
    rect[2] = *std::max_element(points.begin(), points.end(), [&](const auto& a, const auto& b) {
        return sum(a) < sum(b);
    });

    std::vector<cv::Point2f> remain;
    for (const auto& p : points) {
        if (cv::norm(p - rect[0]) > 1e-3 && cv::norm(p - rect[2]) > 1e-3) {
            remain.push_back(p);
        }
    }
    if (remain.size() != 2) {
        return points;
    }
    rect[1] = diff(remain[0]) < diff(remain[1]) ? remain[0] : remain[1];
    rect[3] = diff(remain[0]) < diff(remain[1]) ? remain[1] : remain[0];
    return rect;
}

bool PaddleOcrEngine::clipAndValidate(std::vector<cv::Point2f>* box, int imageWidth, int imageHeight)
{
    if (!box || box->size() != 4) {
        return false;
    }
    for (auto& p : *box) {
        p.x = std::max(0.0f, std::min(static_cast<float>(imageWidth - 1), p.x));
        p.y = std::max(0.0f, std::min(static_cast<float>(imageHeight - 1), p.y));
    }

    const int rectWidth = static_cast<int>(cv::norm((*box)[0] - (*box)[1]));
    const int rectHeight = static_cast<int>(cv::norm((*box)[0] - (*box)[3]));
    return rectWidth > 3 && rectHeight > 3;
}

std::vector<std::vector<cv::Point2f>> PaddleOcrEngine::clipAndFilterBoxes(
    std::vector<std::vector<cv::Point2f>> boxes,
    int imageWidth,
    int imageHeight,
    int minSide)
{
    std::vector<std::vector<cv::Point2f>> filtered;
    filtered.reserve(boxes.size());

    for (auto& box : boxes) {
        if (box.size() != 4) {
            continue;
        }
        if (!clipAndValidate(&box, imageWidth, imageHeight)) {
            continue;
        }

        float minX = box[0].x;
        float maxX = box[0].x;
        float minY = box[0].y;
        float maxY = box[0].y;
        for (const auto& p : box) {
            minX = std::min(minX, p.x);
            maxX = std::max(maxX, p.x);
            minY = std::min(minY, p.y);
            maxY = std::max(maxY, p.y);
        }
        if ((maxX - minX) < static_cast<float>(minSide) || (maxY - minY) < static_cast<float>(minSide)) {
            continue;
        }

        filtered.push_back(std::move(box));
    }

    return filtered;
}

void PaddleOcrEngine::filterValidCrops(
    std::vector<std::vector<cv::Point2f>>* boxes,
    std::vector<cv::Mat>* crops,
    int minSide)
{
    if (!boxes || !crops) {
        return;
    }

    std::vector<std::vector<cv::Point2f>> validBoxes;
    std::vector<cv::Mat> validCrops;
    const std::size_t count = std::min(boxes->size(), crops->size());
    validBoxes.reserve(count);
    validCrops.reserve(count);

    for (std::size_t i = 0; i < count; ++i) {
        const cv::Mat& crop = (*crops)[i];
        if (crop.empty() || crop.cols < minSide || crop.rows < minSide) {
            continue;
        }
        validBoxes.push_back((*boxes)[i]);
        validCrops.push_back(crop.isContinuous() ? crop : crop.clone());
    }

    *boxes = std::move(validBoxes);
    *crops = std::move(validCrops);
}

bool PaddleOcrEngine::isValidCropForRecognition(const cv::Mat& crop)
{
    if (crop.empty()) {
        return false;
    }
    return crop.cols >= kMinCropSide && crop.rows >= kMinCropSide &&
           crop.cols <= kMaxCropSide && crop.rows <= kMaxCropSide;
}

void PaddleOcrEngine::sortBoxes(std::vector<std::vector<cv::Point2f>>* boxes)
{
    if (!boxes) {
        return;
    }
    std::sort(boxes->begin(), boxes->end(), [](const auto& a, const auto& b) {
        if (std::abs(a[0].y - b[0].y) > 1e-5f) {
            return a[0].y < b[0].y;
        }
        return a[0].x < b[0].x;
    });

    for (std::size_t i = 0; i + 1 < boxes->size(); ++i) {
        for (std::size_t j = i + 1; j > 0; --j) {
            auto& prev = (*boxes)[j - 1];
            auto& cur = (*boxes)[j];
            if (std::abs(cur[0].y - prev[0].y) < 10.0f && cur[0].x < prev[0].x) {
                std::swap(prev, cur);
            } else {
                break;
            }
        }
    }
}

}  // namespace camocr
