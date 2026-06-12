#include "ocr_engine.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>

namespace fs = std::filesystem;

namespace camocr {
namespace {

constexpr double kDetThresh = 0.3;
constexpr double kDetBoxThresh = 0.6;
constexpr double kDetUnclipRatio = 1.5;
constexpr int kDetMaxCandidates = 1500;
constexpr double kClsThresh = 0.9;
constexpr double kRecDropScore = 0.3;
constexpr std::size_t kMaxAsyncRecognitionRequests = 10;

double elapsedMs(const std::chrono::steady_clock::time_point& start)
{
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

int product(const std::vector<int64_t>& shape)
{
    int result = 1;
    for (const auto dim : shape) {
        if (dim > 0) {
            result *= static_cast<int>(dim);
        }
    }
    return result;
}

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
      language_(options.language),
      modelProfile_(options.modelProfile),
      enableUvdoc_(options.enableUvdoc)
{
    if (modelProfile_ != "mobile" && modelProfile_ != "server") {
        throw std::runtime_error("Unsupported OCR model profile: " + modelProfile_);
    }
    modelDir_ = resolveModelDir();
    fontDir_ = rootDir_ / "engine" / "fonts";
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

fs::path PaddleOcrEngine::resolveModelDir() const
{
    const std::vector<fs::path> candidates = {
        rootDir_ / "engine" / "model_files" / modelProfile_,
        rootDir_ / ".temp" / modelProfile_,
        rootDir_ / "engine" / "model_files",
    };

    for (const auto& candidate : candidates) {
        if (fs::exists(candidate / detModelFilename(640)) && fs::exists(candidate / recModelFilename(3))) {
            return candidate;
        }
    }

    throw std::runtime_error(
        "OCR model files were not found for profile '" + modelProfile_ +
        "'. Expected models under engine/model_files/" + modelProfile_ + " or .temp/" + modelProfile_ + ".");
}

fs::path PaddleOcrEngine::resolveDictPath() const
{
    fs::path langDict;
    if (language_ == "korean") {
        langDict = modelDir_ / "ppocrv5_korean_dict.txt";
    } else if (language_ == "german") {
        langDict = modelDir_ / "ppocrv5_latin_dict.txt";
    }

    if (!langDict.empty() && fs::exists(langDict)) {
        return langDict;
    }
    if (fs::exists(modelDir_ / "ppocrv5_dict.txt")) {
        return modelDir_ / "ppocrv5_dict.txt";
    }
    return rootDir_ / "engine" / "model_files" / "ppocrv5_dict.txt";
}

std::string PaddleOcrEngine::detModelFilename(int res) const
{
    if (modelProfile_ == "mobile") {
        return "det_mobile_" + std::to_string(res) + ".dxnn";
    }
    return "det_v5_" + std::to_string(res) + ".dxnn";
}

std::string PaddleOcrEngine::recModelFilename(int ratio, const std::string& suffix) const
{
    const std::string base = (modelProfile_ == "mobile") ? "rec_mobile_ratio_" : "rec_v5_ratio_";
    return base + std::to_string(ratio) + suffix + ".dxnn";
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
        fs::path langPath = modelDir_ / recModelFilename(ratio, suffix);
        if (fs::exists(langPath)) {
            return langPath;
        }
    }
    return modelDir_ / recModelFilename(ratio);
}

std::unique_ptr<dxrt::InferenceEngine> PaddleOcrEngine::loadEngine(const fs::path& path) const
{
    if (!fs::exists(path)) {
        throw std::runtime_error("Missing model: " + path.string());
    }
    dxrt::InferenceOption option;
    option.useORT = true;
    return std::make_unique<dxrt::InferenceEngine>(path.string(), option);
}

void PaddleOcrEngine::loadModels()
{
    detModelPaths_[640] = modelDir_ / detModelFilename(640);
    detModelPaths_[960] = modelDir_ / detModelFilename(960);

    for (int ratio : {3, 5, 10, 15, 25, 35}) {
        recModelPaths_[ratio] = resolveRecModelPath(ratio);
    }

    for (const auto& [_, path] : detModelPaths_) {
        if (!fs::exists(path)) {
            throw std::runtime_error("Missing model: " + path.string());
        }
    }
    if (!fs::exists(modelDir_ / "textline_ori.dxnn")) {
        throw std::runtime_error("Missing model: " + (modelDir_ / "textline_ori.dxnn").string());
    }
    for (const auto& [_, path] : recModelPaths_) {
        if (!fs::exists(path)) {
            throw std::runtime_error("Missing model: " + path.string());
        }
    }
    if (enableUvdoc_) {
        for (const auto& path : {modelDir_ / "doc_ori_fixed.dxnn", modelDir_ / "UVDoc_pruned_p3.dxnn"}) {
            if (!fs::exists(path)) {
                throw std::runtime_error("Missing model: " + path.string());
            }
        }
    }

    std::cout << "[OCR] Model profile: " << modelProfile_ << std::endl;
    std::cout << "[OCR] Model directory: " << modelDir_ << std::endl;
    std::cout << "[OCR] Document preprocessing: " << (enableUvdoc_ ? "enabled" : "disabled") << std::endl;
    std::cout << "[OCR] Recognition async max in-flight requests: " << kMaxAsyncRecognitionRequests << std::endl;
    std::cout << "[OCR] Models will be loaded lazily to keep NPU memory usage low." << std::endl;
}

dxrt::InferenceEngine& PaddleOcrEngine::detEngine(int res)
{
    releaseRecEngine();
    if (!activeDetModel_ || activeDetRes_ != res) {
        activeDetModel_.reset();
        activeDetModel_ = loadEngine(detModelPaths_.at(res));
        activeDetRes_ = res;
    }
    return *activeDetModel_;
}

dxrt::InferenceEngine& PaddleOcrEngine::clsEngine()
{
    if (!clsModel_) {
        clsModel_ = loadEngine(modelDir_ / "textline_ori.dxnn");
    }
    return *clsModel_;
}

dxrt::InferenceEngine& PaddleOcrEngine::recEngine(int ratio)
{
    releaseDetEngine();
    auto& model = activeRecModels_[ratio];
    if (!model) {
        model = loadEngine(recModelPaths_.at(ratio));
    }
    return *model;
}

dxrt::InferenceEngine& PaddleOcrEngine::docOriEngine()
{
    releaseDetEngine();
    releaseRecEngine();
    if (!docOriModel_) {
        docOriModel_ = loadEngine(modelDir_ / "doc_ori_fixed.dxnn");
    }
    return *docOriModel_;
}

dxrt::InferenceEngine& PaddleOcrEngine::uvdocEngine()
{
    releaseDetEngine();
    releaseRecEngine();
    if (!uvdocModel_) {
        uvdocModel_ = loadEngine(modelDir_ / "UVDoc_pruned_p3.dxnn");
    }
    return *uvdocModel_;
}

void PaddleOcrEngine::releaseDetEngine()
{
    activeDetModel_.reset();
    activeDetRes_ = 0;
}

void PaddleOcrEngine::releaseRecEngine()
{
    activeRecModels_.clear();
}

void PaddleOcrEngine::releaseDocEngines()
{
    docOriModel_.reset();
    uvdocModel_.reset();
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

    std::vector<cv::Mat> crops;
    crops.reserve(result.boxes.size());
    for (const auto& box : result.boxes) {
        crops.push_back(rotateIfVertical(getRotateCropImage(result.preprocessedImage, box)));
    }

    const auto clsResults = classify(crops, &clsMs);
    for (std::size_t i = 0; i < clsResults.size() && i < crops.size(); ++i) {
        if (clsResults[i].first == "180" && clsResults[i].second > kClsThresh) {
            cv::rotate(crops[i], crops[i], cv::ROTATE_180);
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
    cv::Mat current = bgrImage.clone();
    if (!enableUvdoc_) {
        return current;
    }
    current = runDocOrientation(current);
    current = runDocUnwarp(current);
    releaseDocEngines();
    return current;
}

int PaddleOcrEngine::routeDetection(int width, int height)
{
    return (width < 800 && height < 800) ? 640 : 960;
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

cv::Mat PaddleOcrEngine::resizeShortCenterCrop(const cv::Mat& image, int shortSide, int cropHeight, int cropWidth)
{
    const int h = image.rows;
    const int w = image.cols;
    int newH = shortSide;
    int newW = shortSide;
    if (h < w) {
        const double scale = static_cast<double>(shortSide) / h;
        newW = static_cast<int>(w * scale);
    } else {
        const double scale = static_cast<double>(shortSide) / w;
        newH = static_cast<int>(h * scale);
    }

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(newW, newH), 0.0, 0.0, cv::INTER_LINEAR);

    const int x = std::max(0, static_cast<int>(std::round((newW - cropWidth) / 2.0)));
    const int y = std::max(0, static_cast<int>(std::round((newH - cropHeight) / 2.0)));
    const int width = std::min(cropWidth, resized.cols - x);
    const int height = std::min(cropHeight, resized.rows - y);
    cv::Mat cropped = resized(cv::Rect(x, y, width, height)).clone();
    if (cropped.cols != cropWidth || cropped.rows != cropHeight) {
        cv::resize(cropped, cropped, cv::Size(cropWidth, cropHeight), 0.0, 0.0, cv::INTER_LINEAR);
    }
    return cropped.isContinuous() ? cropped : cropped.clone();
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
    const int res = routeDetection(image.cols, image.rows);
    if (lastLoggedDetRes_ != res) {
        std::cout << "[OCR] Detection model selected: " << modelProfile_
                  << " " << res << " (" << detModelPaths_.at(res).filename().string()
                  << ") for input " << image.cols << "x" << image.rows << std::endl;
        lastLoggedDetRes_ = res;
    }

    PaddingInfo padding;
    padding.origWidth = image.cols;
    padding.origHeight = image.rows;
    padding.paddedWidth = image.cols;
    padding.paddedHeight = image.rows;

    cv::Mat input;
    if (image.cols == res && image.rows == res) {
        input = image.isContinuous() ? image : image.clone();
    } else {
        input = resizePpocr(image, res, res, &padding);
    }

    const auto start = std::chrono::steady_clock::now();
    auto outputs = detEngine(res).Run(input.data);
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

    for (const auto& crop : crops) {
        if (crop.empty()) {
            results.emplace_back("0", 0.0);
            continue;
        }
        cv::Mat input = resizeDefault(crop, 80, 160);
        auto outputs = clsEngine().Run(input.data);
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

    struct AsyncRecognitionJob {
        std::size_t cropIndex = 0;
        int ratio = 0;
        cv::Mat input;
        dxrt::InferenceEngine* engine = nullptr;
        int jobId = -1;
    };

    std::vector<AsyncRecognitionJob> pendingJobs;
    pendingJobs.reserve(kMaxAsyncRecognitionRequests);

    auto flushPendingJobs = [&]() {
        for (auto& job : pendingJobs) {
            job.engine = &recEngine(job.ratio);
            job.jobId = job.engine->RunAsync(job.input.data);
            if (job.jobId < 0) {
                throw std::runtime_error("Failed to submit async recognition job");
            }
        }

        for (auto& job : pendingJobs) {
            auto outputs = job.engine->Wait(job.jobId);
            if (!outputs.empty()) {
                decodedResults[job.cropIndex] = decodeRecognition(outputs.front());
            }
        }
        pendingJobs.clear();
    };

    for (std::size_t i = 0; i < crops.size(); ++i) {
        if (crops[i].empty()) {
            continue;
        }
        const int ratio = routeRecognition(crops[i].cols, crops[i].rows);
        pendingJobs.push_back({i, ratio, resizePpocr(crops[i], 48, 48 * ratio, nullptr), nullptr, -1});
        if (pendingJobs.size() >= kMaxAsyncRecognitionRequests) {
            flushPendingJobs();
        }
    }
    if (!pendingJobs.empty()) {
        flushPendingJobs();
    }

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

cv::Mat PaddleOcrEngine::runDocOrientation(const cv::Mat& image)
{
    if (!enableUvdoc_) {
        return image;
    }

    cv::Mat input = resizeShortCenterCrop(image, 256, 224, 224);
    auto outputs = docOriEngine().Run(input.data);
    const auto logits = tensorToFloatVector(outputs.front());
    if (logits.size() < 4) {
        return image;
    }

    const float maxLogit = *std::max_element(logits.begin(), logits.end());
    double sum = 0.0;
    std::vector<double> probs(logits.size());
    for (std::size_t i = 0; i < logits.size(); ++i) {
        probs[i] = std::exp(logits[i] - maxLogit);
        sum += probs[i];
    }
    for (auto& p : probs) {
        p /= std::max(sum, 1e-12);
    }
    const auto bestIt = std::max_element(probs.begin(), probs.end());
    if (bestIt == probs.end() || *bestIt < 0.4) {
        return image;
    }

    const int idx = static_cast<int>(std::distance(probs.begin(), bestIt));
    cv::Mat rotated;
    if (idx == 1) {
        cv::rotate(image, rotated, cv::ROTATE_90_COUNTERCLOCKWISE);
    } else if (idx == 2) {
        cv::rotate(image, rotated, cv::ROTATE_180);
    } else if (idx == 3) {
        cv::rotate(image, rotated, cv::ROTATE_90_CLOCKWISE);
    } else {
        return image;
    }
    return rotated;
}

cv::Mat PaddleOcrEngine::runDocUnwarp(const cv::Mat& image)
{
    if (!enableUvdoc_) {
        return image;
    }

    cv::Mat input = resizeDefault(image, 712, 488);
    auto outputs = uvdocEngine().Run(input.data);
    const auto& tensor = outputs.front();
    const auto& shape = tensor->shape();
    if (shape.size() != 4 || shape[1] != 2) {
        return image;
    }

    const int mapH = static_cast<int>(shape[2]);
    const int mapW = static_cast<int>(shape[3]);
    const float* data = static_cast<const float*>(tensor->data());
    cv::Mat uvX(mapH, mapW, CV_32F, const_cast<float*>(data));
    cv::Mat uvY(mapH, mapW, CV_32F, const_cast<float*>(data + mapH * mapW));

    cv::Mat mapXNorm;
    cv::Mat mapYNorm;
    cv::resize(uvX, mapXNorm, image.size(), 0.0, 0.0, cv::INTER_LINEAR);
    cv::resize(uvY, mapYNorm, image.size(), 0.0, 0.0, cv::INTER_LINEAR);

    cv::Mat mapX(image.size(), CV_32F);
    cv::Mat mapY(image.size(), CV_32F);
    const float scaleX = static_cast<float>((image.cols - 1) * 0.5);
    const float scaleY = static_cast<float>((image.rows - 1) * 0.5);
    for (int y = 0; y < image.rows; ++y) {
        const float* nx = mapXNorm.ptr<float>(y);
        const float* ny = mapYNorm.ptr<float>(y);
        float* mx = mapX.ptr<float>(y);
        float* my = mapY.ptr<float>(y);
        for (int x = 0; x < image.cols; ++x) {
            mx[x] = (nx[x] + 1.0f) * scaleX;
            my[x] = (ny[x] + 1.0f) * scaleY;
        }
    }

    cv::Mat unwarped;
    cv::remap(image, unwarped, mapX, mapY, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    return unwarped;
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
    const int count = product(tensor->shape());
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
