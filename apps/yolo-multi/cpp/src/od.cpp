#include "od.h"
#include "yolo.h"
#include <utils/common_util.hpp>
#include <algorithm>

extern YoloParam yoloParam;

ObjectDetection::ObjectDetection(std::shared_ptr<dxrt::InferenceEngine> ie, std::pair<std::string, std::string> &videoSrc, int channel, 
        int width, int height, int destWidth, int destHeight,
        int posX, int posY, int numFrames)
: _ie(ie), _profiler(dxrt::Profiler::GetInstance()), _channel(channel + 1),
    _width(width), _height(height), _destWidth(destWidth), _destHeight(destHeight), 
    _posX(posX), _posY(posY), _videoSrc(videoSrc)
{
    AppInputType inputType = AppInputType::VIDEO;
    if(_videoSrc.second == "camera")
        inputType = AppInputType::CAMERA;
    else if(_videoSrc.second == "image")
        inputType = AppInputType::IMAGE;
    else if(_videoSrc.second == "rtsp")
        inputType = AppInputType::RTSP;
#if __riscv
    else if(_videoSrc.second == "isp")
        inputType = AppInputType::ISP;
#endif
    else
        inputType = AppInputType::VIDEO;
    auto inputShape = _ie->GetInputs().front().shape();
    auto npuShape = dxapp::common::Size((int)inputShape[1],(int)inputShape[1]);
    auto dstShape = dxapp::common::Size(_destWidth, _destHeight);

    _vStream = VideoStream(inputType, _videoSrc.first, numFrames, npuShape, AppInputFormat::IMAGE_BGR, dstShape, _ie);
    auto srcShape = _vStream._srcSize;
    _srcWidth = srcShape._width;
    _srcHeight = srcShape._height;
    _name = "app" + std::to_string(_channel);
    dxapp::common::Size_f _postprocRatio;
    _postprocRatio._width = (float)dstShape._width/srcShape._width;
    _postprocRatio._height = (float)dstShape._height/srcShape._height;

    float _preprocRatio = std::min((float)npuShape._width/srcShape._width, (float)npuShape._height/srcShape._height);
    
    if(srcShape == npuShape)
    {
        _postprocPaddedSize._width = 0.f;
        _postprocPaddedSize._height = 0.f;
    }
    else
    {
        dxapp::common::Size resizeShpae((int)(srcShape._width * _preprocRatio), (int)(srcShape._height * _preprocRatio));
        _postprocPaddedSize._width = (npuShape._width - resizeShpae._width) / 2.f;
        _postprocPaddedSize._height = (npuShape._height - resizeShpae._height) / 2.f;
    }

    _postprocScaleRatio = dxapp::common::Size_f(_postprocRatio._width/_preprocRatio, _postprocRatio._height/_preprocRatio);
    
    _resultFrame = cv::Mat(_destHeight, _destWidth, CV_8UC3, cv::Scalar(0, 0, 0));
    yolo = Yolo(yoloParam);
    if(!yolo.LayerReorder(_ie->GetOutputs()))
        return;

    outputMemory = (uint8_t*)operator new(_ie->GetOutputSize());
    output_length = 0;
    for(auto &o:_ie->GetOutputs())
    {
        output_shape.emplace_back(o.shape());
    }
    data_type = _ie->GetOutputs().front().type();

    _fps_time_s = std::chrono::high_resolution_clock::now();
    _fps_time_e = std::chrono::high_resolution_clock::now();
}
ObjectDetection::ObjectDetection(std::shared_ptr<dxrt::InferenceEngine> ie, int channel, int destWidth, int destHeight, int posX, int posY)
: _ie(ie), _profiler(dxrt::Profiler::GetInstance()), _channel(channel+1), _destWidth(destWidth), _destHeight(destHeight), _posX(posX), _posY(posY)
{
    _name = "app" + std::to_string(_channel);
    if(dxdemo::common::pathValidation("../../assets/yolo-multi/sample/dx_colored_log.png"))
    {
        _logo = cv::imread("../../assets/yolo-multi/sample/dx_colored_log.png", cv::IMREAD_COLOR);
        cv::resize(_logo, _resultFrame, cv::Size(_destWidth, _destHeight), 0, 0, cv::INTER_LINEAR);
    }
    else
    {
        _resultFrame = cv::Mat(_destHeight, _destWidth, CV_8UC3, cv::Scalar(0, 0, 0));
    }
    outputMemory = nullptr;
}
ObjectDetection::~ObjectDetection() {}
dxapp::common::DetectObject ObjectDetection::GetScalingBBox(std::vector<BoundingBox>& bboxes)
{
    dxapp::common::DetectObject result;
    result._num_of_detections = bboxes.size();
    for (auto& b : bboxes)
    {
        dxapp::common::BBox box;
        box._xmin = (b.box[0] - _postprocPaddedSize._width) * _postprocScaleRatio._width;
        box._ymin = (b.box[1] - _postprocPaddedSize._height) * _postprocScaleRatio._height;
        box._xmax = (b.box[2] - _postprocPaddedSize._width) * _postprocScaleRatio._width;
        box._ymax = (b.box[3] - _postprocPaddedSize._height) * _postprocScaleRatio._height;
        box._width = (b.box[2] - b.box[0]) * _postprocScaleRatio._width;
        box._height = (b.box[3] - b.box[1]) * _postprocScaleRatio._height;
        box._kpts.emplace_back(dxapp::common::Point_f(-1 , -1, -1));
    
        dxapp::common::Object object;
        object._bbox = box;
        object._conf = b.score;
        object._classId = b.label;
        object._name = b.labelname;
        result._detections.emplace_back(object);
    }
    return result;
}
void ObjectDetection::threadFunc(int period)
{
    std::string cap = "cap" + std::to_string(_channel);
    std::string proc = "proc" + std::to_string(_channel);
#if 0
    char caption[100] = {0,};
    float fps = 0.f; double infCount = 0.0;
#endif
    _profiler.Add(cap);
    _profiler.Add(proc);
    cv::Mat member_temp;

    struct CameraUiCache {
        bool showBorder = false;
        bool showLivePill = false;
        cv::Rect pill;
        cv::Point dotCenter;
        int dotR = 0;
        double fontScale = 0.;
        int fontThick = 1;
        cv::Point textOrg;
    } camUi;
    if(_videoSrc.second == "camera")
    {
        camUi.showBorder = true;
        const int cols = _destWidth;
        const int rows = _destHeight;
        static const char liveLabel[] = "Live";
        const double fontScale =
            std::min(0.62, std::max(0.38, static_cast<double>(cols) * 0.0011));
        const int fontThick = 1;
        int baseline = 0;
        const cv::Size ts =
            cv::getTextSize(liveLabel, cv::FONT_HERSHEY_SIMPLEX, fontScale, fontThick, &baseline);
        const int padX = std::max(5, static_cast<int>(7 * fontScale / 0.55));
        const int padY = std::max(3, static_cast<int>(5 * fontScale / 0.55));
        const int margin = std::max(4, cols / 80);
        const int dotR = std::max(2, static_cast<int>(4 * fontScale / 0.55));
        const int innerGap = std::max(3, static_cast<int>(5 * fontScale / 0.55));
        const int pillW = padX + dotR * 2 + innerGap + ts.width + padX;
        const int pillH = padY + ts.height + padY;
        const int x2 = cols - margin;
        const int x1 = x2 - pillW;
        const int y1 = margin;
        const int y2 = y1 + pillH;
        if(x1 >= 0 && y2 <= rows && pillW + 2 * margin <= cols)
        {
            camUi.showLivePill = true;
            camUi.pill = cv::Rect(x1, y1, pillW, pillH);
            camUi.dotCenter = cv::Point(x1 + padX + dotR, y1 + pillH / 2);
            camUi.dotR = dotR;
            camUi.fontScale = fontScale;
            camUi.fontThick = fontThick;
            camUi.textOrg = cv::Point(x1 + padX + dotR * 2 + innerGap, y1 + padY + ts.height);
        }
    }

    while(1)
    {        
        if(stop) break;
        _profiler.Start(proc);
        _profiler.Start(cap);
        auto input = _vStream.GetInputStream();
        _fps_time_s = std::chrono::high_resolution_clock::now();
        std::ignore = _ie->RunAsync(input, (void*)this, (void*)outputMemory);
        std::vector<BoundingBox> bboxes;
        dxapp::common::DetectObject bboxes_objects;
        {
            std::unique_lock<std::mutex> lk(_lock);
            if(!_bboxes.empty() && _toggleDrawing)
            {
                bboxes = std::vector<BoundingBox>(_bboxes);
                bboxes_objects = GetScalingBBox(bboxes);
            }
        }
        member_temp = _vStream.GetOutputStream(bboxes_objects);
        if(camUi.showBorder)
        {
            cv::rectangle(member_temp,
                          cv::Point(0, 0),
                          cv::Point(_destWidth - 1, _destHeight - 1),
                          cv::Scalar(0, 255, 255),
                          3);
            if(camUi.showLivePill)
            {
                cv::rectangle(member_temp, camUi.pill, cv::Scalar(248, 248, 255), cv::FILLED);
                //cv::rectangle(member_temp, camUi.pill, cv::Scalar(0, 230, 255), 1, cv::LINE_AA);
                cv::circle(member_temp, camUi.dotCenter, camUi.dotR, cv::Scalar(0, 0, 255), cv::FILLED, cv::LINE_AA);
                cv::putText(member_temp, "LIVE", camUi.textOrg, cv::FONT_HERSHEY_SIMPLEX, camUi.fontScale,
                            cv::Scalar(28, 28, 32), camUi.fontThick, cv::LINE_AA);
            }
        }
#if 0
        fps += 1000000.0 / _inferTime;
        infCount++;
        float resultFps = round((fps/infCount) * 100) / 100;
        
        snprintf(caption, sizeof(caption), " / %.2f FPS", _channel, resultFps);
        cv::rectangle(member_temp, cv::Point(0, 0), cv::Point(230, 34), cv::Scalar(0, 0, 0), cv::FILLED);
        cv::putText(member_temp, caption, cv::Point(56, 21), 0, 0.7, cv::Scalar(255,255,255), 2, cv::LINE_AA);
#else
        cv::rectangle(member_temp, cv::Point(0, 0), cv::Point(40, 20), cv::Scalar(0, 0, 0), cv::FILLED);
#endif
        cv::putText(member_temp, std::to_string(_channel), cv::Point(5, 15), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);

        
        _inferenceTime = _ie->GetNpuInferenceTime();
        _latencyTime = _ie->GetLatency();
        
        _profiler.End(cap);
        int64_t t = (period*1000 - _profiler.Get(cap))/1000;
        if(t<0 || t>period) t = 0;
        
        if(_processed_count > 0)
        {
            std::unique_lock<std::mutex> lk(_frameLock);
            member_temp.copyTo(_resultFrame);
            if(_isPause){
                _cv.wait(lk, [this]{return !_isPause;});
            }
        }

        _profiler.End(proc);
        _processTime = _profiler.Get(proc);
#ifdef __linux__
        usleep(t*1000);
#elif _WIN32
        Sleep(t);
#endif
    }
    _profiler.Erase(cap);
    _profiler.Erase(proc);
    std::cout << _channel << " ended." << std::endl;
}
void ObjectDetection::threadFillBlank(int period)
{
    while(1)
    {        
        if(stop) break;
#ifdef __linux__
        usleep(period * 1000);
#elif _WIN32
        Sleep(period);
#endif
    }
    std::cout << _channel << " ended." << std::endl;
}
void ObjectDetection::Run(int period)
{
    stop = false;
    if(_videoSrc.first.empty())
        _thread = std::thread(&ObjectDetection::threadFillBlank, this, period);
    else
        _thread = std::thread(&ObjectDetection::threadFunc, this, period);
}
void ObjectDetection::Stop()
{
    stop = true;
    _thread.join();
}
void ObjectDetection::Pause()
{
    std::unique_lock<std::mutex> lk(_frameLock);
    if(!_isPause)
        _isPause = true;
}
void ObjectDetection::Play()
{
    std::unique_lock<std::mutex> lk(_frameLock);
    if(_isPause){
        _isPause = false;
        _cv.notify_all();
    }
}
cv::Mat ObjectDetection::ResultFrame()
{
    std::unique_lock<std::mutex> lk(_frameLock);
    cv::Mat out = _resultFrame.clone();
    return out;
}
std::pair<int, int> ObjectDetection::Position()
{
    return std::make_pair(_posX, _posY);
}
std::pair<int, int> ObjectDetection::Resolution()
{
    return std::make_pair(_destWidth, _destHeight);
}
uint64_t ObjectDetection::GetLatencyTime()
{
    return _latencyTime;
}
uint64_t ObjectDetection::GetInferenceTime()
{
    return _inferenceTime;
}
uint64_t ObjectDetection::GetProcessingTime()
{
    return _duration_time;
}
int ObjectDetection::Channel()
{
    return _channel;
}
std::string &ObjectDetection::Name()
{
    return _name;
}
void ObjectDetection::Toggle()
{
    _toggleDrawing = !_toggleDrawing;
}
void ObjectDetection::PostProc(std::vector<std::shared_ptr<dxrt::Tensor>> &outputs)
{
    std::unique_lock<std::mutex> lk(_lock);
    _bboxes = yolo.PostProc(outputs);

    // Print bbox count for aging test monitoring
    if (_bboxes.size() >= 500) {
        std::cerr << "[BBOX_ERROR] count=" << _bboxes.size() << std::endl;
    }

    _processed_count++;
    _ret_processed_count++;
}
uint64_t ObjectDetection::GetPostProcessCount()
{
    std::unique_lock<std::mutex> lk(_lock);
    return _ret_processed_count;
}
void ObjectDetection::SetZeroPostProcessCount()
{
    std::unique_lock<std::mutex> lk(_lock);
    _ret_processed_count = 0;
}
std::ostream& operator<<(std::ostream& os, const ObjectDetection& od)
{
    os << od._name << ": " << od._channel << ", "
        << od._videoSrc.first << ", " << od._videoSrc.second << ", "
        << od._channel << ", " << od._targetFps << ", "
        << od._width << ", " << od._height << ", "
        << od._destWidth << ", " << od._destHeight << ", "
        << od._posX << ", " << od._posY << ", "
        << od._offline << ", " << od._cap.get(cv::CAP_PROP_FPS);
    return os;
}
