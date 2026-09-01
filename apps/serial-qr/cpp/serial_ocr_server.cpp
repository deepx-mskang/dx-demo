// DEEPX Serial-QR 데모 서버
//
// apps/paddle-ocr 의 PP-OCRv6 엔진(camocr::PaddleOcrEngine)을 그대로 재사용하되,
// Qt GUI 대신 HTTP 서버로 감싼다.
//
//   - 카메라는 서버가 V4L2 로 직접 잡고 MJPEG 로 브라우저에 스트리밍한다.
//     (브라우저 getUserMedia 를 쓰지 않으므로 HTTPS 가 필요 없다)
//   - POST /api/scan 이 오면 최신 프레임을 복사해 OCR 을 돌리고
//     인식된 텍스트에서 시리얼 번호를 추출해 JSON 으로 돌려준다.

#define CPPHTTPLIB_THREAD_POOL_COUNT 16

#include "device_registry.hpp"
#include "httplib.h"
#include "ocr_engine.hpp"

#include <opencv2/opencv.hpp>

#include <arpa/inet.h>
#include <ifaddrs.h>
#include <net/if.h>
#include <netdb.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

namespace {

// ---------------------------------------------------------------------------
// 최소 JSON 직렬화
// ---------------------------------------------------------------------------

std::string jsonEscape(const std::string& s)
{
    std::string out;
    out.reserve(s.size() + 8);
    for (const unsigned char c : s) {
        switch (c) {
        case '"': out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default:
            if (c < 0x20) {
                char buf[8];
                std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                out += buf;
            } else {
                out += static_cast<char>(c);
            }
        }
    }
    return out;
}

std::string jsonStr(const std::string& s)
{
    return "\"" + jsonEscape(s) + "\"";
}

std::string jsonNum(double v, int precision = 2)
{
    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    oss.precision(precision);
    oss << v;
    return oss.str();
}

// ---------------------------------------------------------------------------
// Base64
// ---------------------------------------------------------------------------

std::string base64Encode(const std::vector<unsigned char>& data)
{
    static constexpr char kTable[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    std::string out;
    out.reserve(((data.size() + 2) / 3) * 4);

    std::size_t i = 0;
    for (; i + 2 < data.size(); i += 3) {
        const std::uint32_t v = (data[i] << 16) | (data[i + 1] << 8) | data[i + 2];
        out += kTable[(v >> 18) & 0x3F];
        out += kTable[(v >> 12) & 0x3F];
        out += kTable[(v >> 6) & 0x3F];
        out += kTable[v & 0x3F];
    }
    if (i + 1 == data.size()) {
        const std::uint32_t v = data[i] << 16;
        out += kTable[(v >> 18) & 0x3F];
        out += kTable[(v >> 12) & 0x3F];
        out += "==";
    } else if (i + 2 == data.size()) {
        const std::uint32_t v = (data[i] << 16) | (data[i + 1] << 8);
        out += kTable[(v >> 18) & 0x3F];
        out += kTable[(v >> 12) & 0x3F];
        out += kTable[(v >> 6) & 0x3F];
        out += "=";
    }
    return out;
}

// ---------------------------------------------------------------------------
// LAN IP 탐색
//
// QR 에 넣을 주소는 반드시 휴대폰이 접근 가능한 LAN 주소여야 한다.
// 프론트가 window.location.origin 을 쓰면 데모 PC 에서 localhost 로 열었을 때
// QR 이 휴대폰에서 열리지 않으므로, 서버가 직접 알려준다.
// ---------------------------------------------------------------------------

std::string detectLanIPv4()
{
    struct ifaddrs* ifaddr = nullptr;
    if (getifaddrs(&ifaddr) == -1) {
        return {};
    }

    std::string fallback;
    std::string preferred;

    for (struct ifaddrs* ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next) {
        if (ifa->ifa_addr == nullptr || ifa->ifa_addr->sa_family != AF_INET) {
            continue;
        }
        if ((ifa->ifa_flags & IFF_UP) == 0 || (ifa->ifa_flags & IFF_LOOPBACK) != 0) {
            continue;
        }

        char host[NI_MAXHOST] = {0};
        auto* addr = reinterpret_cast<struct sockaddr_in*>(ifa->ifa_addr);
        if (!inet_ntop(AF_INET, &addr->sin_addr, host, sizeof(host))) {
            continue;
        }

        const std::string ip = host;
        const std::string name = ifa->ifa_name ? ifa->ifa_name : "";

        // docker/virbr 같은 가상 브릿지는 휴대폰에서 접근 불가하므로 뒤로 미룬다.
        const bool virtualIface = name.rfind("docker", 0) == 0 || name.rfind("virbr", 0) == 0 ||
                                  name.rfind("br-", 0) == 0 || name.rfind("veth", 0) == 0;
        if (virtualIface) {
            if (fallback.empty()) {
                fallback = ip;
            }
            continue;
        }
        if (preferred.empty()) {
            preferred = ip;
        }
    }

    freeifaddrs(ifaddr);
    return !preferred.empty() ? preferred : fallback;
}

// ---------------------------------------------------------------------------
// 시리얼 추출
// ---------------------------------------------------------------------------

struct SerialCandidate {
    std::string text;
    std::string rawText;
    double score = 0.0;
    bool normalized = false;
    int priority = 99;  // 낮을수록 우선
};

// OCR 텍스트를 시리얼 탐색용으로 정규화한다.
//
// 영숫자는 대문자로 남기고, 그 밖의 문자(공백, ':', '/', '.', '_' …)는
// 전부 '-' 하나로 접는다. 삭제하지 않는 것이 중요하다.
// "S/N: DX-M1-A7K3P9V2" 에서 구분자를 지워 버리면 "SNDX-M1-..." 이 되어
// 시리얼 토큰의 시작 경계가 사라진다.
// 덤으로 OCR 이 하이픈을 공백으로 읽은 "DX M1 A7K3P9V2" 도 함께 복구된다.
std::string toUpperAlnum(const std::string& in)
{
    std::string out;
    out.reserve(in.size());
    for (const char c : in) {
        if (std::isalnum(static_cast<unsigned char>(c))) {
            out += static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
        } else if (!out.empty() && out.back() != '-') {
            out += '-';
        }
    }
    while (!out.empty() && out.back() == '-') {
        out.pop_back();
    }
    return out;
}

// 시리얼 본문 구간에서만 OCR 혼동 문자를 보정한다.
// 접두사(DX-M1-)는 알파벳이 정상이므로 건드리지 않는다.
std::string normalizeConfusables(const std::string& body, bool* changed)
{
    std::string out = body;
    *changed = false;
    for (char& c : out) {
        char before = c;
        switch (c) {
        case 'O': c = '0'; break;
        case 'Q': c = '0'; break;
        case 'I': c = '1'; break;
        case 'L': c = '1'; break;
        case 'S': c = '5'; break;
        case 'B': c = '8'; break;
        case 'Z': c = '2'; break;
        default: break;
        }
        if (before != c) {
            *changed = true;
        }
    }
    return out;
}

// 매칭이 영숫자 한가운데서 시작/끝나지 않는지 확인한다.
// "XDX-M1-A7K3P9V2" 처럼 앞에 글자가 더 붙은 경우를 걸러낸다.
bool onTokenBoundary(const std::string& hay, const std::smatch& m)
{
    const auto begin = static_cast<std::size_t>(m.position(0));
    const auto end = begin + static_cast<std::size_t>(m.length(0));
    const auto isBody = [](char c) {
        return std::isalnum(static_cast<unsigned char>(c)) != 0;
    };
    if (begin > 0 && isBody(hay[begin - 1])) {
        return false;
    }
    if (end < hay.size() && isBody(hay[end])) {
        return false;
    }
    return true;
}

// hay 안에서 re 에 맞는 첫 토큰을 찾는다 (경계 조건 포함).
bool findToken(const std::string& hay, const std::regex& re, std::smatch* out)
{
    auto it = std::sregex_iterator(hay.begin(), hay.end(), re);
    const auto last = std::sregex_iterator();
    for (; it != last; ++it) {
        if (onTokenBoundary(hay, *it)) {
            *out = *it;
            return true;
        }
    }
    return false;
}

int countDigits(const std::string& s)
{
    return static_cast<int>(std::count_if(s.begin(), s.end(), [](unsigned char ch) {
        return std::isdigit(ch) != 0;
    }));
}

std::vector<SerialCandidate> extractSerials(const std::vector<camocr::OcrText>& texts)
{
    // regex_match(전체 일치)가 아니라 regex_search(부분 탐색)를 쓴다.
    // 실제 라벨은 "S/N: DX-M1-A7K3P9V2" 처럼 시리얼 앞뒤에 다른 문구가
    // 같은 텍스트 박스로 묶여 나오는 경우가 대부분이기 때문이다.
    //
    //   P0  정식 포맷, 본문 정확히 8자          -> 가장 신뢰
    //   P1  일반 시리얼 형태 + 본문 숫자 3개 이상
    //   P2  박스가 쪼개진 경우 (전체를 이어 붙여 재탐색)
    //   P3  자릿수가 어긋난 정식 포맷           -> 최소한 무엇을 읽었는지 보여준다
    static const std::regex kPrimary(R"(DX-?M1-?([A-Z0-9]{8}))");
    static const std::regex kGeneric(R"(([A-Z]{2,4})-?([A-Z0-9]{6,12}))");
    static const std::regex kLoose(R"(DX-?M1-?([A-Z0-9]{5,12}))");

    std::vector<SerialCandidate> out;

    const auto push = [&out](std::string text,
                             const std::string& rawText,
                             double score,
                             bool normalized,
                             int priority) {
        SerialCandidate c;
        c.text = std::move(text);
        c.rawText = rawText;
        c.score = score;
        c.normalized = normalized;
        c.priority = priority;
        out.push_back(std::move(c));
    };

    // 느슨한 패턴으로 찾은 것은 따로 모아 둔다.
    // 제대로 된 후보가 하나라도 있으면 이건 군더더기이므로 버린다.
    std::vector<SerialCandidate> loose;

    std::string joined;
    joined.reserve(64);

    for (const auto& t : texts) {
        const std::string cleaned = toUpperAlnum(t.text);
        joined += cleaned;
        if (cleaned.size() < 6) {
            continue;
        }

        std::smatch m;

        if (findToken(cleaned, kPrimary, &m)) {
            bool changed = false;
            const std::string body = normalizeConfusables(m[1].str(), &changed);
            push("DX-M1-" + body, t.text, t.score, changed, 0);
            continue;
        }

        if (findToken(cleaned, kGeneric, &m)) {
            const std::string rawBody = m[2].str();
            // 라벨의 설명 문구가 일반 패턴에 걸리는 것을 막는다.
            // "S/N:DX-M1 NPU MODULE" -> "SNDX-M1NPUMODULE" 은 형태상 매칭되지만
            // 시리얼이 아니다. 실제 시리얼 본문은 숫자를 여러 개 포함한다.
            if (countDigits(rawBody) >= 3) {
                bool changed = false;
                const std::string body = normalizeConfusables(rawBody, &changed);
                push(m[1].str() + "-" + body, t.text, t.score, changed, 1);
                continue;
            }
        }

        // 자릿수가 어긋난 정식 포맷. 본문에 숫자가 없으면 설명 문구
        // ("DX-M1 NPU MODULE" 등)일 가능성이 높으므로 제외한다.
        if (findToken(cleaned, kLoose, &m) && countDigits(m[1].str()) >= 2) {
            bool changed = false;
            const std::string body = normalizeConfusables(m[1].str(), &changed);
            SerialCandidate c;
            c.text = "DX-M1-" + body;
            c.rawText = t.text;
            c.score = t.score;
            c.normalized = changed;
            c.priority = 3;
            loose.push_back(std::move(c));
        }
    }

    // 제대로 된 후보가 없을 때만 느슨한 결과를 쓴다.
    if (out.empty()) {
        out = std::move(loose);
    }

    // 시리얼이 여러 박스로 쪼개져 읽힌 경우("DX-M1-" + "A7K3P9V2").
    // 앞선 단계에서 아무것도 못 찾았을 때만 시도한다.
    if (out.empty()) {
        std::smatch m;
        if (findToken(joined, kPrimary, &m)) {
            bool changed = false;
            const std::string body = normalizeConfusables(m[1].str(), &changed);
            push("DX-M1-" + body, joined, 0.0, changed, 2);
        }
    }

    std::stable_sort(out.begin(), out.end(), [](const SerialCandidate& a, const SerialCandidate& b) {
        if (a.priority != b.priority) {
            return a.priority < b.priority;
        }
        return a.score > b.score;
    });

    // 같은 시리얼이 여러 박스에서 잡히면 최고 점수만 남긴다.
    std::vector<SerialCandidate> unique;
    for (auto& c : out) {
        const bool dup = std::any_of(unique.begin(), unique.end(), [&](const SerialCandidate& u) {
            return u.text == c.text;
        });
        if (!dup) {
            unique.push_back(std::move(c));
        }
    }
    return unique;
}

// OCR 결과 + 시리얼 후보를 스캔 응답 JSON 으로 만든다.
// HTTP 핸들러와 --test-image 경로가 같은 포맷을 쓰도록 한 곳에 모아 둔다.
std::string buildScanJson(const camocr::OcrResult& result, const std::string& frameB64)
{
    const std::vector<SerialCandidate> candidates = extractSerials(result.texts);

    std::ostringstream oss;
    oss << "{";
    oss << "\"ok\":" << (candidates.empty() ? "false" : "true");
    if (candidates.empty()) {
        oss << ",\"reason\":\"no_serial_found\"";
        oss << ",\"serial\":null,\"confidence\":0";
    } else {
        oss << ",\"serial\":" << jsonStr(candidates.front().text);
        oss << ",\"confidence\":" << jsonNum(candidates.front().score, 4);
    }

    oss << ",\"candidates\":[";
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        const auto& c = candidates[i];
        if (i > 0) {
            oss << ",";
        }
        oss << "{\"text\":" << jsonStr(c.text) << ",\"rawText\":" << jsonStr(c.rawText)
            << ",\"score\":" << jsonNum(c.score, 4)
            << ",\"normalized\":" << (c.normalized ? "true" : "false") << "}";
    }
    oss << "]";

    oss << ",\"rawTexts\":[";
    for (std::size_t i = 0; i < result.texts.size(); ++i) {
        if (i > 0) {
            oss << ",";
        }
        oss << jsonStr(result.texts[i].text);
    }
    oss << "]";

    oss << ",\"perf\":{"
        << "\"detMs\":" << jsonNum(result.perf.detTimeMs)
        << ",\"recMs\":" << jsonNum(result.perf.recTimeMs)
        << ",\"e2eMs\":" << jsonNum(result.perf.e2eTimeMs)
        << ",\"numBoxes\":" << result.perf.numBoxes
        << ",\"numCrops\":" << result.perf.numCrops
        << ",\"totalChars\":" << result.perf.totalChars
        << ",\"cps\":" << jsonNum(result.perf.cps) << "}";

    oss << ",\"frame\":" << jsonStr(frameB64);
    oss << "}";
    return oss.str();
}

// ---------------------------------------------------------------------------
// 카메라 워커
//
// 전용 스레드가 계속 grab 하고 최신 프레임 1장만 들고 있는다.
// MJPEG 스트리밍과 OCR 스캔이 같은 버퍼를 공유하되, 항상 복사본을 꺼내 쓴다.
// ---------------------------------------------------------------------------

struct CameraConfig {
    int deviceIndex = 0;
    std::string devicePath;  // 비어 있지 않으면 인덱스 대신 경로로 연다
    int width = 1280;
    int height = 720;
    int cropSize = 960;
    double fps = 15.0;
};

cv::Mat centerCropToSize(const cv::Mat& frame, int cropSize)
{
    if (frame.empty() || cropSize <= 0) {
        return frame;
    }
    if (frame.cols >= cropSize && frame.rows >= cropSize) {
        const int x = (frame.cols - cropSize) / 2;
        const int y = (frame.rows - cropSize) / 2;
        return frame(cv::Rect(x, y, cropSize, cropSize)).clone();
    }
    const int square = std::min(frame.cols, frame.rows);
    const int x = (frame.cols - square) / 2;
    const int y = (frame.rows - square) / 2;
    cv::Mat cropped = frame(cv::Rect(x, y, square, square));
    cv::Mat resized;
    cv::resize(cropped, resized, cv::Size(cropSize, cropSize), 0.0, 0.0, cv::INTER_LINEAR);
    return resized;
}

class CameraWorker {
public:
    explicit CameraWorker(CameraConfig config) : config_(std::move(config)) {}

    ~CameraWorker() { stop(); }

    bool start()
    {
        if (!open()) {
            return false;
        }
        running_.store(true);
        thread_ = std::thread([this] { loop(); });
        return true;
    }

    void stop()
    {
        running_.store(false);
        if (thread_.joinable()) {
            thread_.join();
        }
        if (cap_.isOpened()) {
            cap_.release();
        }
    }

    // 최신 프레임의 복사본. 아직 한 장도 못 읽었으면 empty.
    cv::Mat latestFrame() const
    {
        std::lock_guard<std::mutex> lock(frameMutex_);
        return latestFrame_.empty() ? cv::Mat() : latestFrame_.clone();
    }

    bool healthy() const { return running_.load() && frameCount_.load() > 0; }
    std::uint64_t frameCount() const { return frameCount_.load(); }

private:
    bool open()
    {
        if (!config_.devicePath.empty()) {
            cap_.open(config_.devicePath, cv::CAP_V4L2);
        } else {
            cap_.open(config_.deviceIndex, cv::CAP_V4L2);
        }
        if (!cap_.isOpened()) {
            std::cerr << "[camera] Failed to open camera "
                      << (config_.devicePath.empty() ? std::to_string(config_.deviceIndex)
                                                     : config_.devicePath)
                      << std::endl;
            return false;
        }

        cap_.set(cv::CAP_PROP_BUFFERSIZE, 1);
        cap_.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
        cap_.set(cv::CAP_PROP_FRAME_WIDTH, config_.width);
        cap_.set(cv::CAP_PROP_FRAME_HEIGHT, config_.height);
        cap_.set(cv::CAP_PROP_FPS, config_.fps);
        cap_.set(cv::CAP_PROP_AUTOFOCUS, 1.0);

        std::cout << "[camera] opened "
                  << (config_.devicePath.empty() ? "/dev/video" + std::to_string(config_.deviceIndex)
                                                 : config_.devicePath)
                  << " -> " << cap_.get(cv::CAP_PROP_FRAME_WIDTH) << "x"
                  << cap_.get(cv::CAP_PROP_FRAME_HEIGHT) << " @ " << cap_.get(cv::CAP_PROP_FPS)
                  << " FPS, crop " << config_.cropSize << "x" << config_.cropSize << std::endl;
        return true;
    }

    void loop()
    {
        while (running_.load()) {
            cv::Mat frame;
            if (!cap_.read(frame) || frame.empty()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }
            cv::Mat cropped = centerCropToSize(frame, config_.cropSize);
            {
                std::lock_guard<std::mutex> lock(frameMutex_);
                latestFrame_ = std::move(cropped);
            }
            frameCount_.fetch_add(1);
        }
    }

    CameraConfig config_;
    cv::VideoCapture cap_;
    std::thread thread_;
    std::atomic_bool running_{false};
    std::atomic<std::uint64_t> frameCount_{0};
    mutable std::mutex frameMutex_;
    cv::Mat latestFrame_;
};

// ---------------------------------------------------------------------------
// 인자 파싱
// ---------------------------------------------------------------------------

struct Args {
    CameraConfig camera;
    int port = 8090;
    std::string host = "0.0.0.0";
    fs::path webRoot;
    fs::path assetsDir;
    std::string detModel = "det_v6_m_640.dxnn";
    int jpegQuality = 80;
    double streamFps = 15.0;
    /** 지정되면 서버를 띄우지 않고 이 이미지로 OCR 을 한 번 돌린 뒤 결과를 출력한다. */
    fs::path testImage;
    fs::path registryPath;
    fs::path seedPath;
};

fs::path defaultRoot()
{
#ifdef SERIAL_QR_ROOT_DIR
    return SERIAL_QR_ROOT_DIR;
#else
    return fs::current_path();
#endif
}

void printUsage(const char* argv0)
{
    std::cout
        << "Usage: " << argv0 << " [options]\n"
        << "  --camera <idx>       카메라 인덱스 (기본 0)\n"
        << "  --device <path>      카메라 디바이스 경로 (예: /dev/video2, --camera 보다 우선)\n"
        << "  --port <n>           HTTP 포트 (기본 8090)\n"
        << "  --host <addr>        바인딩 주소 (기본 0.0.0.0)\n"
        << "  --width <n>          캡처 폭 (기본 1280)\n"
        << "  --height <n>         캡처 높이 (기본 720)\n"
        << "  --crop <n>           중앙 정사각 크롭 크기 (기본 960)\n"
        << "  --fps <n>            카메라 FPS (기본 15)\n"
        << "  --stream-fps <n>     MJPEG 전송 FPS (기본 15)\n"
        << "  --jpeg-quality <n>   MJPEG JPEG 품질 1-100 (기본 80)\n"
        << "  --web-root <dir>     정적 파일 루트 (기본 <app>/web/dist)\n"
        << "  --assets <dir>       PP-OCRv6 모델 디렉터리\n"
        << "  --det-model <name>   검출 모델 파일명 (기본 det_v6_m_640.dxnn)\n"
        << "  --test-image <path>  카메라 대신 이미지 1장으로 OCR 을 돌리고 종료\n"
        << "  --registry <path>    기기 레지스트리 JSON (기본 <app>/data/registry.json)\n"
        << "  --seed <path>        레지스트리 최초 생성 시 쓸 시드 JSON\n"
        << "  -h, --help           도움말\n";
}

bool parseArgs(int argc, char** argv, Args* args)
{
    const fs::path root = defaultRoot();
    args->webRoot = root / ".." / "web" / "dist";
    args->assetsDir = root / ".." / ".." / ".." / "workspace" / "models" / "ocr" / "v6";
    args->registryPath = root / ".." / "data" / "registry.json";
    args->seedPath = root / ".." / "data" / "seed_devices.json";

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("Missing value for ") + name);
            }
            return argv[++i];
        };

        if (a == "-h" || a == "--help") {
            printUsage(argv[0]);
            return false;
        } else if (a == "--camera") {
            args->camera.deviceIndex = std::stoi(next("--camera"));
        } else if (a == "--device") {
            args->camera.devicePath = next("--device");
        } else if (a == "--port") {
            args->port = std::stoi(next("--port"));
        } else if (a == "--host") {
            args->host = next("--host");
        } else if (a == "--width") {
            args->camera.width = std::stoi(next("--width"));
        } else if (a == "--height") {
            args->camera.height = std::stoi(next("--height"));
        } else if (a == "--crop") {
            args->camera.cropSize = std::stoi(next("--crop"));
        } else if (a == "--fps") {
            args->camera.fps = std::stod(next("--fps"));
        } else if (a == "--stream-fps") {
            args->streamFps = std::stod(next("--stream-fps"));
        } else if (a == "--jpeg-quality") {
            args->jpegQuality = std::clamp(std::stoi(next("--jpeg-quality")), 1, 100);
        } else if (a == "--web-root") {
            args->webRoot = next("--web-root");
        } else if (a == "--assets") {
            args->assetsDir = next("--assets");
        } else if (a == "--det-model") {
            args->detModel = next("--det-model");
        } else if (a == "--test-image") {
            args->testImage = next("--test-image");
        } else if (a == "--registry") {
            args->registryPath = next("--registry");
        } else if (a == "--seed") {
            args->seedPath = next("--seed");
        } else {
            std::cerr << "Unknown argument: " << a << "\n";
            printUsage(argv[0]);
            throw std::runtime_error("bad arguments");
        }
    }
    return true;
}

}  // namespace

int main(int argc, char** argv)
{
    Args args;
    try {
        if (!parseArgs(argc, argv, &args)) {
            return 0;
        }
    } catch (const std::exception& e) {
        std::cerr << "[args] " << e.what() << std::endl;
        return 1;
    }

    args.webRoot = fs::weakly_canonical(args.webRoot);
    args.assetsDir = fs::weakly_canonical(args.assetsDir);
    args.registryPath = fs::weakly_canonical(args.registryPath);
    if (!args.seedPath.empty()) {
        args.seedPath = fs::weakly_canonical(args.seedPath);
    }

    std::cout << "[serial-qr] assets   : " << args.assetsDir << std::endl;
    std::cout << "[serial-qr] web root : " << args.webRoot << std::endl;
    std::cout << "[serial-qr] registry : " << args.registryPath << std::endl;

    // --- OCR 엔진 ---------------------------------------------------------
    std::unique_ptr<camocr::PaddleOcrEngine> engine;
    try {
        camocr::EngineOptions options;
        options.rootDir = defaultRoot();
        options.assetsDir = args.assetsDir;
        options.detModelName = args.detModel;
        engine = std::make_unique<camocr::PaddleOcrEngine>(options);
    } catch (const std::exception& e) {
        std::cerr << "[ocr] Failed to initialize engine: " << e.what() << std::endl;
        std::cerr << "[ocr] workspace/models/ocr/v6 자산이 있는지 확인하세요 "
                     "(./setup_assets.sh)."
                  << std::endl;
        return 1;
    }
    std::cout << "[ocr] engine ready" << std::endl;

    // PaddleOcrEngine 은 내부에 async 콜백과 inflight 카운터를 들고 있어
    // 스레드 안전하지 않다. run() 호출을 반드시 직렬화한다.
    std::mutex ocrMutex;

    serialqr::DeviceRegistry registry(args.registryPath, args.seedPath);

    // --- --test-image: 카메라 없이 OCR/시리얼 추출만 검증하고 종료 -------
    if (!args.testImage.empty()) {
        const cv::Mat image = cv::imread(args.testImage.string(), cv::IMREAD_COLOR);
        if (image.empty()) {
            std::cerr << "[test] 이미지를 열 수 없습니다: " << args.testImage << std::endl;
            return 1;
        }
        try {
            const camocr::OcrResult result = engine->run(image);
            std::cout << buildScanJson(result, std::string()) << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[test] OCR 실패: " << e.what() << std::endl;
            return 1;
        }
        return 0;
    }

    // --- 카메라 -----------------------------------------------------------
    CameraWorker camera(args.camera);
    const bool cameraOk = camera.start();
    if (!cameraOk) {
        std::cerr << "[camera] 카메라를 열지 못했습니다. /api/stream 과 /api/scan 은 "
                     "503 을 반환합니다."
                  << std::endl;
    }

    const std::string lanIp = detectLanIPv4();
    const std::string lanBaseUrl =
        lanIp.empty() ? "" : "http://" + lanIp + ":" + std::to_string(args.port);
    std::cout << "[serial-qr] LAN base URL: " << (lanBaseUrl.empty() ? "(none)" : lanBaseUrl)
              << std::endl;

    const std::vector<int> jpegParams{cv::IMWRITE_JPEG_QUALITY, args.jpegQuality};

    httplib::Server server;
    server.set_payload_max_length(32 * 1024 * 1024);

    // --- /api/health ------------------------------------------------------
    server.Get("/api/health", [&](const httplib::Request&, httplib::Response& res) {
        std::ostringstream oss;
        oss << "{\"status\":" << (camera.healthy() ? "\"ok\"" : "\"degraded\"")
            << ",\"camera\":" << (camera.healthy() ? "true" : "false")
            << ",\"npu\":true"
            << ",\"frames\":" << camera.frameCount() << "}";
        res.set_content(oss.str(), "application/json");
    });

    // --- /api/config ------------------------------------------------------
    server.Get("/api/config", [&](const httplib::Request&, httplib::Response& res) {
        std::ostringstream oss;
        oss << "{\"lanBaseUrl\":" << jsonStr(lanBaseUrl) << ",\"port\":" << args.port << "}";
        res.set_content(oss.str(), "application/json");
    });

    // --- 기기 레지스트리 --------------------------------------------------
    //
    // 프론트엔드가 아니라 서버가 기기 정보를 들고 있다. QR 을 찍은 휴대폰이
    // 같은 데이터를 조회해야 하기 때문이다. 시리얼은 유니크하다.

    server.Get("/api/devices", [&registry](const httplib::Request&, httplib::Response& res) {
        res.set_content(registry.all().dump(), "application/json");
    });

    server.Get(R"(/api/devices/([^/]+))", [&registry](const httplib::Request& req,
                                                      httplib::Response& res) {
        const auto device = registry.find(req.matches[1].str());
        if (!device) {
            res.status = 404;
            res.set_content(R"({"ok":false,"reason":"not_found"})", "application/json");
            return;
        }
        res.set_content(device->dump(), "application/json");
    });

    server.Post("/api/devices", [&registry](const httplib::Request& req, httplib::Response& res) {
        nlohmann::json payload;
        try {
            payload = nlohmann::json::parse(req.body);
        } catch (const nlohmann::json::exception&) {
            res.status = 400;
            res.set_content(R"({"ok":false,"reason":"bad_json"})", "application/json");
            return;
        }

        nlohmann::json stored;
        std::string message;
        const serialqr::RegisterResult result = registry.add(payload, &stored, &message);

        nlohmann::json body;
        body["message"] = message;

        switch (result) {
        case serialqr::RegisterResult::Ok:
            body["ok"] = true;
            body["device"] = stored;
            res.status = 201;
            break;
        case serialqr::RegisterResult::DuplicateSerial:
            body["ok"] = false;
            body["reason"] = "duplicate_serial";
            res.status = 409;
            break;
        case serialqr::RegisterResult::InvalidSerial:
            body["ok"] = false;
            body["reason"] = "invalid_serial";
            res.status = 400;
            break;
        case serialqr::RegisterResult::InvalidPayload:
            body["ok"] = false;
            body["reason"] = "invalid_payload";
            res.status = 400;
            break;
        case serialqr::RegisterResult::WriteFailed:
            body["ok"] = false;
            body["reason"] = "write_failed";
            res.status = 500;
            break;
        }
        res.set_content(body.dump(), "application/json");
    });

    // 데모를 반복 시연할 때 등록분을 되돌리기 위한 경로
    server.Delete(R"(/api/devices/([^/]+))", [&registry](const httplib::Request& req,
                                                         httplib::Response& res) {
        if (!registry.remove(req.matches[1].str())) {
            res.status = 404;
            res.set_content(R"({"ok":false,"reason":"not_found"})", "application/json");
            return;
        }
        res.set_content(R"({"ok":true})", "application/json");
    });

    // --- /api/stream (MJPEG) ---------------------------------------------
    const auto frameInterval =
        std::chrono::milliseconds(static_cast<int>(1000.0 / std::max(1.0, args.streamFps)));

    server.Get("/api/stream", [&, frameInterval, jpegParams](const httplib::Request&,
                                                             httplib::Response& res) {
        if (!cameraOk) {
            res.status = 503;
            res.set_content("{\"error\":\"camera_unavailable\"}", "application/json");
            return;
        }

        res.set_chunked_content_provider(
            "multipart/x-mixed-replace; boundary=frame",
            [&camera, frameInterval, jpegParams](std::size_t, httplib::DataSink& sink) {
                const cv::Mat frame = camera.latestFrame();
                if (frame.empty()) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(30));
                    return true;
                }

                std::vector<unsigned char> buf;
                if (!cv::imencode(".jpg", frame, buf, jpegParams)) {
                    return true;
                }

                std::ostringstream head;
                head << "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " << buf.size()
                     << "\r\n\r\n";
                const std::string header = head.str();

                if (!sink.write(header.data(), header.size())) {
                    return false;
                }
                if (!sink.write(reinterpret_cast<const char*>(buf.data()), buf.size())) {
                    return false;
                }
                if (!sink.write("\r\n", 2)) {
                    return false;
                }

                std::this_thread::sleep_for(frameInterval);
                return true;
            });
    });

    // --- /api/scan --------------------------------------------------------
    auto handleScan = [&](const httplib::Request&, httplib::Response& res) {
        if (!cameraOk) {
            res.status = 503;
            res.set_content("{\"ok\":false,\"reason\":\"camera_unavailable\"}", "application/json");
            return;
        }

        const cv::Mat frame = camera.latestFrame();
        if (frame.empty()) {
            res.status = 503;
            res.set_content("{\"ok\":false,\"reason\":\"no_frame\"}", "application/json");
            return;
        }

        camocr::OcrResult result;
        try {
            std::lock_guard<std::mutex> lock(ocrMutex);
            result = engine->run(frame);
        } catch (const std::exception& e) {
            std::cerr << "[ocr] scan failed: " << e.what() << std::endl;
            res.status = 500;
            res.set_content(std::string("{\"ok\":false,\"reason\":\"ocr_error\",\"message\":") +
                                jsonStr(e.what()) + "}",
                            "application/json");
            return;
        }

        std::vector<unsigned char> jpeg;
        cv::imencode(".jpg", frame, jpeg, jpegParams);

        res.set_content(buildScanJson(result, base64Encode(jpeg)), "application/json");
    };

    server.Post("/api/scan", handleScan);
    server.Get("/api/scan", handleScan);  // 브라우저에서 바로 눌러보기 편하도록

    // --- 정적 파일 + SPA 폴백 --------------------------------------------
    const bool webRootExists = fs::exists(args.webRoot);
    if (webRootExists) {
        server.set_mount_point("/", args.webRoot.string());
    } else {
        std::cerr << "[web] " << args.webRoot << " 가 없습니다. "
                  << "apps/serial-qr/build.sh 로 프론트엔드를 빌드하세요." << std::endl;
    }

    const fs::path indexPath = args.webRoot / "index.html";
    server.set_error_handler([&](const httplib::Request& req, httplib::Response& res) {
        // SPA 라우팅(/device/xxx 직접 진입, 새로고침)을 위해 index.html 로 폴백한다.
        if (res.status != 404 || req.method != "GET" || req.path.rfind("/api/", 0) == 0) {
            return;
        }
        if (!fs::exists(indexPath)) {
            res.set_content(
                "<h1>serial-qr</h1><p>프론트엔드가 빌드되지 않았습니다. "
                "<code>apps/serial-qr/build.sh</code> 를 실행하세요.</p>",
                "text/html; charset=utf-8");
            res.status = 200;
            return;
        }
        std::ifstream in(indexPath, std::ios::binary);
        std::ostringstream body;
        body << in.rdbuf();
        res.set_content(body.str(), "text/html; charset=utf-8");
        res.status = 200;
    });

    // 런처가 붙을 경우를 대비한 준비 완료 신호 (다른 데모와 동일한 규약)
    if (const char* readyPath = std::getenv("DX_LAUNCHER_READY_FILE")) {
        if (readyPath[0] != '\0') {
            std::ofstream(readyPath).put('\n');
        }
    }

    std::cout << "[serial-qr] listening on http://" << args.host << ":" << args.port << std::endl;
    if (!lanBaseUrl.empty()) {
        std::cout << "[serial-qr] 휴대폰에서 접속: " << lanBaseUrl << std::endl;
    }

    if (!server.listen(args.host, args.port)) {
        std::cerr << "[serial-qr] 포트 " << args.port << " 바인딩 실패" << std::endl;
        camera.stop();
        return 1;
    }

    camera.stop();
    return 0;
}
