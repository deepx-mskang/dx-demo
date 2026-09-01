#include "device_registry.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <regex>

namespace fs = std::filesystem;
using nlohmann::json;

namespace serialqr {
namespace {

// 등록 폼에서 받는 필드. 여기 없는 키는 저장하지 않는다.
constexpr const char* kStringFields[] = {
    "serial", "model",         "npu",          "hwRevision",  "firmware",
    "manufacturedAt", "macAddress", "warrantyUntil", "qaStatus", "deployedSite",
};

constexpr const char* kQaValues[] = {"PASS", "FAIL", "PENDING"};

std::string trim(const std::string& s)
{
    const auto notSpace = [](unsigned char c) { return std::isspace(c) == 0; };
    auto begin = std::find_if(s.begin(), s.end(), notSpace);
    auto end = std::find_if(s.rbegin(), s.rend(), notSpace).base();
    return begin < end ? std::string(begin, end) : std::string();
}

/// 문자열 필드를 꺼낸다. 없거나 문자열이 아니면 빈 문자열.
std::string readString(const json& j, const char* key)
{
    if (!j.contains(key) || !j.at(key).is_string()) {
        return {};
    }
    return trim(j.at(key).get<std::string>());
}

double readNumber(const json& j, const char* key, double fallback)
{
    if (!j.contains(key)) {
        return fallback;
    }
    const json& v = j.at(key);
    if (v.is_number()) {
        return v.get<double>();
    }
    if (v.is_string()) {
        try {
            return std::stod(v.get<std::string>());
        } catch (const std::exception&) {
            return fallback;
        }
    }
    return fallback;
}

}  // namespace

DeviceRegistry::DeviceRegistry(fs::path registryPath, fs::path seedPath)
    : registryPath_(std::move(registryPath)), seedPath_(std::move(seedPath))
{
    loadOrSeed();
}

std::string DeviceRegistry::normalizeSerial(const std::string& serial)
{
    std::string out = trim(serial);
    std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) {
        return static_cast<char>(std::toupper(c));
    });
    return out;
}

void DeviceRegistry::loadOrSeed()
{
    std::lock_guard<std::mutex> lock(mutex_);

    const auto readArray = [](const fs::path& path, json* out) {
        std::ifstream in(path);
        if (!in) {
            return false;
        }
        try {
            json parsed = json::parse(in);
            if (!parsed.is_array()) {
                return false;
            }
            *out = std::move(parsed);
            return true;
        } catch (const json::exception& e) {
            std::cerr << "[registry] " << path << " 파싱 실패: " << e.what() << std::endl;
            return false;
        }
    };

    if (fs::exists(registryPath_) && readArray(registryPath_, &devices_)) {
        std::cout << "[registry] " << registryPath_ << " (" << devices_.size() << "대)"
                  << std::endl;
        return;
    }

    if (!seedPath_.empty() && fs::exists(seedPath_) && readArray(seedPath_, &devices_)) {
        std::cout << "[registry] 시드로 초기화: " << seedPath_ << " (" << devices_.size() << "대)"
                  << std::endl;
        if (!saveLocked()) {
            std::cerr << "[registry] 초기 저장 실패: " << registryPath_ << std::endl;
        }
        return;
    }

    std::cout << "[registry] 빈 레지스트리로 시작합니다 (" << registryPath_ << ")" << std::endl;
    devices_ = json::array();
}

bool DeviceRegistry::saveLocked() const
{
    std::error_code ec;
    fs::create_directories(registryPath_.parent_path(), ec);

    // 쓰다가 죽어도 기존 파일이 깨지지 않도록 임시 파일에 쓰고 교체한다.
    const fs::path tmp = registryPath_.string() + ".tmp";
    {
        std::ofstream out(tmp, std::ios::trunc);
        if (!out) {
            return false;
        }
        out << devices_.dump(2) << std::endl;
        if (!out) {
            return false;
        }
    }

    fs::rename(tmp, registryPath_, ec);
    if (ec) {
        fs::remove(tmp, ec);
        return false;
    }
    return true;
}

std::size_t DeviceRegistry::indexOfLocked(const std::string& serial) const
{
    for (std::size_t i = 0; i < devices_.size(); ++i) {
        const json& d = devices_[i];
        if (d.contains("serial") && d["serial"].is_string() &&
            normalizeSerial(d["serial"].get<std::string>()) == serial) {
            return i;
        }
    }
    return static_cast<std::size_t>(-1);
}

json DeviceRegistry::all() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return devices_;
}

std::optional<json> DeviceRegistry::find(const std::string& serial) const
{
    const std::string key = normalizeSerial(serial);
    std::lock_guard<std::mutex> lock(mutex_);
    const std::size_t idx = indexOfLocked(key);
    if (idx == static_cast<std::size_t>(-1)) {
        return std::nullopt;
    }
    return devices_[idx];
}

std::size_t DeviceRegistry::size() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return devices_.size();
}

RegisterResult DeviceRegistry::add(const json& device, json* stored, std::string* message)
{
    if (!device.is_object()) {
        *message = "요청 본문이 JSON 객체가 아닙니다.";
        return RegisterResult::InvalidPayload;
    }

    const std::string serial = normalizeSerial(readString(device, "serial"));
    if (serial.empty()) {
        *message = "시리얼 번호를 입력하세요.";
        return RegisterResult::InvalidSerial;
    }
    // 영숫자와 하이픈만, 4~32자.
    static const std::regex kSerialShape(R"(^[A-Z0-9][A-Z0-9-]{2,30}[A-Z0-9]$)");
    if (!std::regex_match(serial, kSerialShape)) {
        *message = "시리얼은 영문/숫자/하이픈 4~32자여야 합니다.";
        return RegisterResult::InvalidSerial;
    }

    json record = json::object();
    for (const char* key : kStringFields) {
        record[key] = readString(device, key);
    }
    record["serial"] = serial;

    if (record["model"].get<std::string>().empty()) {
        *message = "모델명을 입력하세요.";
        return RegisterResult::InvalidPayload;
    }

    const std::string qa = record["qaStatus"].get<std::string>();
    if (std::find(std::begin(kQaValues), std::end(kQaValues), qa) == std::end(kQaValues)) {
        record["qaStatus"] = "PENDING";
    }

    const json specsIn = device.contains("specs") && device["specs"].is_object()
                             ? device["specs"]
                             : json::object();
    record["specs"] = json{
        {"tops", readNumber(specsIn, "tops", 25)},
        {"memoryGb", readNumber(specsIn, "memoryGb", 4)},
        {"powerW", readNumber(specsIn, "powerW", 5)},
    };

    std::lock_guard<std::mutex> lock(mutex_);

    if (indexOfLocked(serial) != static_cast<std::size_t>(-1)) {
        *message = "이미 등록된 시리얼입니다: " + serial;
        return RegisterResult::DuplicateSerial;
    }

    devices_.push_back(record);
    if (!saveLocked()) {
        devices_.erase(devices_.size() - 1);
        *message = "레지스트리 파일을 저장하지 못했습니다.";
        return RegisterResult::WriteFailed;
    }

    *stored = record;
    *message = "등록되었습니다.";
    return RegisterResult::Ok;
}

bool DeviceRegistry::remove(const std::string& serial)
{
    const std::string key = normalizeSerial(serial);
    std::lock_guard<std::mutex> lock(mutex_);
    const std::size_t idx = indexOfLocked(key);
    if (idx == static_cast<std::size_t>(-1)) {
        return false;
    }
    const json backup = devices_;
    devices_.erase(idx);
    if (!saveLocked()) {
        devices_ = backup;
        return false;
    }
    return true;
}

}  // namespace serialqr
