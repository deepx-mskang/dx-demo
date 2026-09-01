#pragma once

// 기기 레지스트리 — 서버가 소유하는 JSON 파일 기반 저장소.
//
// 프론트엔드가 아니라 서버가 기기 정보를 들고 있는 이유는, QR 을 찍은
// 휴대폰이 조회할 때 같은 데이터를 봐야 하기 때문이다. 브라우저
// localStorage 에 두면 등록한 PC 에서만 보인다.
//
// 시리얼은 유니크하다. 이미 있는 시리얼로 등록하면 실패한다.

#include <filesystem>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace serialqr {

enum class RegisterResult {
    Ok,
    DuplicateSerial,
    InvalidSerial,
    InvalidPayload,
    WriteFailed,
};

class DeviceRegistry {
public:
    /// registryPath 를 읽어 들인다. 파일이 없으면 seedPath 로 초기화한다.
    DeviceRegistry(std::filesystem::path registryPath, std::filesystem::path seedPath);

    /// 등록된 모든 기기 (등록 순서).
    nlohmann::json all() const;

    /// 시리얼로 조회. 없으면 nullopt.
    std::optional<nlohmann::json> find(const std::string& serial) const;

    /// 새 기기를 등록한다. 시리얼이 이미 있으면 DuplicateSerial.
    /// 성공하면 stored 에 저장된 레코드를 채운다.
    RegisterResult add(const nlohmann::json& device, nlohmann::json* stored, std::string* message);

    /// 시리얼로 삭제. 지운 게 있으면 true.
    bool remove(const std::string& serial);

    std::size_t size() const;

    /// 시리얼 표기를 통일한다 (앞뒤 공백 제거 + 대문자).
    static std::string normalizeSerial(const std::string& serial);

private:
    bool saveLocked() const;
    void loadOrSeed();
    /// devices_ 에서 시리얼 위치를 찾는다. 없으면 npos.
    std::size_t indexOfLocked(const std::string& serial) const;

    std::filesystem::path registryPath_;
    std::filesystem::path seedPath_;
    mutable std::mutex mutex_;
    nlohmann::json devices_ = nlohmann::json::array();
};

}  // namespace serialqr
