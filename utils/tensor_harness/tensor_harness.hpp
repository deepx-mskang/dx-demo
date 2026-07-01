#pragma once
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace dx_harness {

/**
 * 텐서 데이터를 바이너리 파일로 덤프합니다.
 * @param data float 포인터
 * @param size float 원소의 개수 (바이트 크기가 아님)
 * @param filename 저장할 파일 경로
 */
inline void dump_tensor(const float* data, size_t size, const std::string& filename) {
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs.is_open()) {
        std::cerr << "[Harness Error] Failed to open file for writing: " << filename << std::endl;
        return;
    }
    ofs.write(reinterpret_cast<const char*>(data), size * sizeof(float));
    ofs.close();
    std::cout << "[Harness] Dumped " << size << " floats to " << filename << std::endl;
}

/**
 * std::vector<float>를 바이너리 파일로 덤프합니다.
 */
inline void dump_tensor(const std::vector<float>& tensor, const std::string& filename) {
    dump_tensor(tensor.data(), tensor.size(), filename);
}

} // namespace dx_harness
