#include "crypto_common.hpp"

#include <algorithm>

int main(int argc, char** argv) {
    try {
        dxenc::Options options = dxenc::parse_args(argc, argv, true);

        const std::vector<unsigned char> encrypted = dxenc::read_all(options.input_path);
        size_t payload_offset = 0;
        const dxenc::Header header = dxenc::read_header(encrypted, payload_offset);

        if (header.algorithm != options.algorithm) {
            throw std::runtime_error(
                "algorithm mismatch: file uses " + dxenc::algorithm_name(header.algorithm) +
                ", but argument is " + dxenc::algorithm_name(options.algorithm));
        }

        std::vector<unsigned char> ciphertext(
            encrypted.begin() + static_cast<std::ptrdiff_t>(payload_offset),
            encrypted.end());

        const auto key = dxenc::derive_key(
            options.user_id,
            options.password,
            options.ssh_key_path,
            header.salt);

        const std::vector<unsigned char> plaintext = dxenc::decrypt_data(
            header.algorithm,
            key,
            header.iv,
            header.tag,
            ciphertext);

        dxenc::write_all(options.output_path, plaintext);

        std::cout << "Decrypted " << options.input_path << " -> " << options.output_path
                  << " using " << dxenc::algorithm_name(header.algorithm) << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "dec error: " << e.what() << "\n";
        return 1;
    }
}

