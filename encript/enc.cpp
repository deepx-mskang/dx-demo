#include "crypto_common.hpp"

#include <algorithm>

int main(int argc, char** argv) {
    try {
        dxenc::Options options = dxenc::parse_args(argc, argv, false);

        const std::vector<unsigned char> plaintext = dxenc::read_all(options.input_path);

        dxenc::Header header;
        header.algorithm = options.algorithm;
        dxenc::fill_random(header.salt.data(), header.salt.size(), "salt");
        header.iv.assign(dxenc::iv_size_for(options.algorithm), 0);
        dxenc::fill_random(header.iv.data(), header.iv.size(), "iv");

        const auto key = dxenc::derive_key(
            options.user_id,
            options.password,
            options.ssh_key_path,
            header.salt);

        std::vector<unsigned char> ciphertext = dxenc::encrypt_data(
            options.algorithm,
            key,
            header.iv,
            plaintext,
            header.tag);

        std::vector<unsigned char> output;
        output.reserve(dxenc::kMagic.size() + 1 + 1 + dxenc::kSaltSize + 8 +
                       header.iv.size() + header.tag.size() + ciphertext.size());
        dxenc::write_header(output, header);
        output.insert(output.end(), ciphertext.begin(), ciphertext.end());

        dxenc::write_all(options.output_path, output);

        std::cout << "Encrypted " << options.input_path << " -> " << options.output_path
                  << " using " << dxenc::algorithm_name(options.algorithm) << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "enc error: " << e.what() << "\n";
        return 1;
    }
}

