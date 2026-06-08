#pragma once

#include <openssl/evp.h>
#include <openssl/rand.h>
#include <openssl/sha.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <termios.h>
#include <unistd.h>
#include <vector>

namespace dxenc {

constexpr std::array<unsigned char, 8> kMagic = {'D', 'X', 'E', 'N', 'C', '0', '1', '\0'};
constexpr uint8_t kVersion = 1;
constexpr size_t kSaltSize = 16;
constexpr size_t kKeySize = 32;
constexpr size_t kTagSize = 16;
constexpr int kPbkdf2Iterations = 200000;

enum class Algorithm : uint8_t {
    Aes256Cbc = 1,
    Aes256Gcm = 2,
    Chacha20Poly1305 = 3,
};

struct Options {
    Algorithm algorithm = Algorithm::Aes256Gcm;
    std::string input_path;
    std::string output_path;
    std::string user_id;
    std::string password;
    std::string ssh_key_path;
};

struct Header {
    Algorithm algorithm = Algorithm::Aes256Gcm;
    std::array<unsigned char, kSaltSize> salt{};
    std::vector<unsigned char> iv;
    std::vector<unsigned char> tag;
};

inline std::string usage_algorithms() {
    return "aes-256-cbc | aes-256-gcm | chacha20-poly1305";
}

inline Algorithm parse_algorithm(const std::string& value) {
    if (value == "aes-256-cbc" || value == "cbc") {
        return Algorithm::Aes256Cbc;
    }
    if (value == "aes-256-gcm" || value == "gcm") {
        return Algorithm::Aes256Gcm;
    }
    if (value == "chacha20-poly1305" || value == "chacha") {
        return Algorithm::Chacha20Poly1305;
    }
    throw std::runtime_error("unsupported algorithm: " + value);
}

inline std::string algorithm_name(Algorithm algorithm) {
    switch (algorithm) {
        case Algorithm::Aes256Cbc:
            return "aes-256-cbc";
        case Algorithm::Aes256Gcm:
            return "aes-256-gcm";
        case Algorithm::Chacha20Poly1305:
            return "chacha20-poly1305";
    }
    throw std::runtime_error("unknown algorithm");
}

inline const EVP_CIPHER* cipher_for(Algorithm algorithm) {
    switch (algorithm) {
        case Algorithm::Aes256Cbc:
            return EVP_aes_256_cbc();
        case Algorithm::Aes256Gcm:
            return EVP_aes_256_gcm();
        case Algorithm::Chacha20Poly1305:
            return EVP_chacha20_poly1305();
    }
    throw std::runtime_error("unknown algorithm");
}

inline size_t iv_size_for(Algorithm algorithm) {
    switch (algorithm) {
        case Algorithm::Aes256Cbc:
            return 16;
        case Algorithm::Aes256Gcm:
        case Algorithm::Chacha20Poly1305:
            return 12;
    }
    throw std::runtime_error("unknown algorithm");
}

inline bool is_aead(Algorithm algorithm) {
    return algorithm == Algorithm::Aes256Gcm || algorithm == Algorithm::Chacha20Poly1305;
}

inline std::string expand_home(const std::string& path) {
    if (path == "~" || path.rfind("~/", 0) == 0) {
        const char* home = std::getenv("HOME");
        if (home == nullptr || std::strlen(home) == 0) {
            throw std::runtime_error("HOME environment variable is not set");
        }
        return std::string(home) + path.substr(1);
    }
    return path;
}

inline std::vector<unsigned char> read_all(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("failed to open input file: " + path);
    }
    return std::vector<unsigned char>(
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>());
}

inline void write_all(const std::string& path, const std::vector<unsigned char>& data) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("failed to open output file: " + path);
    }
    output.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
    if (!output) {
        throw std::runtime_error("failed to write output file: " + path);
    }
}

inline bool file_exists(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    return input.good();
}

inline std::string default_user_id() {
    const char* user = std::getenv("USER");
    return user == nullptr ? "" : std::string(user);
}

inline std::string read_password_hidden(const std::string& prompt) {
    std::cerr << prompt;

    termios old_term{};
    if (tcgetattr(STDIN_FILENO, &old_term) != 0) {
        std::string password;
        std::getline(std::cin, password);
        return password;
    }

    termios new_term = old_term;
    new_term.c_lflag &= static_cast<unsigned int>(~ECHO);
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &new_term);

    std::string password;
    std::getline(std::cin, password);

    tcsetattr(STDIN_FILENO, TCSAFLUSH, &old_term);
    std::cerr << '\n';
    return password;
}

inline std::array<unsigned char, kKeySize> derive_key(
    const std::string& user_id,
    const std::string& password,
    const std::string& ssh_key_path,
    const std::array<unsigned char, kSaltSize>& salt) {
    const std::string expanded_key_path = expand_home(ssh_key_path);
    const std::vector<unsigned char> ssh_key = read_all(expanded_key_path);

    std::vector<unsigned char> material;
    material.reserve(user_id.size() + password.size() + ssh_key.size() + 3);
    material.insert(material.end(), user_id.begin(), user_id.end());
    material.push_back(0);
    material.insert(material.end(), password.begin(), password.end());
    material.push_back(0);
    material.insert(material.end(), ssh_key.begin(), ssh_key.end());

    std::array<unsigned char, SHA256_DIGEST_LENGTH> hash{};
    SHA256(material.data(), material.size(), hash.data());

    std::array<unsigned char, kKeySize> key{};
    if (PKCS5_PBKDF2_HMAC(
            reinterpret_cast<const char*>(hash.data()),
            static_cast<int>(hash.size()),
            salt.data(),
            static_cast<int>(salt.size()),
            kPbkdf2Iterations,
            EVP_sha256(),
            static_cast<int>(key.size()),
            key.data()) != 1) {
        throw std::runtime_error("failed to derive encryption key");
    }
    return key;
}

inline void append_u32(std::vector<unsigned char>& output, uint32_t value) {
    output.push_back(static_cast<unsigned char>((value >> 24) & 0xff));
    output.push_back(static_cast<unsigned char>((value >> 16) & 0xff));
    output.push_back(static_cast<unsigned char>((value >> 8) & 0xff));
    output.push_back(static_cast<unsigned char>(value & 0xff));
}

inline uint32_t read_u32(const std::vector<unsigned char>& data, size_t& offset) {
    if (offset + 4 > data.size()) {
        throw std::runtime_error("truncated encrypted file header");
    }
    uint32_t value = (static_cast<uint32_t>(data[offset]) << 24) |
                     (static_cast<uint32_t>(data[offset + 1]) << 16) |
                     (static_cast<uint32_t>(data[offset + 2]) << 8) |
                     static_cast<uint32_t>(data[offset + 3]);
    offset += 4;
    return value;
}

inline void write_header(std::vector<unsigned char>& output, const Header& header) {
    output.insert(output.end(), kMagic.begin(), kMagic.end());
    output.push_back(kVersion);
    output.push_back(static_cast<uint8_t>(header.algorithm));
    output.insert(output.end(), header.salt.begin(), header.salt.end());

    if (header.iv.size() > std::numeric_limits<uint32_t>::max() ||
        header.tag.size() > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error("header field is too large");
    }

    append_u32(output, static_cast<uint32_t>(header.iv.size()));
    output.insert(output.end(), header.iv.begin(), header.iv.end());
    append_u32(output, static_cast<uint32_t>(header.tag.size()));
    output.insert(output.end(), header.tag.begin(), header.tag.end());
}

inline Header read_header(const std::vector<unsigned char>& data, size_t& offset) {
    if (data.size() < kMagic.size() + 1 + 1 + kSaltSize + 4 + 4) {
        throw std::runtime_error("encrypted file is too small");
    }
    if (!std::equal(kMagic.begin(), kMagic.end(), data.begin())) {
        throw std::runtime_error("invalid encrypted file magic");
    }
    offset = kMagic.size();

    const uint8_t version = data[offset++];
    if (version != kVersion) {
        throw std::runtime_error("unsupported encrypted file version");
    }

    Header header;
    header.algorithm = static_cast<Algorithm>(data[offset++]);
    if (header.algorithm != Algorithm::Aes256Cbc &&
        header.algorithm != Algorithm::Aes256Gcm &&
        header.algorithm != Algorithm::Chacha20Poly1305) {
        throw std::runtime_error("unsupported algorithm in encrypted file");
    }

    std::copy(data.begin() + static_cast<std::ptrdiff_t>(offset),
              data.begin() + static_cast<std::ptrdiff_t>(offset + kSaltSize),
              header.salt.begin());
    offset += kSaltSize;

    const uint32_t iv_size = read_u32(data, offset);
    if (offset + iv_size > data.size()) {
        throw std::runtime_error("truncated iv in encrypted file");
    }
    header.iv.assign(data.begin() + static_cast<std::ptrdiff_t>(offset),
                     data.begin() + static_cast<std::ptrdiff_t>(offset + iv_size));
    offset += iv_size;

    const uint32_t tag_size = read_u32(data, offset);
    if (offset + tag_size > data.size()) {
        throw std::runtime_error("truncated tag in encrypted file");
    }
    header.tag.assign(data.begin() + static_cast<std::ptrdiff_t>(offset),
                      data.begin() + static_cast<std::ptrdiff_t>(offset + tag_size));
    offset += tag_size;

    return header;
}

inline std::vector<unsigned char> encrypt_data(
    Algorithm algorithm,
    const std::array<unsigned char, kKeySize>& key,
    const std::vector<unsigned char>& iv,
    const std::vector<unsigned char>& plaintext,
    std::vector<unsigned char>& tag) {
    using CtxPtr = std::unique_ptr<EVP_CIPHER_CTX, decltype(&EVP_CIPHER_CTX_free)>;
    CtxPtr ctx(EVP_CIPHER_CTX_new(), EVP_CIPHER_CTX_free);
    if (!ctx) {
        throw std::runtime_error("failed to allocate cipher context");
    }

    const EVP_CIPHER* cipher = cipher_for(algorithm);
    if (EVP_EncryptInit_ex(ctx.get(), cipher, nullptr, nullptr, nullptr) != 1) {
        throw std::runtime_error("failed to initialize encryption");
    }
    if (is_aead(algorithm) &&
        EVP_CIPHER_CTX_ctrl(ctx.get(), EVP_CTRL_AEAD_SET_IVLEN, static_cast<int>(iv.size()), nullptr) != 1) {
        throw std::runtime_error("failed to set iv length");
    }
    if (EVP_EncryptInit_ex(ctx.get(), nullptr, nullptr, key.data(), iv.data()) != 1) {
        throw std::runtime_error("failed to set encryption key and iv");
    }

    std::vector<unsigned char> ciphertext(plaintext.size() + EVP_CIPHER_block_size(cipher));
    int out_len = 0;
    int total_len = 0;
    if (EVP_EncryptUpdate(
            ctx.get(),
            ciphertext.data(),
            &out_len,
            plaintext.data(),
            static_cast<int>(plaintext.size())) != 1) {
        throw std::runtime_error("encryption failed");
    }
    total_len += out_len;

    if (EVP_EncryptFinal_ex(ctx.get(), ciphertext.data() + total_len, &out_len) != 1) {
        throw std::runtime_error("failed to finalize encryption");
    }
    total_len += out_len;
    ciphertext.resize(static_cast<size_t>(total_len));

    if (is_aead(algorithm)) {
        tag.assign(kTagSize, 0);
        if (EVP_CIPHER_CTX_ctrl(ctx.get(), EVP_CTRL_AEAD_GET_TAG, static_cast<int>(tag.size()), tag.data()) != 1) {
            throw std::runtime_error("failed to get authentication tag");
        }
    } else {
        tag.clear();
    }

    return ciphertext;
}

inline std::vector<unsigned char> decrypt_data(
    Algorithm algorithm,
    const std::array<unsigned char, kKeySize>& key,
    const std::vector<unsigned char>& iv,
    const std::vector<unsigned char>& tag,
    const std::vector<unsigned char>& ciphertext) {
    using CtxPtr = std::unique_ptr<EVP_CIPHER_CTX, decltype(&EVP_CIPHER_CTX_free)>;
    CtxPtr ctx(EVP_CIPHER_CTX_new(), EVP_CIPHER_CTX_free);
    if (!ctx) {
        throw std::runtime_error("failed to allocate cipher context");
    }

    const EVP_CIPHER* cipher = cipher_for(algorithm);
    if (EVP_DecryptInit_ex(ctx.get(), cipher, nullptr, nullptr, nullptr) != 1) {
        throw std::runtime_error("failed to initialize decryption");
    }
    if (is_aead(algorithm) &&
        EVP_CIPHER_CTX_ctrl(ctx.get(), EVP_CTRL_AEAD_SET_IVLEN, static_cast<int>(iv.size()), nullptr) != 1) {
        throw std::runtime_error("failed to set iv length");
    }
    if (EVP_DecryptInit_ex(ctx.get(), nullptr, nullptr, key.data(), iv.data()) != 1) {
        throw std::runtime_error("failed to set decryption key and iv");
    }

    std::vector<unsigned char> plaintext(ciphertext.size() + EVP_CIPHER_block_size(cipher));
    int out_len = 0;
    int total_len = 0;
    if (EVP_DecryptUpdate(
            ctx.get(),
            plaintext.data(),
            &out_len,
            ciphertext.data(),
            static_cast<int>(ciphertext.size())) != 1) {
        throw std::runtime_error("decryption failed");
    }
    total_len += out_len;

    if (is_aead(algorithm)) {
        if (tag.size() != kTagSize) {
            throw std::runtime_error("invalid authentication tag size");
        }
        if (EVP_CIPHER_CTX_ctrl(ctx.get(), EVP_CTRL_AEAD_SET_TAG, static_cast<int>(tag.size()),
                                const_cast<unsigned char*>(tag.data())) != 1) {
            throw std::runtime_error("failed to set authentication tag");
        }
    }

    if (EVP_DecryptFinal_ex(ctx.get(), plaintext.data() + total_len, &out_len) != 1) {
        throw std::runtime_error("failed to finalize decryption; credentials, ssh key, or file may be wrong");
    }
    total_len += out_len;
    plaintext.resize(static_cast<size_t>(total_len));

    return plaintext;
}

inline void fill_random(unsigned char* data, size_t size, const std::string& name) {
    if (RAND_bytes(data, static_cast<int>(size)) != 1) {
        throw std::runtime_error("failed to generate random " + name);
    }
}

inline std::string strip_enc_suffix(const std::string& path) {
    const std::string suffix = ".enc";
    if (path.size() >= suffix.size() &&
        path.compare(path.size() - suffix.size(), suffix.size(), suffix) == 0) {
        return path.substr(0, path.size() - suffix.size());
    }
    return path + ".dec";
}

inline void print_common_usage(const char* program, bool decrypt) {
    std::cerr
        << "Usage: " << program << " --alg <" << usage_algorithms() << "> --user <id> [options] <model>\n"
        << "\n"
        << "Options:\n"
        << "  -a, --alg <name>       Algorithm. Default: aes-256-gcm\n"
        << "  -u, --user <id>        User ID. Default: $USER\n"
        << "  -p, --password <pw>    User password. If omitted, prompts securely.\n"
        << "  -k, --key <path>       SSH private key path. Default: ~/.ssh/id_ed25519\n"
        << "  -o, --output <path>    Output path. Default: "
        << (decrypt ? "remove .enc suffix or append .dec" : "append .enc") << "\n"
        << "  -h, --help             Show this help.\n";
}

inline Options parse_args(int argc, char** argv, bool decrypt) {
    Options options;
    options.user_id = default_user_id();
    options.ssh_key_path = "~/.ssh/id_ed25519";

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto need_value = [&](const std::string& flag) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("missing value for " + flag);
            }
            return argv[++i];
        };

        if (arg == "-h" || arg == "--help") {
            print_common_usage(argv[0], decrypt);
            std::exit(0);
        } else if (arg == "-a" || arg == "--alg") {
            options.algorithm = parse_algorithm(need_value(arg));
        } else if (arg == "-u" || arg == "--user") {
            options.user_id = need_value(arg);
        } else if (arg == "-p" || arg == "--password") {
            options.password = need_value(arg);
        } else if (arg == "-k" || arg == "--key") {
            options.ssh_key_path = need_value(arg);
        } else if (arg == "-o" || arg == "--output") {
            options.output_path = need_value(arg);
        } else if (!arg.empty() && arg[0] == '-') {
            throw std::runtime_error("unknown option: " + arg);
        } else if (options.input_path.empty()) {
            options.input_path = arg;
        } else {
            throw std::runtime_error("unexpected extra argument: " + arg);
        }
    }

    if (options.input_path.empty()) {
        throw std::runtime_error("model input path is required");
    }
    if (options.user_id.empty()) {
        throw std::runtime_error("user id is required; pass --user <id>");
    }
    if (options.password.empty()) {
        options.password = read_password_hidden("User password: ");
    }
    if (options.output_path.empty()) {
        options.output_path = decrypt ? strip_enc_suffix(options.input_path) : options.input_path + ".enc";
    }

    return options;
}

}  // namespace dxenc

