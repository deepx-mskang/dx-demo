#include "encrypted_model_engine.h"

#include <openssl/evp.h>
#include <openssl/sha.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <pwd.h>
#include <stdexcept>
#include <unistd.h>

namespace
{

constexpr std::array<unsigned char, 8> kMagic = {{'D', 'X', 'E', 'N', 'C', '0', '1', '\0'}};
constexpr unsigned char kVersion = 1;
constexpr unsigned char kChacha20Poly1305 = 3;
constexpr std::size_t kSaltSize = 16;
constexpr std::size_t kKeySize = 32;
constexpr std::size_t kTagSize = 16;
constexpr int kPbkdf2Iterations = 200000;

std::runtime_error model_error(const char *code)
{
    return std::runtime_error(code);
}

std::string reveal(const unsigned char *bytes, std::size_t size, unsigned char mask)
{
    std::string value;
    value.reserve(size);
    for (std::size_t i = 0; i < size; ++i)
    {
        value.push_back(static_cast<char>(bytes[i] ^ mask));
    }
    return value;
}

std::string trim_copy(const std::string &value)
{
    std::size_t first = 0;
    while (first < value.size() && std::isspace(static_cast<unsigned char>(value[first])))
    {
        ++first;
    }

    std::size_t last = value.size();
    while (last > first && std::isspace(static_cast<unsigned char>(value[last - 1])))
    {
        --last;
    }

    return value.substr(first, last - first);
}

std::string unquote_copy(const std::string &value)
{
    if (value.size() >= 2 &&
        ((value.front() == '"' && value.back() == '"') ||
         (value.front() == '\'' && value.back() == '\'')))
    {
        return value.substr(1, value.size() - 2);
    }
    return value;
}

struct EncHeader
{
    unsigned char algorithm = 0;
    std::array<unsigned char, kSaltSize> salt;
    std::vector<unsigned char> iv;
    std::vector<unsigned char> tag;
};

std::vector<unsigned char> load_model_data(const std::string &model_path);

struct EngineState
{
    explicit EngineState(const std::string &model_path)
        : model_data(load_model_data(model_path))
    {
        if (model_data.empty())
        {
            throw model_error("E21");
        }
        engine.reset(new dxrt::InferenceEngine(
            reinterpret_cast<const uint8_t *>(model_data.data()),
            model_data.size()));
    }

    std::vector<unsigned char> model_data;
    std::unique_ptr<dxrt::InferenceEngine> engine;
};

bool file_exists(const std::string &path)
{
    std::ifstream input(path.c_str(), std::ios::binary);
    return input.good();
}

std::string encrypted_path_for(const std::string &model_path)
{
    if (file_exists(model_path))
    {
        return model_path;
    }

    const std::string enc_path = model_path + ".enc";
    if (file_exists(enc_path))
    {
        return enc_path;
    }

    throw model_error("E01");
}

std::string expand_home(const std::string &path)
{
    if (path == "~" || path.find("~/") == 0)
    {
        const unsigned char encoded_home[] = {0x6e, 0x69, 0x6b, 0x63};
        const char *home = std::getenv(reveal(encoded_home, sizeof(encoded_home), 0x26).c_str());
        if (home == nullptr || std::strlen(home) == 0)
        {
            throw model_error("E02");
        }
        return std::string(home) + path.substr(1);
    }
    return path;
}

std::vector<unsigned char> read_all(const std::string &path)
{
    std::ifstream input(path.c_str(), std::ios::binary);
    if (!input)
    {
        throw model_error("E03");
    }

    return std::vector<unsigned char>(
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>());
}

uint32_t read_u32(const std::vector<unsigned char> &data, std::size_t *offset)
{
    if (*offset + 4 > data.size())
    {
        throw model_error("E04");
    }

    const uint32_t value = (static_cast<uint32_t>(data[*offset]) << 24) |
                           (static_cast<uint32_t>(data[*offset + 1]) << 16) |
                           (static_cast<uint32_t>(data[*offset + 2]) << 8) |
                           static_cast<uint32_t>(data[*offset + 3]);
    *offset += 4;
    return value;
}

bool has_encrypted_magic(const std::vector<unsigned char> &data)
{
    return data.size() >= kMagic.size() &&
           std::equal(kMagic.begin(), kMagic.end(), data.begin());
}

EncHeader read_header(const std::vector<unsigned char> &data, std::size_t *offset)
{
    if (data.size() < kMagic.size() + 1 + 1 + kSaltSize + 4 + 4)
    {
        throw model_error("E05");
    }
    if (!has_encrypted_magic(data))
    {
        throw model_error("E06");
    }

    *offset = kMagic.size();
    const unsigned char version = data[(*offset)++];
    if (version != kVersion)
    {
        throw model_error("E07");
    }

    EncHeader header;
    header.algorithm = data[(*offset)++];
    if (header.algorithm != kChacha20Poly1305)
    {
        throw model_error("E08");
    }

    std::copy(data.begin() + static_cast<std::ptrdiff_t>(*offset),
              data.begin() + static_cast<std::ptrdiff_t>(*offset + kSaltSize),
              header.salt.begin());
    *offset += kSaltSize;

    const uint32_t iv_size = read_u32(data, offset);
    if (*offset + iv_size > data.size())
    {
        throw model_error("E09");
    }
    header.iv.assign(data.begin() + static_cast<std::ptrdiff_t>(*offset),
                     data.begin() + static_cast<std::ptrdiff_t>(*offset + iv_size));
    *offset += iv_size;

    const uint32_t tag_size = read_u32(data, offset);
    if (*offset + tag_size > data.size())
    {
        throw model_error("E10");
    }
    header.tag.assign(data.begin() + static_cast<std::ptrdiff_t>(*offset),
                      data.begin() + static_cast<std::ptrdiff_t>(*offset + tag_size));
    *offset += tag_size;

    return header;
}

std::string model_user_id()
{
    const passwd *entry = getpwuid(geteuid());
    if (entry != nullptr && entry->pw_name != nullptr && entry->pw_name[0] != '\0')
    {
        return entry->pw_name;
    }

    const unsigned char encoded_user[] = {0x72, 0x74, 0x62, 0x75};
    const char *user = std::getenv(reveal(encoded_user, sizeof(encoded_user), 0x27).c_str());
    if (user != nullptr && *user != '\0')
    {
        return user;
    }

    throw model_error("E11");
}

std::string model_password()
{
    const unsigned char encoded_env[] = {
        0x7d, 0x61, 0x66, 0x74, 0x76, 0x7d, 0x7c, 0x75, 0x66, 0x7d, 0x7c, 0x7a, 0x6b,
        0x60, 0x69, 0x6d, 0x66, 0x69, 0x78, 0x6a, 0x6a, 0x6e, 0x76, 0x6b, 0x7d};
    const std::string key = reveal(encoded_env, sizeof(encoded_env), 0x39);

    const unsigned char encoded_path[] = {
        0x7e, 0x34, 0x25, 0x32, 0x7e, 0x7f, 0x35, 0x29, 0x7c, 0x35, 0x34, 0x3c, 0x3e,
        0x22, 0x7e, 0x7f, 0x22, 0x34, 0x32, 0x23, 0x34, 0x25, 0x7f, 0x34, 0x3f, 0x27};
    std::ifstream input(reveal(encoded_path, sizeof(encoded_path), 0x51).c_str());
    if (!input)
    {
        throw model_error("E24");
    }

    std::string line;
    while (std::getline(input, line))
    {
        line = trim_copy(line);
        if (line.empty() || line[0] == '#')
        {
            continue;
        }

        const std::size_t equal_pos = line.find('=');
        if (equal_pos == std::string::npos)
        {
            continue;
        }

        const std::string name = trim_copy(line.substr(0, equal_pos));
        if (name != key)
        {
            continue;
        }

        const std::string password = unquote_copy(trim_copy(line.substr(equal_pos + 1)));
        if (password.empty())
        {
            throw model_error("E25");
        }
        return password;
    }

    throw model_error("E26");
}

std::string ssh_key_path()
{
    const unsigned char encoded_env[] = {
        0x7e, 0x62, 0x65, 0x77, 0x75, 0x7e, 0x7f, 0x76, 0x65, 0x7e, 0x7f, 0x79, 0x68,
        0x63, 0x6a, 0x6e, 0x65, 0x71, 0x7f, 0x63, 0x65, 0x6a, 0x7b, 0x6e, 0x72};
    const char *override_path = std::getenv(reveal(encoded_env, sizeof(encoded_env), 0x3a).c_str());
    if (override_path != nullptr && *override_path != '\0')
    {
        return override_path;
    }

    const unsigned char encoded_path[] = {
        0x3a, 0x6b, 0x6a, 0x37, 0x37, 0x2c, 0x6b, 0x2d, 0x20,
        0x1b, 0x21, 0x20, 0x76, 0x71, 0x71, 0x75, 0x7d};
    return reveal(encoded_path, sizeof(encoded_path), 0x44);
}

std::array<unsigned char, kKeySize> derive_key(const EncHeader &header)
{
    const std::vector<unsigned char> ssh_key = read_all(expand_home(ssh_key_path()));
    const std::string user_id = model_user_id();
    const std::string password = model_password();

    std::vector<unsigned char> material;
    material.reserve(user_id.size() + password.size() + ssh_key.size() + 3);
    material.insert(material.end(), user_id.begin(), user_id.end());
    material.push_back(0);
    material.insert(material.end(), password.begin(), password.end());
    material.push_back(0);
    material.insert(material.end(), ssh_key.begin(), ssh_key.end());

    std::array<unsigned char, SHA256_DIGEST_LENGTH> hash;
    SHA256(material.data(), material.size(), hash.data());

    std::array<unsigned char, kKeySize> key;
    if (PKCS5_PBKDF2_HMAC(
            reinterpret_cast<const char *>(hash.data()),
            static_cast<int>(hash.size()),
            header.salt.data(),
            static_cast<int>(header.salt.size()),
            kPbkdf2Iterations,
            EVP_sha256(),
            static_cast<int>(key.size()),
            key.data()) != 1)
    {
        throw model_error("E12");
    }
    return key;
}

std::vector<unsigned char> decrypt_chacha20_poly1305(
    const std::array<unsigned char, kKeySize> &key,
    const EncHeader &header,
    const std::vector<unsigned char> &ciphertext)
{
    if (header.tag.size() != kTagSize)
    {
        throw model_error("E13");
    }

    typedef std::unique_ptr<EVP_CIPHER_CTX, decltype(&EVP_CIPHER_CTX_free)> CipherCtx;
    CipherCtx ctx(EVP_CIPHER_CTX_new(), EVP_CIPHER_CTX_free);
    if (!ctx)
    {
        throw model_error("E14");
    }

    if (EVP_DecryptInit_ex(ctx.get(), EVP_chacha20_poly1305(), nullptr, nullptr, nullptr) != 1)
    {
        throw model_error("E15");
    }
    if (EVP_CIPHER_CTX_ctrl(ctx.get(), EVP_CTRL_AEAD_SET_IVLEN,
                            static_cast<int>(header.iv.size()), nullptr) != 1)
    {
        throw model_error("E16");
    }
    if (EVP_DecryptInit_ex(ctx.get(), nullptr, nullptr, key.data(), header.iv.data()) != 1)
    {
        throw model_error("E17");
    }

    std::vector<unsigned char> plaintext(ciphertext.size() + EVP_CIPHER_block_size(EVP_chacha20_poly1305()));
    int out_len = 0;
    int total_len = 0;
    if (EVP_DecryptUpdate(ctx.get(), plaintext.data(), &out_len, ciphertext.data(),
                          static_cast<int>(ciphertext.size())) != 1)
    {
        throw model_error("E18");
    }
    total_len += out_len;

    if (EVP_CIPHER_CTX_ctrl(ctx.get(), EVP_CTRL_AEAD_SET_TAG, static_cast<int>(header.tag.size()),
                            const_cast<unsigned char *>(header.tag.data())) != 1)
    {
        throw model_error("E19");
    }

    if (EVP_DecryptFinal_ex(ctx.get(), plaintext.data() + total_len, &out_len) != 1)
    {
        throw model_error("E20");
    }
    total_len += out_len;
    plaintext.resize(static_cast<std::size_t>(total_len));
    return plaintext;
}

std::vector<unsigned char> load_model_data(const std::string &model_path)
{
    const std::string actual_path = encrypted_path_for(model_path);
    std::vector<unsigned char> data = read_all(actual_path);
    if (!has_encrypted_magic(data))
    {
        return data;
    }

    std::size_t payload_offset = 0;
    const EncHeader header = read_header(data, &payload_offset);
    std::vector<unsigned char> ciphertext(
        data.begin() + static_cast<std::ptrdiff_t>(payload_offset),
        data.end());

    return decrypt_chacha20_poly1305(derive_key(header), header, ciphertext);
}

}  // namespace

extern "C" void *dxm_0(const char *model_path)
{
    if (model_path == nullptr || *model_path == '\0')
    {
        throw model_error("E22");
    }
    return new EngineState(model_path);
}

extern "C" void dxm_1(void *handle)
{
    delete static_cast<EngineState *>(handle);
}

extern "C" dxrt::InferenceEngine *dxm_2(void *handle)
{
    if (handle == nullptr)
    {
        throw model_error("E23");
    }
    return static_cast<EngineState *>(handle)->engine.get();
}
