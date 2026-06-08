#pragma once

#include <memory>
#include <string>
#include <vector>

#include <dxrt/dxrt_api.h>

#if defined(_WIN32)
#if defined(DX_MODEL_CRYPTO_BUILD)
#define DX_MODEL_CRYPTO_API __declspec(dllexport)
#else
#define DX_MODEL_CRYPTO_API __declspec(dllimport)
#endif
#else
#define DX_MODEL_CRYPTO_API __attribute__((visibility("default")))
#endif

extern "C" {
DX_MODEL_CRYPTO_API void *dxm_0(const char *model_path);
DX_MODEL_CRYPTO_API void dxm_1(void *handle);
DX_MODEL_CRYPTO_API dxrt::InferenceEngine *dxm_2(void *handle);
}

class EncryptedModelEngine
{
public:
    explicit EncryptedModelEngine(const std::string &model_path)
        : handle_(dxm_0(model_path.c_str()))
    {
    }

    ~EncryptedModelEngine()
    {
        dxm_1(handle_);
    }

    EncryptedModelEngine(const EncryptedModelEngine &) = delete;
    EncryptedModelEngine &operator=(const EncryptedModelEngine &) = delete;

    dxrt::InferenceEngine *get()
    {
        return dxm_2(handle_);
    }

    const dxrt::InferenceEngine *get() const
    {
        return dxm_2(handle_);
    }

private:
    void *handle_;
};
