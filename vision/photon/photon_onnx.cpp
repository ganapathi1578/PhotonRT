#include "photon/photon_onnx.h"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

namespace photon {
namespace {

constexpr std::size_t D_MODEL = 256;
constexpr std::size_t NUM_LAYERS = 6;
constexpr std::size_t NUM_HEADS = 8;
constexpr std::size_t HEAD_DIM = 32;
constexpr std::size_t VOCAB_SIZE = 8000;
constexpr std::size_t IMAGE_EMBED_DIM = 512;
constexpr std::size_t IMAGE_TOKENS = 8;

std::size_t configured_threads()
{
    const char* raw = std::getenv("PHOTON_THREADS");

    if (!raw || !*raw)
        return 1;

    char* end = nullptr;

    const unsigned long n =
        std::strtoul(raw, &end, 10);

    if (end == raw || *end != '\0' || n == 0)
        return 1;

    return static_cast<std::size_t>(
        std::min<unsigned long>(n, 256UL)
    );
}

#ifdef _WIN32

std::wstring widen_utf8(const std::string& s)
{
    if (s.empty())
        return {};

    const int n =
        MultiByteToWideChar(
            CP_UTF8,
            MB_ERR_INVALID_CHARS,
            s.data(),
            static_cast<int>(s.size()),
            nullptr,
            0
        );

    if (n <= 0)
        throw std::runtime_error(
            "Failed to convert ONNX path to UTF-16"
        );

    std::wstring out(
        static_cast<std::size_t>(n),
        L'\0'
    );

    if (
        MultiByteToWideChar(
            CP_UTF8,
            MB_ERR_INVALID_CHARS,
            s.data(),
            static_cast<int>(s.size()),
            out.data(),
            n
        ) != n
    ) {
        throw std::runtime_error(
            "Failed to convert ONNX path to UTF-16"
        );
    }

    return out;
}

#endif

Ort::SessionOptions make_session_options()
{
    Ort::SessionOptions options;

    options.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL
    );

    options.SetExecutionMode(
        ExecutionMode::ORT_SEQUENTIAL
    );

    options.SetIntraOpNumThreads(
        static_cast<int>(configured_threads())
    );

    options.SetInterOpNumThreads(1);

    options.EnableCpuMemArena();
    options.EnableMemPattern();

    return options;
}

Ort::MemoryInfo cpu_memory()
{
    return Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator,
        OrtMemTypeDefault
    );
}

int32_t argmax(
    const float* values,
    std::size_t n
)
{
    std::size_t best = 0;

    for (std::size_t i = 1; i < n; ++i) {
        if (values[i] > values[best])
            best = i;
    }

    return static_cast<int32_t>(best);
}

std::vector<float> read_logits(
    const Ort::Value& value
)
{
    if (!value.IsTensor())
        throw std::runtime_error(
            "Photon ONNX logits output is not a tensor"
        );

    const auto info =
        value.GetTensorTypeAndShapeInfo();

    if (
        info.GetElementType() !=
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
    ) {
        throw std::runtime_error(
            "Photon logits must be float32"
        );
    }

    const auto shape = info.GetShape();

    std::size_t count = 1;

    for (const int64_t d : shape) {
        if (d <= 0)
            throw std::runtime_error(
                "Invalid Photon logits shape"
            );

        count *= static_cast<std::size_t>(d);
    }

    if (count != VOCAB_SIZE)
        throw std::runtime_error(
            "Photon logits must contain 8000 values"
        );

    const float* p =
        value.GetTensorData<float>();

    return std::vector<float>(
        p,
        p + VOCAB_SIZE
    );
}

} // namespace

struct PhotonONNX::Impl {

    static Ort::Env& env()
    {
        static Ort::Env e(
            ORT_LOGGING_LEVEL_WARNING,
            "PhotonRT-Photon"
        );

        return e;
    }

    Ort::SessionOptions prefill_options;
    Ort::SessionOptions decode_options;

    std::unique_ptr<Ort::Session> prefill;
    std::unique_ptr<Ort::Session> decode;

    bool loaded = false;
};

PhotonONNX::PhotonONNX()
    : impl_(std::make_unique<Impl>())
{
}

PhotonONNX::~PhotonONNX() = default;

PhotonONNX::PhotonONNX(
    PhotonONNX&&
) noexcept = default;

PhotonONNX& PhotonONNX::operator=(
    PhotonONNX&&
) noexcept = default;

void PhotonONNX::load(
    const std::string& prefill_path,
    const std::string& decode_path
)
{
    auto impl =
        std::make_unique<Impl>();

    impl->prefill_options =
        make_session_options();

    impl->decode_options =
        make_session_options();

#ifdef _WIN32

    const std::wstring prefill_w =
        widen_utf8(prefill_path);

    const std::wstring decode_w =
        widen_utf8(decode_path);

    impl->prefill =
        std::make_unique<Ort::Session>(
            Impl::env(),
            prefill_w.c_str(),
            impl->prefill_options
        );

    impl->decode =
        std::make_unique<Ort::Session>(
            Impl::env(),
            decode_w.c_str(),
            impl->decode_options
        );

#else

    impl->prefill =
        std::make_unique<Ort::Session>(
            Impl::env(),
            prefill_path.c_str(),
            impl->prefill_options
        );

    impl->decode =
        std::make_unique<Ort::Session>(
            Impl::env(),
            decode_path.c_str(),
            impl->decode_options
        );

#endif

    if (
        impl->prefill->GetInputCount() != 2 ||
        impl->prefill->GetOutputCount() != 13
    ) {
        throw std::runtime_error(
            "Invalid Photon prefill ONNX graph"
        );
    }

    if (
        impl->decode->GetInputCount() != 14 ||
        impl->decode->GetOutputCount() != 13
    ) {
        throw std::runtime_error(
            "Invalid Photon decode ONNX graph"
        );
    }

    impl->loaded = true;

    impl_ = std::move(impl);
}

std::vector<int32_t> PhotonONNX::generate(
    const std::vector<float>& image_embedding,
    int32_t bos_token_id,
    int32_t eos_token_id,
    uint32_t max_new_tokens
) const
{
    if (
        !impl_ ||
        !impl_->loaded
    ) {
        throw std::runtime_error(
            "Photon ONNX model is not loaded"
        );
    }

    if (
        image_embedding.size() !=
        IMAGE_EMBED_DIM
    ) {
        throw std::invalid_argument(
            "Photon expects a 512-D image embedding"
        );
    }

    if (max_new_tokens == 0)
        return {};

    const Ort::MemoryInfo mem =
        cpu_memory();

    // ========================================================
    // PREFILL
    //
    // [image tokens] + [CLS]
    // ========================================================

    std::array<int64_t, 2> image_shape{
        1,
        IMAGE_EMBED_DIM
    };

    Ort::Value image_value =
        Ort::Value::CreateTensor<float>(
            mem,
            const_cast<float*>(
                image_embedding.data()
            ),
            image_embedding.size(),
            image_shape.data(),
            image_shape.size()
        );

    std::array<int64_t, 2> token_shape{
        1,
        1
    };

    int64_t bos =
        static_cast<int64_t>(
            bos_token_id
        );

    Ort::Value bos_value =
        Ort::Value::CreateTensor<int64_t>(
            mem,
            &bos,
            1,
            token_shape.data(),
            token_shape.size()
        );

    const char* prefill_inputs[] = {
        "image_embedding",
        "input_ids"
    };

    const char* prefill_outputs[] = {
        "logits",
        "present_key_0",
        "present_value_0",
        "present_key_1",
        "present_value_1",
        "present_key_2",
        "present_value_2",
        "present_key_3",
        "present_value_3",
        "present_key_4",
        "present_value_4",
        "present_key_5",
        "present_value_5"
    };

    std::array<Ort::Value, 2> prefill_values;

    prefill_values[0] = std::move(image_value);
    prefill_values[1] = std::move(bos_value);

    auto prefill_outputs_values =
        impl_->prefill->Run(
            Ort::RunOptions{nullptr},
            prefill_inputs,
            prefill_values.data(),
            prefill_values.size(),
            prefill_outputs,
            13
        );
    if (prefill_outputs_values.size() != 13)
        throw std::runtime_error(
            "Invalid Photon prefill outputs"
        );

    std::vector<float> logits =
        read_logits(
            prefill_outputs_values[0]
        );

    // Keep KV cache alive for decode.
    std::vector<Ort::Value> cache;
    cache.reserve(NUM_LAYERS * 2);

    for (std::size_t i = 0; i < NUM_LAYERS * 2; ++i) {
        cache.emplace_back(
            std::move(
                prefill_outputs_values[i + 1]
            )
        );
    }

    // First generated token comes directly
    // from the BOS position in the prefill.
    int32_t next_token =
        argmax(
            logits.data(),
            logits.size()
        );

    std::vector<int32_t> generated;
    generated.reserve(max_new_tokens);

    // Image tokens occupy 0..7.
    // CLS occupies position 8.
    // First generated token is position 9.
    int64_t position =
        static_cast<int64_t>(
            IMAGE_TOKENS + 1
        );

    // ========================================================
    // DECODE
    // ========================================================

    const char* decode_inputs[] = {
        "input_ids",
        "position_ids",
        "past_key_0",
        "past_value_0",
        "past_key_1",
        "past_value_1",
        "past_key_2",
        "past_value_2",
        "past_key_3",
        "past_value_3",
        "past_key_4",
        "past_value_4",
        "past_key_5",
        "past_value_5"
    };

    const char* decode_outputs[] = {
        "logits",
        "present_key_0",
        "present_value_0",
        "present_key_1",
        "present_value_1",
        "present_key_2",
        "present_value_2",
        "present_key_3",
        "present_value_3",
        "present_key_4",
        "present_value_4",
        "present_key_5",
        "present_value_5"
    };

    for (
        uint32_t step = 0;
        step < max_new_tokens;
        ++step
    ) {
        if (next_token == eos_token_id)
            break;

        generated.push_back(
            next_token
        );

        int64_t token =
            static_cast<int64_t>(
                next_token
            );

        std::array<int64_t, 2> one_shape{
            1,
            1
        };

        Ort::Value token_value =
            Ort::Value::CreateTensor<int64_t>(
                mem,
                &token,
                1,
                one_shape.data(),
                one_shape.size()
            );

        Ort::Value position_value =
            Ort::Value::CreateTensor<int64_t>(
                mem,
                &position,
                1,
                one_shape.data(),
                one_shape.size()
            );

        // ORT 1.30 expects a contiguous array of Ort::Value
        // objects, not an array of Ort::Value pointers.

        std::array<Ort::Value, 14> decode_values;

        decode_values[0] =
            std::move(token_value);

        decode_values[1] =
            std::move(position_value);

        // The old cache is no longer needed after this Run(),
        // because the decoder returns a new cache.
        for (
            std::size_t i = 0;
            i < cache.size();
            ++i
        ) {
            decode_values[i + 2] =
                std::move(cache[i]);
        }

        auto outputs =
            impl_->decode->Run(
                Ort::RunOptions{nullptr},
                decode_inputs,
                decode_values.data(),
                decode_values.size(),
                decode_outputs,
                13
            );

        if (outputs.size() != 13)
            throw std::runtime_error(
                "Invalid Photon decode outputs"
            );

        logits =
            read_logits(outputs[0]);

        std::vector<Ort::Value>
            new_cache;

        new_cache.reserve(
            NUM_LAYERS * 2
        );

        for (
            std::size_t i = 0;
            i < NUM_LAYERS * 2;
            ++i
        ) {
            new_cache.emplace_back(
                std::move(
                    outputs[i + 1]
                )
            );
        }

        cache.swap(new_cache);

        next_token =
            argmax(
                logits.data(),
                logits.size()
            );

        ++position;
    }

    return generated;
}

} // namespace photon