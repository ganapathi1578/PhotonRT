// #include "photon/captioner.h"

// #include "photon/image.h"
// #include "photon/preprocess.h"

// #include <stdexcept>
// #include <vector>

// namespace photon {

// Captioner::Captioner(
//     const std::string& mobileclip_model,
//     const std::string& photon_prefill_model,
//     const std::string& photon_decode_model,
//     const std::string& tokenizer_model
// )
// {
//     if (!mobileclip_.load(mobileclip_model)) {
//         throw std::runtime_error(
//             "Failed to load MobileCLIP: " +
//             mobileclip_model
//         );
//     }

//     photon_.load(
//         photon_prefill_model,
//         photon_decode_model
//     );

//     tokenizer_.load(
//         tokenizer_model
//     );
// }

// std::string Captioner::caption(
//     const std::string& image_path,
//     const CaptionOptions& options
// )
// {
//     // ========================================================
//     // IMAGE LOAD
//     // ========================================================

//     const RGBImage image =
//         load_image(image_path);

//     // ========================================================
//     // NATIVE PREPROCESS
//     // ========================================================

//     const ImageTensor tensor =
//         mobileclip_preprocess(
//             image,
//             mobileclip_.config().image_size
//         );

//     // ========================================================
//     // MOBILECLIP ONNX
//     // ========================================================

//     const std::vector<float> embedding =
//         mobileclip_.encode(
//             tensor,
//             true
//         );

//     // ========================================================
//     // PHOTON ONNX
//     // ========================================================

//     const std::vector<int32_t> ids =
//         photon_.generate(
//             embedding,
//             tokenizer_.bos_token_id(),
//             tokenizer_.eos_token_id(),
//             options.max_new_tokens
//         );

//     // ========================================================
//     // TOKENIZER
//     // ========================================================

//     return tokenizer_.decode(
//         ids,
//         true
//     );
// }

// } // namespace photon

#include "photon/captioner.h"

#include "photon/image.h"
#include "photon/preprocess.h"

#include <cstring>
#include <stdexcept>
#include <vector>

namespace photon {

Captioner::Captioner(
    const std::string& mobileclip_model,
    const std::string& photon_prefill_model,
    const std::string& photon_decode_model,
    const std::string& tokenizer_model
)
{
    if (!mobileclip_.load(mobileclip_model)) {
        throw std::runtime_error(
            "Failed to load MobileCLIP: " + mobileclip_model
        );
    }

    photon_.load(
        photon_prefill_model,
        photon_decode_model
    );

    tokenizer_.load(tokenizer_model);
}

std::string Captioner::caption(
    const std::string& image_path,
    const CaptionOptions& options
)
{
    const RGBImage image = load_image(image_path);

    const ImageTensor tensor = mobileclip_preprocess(
        image,
        mobileclip_.config().image_size
    );

    const std::vector<float> embedding = mobileclip_.encode(tensor, true);

    const std::vector<int32_t> ids = photon_.generate(
        embedding,
        tokenizer_.bos_token_id(),
        tokenizer_.eos_token_id(),
        options.max_new_tokens
    );

    return tokenizer_.decode(ids, true);
}

std::string Captioner::caption_rgb(
    const uint8_t* data,
    uint32_t width,
    uint32_t height,
    const CaptionOptions& options
)
{
    return caption_rgb_image(data, width, height, false, options);
}

std::string Captioner::caption_bgr(
    const uint8_t* data,
    uint32_t width,
    uint32_t height,
    const CaptionOptions& options
)
{
    return caption_rgb_image(data, width, height, true, options);
}

std::string Captioner::caption_rgb_image(
    const uint8_t* data,
    uint32_t width,
    uint32_t height,
    bool bgr,
    const CaptionOptions& options
)
{
    if (data == nullptr) {
        throw std::invalid_argument("Camera frame data is null");
    }

    if (width == 0 || height == 0) {
        throw std::invalid_argument(
            "Camera frame dimensions must be greater than zero"
        );
    }

    const size_t pixel_count =
        static_cast<size_t>(width) * static_cast<size_t>(height);
    const size_t byte_count = pixel_count * 3u;

    // Reuse the existing PhotonRT RGBImage + native preprocessing path.
    // This is one copy from the camera/Python buffer into RGBImage.
    RGBImage image;
    image.width = width;
    image.height = height;
    image.pixels.resize(byte_count);

    if (!bgr) {
        std::memcpy(image.pixels.data(), data, byte_count);
    } else {
        // OpenCV normally returns BGR. Convert while copying so Python
        // does not need an additional cvtColor() allocation/copy.
        for (size_t i = 0; i < byte_count; i += 3u) {
            image.pixels[i + 0] = data[i + 2];
            image.pixels[i + 1] = data[i + 1];
            image.pixels[i + 2] = data[i + 0];
        }
    }

    const ImageTensor tensor = mobileclip_preprocess(
        image,
        mobileclip_.config().image_size
    );

    const std::vector<float> embedding = mobileclip_.encode(tensor, true);

    const std::vector<int32_t> ids = photon_.generate(
        embedding,
        tokenizer_.bos_token_id(),
        tokenizer_.eos_token_id(),
        options.max_new_tokens
    );

    return tokenizer_.decode(ids, true);
}

} // namespace photon
