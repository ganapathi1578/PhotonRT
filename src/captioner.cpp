#include "photon/captioner.h"

#include "photon/image.h"
#include "photon/preprocess.h"

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
            "Failed to load MobileCLIP: " +
            mobileclip_model
        );
    }

    photon_.load(
        photon_prefill_model,
        photon_decode_model
    );

    tokenizer_.load(
        tokenizer_model
    );
}

std::string Captioner::caption(
    const std::string& image_path,
    const CaptionOptions& options
)
{
    // ========================================================
    // IMAGE LOAD
    // ========================================================

    const RGBImage image =
        load_image(image_path);

    // ========================================================
    // NATIVE PREPROCESS
    // ========================================================

    const ImageTensor tensor =
        mobileclip_preprocess(
            image,
            mobileclip_.config().image_size
        );

    // ========================================================
    // MOBILECLIP ONNX
    // ========================================================

    const std::vector<float> embedding =
        mobileclip_.encode(
            tensor,
            true
        );

    // ========================================================
    // PHOTON ONNX
    // ========================================================

    const std::vector<int32_t> ids =
        photon_.generate(
            embedding,
            tokenizer_.bos_token_id(),
            tokenizer_.eos_token_id(),
            options.max_new_tokens
        );

    // ========================================================
    // TOKENIZER
    // ========================================================

    return tokenizer_.decode(
        ids,
        true
    );
}

} // namespace photon