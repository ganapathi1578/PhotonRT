// #pragma once

// #include "photon/mobileclip.h"
// #include "photon/photon_onnx.h"
// #include "photon/tokenizer.h"

// #include <cstdint>
// #include <string>

// namespace photon {

// struct CaptionOptions {
//     uint32_t max_new_tokens = 32;
// };

// class Captioner {
// public:
//     Captioner(
//         const std::string& mobileclip_model,
//         const std::string& photon_prefill_model,
//         const std::string& photon_decode_model,
//         const std::string& tokenizer_model
//     );

//     std::string caption(
//         const std::string& image_path,
//         const CaptionOptions& options = {}
//     );

// private:
//     MobileCLIPS1 mobileclip_;
//     PhotonONNX photon_;
//     Tokenizer tokenizer_;
// };

// } // namespace photon


#pragma once

#include "photon/mobileclip.h"
#include "photon/photon_onnx.h"
#include "photon/tokenizer.h"

#include <cstdint>
#include <string>

namespace photon {

struct CaptionOptions {
    uint32_t max_new_tokens = 32;
};

class Captioner {
public:
    Captioner(
        const std::string& mobileclip_model,
        const std::string& photon_prefill_model,
        const std::string& photon_decode_model,
        const std::string& tokenizer_model
    );

    // Existing file-path API.
    std::string caption(
        const std::string& image_path,
        const CaptionOptions& options = {}
    );

    // Raw HWC uint8 frame APIs.
    // RGB: R,G,B,R,G,B,...
    // BGR: B,G,R,B,G,R,... (OpenCV camera frames).
    std::string caption_rgb(
        const uint8_t* data,
        uint32_t width,
        uint32_t height,
        const CaptionOptions& options = {}
    );

    std::string caption_bgr(
        const uint8_t* data,
        uint32_t width,
        uint32_t height,
        const CaptionOptions& options = {}
    );

private:
    std::string caption_rgb_image(
        const uint8_t* data,
        uint32_t width,
        uint32_t height,
        bool bgr,
        const CaptionOptions& options
    );

    MobileCLIPS1 mobileclip_;
    PhotonONNX photon_;
    Tokenizer tokenizer_;
};

} // namespace photon
