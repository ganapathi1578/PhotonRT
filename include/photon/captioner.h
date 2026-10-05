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

    std::string caption(
        const std::string& image_path,
        const CaptionOptions& options = {}
    );

private:
    MobileCLIPS1 mobileclip_;
    PhotonONNX photon_;
    Tokenizer tokenizer_;
};

} // namespace photon