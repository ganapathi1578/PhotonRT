#include "photon/captioner.h"

#include <cstdlib>
#include <iostream>
#include <string>

int main(int argc, char** argv)
{
    if (argc < 6 || argc > 7) {
        std::cerr
            << "Usage:\n"
            << "  " << argv[0]
            << " mobileclip.onnx"
            << " photon_prefill.onnx"
            << " photon.onnx"
            << " photon.tokenizer"
            << " image.jpg"
            << " [max_tokens]\n";

        return 1;
    }

    try {
        photon::Captioner captioner(
            argv[1],
            argv[2],
            argv[3],
            argv[4]
        );

        photon::CaptionOptions options;

        if (argc == 7) {
            options.max_new_tokens =
                static_cast<uint32_t>(
                    std::stoul(argv[6])
                );
        }

        const std::string result =
            captioner.caption(
                argv[5],
                options
            );

        std::cout
            << result
            << '\n';

        return 0;

    } catch (const std::exception& e) {

        std::cerr
            << "image-caption: "
            << e.what()
            << '\n';

        return 1;
    }
}