#define STB_IMAGE_IMPLEMENTATION
#define STBI_FAILURE_USERMSG

#include "third_party/stb/stb_image.h"
#include "photon/image.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace photon {

RGBImage load_image(const std::string& path) {
    int width = 0;
    int height = 0;
    int source_channels = 0;

    // Force 3-channel RGB output.
    unsigned char* raw = stbi_load(
        path.c_str(),
        &width,
        &height,
        &source_channels,
        3
    );

    if (raw == nullptr) {
        const char* reason = stbi_failure_reason();

        throw std::runtime_error(
            "Failed to load image '" + path + "': " +
            (reason
                ? std::string(reason)
                : std::string("unknown stb_image error"))
        );
    }

    if (width <= 0 || height <= 0) {
        stbi_image_free(raw);

        throw std::runtime_error(
            "Invalid image dimensions: " + path
        );
    }

    const size_t w =
        static_cast<size_t>(width);

    const size_t h =
        static_cast<size_t>(height);

    if (
        h != 0 &&
        w > std::numeric_limits<size_t>::max() / h / 3
    ) {
        stbi_image_free(raw);

        throw std::runtime_error(
            "Image is too large: " + path
        );
    }

    const size_t count = w * h * 3;

    RGBImage image;

    image.width =
        static_cast<uint32_t>(width);

    image.height =
        static_cast<uint32_t>(height);

    image.pixels.assign(
        raw,
        raw + count
    );

    stbi_image_free(raw);

    (void)source_channels;

    return image;
}

} // namespace photon