#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace photon {

// Interleaved RGB uint8 image loaded from disk.
struct RGBImage {
    uint32_t width = 0;
    uint32_t height = 0;

    // RGBRGBRGB...
    std::vector<uint8_t> pixels;
};

// Load an image from disk and convert it to RGB uint8.
//
// stb_image supports common formats such as JPEG and PNG.
RGBImage load_image(const std::string& path);

} // namespace photon