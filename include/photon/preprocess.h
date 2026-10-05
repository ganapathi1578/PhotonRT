#pragma once

#include "photon/image.h"

#include <cstdint>
#include <vector>

namespace photon {

// Single image, CHW float32 tensor.
//
// Layout:
//   [channel][height][width]
//
// Values are in [0, 1], matching torchvision ToTensor()
// for uint8 RGB images.
struct ImageTensor {
    uint32_t channels = 3;
    uint32_t height = 256;
    uint32_t width = 256;

    std::vector<float> data;
};

// MobileCLIP v1 preprocessing:
//
//   Resize(shorter edge -> size, bilinear)
//   CenterCrop(size)
//   ToTensor()
//
ImageTensor mobileclip_preprocess(
    const RGBImage& image,
    uint32_t size = 256
);

} // namespace photon