#pragma once

#include "photon/preprocess.h"

#include <cstddef>
#include <vector>

namespace photon {

class ImageEncoder {
public:
    virtual ~ImageEncoder() = default;

    virtual std::vector<float> encode(
        const ImageTensor& image,
        bool normalize = true
    ) const = 0;

    virtual std::size_t embedding_dim() const noexcept = 0;
};

} // namespace photon