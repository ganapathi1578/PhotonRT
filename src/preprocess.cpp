#include "photon/preprocess.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace photon {
namespace {

struct AxisTap {
    int index;
    float weight;
};

using AxisTable = std::vector<std::vector<AxisTap>>;

// Build bilinear/antialiased weights for one dimension.
static AxisTable make_axis_table(
    int src_size,
    int dst_size
) {
    if (src_size <= 0 || dst_size <= 0) {
        throw std::invalid_argument(
            "Invalid resize dimensions"
        );
    }

    AxisTable table(
        static_cast<size_t>(dst_size)
    );

    // Source pixels per destination pixel.
    const double scale =
        static_cast<double>(src_size) /
        static_cast<double>(dst_size);

    // Bilinear triangle kernel.
    // For downsampling, widen the kernel.
    const double support =
        std::max(1.0, scale);

    const double kernel_scale =
        std::max(1.0, scale);

    for (int d = 0; d < dst_size; ++d) {

        // Pixel-center mapping.
        const double center =
            (static_cast<double>(d) + 0.5) * scale - 0.5;

        const int left =
            static_cast<int>(
                std::ceil(center - support)
            );

        const int right =
            static_cast<int>(
                std::floor(center + support)
            );

        auto& taps = table[
            static_cast<size_t>(d)
        ];

        double weight_sum = 0.0;

        for (int s = left; s <= right; ++s) {

            const double distance =
                std::abs(
                    center -
                    static_cast<double>(s)
                );

            const double w =
                std::max(
                    0.0,
                    1.0 - distance / kernel_scale
                );

            if (w <= 0.0) {
                continue;
            }

            const int clamped =
                std::clamp(
                    s,
                    0,
                    src_size - 1
                );

            taps.push_back({
                clamped,
                static_cast<float>(w)
            });

            weight_sum += w;
        }

        if (weight_sum <= 0.0) {
            const int nearest =
                std::clamp(
                    static_cast<int>(
                        std::floor(center + 0.5)
                    ),
                    0,
                    src_size - 1
                );

            taps.clear();

            taps.push_back({
                nearest,
                1.0f
            });

        } else {

            for (auto& tap : taps) {
                tap.weight =
                    static_cast<float>(
                        tap.weight / weight_sum
                    );
            }
        }
    }

    return table;
}


static std::vector<uint8_t> resize_rgb(
    const RGBImage& image,
    uint32_t dst_width,
    uint32_t dst_height
) {
    const AxisTable wx = make_axis_table(
        static_cast<int>(image.width),
        static_cast<int>(dst_width)
    );

    const AxisTable wy = make_axis_table(
        static_cast<int>(image.height),
        static_cast<int>(dst_height)
    );

    std::vector<uint8_t> result(
        static_cast<size_t>(dst_width) *
        static_cast<size_t>(dst_height) *
        3
    );

    for (uint32_t y = 0; y < dst_height; ++y) {

        const auto& yt =
            wy[static_cast<size_t>(y)];

        for (uint32_t x = 0; x < dst_width; ++x) {

            const auto& xt =
                wx[static_cast<size_t>(x)];

            for (int c = 0; c < 3; ++c) {

                double value = 0.0;

                for (const auto& yy : yt) {
                    for (const auto& xx : xt) {

                        const size_t idx =
                            (
                                static_cast<size_t>(yy.index) *
                                image.width +
                                static_cast<size_t>(xx.index)
                            ) * 3 +
                            static_cast<size_t>(c);

                        value +=
                            static_cast<double>(
                                image.pixels[idx]
                            ) *
                            static_cast<double>(
                                yy.weight
                            ) *
                            static_cast<double>(
                                xx.weight
                            );
                    }
                }

                value =
                    std::clamp(
                        value,
                        0.0,
                        255.0
                    );

                // Convert back to uint8 before ToTensor.
                const int rounded =
                    static_cast<int>(
                        std::floor(value + 0.5)
                    );

                const size_t dst_idx =
                    (
                        static_cast<size_t>(y) *
                        dst_width +
                        x
                    ) * 3 +
                    static_cast<size_t>(c);

                result[dst_idx] =
                    static_cast<uint8_t>(
                        std::clamp(
                            rounded,
                            0,
                            255
                        )
                    );
            }
        }
    }

    return result;
}

} // namespace


ImageTensor mobileclip_preprocess(
    const RGBImage& image,
    uint32_t size
) {
    if (size == 0) {
        throw std::invalid_argument(
            "Image target size cannot be zero"
        );
    }

    if (image.width == 0 ||
        image.height == 0) {
        throw std::invalid_argument(
            "Image dimensions cannot be zero"
        );
    }

    const size_t expected =
        static_cast<size_t>(image.width) *
        static_cast<size_t>(image.height) *
        3;

    if (image.pixels.size() != expected) {
        throw std::invalid_argument(
            "RGB image buffer has unexpected size"
        );
    }

    // torchvision Resize(size=int):
    // shorter edge becomes `size`;
    // longer edge is floor(size * long / short).
    const uint32_t short_edge =
        std::min(
            image.width,
            image.height
        );

    const uint32_t long_edge =
        std::max(
            image.width,
            image.height
        );

    const uint32_t resized_long =
        std::max(
            size,
            static_cast<uint32_t>(
                (
                    static_cast<uint64_t>(size) *
                    long_edge
                ) / short_edge
            )
        );

    const bool width_is_long =
        image.width >= image.height;

    const uint32_t resized_width =
        width_is_long
            ? resized_long
            : size;

    const uint32_t resized_height =
        width_is_long
            ? size
            : resized_long;

    const std::vector<uint8_t> resized =
        resize_rgb(
            image,
            resized_width,
            resized_height
        );

    // CenterCrop(size)
    const uint32_t left =
        (resized_width - size) / 2;

    const uint32_t top =
        (resized_height - size) / 2;

    ImageTensor output;

    output.channels = 3;
    output.height = size;
    output.width = size;

    output.data.resize(
        static_cast<size_t>(3) *
        size *
        size
    );

    // ToTensor(): HWC uint8 → CHW float / 255.
    for (uint32_t y = 0; y < size; ++y) {
        for (uint32_t x = 0; x < size; ++x) {

            const size_t src_idx =
                (
                    static_cast<size_t>(y + top) *
                    resized_width +
                    (x + left)
                ) * 3;

            for (uint32_t c = 0; c < 3; ++c) {

                const size_t dst_idx =
                    (
                        static_cast<size_t>(c) *
                        size +
                        y
                    ) * size +
                    x;

                output.data[dst_idx] =
                    static_cast<float>(
                        resized[src_idx + c]
                    ) / 255.0f;
            }
        }
    }

    return output;
}

} // namespace photon