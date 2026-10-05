#include "photon/preprocess.h"

#include <cassert>
#include <cmath>
#include <iostream>

int main() {

    photon::RGBImage image;

    image.width = 2;
    image.height = 2;

    image.pixels = {
        0,   0,   0,
        255, 0,   0,
        0,   255, 0,
        0,   0,   255
    };

    const auto result =
        photon::mobileclip_preprocess(
            image,
            4
        );

    assert(
        result.channels == 3
    );

    assert(
        result.height == 4
    );

    assert(
        result.width == 4
    );

    assert(
        result.data.size() ==
        3ull * 4ull * 4ull
    );

    for (float value : result.data) {

        assert(
            std::isfinite(value)
        );

        assert(
            value >= 0.0f
        );

        assert(
            value <= 1.0f
        );
    }

    std::cout
        << "preprocess test passed\n";

    return 0;
}