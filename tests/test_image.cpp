#include "photon/image.h"

#include <cassert>
#include <iostream>

int main() {
    photon::RGBImage image;

    image.width = 2;
    image.height = 1;

    image.pixels = {
        255, 0, 0,
        0, 255, 0
    };

    assert(image.width == 2);
    assert(image.height == 1);

    assert(image.pixels.size() == 6);

    assert(image.pixels[0] == 255);
    assert(image.pixels[1] == 0);
    assert(image.pixels[2] == 0);

    std::cout
        << "image API test passed\n";

    return 0;
}