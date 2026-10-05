#include "photon/tensor.h"

#include <cmath>
#include <iostream>

int main() {
    using namespace photon;
    Tensor a = Tensor::from_vector({2, 2}, {1, 2, 3, 4});
    Tensor b = Tensor::from_vector({2, 2}, {5, 6, 7, 8});
    Tensor c = matmul(a, b);
    if (std::fabs(c[0] - 19.0f) > 1e-5f || std::fabs(c[3] - 50.0f) > 1e-5f) return 1;

    Tensor x = Tensor::from_vector({4}, {1, 2, 3, 4});
    Tensor w = Tensor::from_vector({4}, {1, 1, 1, 1});
    Tensor n = rms_norm(x, w, 1e-6f);
    float ss = 0;
    for (size_t i = 0; i < 4; ++i) ss += n[i] * n[i];
    if (std::fabs(ss - 4.0f) > 1e-4f) return 1;
    std::cout << "test_tensor: OK\n";
    return 0;
}
