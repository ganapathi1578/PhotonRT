#pragma once

#include "photon/tensor.h"

#include <cstddef>
#include <vector>

namespace photon {

class CPUBackend {
public:
    explicit CPUBackend(int num_threads = 0);

    int num_threads() const noexcept { return num_threads_; }
    void set_num_threads(int n);

    Tensor matmul(const Tensor& a, const Tensor& b) const;
    Tensor matvec(const Tensor& w, const Tensor& x) const;

private:
    int num_threads_;
};

} // namespace photon
