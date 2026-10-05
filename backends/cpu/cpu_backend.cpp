#include "backends/cpu/cpu_backend.h"

#include <algorithm>
#include <stdexcept>
#include <thread>

namespace photon {

CPUBackend::CPUBackend(int num_threads) {
    if (num_threads <= 0) {
        num_threads = static_cast<int>(std::thread::hardware_concurrency());
        if (num_threads <= 0) num_threads = 1;
    }
    num_threads_ = num_threads;
}

void CPUBackend::set_num_threads(int n) {
    if (n <= 0) throw std::invalid_argument("CPU thread count must be positive");
    num_threads_ = n;
}

Tensor CPUBackend::matmul(const Tensor& a, const Tensor& b) const {
    if (a.ndim() != 2 || b.ndim() != 2 || a.shape()[1] != b.shape()[0])
        throw std::invalid_argument("CPUBackend::matmul shape mismatch");
    const size_t M = a.shape()[0], K = a.shape()[1], N = b.shape()[1];
    Tensor out({M, N}, 0.0f);

    const size_t workers = std::min<size_t>(static_cast<size_t>(num_threads_), M);
    auto job = [&](size_t begin, size_t end) {
        for (size_t i = begin; i < end; ++i) {
            for (size_t k = 0; k < K; ++k) {
                const float av = a[i * K + k];
                const float* bp = b.data() + k * N;
                float* op = out.data() + i * N;
                for (size_t j = 0; j < N; ++j) op[j] += av * bp[j];
            }
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(workers);
    for (size_t t = 0; t < workers; ++t) {
        const size_t begin = M * t / workers;
        const size_t end = M * (t + 1) / workers;
        threads.emplace_back(job, begin, end);
    }
    for (auto& th : threads) th.join();
    return out;
}

Tensor CPUBackend::matvec(const Tensor& w, const Tensor& x) const {
    if (w.ndim() != 2 || x.ndim() != 1 || w.shape()[1] != x.shape()[0])
        throw std::invalid_argument("CPUBackend::matvec shape mismatch");
    const size_t O = w.shape()[0], I = w.shape()[1];
    Tensor out({O}, 0.0f);
    const size_t workers = std::min<size_t>(static_cast<size_t>(num_threads_), O);
    auto job = [&](size_t begin, size_t end) {
        for (size_t o = begin; o < end; ++o) {
            float s = 0.0f;
            const float* wp = w.data() + o * I;
            for (size_t i = 0; i < I; ++i) s += wp[i] * x[i];
            out[o] = s;
        }
    };
    std::vector<std::thread> threads;
    threads.reserve(workers);
    for (size_t t = 0; t < workers; ++t) {
        const size_t begin = O * t / workers;
        const size_t end = O * (t + 1) / workers;
        threads.emplace_back(job, begin, end);
    }
    for (auto& th : threads) th.join();
    return out;
}

} // namespace photon
