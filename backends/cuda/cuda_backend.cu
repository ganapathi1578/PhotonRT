#include "backends/cuda/cuda_backend.h"

#include <cuda_runtime.h>

namespace photon {

bool cuda_backend_available() noexcept {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

const char* cuda_backend_name() noexcept { return "PhotonRT CUDA scaffold"; }

} // namespace photon
