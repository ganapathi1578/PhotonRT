#pragma once

namespace photon {

bool cuda_backend_available() noexcept;
const char* cuda_backend_name() noexcept;

} // namespace photon
