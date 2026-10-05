#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace photon {

class PhotonONNX {
public:
    PhotonONNX();
    ~PhotonONNX();

    PhotonONNX(const PhotonONNX&) = delete;
    PhotonONNX& operator=(const PhotonONNX&) = delete;

    PhotonONNX(PhotonONNX&&) noexcept;
    PhotonONNX& operator=(PhotonONNX&&) noexcept;

    void load(
        const std::string& prefill_path,
        const std::string& decode_path
    );

    std::vector<int32_t> generate(
        const std::vector<float>& image_embedding,
        int32_t bos_token_id,
        int32_t eos_token_id,
        uint32_t max_new_tokens = 32
    ) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace photon