#pragma once

#include "photon/image_encoder.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace photon {

struct MobileCLIPConfig {
    uint32_t version = 1;
    uint32_t image_size = 256;
    uint32_t embedding_dim = 512;
    uint32_t stem_dim = 64;
    uint32_t final_dim = 512;
    uint32_t stage0_blocks = 4;
    uint32_t stage1_blocks = 12;
    uint32_t stage2_blocks = 20;
    uint32_t stage3_blocks = 4;
};

class MobileCLIPS1 final : public ImageEncoder {
public:
    MobileCLIPS1();
    ~MobileCLIPS1() override;

    MobileCLIPS1(const MobileCLIPS1&) = delete;
    MobileCLIPS1& operator=(const MobileCLIPS1&) = delete;

    MobileCLIPS1(MobileCLIPS1&&) noexcept;
    MobileCLIPS1& operator=(MobileCLIPS1&&) noexcept;

    bool load(const std::string& path);

    const MobileCLIPConfig& config() const noexcept {
        return cfg_;
    }

    std::size_t thread_count() const noexcept {
        return thread_count_;
    }

    std::vector<float> encode(
        const ImageTensor& image,
        bool normalize = true
    ) const override;

    std::size_t embedding_dim() const noexcept override {
        return cfg_.embedding_dim;
    }

private:
    struct Impl;

    MobileCLIPConfig cfg_;
    std::unique_ptr<Impl> impl_;
    std::size_t thread_count_ = 1;
};

} // namespace photon