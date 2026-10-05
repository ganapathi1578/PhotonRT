#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace photon {

class Tokenizer {
public:
    Tokenizer() = default;
    explicit Tokenizer(const std::string& path) { load(path); }

    void load(const std::string& path);

    // The Photon v0.1 native tokenizer is decode-first because generation starts
    // from BOS and only needs token-id -> text conversion. Encode support can be
    // added without changing the public interface.
    std::string decode(const std::vector<int32_t>& ids,
                       bool skip_special_tokens = true) const;

    uint32_t vocab_size() const noexcept { return static_cast<uint32_t>(pieces_.size()); }
    int32_t bos_token_id() const noexcept { return bos_token_id_; }
    int32_t eos_token_id() const noexcept { return eos_token_id_; }
    int32_t pad_token_id() const noexcept { return pad_token_id_; }
    int32_t unk_token_id() const noexcept { return unk_token_id_; }

private:
    std::vector<std::string> pieces_;
    std::vector<uint8_t> special_;
    int32_t bos_token_id_ = 1;
    int32_t eos_token_id_ = 2;
    int32_t pad_token_id_ = 3;
    int32_t unk_token_id_ = 0;
};

} // namespace photon
