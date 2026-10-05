#include "photon/tokenizer.h"

#include <cstdint>
#include <fstream>
#include <stdexcept>

namespace {

uint32_t read_u32(std::istream& in) {
    uint32_t v = 0;
    in.read(reinterpret_cast<char*>(&v), sizeof(v));
    if (!in) throw std::runtime_error("Failed to read tokenizer file");
    return v;
}

int32_t read_i32(std::istream& in) {
    int32_t v = 0;
    in.read(reinterpret_cast<char*>(&v), sizeof(v));
    if (!in) throw std::runtime_error("Failed to read tokenizer file");
    return v;
}

} // namespace

namespace photon {

void Tokenizer::load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open tokenizer: " + path);

    char magic[8]{};
    in.read(magic, 8);
    if (std::string(magic, 8) != "PHOTOKN1") throw std::runtime_error("Invalid Photon tokenizer magic");

    const uint32_t version = read_u32(in);
    if (version != 1) throw std::runtime_error("Unsupported Photon tokenizer version");

    const uint32_t vocab = read_u32(in);
    bos_token_id_ = read_i32(in);
    eos_token_id_ = read_i32(in);
    pad_token_id_ = read_i32(in);
    unk_token_id_ = read_i32(in);

    pieces_.clear();
    special_.clear();
    pieces_.reserve(vocab);
    special_.reserve(vocab);

    for (uint32_t id = 0; id < vocab; ++id) {
        const uint32_t len = read_u32(in);
        uint8_t special = 0;
        in.read(reinterpret_cast<char*>(&special), sizeof(special));
        if (!in) throw std::runtime_error("Failed to read tokenizer token metadata");
        std::string piece(len, '\0');
        in.read(piece.data(), static_cast<std::streamsize>(len));
        if (!in) throw std::runtime_error("Failed to read tokenizer token text");
        pieces_.push_back(std::move(piece));
        special_.push_back(special);
    }
}

std::string Tokenizer::decode(const std::vector<int32_t>& ids, bool skip_special_tokens) const {
    std::string out;
    for (int32_t id : ids) {
        if (id < 0 || static_cast<size_t>(id) >= pieces_.size()) continue;
        if (skip_special_tokens && special_[static_cast<size_t>(id)]) continue;
        out += pieces_[static_cast<size_t>(id)];
    }
    return out;
}

} // namespace photon
