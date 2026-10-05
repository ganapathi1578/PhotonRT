#include "photon/mobileclip.h"
#include <onnxruntime_cxx_api.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef _WIN32
#include <windows.h>
#endif
namespace photon {
namespace {
size_t configured_threads() {
    const char* raw = std::getenv("PHOTON_THREADS");
    if (!raw || !*raw) return 1;
    char* end = nullptr;
    const unsigned long n = std::strtoul(raw, &end, 10);
    if (end == raw || *end != '\0' || n == 0) return 1;
    return static_cast<size_t>(std::min<unsigned long>(n, 256UL));
}
#ifdef _WIN32
std::wstring widen_utf8(const std::string& s) {
    if (s.empty()) return {};
    const int n = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s.data(), (int)s.size(), nullptr, 0);
    if (n <= 0) throw std::runtime_error("Failed to convert ONNX model path to UTF-16");
    std::wstring out((size_t)n, L'\0');
    if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s.data(), (int)s.size(), out.data(), n) != n)
        throw std::runtime_error("Failed to convert ONNX model path to UTF-16");
    return out;
}
#endif
std::string name_string(const Ort::AllocatedStringPtr& p) {
    if (!p || !p.get()) throw std::runtime_error("ONNX Runtime returned an empty tensor name");
    return std::string(p.get());
}
void check_shape(const std::vector<int64_t>& got, const std::vector<int64_t>& want, const char* what) {
    if (got.size() != want.size()) throw std::runtime_error(std::string("Unexpected ") + what + " rank");
    for (size_t i = 0; i < want.size(); ++i)
        if (got[i] >= 0 && got[i] != want[i]) throw std::runtime_error(std::string("Unexpected ") + what + " shape");
}
}
struct MobileCLIPS1::Impl {
    static Ort::Env& env() { static Ort::Env e(ORT_LOGGING_LEVEL_WARNING, "PhotonRT-MobileCLIP"); return e; }
    Ort::SessionOptions options;
    std::unique_ptr<Ort::Session> session;
    std::string input_name, output_name;
};
MobileCLIPS1::MobileCLIPS1() = default;
MobileCLIPS1::~MobileCLIPS1() = default;
MobileCLIPS1::MobileCLIPS1(MobileCLIPS1&&) noexcept = default;
MobileCLIPS1& MobileCLIPS1::operator=(MobileCLIPS1&&) noexcept = default;

bool MobileCLIPS1::load(const std::string& path) {
    auto impl = std::make_unique<Impl>();
    thread_count_ = configured_threads();
    impl->options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    impl->options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    impl->options.SetIntraOpNumThreads((int)thread_count_);
    impl->options.SetInterOpNumThreads(1);
    impl->options.EnableCpuMemArena();
    impl->options.EnableMemPattern();
#ifdef _WIN32
    const auto wide = widen_utf8(path);
    impl->session = std::make_unique<Ort::Session>(Impl::env(), wide.c_str(), impl->options);
#else
    impl->session = std::make_unique<Ort::Session>(Impl::env(), path.c_str(), impl->options);
#endif
    if (impl->session->GetInputCount() != 1 || impl->session->GetOutputCount() != 1)
        throw std::runtime_error("MobileCLIP ONNX model must have exactly one input and one output");
    Ort::AllocatorWithDefaultOptions allocator;
    impl->input_name = name_string(impl->session->GetInputNameAllocated(0, allocator));
    impl->output_name = name_string(impl->session->GetOutputNameAllocated(0, allocator));
    const auto in_info = impl->session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
    const auto out_info = impl->session->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo();
    if (in_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
        throw std::runtime_error("MobileCLIP input must be float32");
    if (out_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
        throw std::runtime_error("MobileCLIP output must be float32");
    check_shape(in_info.GetShape(), {1,3,256,256}, "input");
    check_shape(out_info.GetShape(), {1,512}, "output");
    cfg_ = MobileCLIPConfig{};
    impl_ = std::move(impl);
    return true;
}
std::vector<float> MobileCLIPS1::encode(const ImageTensor& image, bool normalize) const {
    if (!impl_ || !impl_->session) throw std::runtime_error("MobileCLIP-S1 ONNX model is not loaded");
    if (image.channels != 3 || image.height != 256 || image.width != 256 || image.data.size() != 3ull*256ull*256ull)
        throw std::invalid_argument("MobileCLIP-S1 expects a 3x256x256 float32 tensor");
    const std::array<int64_t,4> shape{1,3,256,256};
    Ort::MemoryInfo mem("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
    auto in = Ort::Value::CreateTensor<float>(mem, const_cast<float*>(image.data.data()), image.data.size(), shape.data(), shape.size());
    const char* ins[] = {impl_->input_name.c_str()};
    const char* outs[] = {impl_->output_name.c_str()};
    auto result = impl_->session->Run(Ort::RunOptions{nullptr}, ins, &in, 1, outs, 1);
    if (result.size() != 1 || !result[0].IsTensor()) throw std::runtime_error("Invalid MobileCLIP ONNX output");
    const auto out_info = result[0].GetTensorTypeAndShapeInfo();
    const auto shape_out = out_info.GetShape();
    size_t count = 1;
    for (int64_t d : shape_out) { if (d < 0) throw std::runtime_error("Dynamic output unsupported"); count *= (size_t)d; }
    if (count != 512) throw std::runtime_error("MobileCLIP output is not 512-D");
    const float* p = result[0].GetTensorData<float>();
    std::vector<float> v(p, p + 512);
    if (normalize) {
        double ss = 0.0; for (float x : v) ss += (double)x * x;
        const float n = (float)std::sqrt(ss);
        if (n > 1e-12f) { const float inv = 1.0f / n; for (float& x : v) x *= inv; }
    }
    return v;
}
} // namespace photon
