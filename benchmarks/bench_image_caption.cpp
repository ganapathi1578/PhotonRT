#include "photon/captioner.h"

#include <chrono>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

double ms(
    Clock::time_point a,
    Clock::time_point b
)
{
    return std::chrono::duration<double, std::milli>(
        b - a
    ).count();
}

void print_stats(
    const char* name,
    const std::vector<double>& values
)
{
    if (values.empty())
        return;

    double sum = 0.0;
    double min_v = values[0];
    double max_v = values[0];

    for (double v : values) {
        sum += v;

        if (v < min_v)
            min_v = v;

        if (v > max_v)
            max_v = v;
    }

    const double avg =
        sum / static_cast<double>(values.size());

    std::cout
        << std::left
        << std::setw(24)
        << name
        << " avg="
        << std::fixed
        << std::setprecision(3)
        << avg
        << " ms  min="
        << min_v
        << " ms  max="
        << max_v
        << " ms\n";
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 6 || argc > 7) {

        std::cerr
            << "Usage:\n"
            << "  " << argv[0]
            << " mobileclip.onnx"
            << " photon_prefill.onnx"
            << " photon.onnx"
            << " photon.tokenizer"
            << " image.jpg"
            << " [iterations]\n";

        return 1;
    }

    const std::string mobileclip_model = argv[1];
    const std::string photon_prefill_model = argv[2];
    const std::string photon_decode_model = argv[3];
    const std::string tokenizer_model = argv[4];
    const std::string image_path = argv[5];

    int iterations = 50;

    if (argc == 7) {
        try {
            iterations = std::stoi(argv[6]);
        }
        catch (...) {
            std::cerr
                << "Invalid iterations\n";
            return 1;
        }
    }

    if (iterations <= 0) {
        std::cerr
            << "Iterations must be > 0\n";
        return 1;
    }

    try {

        std::cout
            << "\n"
            << "============================================================\n"
            << "PhotonRT C++ end-to-end benchmark\n"
            << "============================================================\n";

        std::cout
            << "MobileCLIP : "
            << mobileclip_model
            << "\n";

        std::cout
            << "Prefill    : "
            << photon_prefill_model
            << "\n";

        std::cout
            << "Decode     : "
            << photon_decode_model
            << "\n";

        std::cout
            << "Tokenizer  : "
            << tokenizer_model
            << "\n";

        std::cout
            << "Image      : "
            << image_path
            << "\n";

        std::cout
            << "Iterations : "
            << iterations
            << "\n";

        // ----------------------------------------------------
        // Load the entire production pipeline ONCE.
        // ----------------------------------------------------

        std::cout
            << "\nLoading models...\n";

        photon::Captioner captioner(
            mobileclip_model,
            photon_prefill_model,
            photon_decode_model,
            tokenizer_model
        );

        std::cout
            << "Models loaded.\n";

        // ----------------------------------------------------
        // Warmup
        // ----------------------------------------------------

        constexpr int warmup = 5;

        std::cout
            << "\nWarmup: "
            << warmup
            << " runs\n";

        for (int i = 0; i < warmup; ++i) {

            const std::string caption =
                captioner.caption(
                    image_path
                );

            (void)caption;
        }

        // ----------------------------------------------------
        // Benchmark
        // ----------------------------------------------------

        std::vector<double> total_times;
        total_times.reserve(iterations);

        std::string last_caption;

        std::cout
            << "\nBenchmarking...\n";

        for (int i = 0; i < iterations; ++i) {

            const auto start =
                Clock::now();

            last_caption =
                captioner.caption(
                    image_path
                );

            const auto end =
                Clock::now();

            total_times.push_back(
                ms(start, end)
            );

            if (
                (i + 1) % 10 == 0 ||
                i + 1 == iterations
            ) {
                std::cout
                    << "  "
                    << (i + 1)
                    << "/"
                    << iterations
                    << "\n";
            }
        }

        // ----------------------------------------------------
        // Results
        // ----------------------------------------------------

        std::cout
            << "\n"
            << "============================================================\n"
            << "RESULTS\n"
            << "============================================================\n\n";

        print_stats(
            "End-to-end",
            total_times
        );

        double sum = 0.0;

        for (double v : total_times)
            sum += v;

        const double avg =
            sum /
            static_cast<double>(
                total_times.size()
            );

        if (avg > 0.0) {

            std::cout
                << "\n"
                << "Throughput : "
                << std::fixed
                << std::setprecision(3)
                << (1000.0 / avg)
                << " images/sec\n";
        }

        std::cout
            << "\nCaption:\n"
            << last_caption
            << "\n";

        std::cout
            << "\n"
            << "============================================================\n"
            << "DONE\n"
            << "============================================================\n";

        return 0;

    }
    catch (const std::exception& e) {

        std::cerr
            << "\nPhotonRT benchmark error: "
            << e.what()
            << "\n";

        return 1;
    }
}