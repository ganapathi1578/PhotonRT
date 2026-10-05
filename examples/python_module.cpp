#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "photon/captioner.h"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace py = pybind11;

namespace {

using UInt8Array =
    py::array_t<
        uint8_t,
        py::array::c_style
    >;

void validate_frame(
    const UInt8Array& frame,
    const char* name
)
{
    if (frame.ndim() != 3) {
        throw std::runtime_error(
            std::string(name) +
            " must have shape (height, width, 3)"
        );
    }

    const py::ssize_t height = frame.shape(0);
    const py::ssize_t width  = frame.shape(1);
    const py::ssize_t channels = frame.shape(2);

    if (height <= 0 || width <= 0) {
        throw std::runtime_error(
            std::string(name) +
            " has invalid dimensions"
        );
    }

    if (channels != 3) {
        throw std::runtime_error(
            std::string(name) +
            " must have 3 channels"
        );
    }

    if (
        static_cast<uint64_t>(width) >
            std::numeric_limits<uint32_t>::max() ||
        static_cast<uint64_t>(height) >
            std::numeric_limits<uint32_t>::max()
    ) {
        throw std::runtime_error(
            std::string(name) +
            " dimensions exceed uint32_t range"
        );
    }
}

} // namespace

PYBIND11_MODULE(_photonrt, m)
{
    m.doc() =
        "PhotonRT - native C++ image captioning runtime";

    py::class_<photon::CaptionOptions>(
        m,
        "CaptionOptions"
    )
        .def(
            py::init<>()
        )

        .def_readwrite(
            "max_new_tokens",
            &photon::CaptionOptions::max_new_tokens
        );

    py::class_<photon::Captioner>(
        m,
        "Captioner"
    )

        .def(
            py::init<
                const std::string&,
                const std::string&,
                const std::string&,
                const std::string&
            >(),

            py::arg("mobileclip_model"),
            py::arg("photon_prefill_model"),
            py::arg("photon_decode_model"),
            py::arg("tokenizer")
        )

        .def(
            "caption",
            [](photon::Captioner& self,
               const std::string& image_path,
               const photon::CaptionOptions& options)
            {
                py::gil_scoped_release release;

                return self.caption(
                    image_path,
                    options
                );
            },

            py::arg("image"),
            py::arg("options") =
                photon::CaptionOptions{}
        )

        .def(
            "caption_rgb",
            [](photon::Captioner& self,
               UInt8Array frame,
               const photon::CaptionOptions& options)
            {
                validate_frame(frame, "RGB frame");

                const uint32_t height =
                    static_cast<uint32_t>(frame.shape(0));

                const uint32_t width =
                    static_cast<uint32_t>(frame.shape(1));

                const uint8_t* data =
                    frame.data();

                py::gil_scoped_release release;

                return self.caption_rgb(
                    data,
                    width,
                    height,
                    options
                );
            },

            py::arg("frame"),
            py::arg("options") =
                photon::CaptionOptions{}
        )

        .def(
            "caption_bgr",
            [](photon::Captioner& self,
               UInt8Array frame,
               const photon::CaptionOptions& options)
            {
                validate_frame(frame, "BGR frame");

                const uint32_t height =
                    static_cast<uint32_t>(frame.shape(0));

                const uint32_t width =
                    static_cast<uint32_t>(frame.shape(1));

                const uint8_t* data =
                    frame.data();

                py::gil_scoped_release release;

                return self.caption_bgr(
                    data,
                    width,
                    height,
                    options
                );
            },

            py::arg("frame"),
            py::arg("options") =
                photon::CaptionOptions{}
        );

    m.attr("__version__") = "0.2.0";
}