#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "photon/captioner.h"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace py = pybind11;

namespace {

py::array get_frame_array(py::handle obj)
{
    py::array array = py::array::ensure(obj);

    if (!array) {
        throw std::invalid_argument(
            "Frame must be convertible to a NumPy array"
        );
    }

    return array;
}

void validate_frame(const py::buffer_info& info)
{
    if (info.ndim != 3) {
        throw std::invalid_argument(
            "Frame must have shape (height, width, 3)"
        );
    }

    if (info.shape[0] <= 0 ||
        info.shape[1] <= 0 ||
        info.shape[2] != 3) {
        throw std::invalid_argument(
            "Frame must have shape (height, width, 3)"
        );
    }

    if (info.itemsize != 1) {
        throw std::invalid_argument(
            "Frame must use uint8 data"
        );
    }
}

} // namespace

PYBIND11_MODULE(_photonrt, m)
{
    m.doc() =
        "PhotonRT - native C++ image and camera captioning runtime";

    // ------------------------------------------------------------
    // CaptionOptions
    // ------------------------------------------------------------

    py::class_<photon::CaptionOptions>(
        m,
        "CaptionOptions"
    )
        .def(py::init<>())

        .def_readwrite(
            "max_new_tokens",
            &photon::CaptionOptions::max_new_tokens
        );

    // ------------------------------------------------------------
    // Captioner
    // ------------------------------------------------------------

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

        // --------------------------------------------------------
        // Image-path captioning
        // --------------------------------------------------------

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

        // --------------------------------------------------------
        // RGB frame captioning
        // --------------------------------------------------------

        .def(
            "caption_rgb",

            [](photon::Captioner& self,
               py::handle frame,
               const photon::CaptionOptions& options)
            {
                py::array array =
                    get_frame_array(frame);

                py::buffer_info info =
                    array.request();

                validate_frame(info);

                if (info.strides[2] != 1) {
                    throw std::invalid_argument(
                        "Frame must be contiguous in memory"
                    );
                }

                const auto* data =
                    static_cast<const uint8_t*>(info.ptr);

                const uint32_t width =
                    static_cast<uint32_t>(info.shape[1]);

                const uint32_t height =
                    static_cast<uint32_t>(info.shape[0]);

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

        // --------------------------------------------------------
        // BGR frame captioning
        // --------------------------------------------------------

        .def(
            "caption_bgr",

            [](photon::Captioner& self,
               py::handle frame,
               const photon::CaptionOptions& options)
            {
                py::array array =
                    get_frame_array(frame);

                py::buffer_info info =
                    array.request();

                validate_frame(info);

                if (info.strides[2] != 1) {
                    throw std::invalid_argument(
                        "Frame must be contiguous in memory"
                    );
                }

                const auto* data =
                    static_cast<const uint8_t*>(info.ptr);

                const uint32_t width =
                    static_cast<uint32_t>(info.shape[1]);

                const uint32_t height =
                    static_cast<uint32_t>(info.shape[0]);

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

    m.attr("__version__") = "0.2.1";
}