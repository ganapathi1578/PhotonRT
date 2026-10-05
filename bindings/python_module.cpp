#include <pybind11/pybind11.h>

#include "photon/captioner.h"

#include <string>

namespace py = pybind11;

PYBIND11_MODULE(_photonrt, m)
{
    m.doc() =
        "PhotonRT - native C++ image captioning runtime";

    // --------------------------------------------------------
    // CaptionOptions
    // --------------------------------------------------------

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

    // --------------------------------------------------------
    // Captioner
    // --------------------------------------------------------

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
                // Do not hold the Python GIL while doing
                // image inference.
                py::gil_scoped_release release;

                return self.caption(
                    image_path,
                    options
                );
            },

            py::arg("image"),
            py::arg("options") =
                photon::CaptionOptions{}
        );

    m.attr("__version__") = "0.1.0";
}