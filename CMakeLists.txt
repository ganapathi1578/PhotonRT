cmake_minimum_required(VERSION 3.20)

project(
    PhotonRT
    VERSION 0.2.1
    LANGUAGES CXX
)

# ============================================================
# Options
# ============================================================

option(
    PHOTON_ENABLE_ONNXRUNTIME
    "Enable ONNX Runtime"
    ON
)

option(
    PHOTON_BUILD_EXAMPLES
    "Build examples"
    ON
)

option(
    PHOTON_BUILD_TESTS
    "Build tests"
    ON
)

option(
    PHOTON_BUILD_BENCHMARKS
    "Build benchmarks"
    ON
)

option(
    PHOTON_BUILD_PYTHON
    "Build Python bindings"
    OFF
)

# ============================================================
# C++
# ============================================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ============================================================
# ONNX Runtime
# ============================================================

set(
    PHOTON_ONNXRUNTIME_ROOT
    ""
    CACHE PATH
    "ONNX Runtime root"
)

# Allow environment variable as a fallback.
if(
    PHOTON_ONNXRUNTIME_ROOT STREQUAL ""
    AND DEFINED ENV{ONNXRUNTIME_ROOT}
)
    set(
        PHOTON_ONNXRUNTIME_ROOT
        "$ENV{ONNXRUNTIME_ROOT}"
        CACHE PATH
        "ONNX Runtime root"
        FORCE
    )
endif()

if(
    PHOTON_ENABLE_ONNXRUNTIME
    AND PHOTON_ONNXRUNTIME_ROOT STREQUAL ""
)
    message(
        FATAL_ERROR
        "PHOTON_ONNXRUNTIME_ROOT is required."
    )
endif()

# ------------------------------------------------------------
# Locate ONNX Runtime headers
# ------------------------------------------------------------

if(PHOTON_ENABLE_ONNXRUNTIME)

    find_path(
        PHOTON_ORT_INCLUDE_DIR
        NAMES
            onnxruntime_cxx_api.h
        PATHS
            "${PHOTON_ONNXRUNTIME_ROOT}/include"
        NO_DEFAULT_PATH
    )

    # --------------------------------------------------------
    # Locate ONNX Runtime library
    #
    # Windows:
    #   onnxruntime.lib
    #
    # Linux:
    #   libonnxruntime.so
    # --------------------------------------------------------

    find_library(
        PHOTON_ORT_LIBRARY
        NAMES
            onnxruntime
        PATHS
            "${PHOTON_ONNXRUNTIME_ROOT}/lib"
        NO_DEFAULT_PATH
    )

    # --------------------------------------------------------
    # Windows DLL
    # --------------------------------------------------------

    if(WIN32)

        find_file(
            PHOTON_ORT_DLL
            NAMES
                onnxruntime.dll
            PATHS
                "${PHOTON_ONNXRUNTIME_ROOT}/lib"
                "${PHOTON_ONNXRUNTIME_ROOT}/bin"
            NO_DEFAULT_PATH
        )

    else()

        # Linux/macOS do not use a Windows DLL.
        set(
            PHOTON_ORT_DLL
            ""
        )

    endif()

    # --------------------------------------------------------
    # Validate ONNX Runtime
    # --------------------------------------------------------

    if(
        NOT PHOTON_ORT_INCLUDE_DIR
        OR NOT PHOTON_ORT_LIBRARY
        OR (WIN32 AND NOT PHOTON_ORT_DLL)
    )
        message(
            STATUS
            "PHOTON_ONNXRUNTIME_ROOT = ${PHOTON_ONNXRUNTIME_ROOT}"
        )

        message(
            STATUS
            "PHOTON_ORT_INCLUDE_DIR = ${PHOTON_ORT_INCLUDE_DIR}"
        )

        message(
            STATUS
            "PHOTON_ORT_LIBRARY = ${PHOTON_ORT_LIBRARY}"
        )

        if(WIN32)
            message(
                STATUS
                "PHOTON_ORT_DLL = ${PHOTON_ORT_DLL}"
            )
        endif()

        message(
            FATAL_ERROR
            "ONNX Runtime was not found."
        )
    endif()

    # --------------------------------------------------------
    # Print detected paths
    # --------------------------------------------------------

    message(
        STATUS
        "ONNX Runtime include: ${PHOTON_ORT_INCLUDE_DIR}"
    )

    message(
        STATUS
        "ONNX Runtime library: ${PHOTON_ORT_LIBRARY}"
    )

    if(WIN32)

        message(
            STATUS
            "ONNX Runtime DLL: ${PHOTON_ORT_DLL}"
        )

    endif()

    # --------------------------------------------------------
    # Linux/macOS runtime search path
    #
    # Allows executables/tests to find:
    #   libonnxruntime.so
    #
    # without requiring LD_LIBRARY_PATH.
    # --------------------------------------------------------

    get_filename_component(
        PHOTON_ORT_LIBRARY_DIR
        "${PHOTON_ORT_LIBRARY}"
        DIRECTORY
    )

    set(
        CMAKE_BUILD_RPATH
        "${PHOTON_ORT_LIBRARY_DIR}"
    )

endif()

# ============================================================
# PhotonRT
# ============================================================

add_library(
    photonrt

    src/image.cpp
    src/preprocess.cpp
    src/tokenizer.cpp
    src/captioner.cpp

    vision/mobileclip/mobileclip_onnx.cpp
    vision/photon/photon_onnx.cpp
)

add_library(
    PhotonRT::photonrt
    ALIAS photonrt
)

target_include_directories(
    photonrt
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include

    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

if(PHOTON_ENABLE_ONNXRUNTIME)

    target_include_directories(
        photonrt
        PRIVATE
            ${PHOTON_ORT_INCLUDE_DIR}
    )

endif()

target_compile_features(
    photonrt
    PUBLIC
        cxx_std_17
)

if(MSVC)

    target_compile_options(
        photonrt
        PRIVATE
            /W4
            /fp:precise
    )

else()

    target_compile_options(
        photonrt
        PRIVATE
            -Wall
            -Wextra
            -Wpedantic
    )

endif()

if(PHOTON_ENABLE_ONNXRUNTIME)

    target_link_libraries(
        photonrt
        PRIVATE
            ${PHOTON_ORT_LIBRARY}
    )

    target_compile_definitions(
        photonrt
        PUBLIC
            PHOTON_USE_ONNXRUNTIME=1
    )

endif()

if(MSVC)

    target_compile_definitions(
        photonrt
        PRIVATE
            _CRT_SECURE_NO_WARNINGS
    )

endif()

# ============================================================
# Copy ONNX Runtime DLL
# ============================================================

function(
    photon_copy_onnxruntime
    target
)

    if(WIN32 AND PHOTON_ENABLE_ONNXRUNTIME)

        add_custom_command(
            TARGET ${target}
            POST_BUILD

            COMMAND
                ${CMAKE_COMMAND}
                -E
                copy_if_different
                "${PHOTON_ORT_DLL}"
                "$<TARGET_FILE_DIR:${target}>"

            VERBATIM
        )

    endif()

endfunction()

# ============================================================
# Examples
# ============================================================

if(PHOTON_BUILD_EXAMPLES)

    add_executable(
        image-caption
        examples/image-caption.cpp
    )

    target_link_libraries(
        image-caption
        PRIVATE
            photonrt
    )

    photon_copy_onnxruntime(
        image-caption
    )

    if(
        EXISTS
        "${CMAKE_CURRENT_SOURCE_DIR}/examples/mobileclip-embedding.cpp"
    )

        add_executable(
            mobileclip-embedding
            examples/mobileclip-embedding.cpp
        )

        target_link_libraries(
            mobileclip-embedding
            PRIVATE
                photonrt
        )

        photon_copy_onnxruntime(
            mobileclip-embedding
        )

    endif()

endif()

# ============================================================
# Tests
# ============================================================

if(PHOTON_BUILD_TESTS)

    enable_testing()

    if(
        EXISTS
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_image.cpp"
    )

        add_executable(
            test_image
            tests/test_image.cpp
        )

        target_link_libraries(
            test_image
            PRIVATE
                photonrt
        )

        add_test(
            NAME image
            COMMAND test_image
        )

    endif()

    if(
        EXISTS
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_preprocess.cpp"
    )

        add_executable(
            test_preprocess
            tests/test_preprocess.cpp
        )

        target_link_libraries(
            test_preprocess
            PRIVATE
                photonrt
        )

        add_test(
            NAME preprocess
            COMMAND test_preprocess
        )

    endif()

    if(
        EXISTS
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_mobileclip_onnx.cpp"
    )

        add_executable(
            test_mobileclip_onnx
            tests/test_mobileclip_onnx.cpp
        )

        target_link_libraries(
            test_mobileclip_onnx
            PRIVATE
                photonrt
        )

        photon_copy_onnxruntime(
            test_mobileclip_onnx
        )

        add_test(
            NAME mobileclip_onnx
            COMMAND test_mobileclip_onnx
        )

    endif()

endif()

# ============================================================
# Benchmark
# ============================================================

if(PHOTON_BUILD_BENCHMARKS)

    add_executable(
        bench_image_caption
        benchmarks/bench_image_caption.cpp
    )

    target_link_libraries(
        bench_image_caption
        PRIVATE
            photonrt
    )

    photon_copy_onnxruntime(
        bench_image_caption
    )

endif()

# ============================================================
# Optional precision comparison benchmark
# ============================================================

# if(PHOTON_ENABLE_ONNXRUNTIME)

#     add_executable(
#         compare_photon_precision
#         benchmarks/compare_photon_precision.cpp
#     )

#     target_link_libraries(
#         compare_photon_precision
#         PRIVATE
#             photonrt
#             "${PHOTON_ORT_LIBRARY}"
#     )

#     target_include_directories(
#         compare_photon_precision
#         PRIVATE
#             ${CMAKE_CURRENT_SOURCE_DIR}
#             ${PHOTON_ORT_INCLUDE_DIR}
#     )

#     photon_copy_onnxruntime(
#         compare_photon_precision
#     )

#     add_executable(
#         compare_photon_decoder
#         benchmarks/compare_photon_decoder.cpp
#     )

#     target_link_libraries(
#         compare_photon_decoder
#         PRIVATE
#             photonrt
#             "${PHOTON_ORT_LIBRARY}"
#     )

#     target_include_directories(
#         compare_photon_decoder
#         PRIVATE
#             ${CMAKE_CURRENT_SOURCE_DIR}
#             ${PHOTON_ORT_INCLUDE_DIR}
#     )

#     photon_copy_onnxruntime(
#         compare_photon_decoder
#     )

# endif()

# ============================================================
# Python bindings
# ============================================================

if(PHOTON_BUILD_PYTHON)

    find_package(
        Python
        COMPONENTS
            Interpreter
            Development.Module
        REQUIRED
    )

    find_package(
        pybind11
        CONFIG
        REQUIRED
    )

    pybind11_add_module(
        _photonrt
        bindings/python_module.cpp
    )

    target_link_libraries(
        _photonrt
        PRIVATE
            photonrt
    )

    target_include_directories(
        _photonrt
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
    )

    if(PHOTON_ENABLE_ONNXRUNTIME)

        target_include_directories(
            _photonrt
            PRIVATE
                ${PHOTON_ORT_INCLUDE_DIR}
        )

    endif()

    target_compile_features(
        _photonrt
        PRIVATE
            cxx_std_17
    )

    photon_copy_onnxruntime(
        _photonrt
    )

    install(
        TARGETS
            _photonrt

        LIBRARY
            DESTINATION photonrt

        RUNTIME
            DESTINATION photonrt
    )

    if(WIN32 AND PHOTON_ENABLE_ONNXRUNTIME)

        install(
            FILES
                "${PHOTON_ORT_DLL}"

            DESTINATION
                photonrt
        )

    endif()

endif()