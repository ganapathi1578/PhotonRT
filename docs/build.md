# Building PhotonRT

## Toolchain

PhotonRT uses:

- CMake
- a C++ compiler/toolchain
- Python
- pybind11
- ONNX Runtime
- scikit-build-core for Python packaging

## ONNX Runtime SDK

Set:

```text
PHOTON_ONNXRUNTIME_ROOT
```

Example on Windows Git Bash:

```bash
export PHOTON_ONNXRUNTIME_ROOT="C:/path/to/onnxruntime"
```

The SDK should provide the required:

```text
include/
lib/onnxruntime.lib
lib/onnxruntime.dll
```

## Native configure

```bash
cmake -S . -B build \
  -DPHOTON_ENABLE_ONNXRUNTIME=ON \
  -DPHOTON_BUILD_PYTHON=ON \
  -DPHOTON_BUILD_EXAMPLES=OFF \
  -DPHOTON_BUILD_TESTS=OFF \
  -DPHOTON_BUILD_BENCHMARKS=OFF \
  -DPHOTON_ONNXRUNTIME_ROOT="$PHOTON_ONNXRUNTIME_ROOT"
```

## Native build

```bash
cmake --build build --config Release
```

## Python build

```bash
python -m build
```

## Validate

```bash
python -m twine check dist/*
```

## Wheel contents

A Python wheel should contain the Python package plus native runtime components such as:

```text
photonrt/
├── __init__.py
├── hub.py
├── stream.py
├── _photonrt.<python-tag>-win_amd64.pyd
└── onnxruntime.dll
```

The exact extension filename changes with the supported Python ABI.

## Source distribution

The source distribution should contain the C++ and Python source required to rebuild the package.

Avoid packaging generated build trees or unrelated large model artifacts.

## Clean package test

Create a fresh environment:

```bash
python -m venv test-env
```

Activate it and install:

```bash
python -m pip install dist/photonrt-<version>-*.whl
```

Then:

```bash
python -c "import photonrt; print(photonrt.__version__)"
python -c "from photonrt import Captioner, CameraCaptioner; print('OK')"
```

Test Hugging Face loading:

```bash
python -c "from photonrt import Captioner; m=Captioner.from_pretrained(); print(m.caption('image.jpg'))"
```

## Environment variables

PhotonRT source builds use:

```text
PHOTON_ONNXRUNTIME_ROOT
```

Network certificate behavior is controlled by the surrounding Python/HTTPS environment, not by the PhotonRT model API.

