# PhotonRT Release Guide

## Release layout

PhotonRT uses three distribution channels:

```text
GitHub
  |
  +-- source code
  +-- documentation
  +-- tags/releases

PyPI
  |
  +-- Python package
  +-- native extension wheels

Hugging Face
  |
  +-- model artifacts
```

## Version

Update the version consistently in the release metadata and package.

For 0.2.0:

```text
0.2.0
```

## Build

```bash
rm -rf dist
python -m build
```

## Validate

```bash
python -m twine check dist/*
```

## Test the wheel

Use a clean environment and install the wheel directly:

```bash
python -m pip install dist/photonrt-0.2.0-cp310-cp310-win_amd64.whl
```

Then:

```bash
python -c "import photonrt; print(photonrt.__version__)"
```

Test:

```python
from photonrt import Captioner, CameraCaptioner

print("PhotonRT OK")
```

Then test model retrieval:

```python
from photonrt import Captioner

model = Captioner.from_pretrained()

print(model.caption("image.jpg"))
```

## Git

Recommended release sequence:

```bash
git status
git add .
git commit -m "Release PhotonRT 0.2.0"
git push
```

Create an annotated tag:

```bash
git tag -a v0.2.0 -m "PhotonRT 0.2.0"
git push origin v0.2.0
```

## GitHub Release

Create a GitHub release from:

```text
v0.2.0
```

Release notes should include:

- package version
- installation command
- key features
- model repository
- supported wheel/platform information
- important API examples

## PyPI

Upload:

```bash
python -m twine upload dist/*
```

Do not upload a package build that contains credentials or private local files.

## Important packaging rule

Do not treat:

```text
build/
dist/
*.pyd
*.dll
large model files
```

as ordinary source files for Git distribution.

Generated native binaries belong in wheels; model artifacts belong in the model distribution.

## Release verification

After PyPI publication, test in a fresh environment:

```bash
python -m pip install photonrt
```

Then:

```bash
python -c "import photonrt; print(photonrt.__version__)"
```

and:

```bash
python -c "from photonrt import Captioner, CameraCaptioner; print('PACKAGE OK')"
```

Finally verify:

```python
from photonrt import Captioner

model = Captioner.from_pretrained()
print(model.caption("image.jpg"))
```

A release is considered operational when:

```text
pip install
    |
    v
Python import
    |
    v
native extension load
    |
    v
Hugging Face model retrieval
    |
    v
native inference
    |
    v
caption
```

all succeed.
