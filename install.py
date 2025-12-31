import launch
import sys

python = sys.executable

# CUDA 12.x (includes CUDA 12.8)
TRT_PKGS = [
    "tensorrt-cu12",
    "tensorrt-cu12-libs",
    "tensorrt-cu12-bindings",
]

NVIDIA_PYPI = "https://pypi.nvidia.com"
NVIDIA_NGC_PYPI = "https://pypi.ngc.nvidia.com"


def _pip_install(args: str, pkgname: str):
    # launch.run_pip takes the pip args string without the leading "pip"
    launch.run_pip(args, pkgname, live=True)


def install():
    print("[TensorRT Enhanced] Installing dependencies...")

    # importlib_metadata (kept from your original)
    if not launch.is_installed("importlib_metadata"):
        print("[TensorRT Enhanced] importlib_metadata is not installed! Installing...")
        _pip_install("install importlib_metadata", "importlib_metadata")

    from importlib_metadata import version

    # ---- TensorRT (CUDA 12) ----
    # Remove older/legacy packages that can break imports or cause placeholder installs.
    # Your old script pinned tensorrt==9.0.1.post11.dev4. [file:122]
    if launch.is_installed("tensorrt"):
        print("[TensorRT Enhanced] Removing legacy 'tensorrt' package (will use tensorrt-cu12*)...")
        launch.run([python, "-m", "pip", "uninstall", "-y", "tensorrt"], "removing legacy tensorrt")

    # Install CUDA-12 TensorRT wheels (CUDA 12.8 is CUDA-major 12) [web:13]
    for pkg in TRT_PKGS:
        if not launch.is_installed(pkg):
            print(f"[TensorRT Enhanced] {pkg} is not installed! Installing...")
            _pip_install(
                f"install --upgrade --no-cache-dir --extra-index-url {NVIDIA_PYPI} {pkg}",
                pkg,
            )

    # Quick sanity check: the import name is still "tensorrt"
    try:
        import tensorrt as trt  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "[TensorRT Enhanced] TensorRT import failed after install. "
            "Check that tensorrt-cu12-bindings installed a compatible wheel for your Python/OS."
        ) from e

    # ---- Polygraphy (kept) ----
    if not launch.is_installed("polygraphy"):
        print("[TensorRT Enhanced] Polygraphy is not installed! Installing...")
        _pip_install(f"install polygraphy --extra-index-url {NVIDIA_NGC_PYPI}", "polygraphy")

    # ---- ONNX GraphSurgeon (kept) ----
    if not launch.is_installed("onnx_graphsurgeon"):
        print("[TensorRT Enhanced] ONNX GS is not installed! Installing...")
        _pip_install("install protobuf==3.20.2", "protobuf")
        _pip_install(
            f"install onnx-graphsurgeon --extra-index-url {NVIDIA_NGC_PYPI}",
            "onnx-graphsurgeon",
        )

    # ---- ONNX (kept) ----
    if launch.is_installed("onnx"):
        if version("onnx") != "1.16.1":
            print("[TensorRT Enhanced] ONNX is not the correct version! Uninstalling...")
            launch.run([python, "-m", "pip", "uninstall", "-y", "onnx"], "removing old version of onnx")

    if not launch.is_installed("onnx"):
        print("[TensorRT Enhanced] ONNX is not installed! Installing...")
        _pip_install("install onnx==1.16.1", "onnx")

    # ---- OPTIMUM (kept) ----
    if not launch.is_installed("optimum"):
        print("[TensorRT Enhanced] Optimum is not installed! Installing...")
        _pip_install("install optimum", "optimum")

    print("[TensorRT Enhanced] Dependencies all installed!")


install()
