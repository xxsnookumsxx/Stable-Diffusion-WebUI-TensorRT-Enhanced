import os
import shutil
import sys
import site

from modules import scripts


def detect_webui_version():
    """
    Detect WebUI variant by checking for modules_forge folder.
    Returns: 'forge' or 'automatic'
    """
    if os.path.exists(os.path.join(os.getcwd(), "modules_forge")):
        return "forge"
    return "automatic"


def _prepend_env_path(var: str, p: str):
    if not p or not os.path.isdir(p):
        return
    cur = os.environ.get(var, "")
    parts = [x for x in cur.split(os.pathsep) if x]
    if p not in parts:
        os.environ[var] = p + (os.pathsep + cur if cur else "")


def ensure_tensorrt_runtime_env():
    """
    Make sure Windows can locate CUDA/TensorRT DLLs when TensorRT is installed via pip wheels
    (e.g. tensorrt-cu12-libs / tensorrt-cu12-bindings).
    """
    # Only really needed on Windows DLL search order
    if os.name != "nt":
        return

    try:
        import torch
    except Exception:
        # If torch isn't importable yet, do nothing here.
        return

    # 1) Torch CUDA DLLs typically live here
    torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
    _prepend_env_path("PATH", torch_lib)

    # 2) Add site-packages roots (best-effort, harmless if redundant)
    candidates = set()

    try:
        for p in site.getsitepackages():
            candidates.add(p)
    except Exception:
        pass

    try:
        candidates.add(site.getusersitepackages())
    except Exception:
        pass

    for p in sys.path:
        if p and "site-packages" in p.lower():
            candidates.add(p)

    for sp in sorted(candidates):
        _prepend_env_path("PATH", sp)
        # common NVIDIA wheel layouts
        _prepend_env_path("PATH", os.path.join(sp, "nvidia"))
        _prepend_env_path("PATH", os.path.join(sp, "nvidia", "cudnn", "bin"))
        _prepend_env_path("PATH", os.path.join(sp, "nvidia", "cuda_runtime", "bin"))

    # 3) Fail fast if TensorRT still cannot import
    try:
        import tensorrt as trt  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "[TensorRT Enhanced] TensorRT import failed. "
            "Install tensorrt-cu12/tensorrt-cu12-libs/tensorrt-cu12-bindings into this python, "
            "and ensure torch\\lib is on PATH."
        ) from e


def clean_pycache():
    print("[TensorRT Enhanced] Cleaning __pycache__ folder...")

    for root, dirs, files in os.walk(scripts.basedir()):
        for dir_name in dirs:
            if dir_name == "__pycache__":
                dir_path = os.path.join(root, dir_name)
                print(f"[TensorRT Enhanced] Deleting: {dir_path}")
                shutil.rmtree(dir_path)

    print("[TensorRT Enhanced] Cleaned __pycache__ folder.")
