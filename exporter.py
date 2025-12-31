import os
import time
import shutil
import json
from pathlib import Path
from logging import info, error
from collections import OrderedDict
from typing import List, Tuple

import torch
import torch.nn.functional as F
import numpy as np
import onnx
from onnx import numpy_helper
from optimum.onnx.utils import (
    _get_onnx_external_data_tensors,
    check_model_uses_external_data,
)

from modules import shared

from utilities import Engine
from datastructures import ProfileSettings
from model_helper import UNetModel


def apply_lora(model: torch.nn.Module, lora_path: str, inputs: Tuple[torch.Tensor]) -> torch.nn.Module:
    """
    Applies a LoRA by importing the A1111 LoRA modules and calling networks.load_networks().
    If LoRA modules can't be imported, fail fast with a clear error (instead of crashing later).
    """
    try:
        import sys
        import importlib

        # A1111-style built-in LoRA path; may differ on some forks.
        sys.path.append("extensions-builtin/Lora")

        networks = importlib.import_module("networks")
        # Imported for side-effects / compatibility with some installs
        importlib.import_module("network")
        importlib.import_module("extra_networks_lora")
    except Exception as e:
        error(e)
        raise RuntimeError(
            "LoRA modules not found/import failed. "
            "Ensure the LoRA extension exists and is importable (extensions-builtin/Lora)."
        )

    # Warm-up forward (some webui/LoRA loaders expect model to be 'touched' first)
    model.forward(*inputs)

    # NOTE: caller currently passes os.path.splitext(lora_name)[0], keep that behavior as-is.
    networks.load_networks([lora_path], [1.0], [1.0], [None])

    # Forward after loading so LoRA layers actually get applied
    model.forward(*inputs)
    return model


def get_refit_weights(
    state_dict: dict, onnx_opt_path: str, weight_name_mapping: dict, weight_shape_mapping: dict
) -> dict:
    refit_weights = OrderedDict()
    onnx_opt_dir = os.path.dirname(onnx_opt_path)
    onnx_opt_model = onnx.load(onnx_opt_path)

    # Create initializer data hashes
    initializer_hash_mapping = {}
    onnx_data_mapping = {}
    for initializer in onnx_opt_model.graph.initializer:
        initializer_data = numpy_helper.to_array(
            initializer, base_dir=onnx_opt_dir
        ).astype(np.float16)
        initializer_hash = hash(initializer_data.data.tobytes())
        initializer_hash_mapping[initializer.name] = initializer_hash
        onnx_data_mapping[initializer.name] = initializer_data

    for torch_name, initializer_name in weight_name_mapping.items():
        initializer_hash = initializer_hash_mapping[initializer_name]
        wt = state_dict[torch_name]

        # get shape transform info
        initializer_shape, is_transpose = weight_shape_mapping[torch_name]
        if is_transpose:
            wt = torch.transpose(wt, 0, 1)
        else:
            wt = torch.reshape(wt, initializer_shape)

        # include weight if hashes differ
        wt_hash = hash(wt.cpu().detach().numpy().astype(np.float16).data.tobytes())
        if initializer_hash != wt_hash:
            delta = wt - torch.tensor(onnx_data_mapping[initializer_name]).to(wt.device)
            refit_weights[initializer_name] = delta.contiguous()

    return refit_weights


def export_lora(
    modelobj: UNetModel,
    onnx_path: str,
    weights_map_path: str,
    lora_name: str,
    profile: ProfileSettings,
) -> dict:
    info("Exporting to ONNX...")
    inputs = modelobj.get_sample_input(
        profile.bs_opt * 2,
        profile.h_opt // 8,
        profile.w_opt // 8,
        profile.t_opt,
    )

    with open(weights_map_path, "r") as fp_wts:
        print(f"[I] Loading weights map: {weights_map_path} ")
        [weights_name_mapping, weights_shape_mapping] = json.load(fp_wts)

    with torch.inference_mode(), torch.autocast("cuda"):
        modelobj.unet = apply_lora(
            modelobj.unet, os.path.splitext(lora_name)[0], inputs
        )

        refit_dict = get_refit_weights(
            modelobj.unet.state_dict(),
            onnx_path,
            weights_name_mapping,
            weights_shape_mapping,
        )

    return refit_dict


def swap_sdpa(func):
    def wrapper(*args, **kwargs):
        swap_sdpa_flag = hasattr(F, "scaled_dot_product_attention")
        old_sdpa = (
            getattr(F, "scaled_dot_product_attention", None) if swap_sdpa_flag else None
        )
        if swap_sdpa_flag:
            delattr(F, "scaled_dot_product_attention")
        ret = func(*args, **kwargs)
        if swap_sdpa_flag and old_sdpa:
            setattr(F, "scaled_dot_product_attention", old_sdpa)
        return ret

    return wrapper


@swap_sdpa
def export_onnx(
    onnx_path: str,
    modelobj: UNetModel,
    profile: ProfileSettings,
    opset: int = 17,
    diable_optimizations: bool = False,
):
    info("Exporting to ONNX...")
    inputs = modelobj.get_sample_input(
        profile.bs_opt * 2,
        profile.h_opt // 8,
        profile.w_opt // 8,
        profile.t_opt,
    )

    if not os.path.exists(onnx_path):
        _export_onnx(
            modelobj.unet,
            inputs,
            Path(onnx_path),
            opset,
            modelobj.get_input_names(),
            modelobj.get_output_names(),
            modelobj.get_dynamic_axes(),
            modelobj.optimize if not diable_optimizations else None,
        )


def _export_onnx(
    model: torch.nn.Module,
    inputs: Tuple[torch.Tensor],
    path: str,
    opset: int,
    in_names: List[str],
    out_names: List[str],
    dyn_axes: dict,
    optimizer=None,
):
    tmp_dir = os.path.abspath("onnx_tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_path = os.path.join(tmp_dir, "model.onnx")

    try:
        info("Exporting to ONNX...")
        with torch.inference_mode(), torch.autocast("cuda"):
            torch.onnx.export(
                model,
                inputs,
                tmp_path,
                export_params=True,
                opset_version=opset,
                do_constant_folding=True,
                input_names=in_names,
                output_names=out_names,
                dynamic_axes=dyn_axes,
            )
    except Exception as e:
        error("Exporting to ONNX failed. {}".format(e))
        return

    info("Optimize ONNX.")
    os.makedirs(path.parent, exist_ok=True)
    onnx_model = onnx.load(tmp_path, load_external_data=False)
    model_uses_external_data = check_model_uses_external_data(onnx_model)

    if model_uses_external_data:
        info("ONNX model uses external data. Saving as external data.")
        _ = _get_onnx_external_data_tensors(onnx_model)
        onnx_model = onnx.load(tmp_path, load_external_data=True)
        onnx.save(
            onnx_model,
            str(path),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=path.name + "_data",
            size_threshold=1024,
        )

    if optimizer is not None:
        try:
            onnx_opt_graph = optimizer("unet", onnx_model)
            onnx.save(onnx_opt_graph, path)
        except Exception as e:
            error("Optimizing ONNX failed. {}".format(e))
            return

    if not model_uses_external_data and optimizer is None:
        shutil.move(tmp_path, str(path))

    shutil.rmtree(tmp_dir)


def export_trt(trt_path: str, onnx_path: str, timing_cache: str, profile: dict, use_fp16: bool):
    engine = Engine(trt_path)

    # Preserve whatever is currently stored (Forge/A1111 may swap this during runtime).
    orig_sd_model = getattr(shared, "sd_model", None)

    moved_to_cpu = False
    cpu_model_obj = None

    try:
        # TODO Still approx. 2gb of VRAM unaccounted for...
        if orig_sd_model is None:
            raise RuntimeError("shared.sd_model is None (model not loaded yet).")

        # Best-effort: move model off VRAM during build.
        if hasattr(orig_sd_model, "cpu"):
            cpu_model_obj = orig_sd_model.cpu()
            moved_to_cpu = True
            shared.sd_model = cpu_model_obj

        torch.cuda.empty_cache()

        s = time.time()
        ret = engine.build(
            onnx_path,
            use_fp16,
            enable_refit=True,
            enable_preview=True,
            timing_cache=timing_cache,
            input_profile=[profile],
        )
        e = time.time()
        info(f"Time taken to build: {(e - s)}s")

        return ret

    except Exception as e:
        error(f"TensorRT build failed: {e}")
        return 1

    finally:
        # Always restore shared.sd_model to a sane value
        try:
            if moved_to_cpu and cpu_model_obj is not None and hasattr(cpu_model_obj, "cuda"):
                shared.sd_model = cpu_model_obj.cuda()
            else:
                shared.sd_model = orig_sd_model
        except Exception as e:
            error(f"Failed to restore shared.sd_model after TRT build: {e}")
            shared.sd_model = orig_sd_model
