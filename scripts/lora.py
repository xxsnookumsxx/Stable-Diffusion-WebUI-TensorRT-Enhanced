import os
from typing import List, Union

import numpy as np
import torch
from safetensors.torch import load_file

import onnx
from onnx import numpy_helper
from tqdm import tqdm


def _to_float(x: Union[str, float, int]) -> float:
    if isinstance(x, (float, int)):
        return float(x)
    if isinstance(x, str):
        return float(x.strip())
    raise TypeError(f"LoRA scale must be a number or numeric string, got {type(x)}")


def merge_loras(loras: List[str], scales: List[Union[str, float, int]]) -> dict:
    if len(loras) != len(scales):
        raise ValueError(f"loras/scales length mismatch: {len(loras)} vs {len(scales)}")

    refit_dict = {}

    # only output the file name (for progress display)
    lora_names = [os.path.splitext(os.path.basename(lora))[0] for lora in loras]

    for lora, scale_raw, lora_name in zip(loras, scales, lora_names):
        scale = _to_float(scale_raw)

        lora_dict = load_file(lora)

        with tqdm(total=len(lora_dict), desc=f" > Loading '{lora_name}' LoRA", mininterval=1) as pbar:
            for k, v in lora_dict.items():
                vv = v.contiguous()
                if k in refit_dict:
                    refit_dict[k] += (scale * vv)
                else:
                    refit_dict[k] = (scale * vv)
                pbar.update(1)
            pbar.refresh()

    print("[TensorRT Enhanced] LoRA loading completed.")
    return refit_dict


def apply_loras(base_path: str, loras: List[str], scales: List[Union[str, float, int]]) -> dict:
    refit_dict = merge_loras(loras, scales)

    base_name = os.path.basename(base_path)
    lora_names = [os.path.basename(lora) for lora in loras]
    print(f"[TensorRT Enhanced] apply base model from {base_name} for refit lora: {lora_names}")

    base = onnx.load(base_path)
    onnx_opt_dir = os.path.dirname(base_path)

    total_initializers = sum(
        1 for initializer in base.graph.initializer if initializer.name in refit_dict
    )

    with tqdm(total=total_initializers, desc="Updating weights", mininterval=1) as pbar:
        for initializer in base.graph.initializer:
            if initializer.name not in refit_dict:
                continue

            wt = refit_dict[initializer.name]

            initializer_data = numpy_helper.to_array(
                initializer, base_dir=onnx_opt_dir
            ).astype(np.float16)

            # base + delta
            base_wt = torch.tensor(initializer_data, device=wt.device)
            refit_dict[initializer.name] = (base_wt + wt).contiguous()

            pbar.update(1)
        pbar.refresh()

    print("[TensorRT Enhanced] LoRA apply completed.")
    return refit_dict
