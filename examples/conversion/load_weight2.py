#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Round-trip conversion between Hugging Face and Megatron FSDP.

Usage examples:
    python -m torch.distributed.run --nproc_per_node=8 examples/conversion/hf_fsdp_roundtrip.py --hf-model-id Qwen/Qwen3-30B-A3B --tp 2 --cp 2 --ep 2
"""

from __future__ import annotations

import argparse
import os

import torch
from megatron.core.distributed import DistributedDataParallelConfig
from rich.console import Console

from megatron.bridge import AutoBridge
from megatron.bridge.models.decorators import torchrun_main


console = Console()
HF_MODEL_ID = "Qwen/Qwen3-VL-30B-A3B-Instruct"


def _maybe_barrier() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def _get_world_size() -> int:
    # torchrun exports WORLD_SIZE; fall back to 1 for single-process runs.
    try:
        return int(os.environ.get("WORLD_SIZE", "1"))
    except ValueError:
        return 1


def _configure_model_provider(model_provider, tp: int, cp: int, ep: int) -> None:
    world_size = _get_world_size()
    mp_size = tp * cp * ep
    if mp_size <= 0:
        raise ValueError(f"Invalid parallel sizes: tp={tp}, cp={cp}, ep={ep}")
    if world_size % mp_size != 0:
        raise ValueError(
            f"WORLD_SIZE ({world_size}) must be divisible by tp*cp*ep ({mp_size}). Got tp={tp}, cp={cp}, ep={ep}."
        )

    model_provider.tensor_model_parallel_size = tp
    model_provider.context_parallel_size = cp
    model_provider.expert_model_parallel_size = ep
    model_provider.finalize()
    model_provider.initialize_model_parallel(seed=0)


@torchrun_main
def main(
    hf_model_id: str = HF_MODEL_ID,
    output_dir: str | None = None,
    tp: int = 1,
    cp: int = 1,
    ep: int = 1,
    trust_remote_code: bool = False,
) -> None:
    """Perform round-trip conversion between HuggingFace and Megatron-FSDP models."""
    bridge = AutoBridge.from_hf_pretrained(
        hf_model_id, trust_remote_code=trust_remote_code, torch_dtype=torch.bfloat16
    )

    model_provider = bridge.to_megatron_provider(load_weights=True)
    _configure_model_provider(model_provider, tp=tp, cp=cp, ep=ep)

    ddp_config = DistributedDataParallelConfig(
        use_distributed_optimizer=True,
        check_for_nan_in_grad=True,
        use_megatron_fsdp=True,
        data_parallel_sharding_strategy="optim_grads_params",
    )

    megatron_model = model_provider.provide_distributed_model(
        ddp_config=ddp_config,
        use_megatron_fsdp=True,
        use_torch_fsdp2=False,
        overlap_param_gather_with_optimizer_step=False,
        data_parallel_random_init=False,
    )

    # Save weights for comparison - each rank saves its own weights
    rank = torch.distributed.get_rank()
    weights_to_save = {}
    for name, param in megatron_model[0].named_parameters():
        weights_to_save[name] = param.detach().cpu().clone()

    output_file = f"weights_method2_rank{rank}.pt"
    torch.save(weights_to_save, output_file)
    console.print(f"[green]Rank {rank}: Saved {len(weights_to_save)} weights to {output_file}[/green]")

    _maybe_barrier()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert between HuggingFace and Megatron-FSDP model formats.")
    parser.add_argument("--hf-model-id", type=str, default=HF_MODEL_ID, help="HuggingFace model ID to convert")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory where the converted model directory will be created. Defaults to CWD.",
    )
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallelism size.")
    parser.add_argument("--cp", type=int, default=1, help="Context parallelism size.")
    parser.add_argument("--ep", type=int, default=1, help="Expert parallelism size.")
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow executing custom modeling/tokenizer code when loading from a model repository.",
    )

    args = parser.parse_args()
    main(
        hf_model_id=args.hf_model_id,
        output_dir=args.output_dir,
        tp=args.tp,
        cp=args.cp,
        ep=args.ep,
        trust_remote_code=args.trust_remote_code,
    )

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
