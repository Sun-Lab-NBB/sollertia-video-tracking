"""Contains tests for the training optimization profile resolver and the process-global optimization applier."""

import os
import importlib.util

import torch
import pytest

from sollertia_video_tracking.hardware import Toggle, AmpMode, DeviceType
from sollertia_video_tracking.training import optimization
from sollertia_video_tracking.training.optimization import (
    MultiGpuStrategy,
    OptimizationProfile,
    _resolve_multi_gpu,
    apply_runtime_optimizations,
    resolve_optimization_profile,
    _choose_dataloader_worker_count,
)


# Helpers
def _set_cuda(monkeypatch: pytest.MonkeyPatch, count: int) -> None:
    """Fakes the visible CUDA device count so the resolvers see a deterministic machine."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: count > 0)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: count)


def _set_capabilities(monkeypatch: pytest.MonkeyPatch, capabilities: dict[int, tuple[int, int]]) -> None:
    """Fakes per-index CUDA compute capabilities for the Ampere check."""
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: capabilities[device])


def _set_cpu_count(monkeypatch: pytest.MonkeyPatch, count: int | None) -> None:
    """Pins the reported CPU core count so worker and thread budgets are deterministic."""
    monkeypatch.setattr(os, "cpu_count", lambda: count)


def _profile(**overrides: object) -> OptimizationProfile:
    """Builds an OptimizationProfile with sensible defaults, overriding only the fields a test cares about."""
    fields: dict[str, object] = {
        "device": "cpu",
        "gpus": (),
        "multi_gpu_strategy": MultiGpuStrategy.SINGLE,
        "amp_dtype": None,
        "use_gradient_scaler": False,
        "tf32": False,
        "cudnn_benchmark": False,
        "torch_compile": False,
        "dataloader_workers": 0,
        "pin_memory": False,
        "cpu_threads": None,
    }
    fields.update(overrides)
    return OptimizationProfile(**fields)  # type: ignore[arg-type]


# MultiGpuStrategy enum
def test_multi_gpu_strategy_string_values() -> None:
    """Verifies that the multi-GPU strategy enum is a string enum with the documented literal values."""
    assert MultiGpuStrategy.AUTO == "auto"
    assert MultiGpuStrategy.DDP == "ddp"
    assert MultiGpuStrategy.DP == "dp"
    assert MultiGpuStrategy.SINGLE == "single"


def test_module_constants() -> None:
    """Verifies that the multi-GPU floor and the automatic dataloader-worker ceiling hold their documented values."""
    assert optimization._MIN_MULTI_GPU_COUNT == 2
    assert optimization._MAX_AUTO_DATALOADER_WORKERS == 8


# OptimizationProfile properties
def test_use_amp_reflects_dtype_presence() -> None:
    """Verifies that mixed precision is on exactly when an autocast dtype is set."""
    assert _profile(amp_dtype=torch.bfloat16).use_amp is True
    assert _profile(amp_dtype=None).use_amp is False


def test_use_ddp_only_for_ddp_strategy() -> None:
    """Verifies that the DDP flag is true only under the DDP strategy."""
    assert _profile(multi_gpu_strategy=MultiGpuStrategy.DDP).use_ddp is True
    assert _profile(multi_gpu_strategy=MultiGpuStrategy.SINGLE).use_ddp is False
    assert _profile(multi_gpu_strategy=MultiGpuStrategy.DP).use_ddp is False


def test_world_size_is_gpu_count_under_ddp_else_one() -> None:
    """Verifies that world size counts every GPU under DDP and collapses to one otherwise."""
    assert _profile(multi_gpu_strategy=MultiGpuStrategy.DDP, gpus=(0, 1, 2)).world_size == 3
    # Non-DDP: even with multiple GPUs selected, the run is a single process.
    assert _profile(multi_gpu_strategy=MultiGpuStrategy.DP, gpus=(0, 1)).world_size == 1
    assert _profile(multi_gpu_strategy=MultiGpuStrategy.SINGLE, gpus=(0,)).world_size == 1


def test_amp_device_type_maps_device_to_autocast_string() -> None:
    """Verifies that the autocast device-type string is cuda for cuda and passes other devices through unchanged."""
    assert _profile(device="cuda").amp_device_type == "cuda"
    assert _profile(device="cpu").amp_device_type == "cpu"
    assert _profile(device="mps").amp_device_type == "mps"


# OptimizationProfile.describe
def test_describe_cuda_with_all_extras() -> None:
    """Verifies the CUDA summary lists the indices, strategy, precision, workers, pinning, and every enabled extra."""
    profile = _profile(
        device="cuda",
        gpus=(0, 1),
        multi_gpu_strategy=MultiGpuStrategy.DDP,
        amp_dtype=torch.bfloat16,
        tf32=True,
        cudnn_benchmark=True,
        torch_compile=True,
        dataloader_workers=4,
        pin_memory=True,
    )
    assert profile.describe() == "CUDA [0, 1] (ddp) | bfloat16 | workers=4 pin=True, tf32+cudnn.benchmark+compile"


def test_describe_cpu_without_extras() -> None:
    """Verifies the non-CUDA summary uppercases the device, reports fp32, and omits extras when none are enabled."""
    profile = _profile(device="cpu", dataloader_workers=0, pin_memory=False)
    assert profile.describe() == "CPU | fp32 | workers=0 pin=False"


# _resolve_multi_gpu
def test_resolve_multi_gpu_single_gpu_auto_is_silent_single(capsys: pytest.CaptureFixture[str]) -> None:
    """Verifies that a single GPU under the auto request resolves to single-device training without a warning."""
    assert _resolve_multi_gpu(MultiGpuStrategy.AUTO, gpus=(0,)) == MultiGpuStrategy.SINGLE
    assert capsys.readouterr().err == ""


def test_resolve_multi_gpu_single_gpu_ddp_warns_and_falls_back(capsys: pytest.CaptureFixture[str]) -> None:
    """Verifies that explicitly requesting DDP against one GPU warns and falls back to single-device training."""
    assert _resolve_multi_gpu(MultiGpuStrategy.DDP, gpus=(0,)) == MultiGpuStrategy.SINGLE
    assert "only 1 GPU is selected" in capsys.readouterr().err


def test_resolve_multi_gpu_single_gpu_dp_warns_and_falls_back(capsys: pytest.CaptureFixture[str]) -> None:
    """Verifies that explicitly requesting DP against one GPU also warns and falls back to single-device training."""
    assert _resolve_multi_gpu(MultiGpuStrategy.DP, gpus=(0,)) == MultiGpuStrategy.SINGLE
    assert "Requested 'dp'" in capsys.readouterr().err


def test_resolve_multi_gpu_two_gpus_dp_stays_dp() -> None:
    """Verifies that two GPUs with an explicit DP request keep DataParallel."""
    assert _resolve_multi_gpu(MultiGpuStrategy.DP, gpus=(0, 1)) == MultiGpuStrategy.DP


def test_resolve_multi_gpu_two_gpus_auto_becomes_ddp() -> None:
    """Verifies that two GPUs under the auto request resolve to DDP."""
    assert _resolve_multi_gpu(MultiGpuStrategy.AUTO, gpus=(0, 1)) == MultiGpuStrategy.DDP


def test_resolve_multi_gpu_two_gpus_ddp_stays_ddp() -> None:
    """Verifies that two GPUs with an explicit DDP request keep DDP."""
    assert _resolve_multi_gpu(MultiGpuStrategy.DDP, gpus=(0, 1)) == MultiGpuStrategy.DDP


# _choose_dataloader_worker_count
def test_choose_workers_caps_at_ceiling(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that a single rank on a big box is capped at the automatic worker ceiling."""
    _set_cpu_count(monkeypatch, 32)  # usable = 30, but ceiling is 8.
    assert _choose_dataloader_worker_count(world_size=1) == 8


def test_choose_workers_splits_across_ranks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that the usable cores are divided evenly across the training ranks."""
    _set_cpu_count(monkeypatch, 12)  # usable = 10, split across 4 ranks -> 2 each.
    assert _choose_dataloader_worker_count(world_size=4) == 2


def test_choose_workers_floors_at_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that a machine with fewer cores than the reserve floors the worker count at zero."""
    _set_cpu_count(monkeypatch, 1)  # usable = -1, floored to 0.
    assert _choose_dataloader_worker_count(world_size=1) == 0


def test_choose_workers_handles_unknown_cpu_count(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that when os.cpu_count returns None the fallback of one core is used, yielding zero workers."""
    _set_cpu_count(monkeypatch, None)  # (None or 1) - 2 = -1, floored to 0.
    assert _choose_dataloader_worker_count(world_size=1) == 0


# resolve_optimization_profile
def test_resolve_profile_cpu_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies a CPU run disables CUDA-only optimizations, uses zero workers, and restores the held thread budget."""
    _set_cuda(monkeypatch, 0)  # No CUDA, so even 'auto' cascades to CPU.
    _set_cpu_count(monkeypatch, 10)
    profile = resolve_optimization_profile(device=DeviceType.CPU)
    assert profile.device == "cpu"
    assert profile.gpus == ()
    assert profile.multi_gpu_strategy == MultiGpuStrategy.SINGLE
    assert profile.amp_dtype is None
    assert profile.use_gradient_scaler is False
    assert profile.tf32 is False
    assert profile.cudnn_benchmark is False
    assert profile.pin_memory is False
    assert profile.torch_compile is False
    assert profile.dataloader_workers == 0  # CPU default holds workers at zero.
    assert profile.cpu_threads == 8  # 10 cores minus the 2 reserved.


def test_resolve_profile_cpu_explicit_worker_count(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that an explicit non-negative worker count overrides the CPU zero-worker default."""
    _set_cuda(monkeypatch, 0)
    _set_cpu_count(monkeypatch, 10)
    profile = resolve_optimization_profile(device=DeviceType.CPU, dataloader_workers=4)
    assert profile.dataloader_workers == 4


def test_resolve_profile_cuda_single_gpu_ampere(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies a single Ampere GPU auto-enables bfloat16, TF32, and pinning, and auto-picks the worker count."""
    _set_cuda(monkeypatch, 1)
    _set_capabilities(monkeypatch, {0: (8, 0)})
    _set_cpu_count(monkeypatch, 12)
    profile = resolve_optimization_profile(device=DeviceType.CUDA)
    assert profile.device == "cuda"
    assert profile.gpus == (0,)
    assert profile.multi_gpu_strategy == MultiGpuStrategy.SINGLE
    assert profile.amp_dtype is torch.bfloat16
    assert profile.use_gradient_scaler is False  # bfloat16 needs no gradient scaler.
    assert profile.tf32 is True
    assert profile.pin_memory is True
    assert profile.cudnn_benchmark is False  # 'auto' follows fixed_input_size, which defaults False.
    assert profile.cpu_threads is None  # No CPU thread budget restored on the CUDA path.
    assert profile.dataloader_workers == 8  # Single rank on 12 cores minus reserve, capped at 8.


def test_resolve_profile_cuda_disables_compile_without_triton(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verifies that a CUDA run requesting compilation falls back to eager when Triton is not importable."""
    _set_cuda(monkeypatch, 1)
    _set_capabilities(monkeypatch, {0: (8, 0)})
    _set_cpu_count(monkeypatch, 12)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None if name == "triton" else object())
    profile = resolve_optimization_profile(device=DeviceType.CUDA, torch_compile=Toggle.ON)
    assert profile.torch_compile is False
    assert "Triton" in capsys.readouterr().err


def test_resolve_profile_cuda_keeps_compile_with_triton(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that a CUDA run requesting compilation keeps it when Triton is importable."""
    _set_cuda(monkeypatch, 1)
    _set_capabilities(monkeypatch, {0: (8, 0)})
    _set_cpu_count(monkeypatch, 12)
    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: object())
    profile = resolve_optimization_profile(device=DeviceType.CUDA, torch_compile=Toggle.ON)
    assert profile.torch_compile is True


def test_resolve_profile_cuda_fp16_enables_gradient_scaler(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that forced float16 on CUDA sets the autocast dtype and requires the gradient scaler."""
    _set_cuda(monkeypatch, 1)
    _set_capabilities(monkeypatch, {0: (8, 0)})
    _set_cpu_count(monkeypatch, 12)
    profile = resolve_optimization_profile(device=DeviceType.CUDA, amp=AmpMode.FP16)
    assert profile.amp_dtype is torch.float16
    assert profile.use_gradient_scaler is True


def test_resolve_profile_cuda_ddp_world_size(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies two GPUs under DDP split the worker budget by the world size rather than treating it as one rank."""
    _set_cuda(monkeypatch, 2)
    _set_capabilities(monkeypatch, {0: (8, 0), 1: (8, 0)})
    _set_cpu_count(monkeypatch, 12)
    profile = resolve_optimization_profile(device=DeviceType.CUDA, gpus=(0, 1), multi_gpu=MultiGpuStrategy.DDP)
    assert profile.multi_gpu_strategy == MultiGpuStrategy.DDP
    assert profile.gpus == (0, 1)
    # World size 2 -> usable 10 // 2 = 5 workers per rank.
    assert profile.dataloader_workers == 5


def test_resolve_profile_dp_with_amp_warns_and_disables(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verifies that DataParallel with mixed precision requested is warned about and reported as full float32."""
    _set_cuda(monkeypatch, 2)
    _set_capabilities(monkeypatch, {0: (8, 0), 1: (8, 0)})
    _set_cpu_count(monkeypatch, 12)
    profile = resolve_optimization_profile(
        device=DeviceType.CUDA, gpus=(0, 1), multi_gpu=MultiGpuStrategy.DP, amp=AmpMode.AUTO
    )
    assert profile.multi_gpu_strategy == MultiGpuStrategy.DP
    assert profile.amp_dtype is None  # Disabled because autocast cannot reach DataParallel replica threads.
    assert profile.use_gradient_scaler is False
    assert "Mixed precision has no effect under DataParallel" in capsys.readouterr().err
    # Under DP the world size is one, so a single rank claims the full worker budget (capped at 8).
    assert profile.dataloader_workers == 8


def test_resolve_profile_cudnn_benchmark_forced_on_without_fixed_size_warns(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verifies that forcing the cuDNN autotuner on without a detected fixed input size enables it but warns."""
    _set_cuda(monkeypatch, 1)
    _set_capabilities(monkeypatch, {0: (8, 0)})
    _set_cpu_count(monkeypatch, 12)
    profile = resolve_optimization_profile(device=DeviceType.CUDA, cudnn_benchmark=Toggle.ON, fixed_input_size=False)
    assert profile.cudnn_benchmark is True
    assert "cuDNN benchmark was forced on" in capsys.readouterr().err


def test_resolve_profile_cudnn_benchmark_auto_with_fixed_size_is_silent(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verifies that the autotuner's 'auto' default enables it for a fixed input size without a warning."""
    _set_cuda(monkeypatch, 1)
    _set_capabilities(monkeypatch, {0: (8, 0)})
    _set_cpu_count(monkeypatch, 12)
    profile = resolve_optimization_profile(device=DeviceType.CUDA, fixed_input_size=True)
    assert profile.cudnn_benchmark is True
    assert capsys.readouterr().err == ""


# apply_runtime_optimizations
def test_apply_runtime_optimizations_sets_cpu_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies a profile with a CPU thread budget forwards the backend flags and restores the intra-op thread count."""
    forwarded: dict[str, object] = {}
    monkeypatch.setattr(
        optimization,
        "apply_backend_flags",
        lambda *, device, tf32, cudnn_benchmark: forwarded.update(
            device=device, tf32=tf32, cudnn_benchmark=cudnn_benchmark
        ),
    )
    threads: list[int] = []
    monkeypatch.setattr(torch, "set_num_threads", threads.append)
    profile = _profile(device="cpu", tf32=False, cudnn_benchmark=False, cpu_threads=6)
    apply_runtime_optimizations(profile)
    assert forwarded == {"device": "cpu", "tf32": False, "cudnn_benchmark": False}
    assert threads == [6]


def test_apply_runtime_optimizations_skips_thread_count_when_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies a profile without a CPU thread budget applies backend flags but leaves the thread count untouched."""
    forwarded: dict[str, object] = {}
    monkeypatch.setattr(
        optimization,
        "apply_backend_flags",
        lambda *, device, tf32, cudnn_benchmark: forwarded.update(
            device=device, tf32=tf32, cudnn_benchmark=cudnn_benchmark
        ),
    )
    threads: list[int] = []
    monkeypatch.setattr(torch, "set_num_threads", threads.append)
    # Distinct tf32/cudnn_benchmark values pin the two flags to their own keyword slots, so a source-side argument
    # swap (forwarding tf32 as cudnn_benchmark or vice versa) is caught rather than masked by equal booleans.
    profile = _profile(device="cuda", tf32=True, cudnn_benchmark=False, cpu_threads=None)
    apply_runtime_optimizations(profile)
    assert forwarded == {"device": "cuda", "tf32": True, "cudnn_benchmark": False}
    assert threads == []  # None thread budget means the intra-op thread count is left untouched.
