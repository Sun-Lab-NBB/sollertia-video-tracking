"""Contains tests for the inference optimization-profile resolver and the resolved profile's derived views.

These tests drive the real ``resolve_inference_profile``/``apply_runtime_optimizations`` logic on a headless, GPU-free
box. Monkeypatching covers the capability probes the code consults (``torch.cuda.*``, ``psutil.cpu_count``, and
``os.cpu_count``), the ``torch.set_num_threads`` effector, and the ``apply_backend_flags`` collaborator, whose real
body therefore never runs. No GPU, network, or DLC runtime is touched.
"""

import os

import torch
import psutil

from sollertia_video_tracking.hardware import Toggle, DeviceType
from sollertia_video_tracking.inference import optimization
from sollertia_video_tracking.inference.optimization import (
    _DEFAULT_GPU_PROCESSES,
    _DEFAULT_CPU_THREADS_PER_WORKER,
    InferenceProfile,
    _resolve_gpu_processes,
    _resolve_cpu_parallelism,
    resolve_inference_profile,
    apply_runtime_optimizations,
)


def _profile(**overrides) -> InferenceProfile:
    """Builds an InferenceProfile from an all-enabled CUDA baseline, overriding only the named fields."""
    defaults = {
        "device": "cuda",
        "gpus": (0, 1),
        "gpu_processes": 2,
        "chunks": 1,
        "cpu_workers": 0,
        "cpu_threads_per_worker": None,
        "amp_dtype": torch.bfloat16,
        "tf32": True,
        "cudnn_benchmark": True,
        "channels_last": True,
        "torch_compile": True,
    }
    defaults.update(overrides)
    return InferenceProfile(**defaults)


def _patch_cuda(monkeypatch, *, count, capability) -> None:
    """Makes the hardware probes report ``count`` CUDA devices, each at the given compute capability."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: count)
    # supports_ampere calls get_device_capability(device=index); accept any call.
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda **_kwargs: capability)


# InferenceProfile derived properties and describe()


def test_use_amp_reflects_amp_dtype_presence() -> None:
    """Verifies that use_amp is True when an autocast dtype is set and False when it is None."""
    assert _profile(amp_dtype=torch.bfloat16).use_amp is True
    assert _profile(amp_dtype=None).use_amp is False


def test_on_cuda_true_only_for_cuda_device() -> None:
    """Verifies that on_cuda tracks the base device string exactly."""
    assert _profile(device="cuda").on_cuda is True
    assert _profile(device="cpu").on_cuda is False
    assert _profile(device="mps").on_cuda is False


def test_amp_device_type_maps_each_device() -> None:
    """Verifies that amp_device_type returns 'cuda' for CUDA and the raw device string otherwise."""
    assert _profile(device="cuda").amp_device_type == "cuda"
    assert _profile(device="cpu").amp_device_type == "cpu"
    assert _profile(device="mps").amp_device_type == "mps"


def test_total_workers_cuda_multiplies_gpus_by_processes() -> None:
    """Verifies that on CUDA the worker cap is gpu count x gpu_processes x chunks, exercised here at chunks=1."""
    assert _profile(device="cuda", gpus=(0, 1), gpu_processes=2).total_workers == 4


def test_total_workers_cpu_uses_cpu_worker_count() -> None:
    """Verifies that on CPU the worker cap is cpu_workers x chunks, exercised here at chunks=1."""
    assert _profile(device="cpu", gpus=(), cpu_workers=3).total_workers == 3


def test_total_workers_other_device_is_single() -> None:
    """Verifies that on a non-CUDA, non-CPU device (MPS) the worker cap is chunks alone, exercised here at chunks=1."""
    assert _profile(device="mps", gpus=(), cpu_workers=0).total_workers == 1


def test_describe_cuda_with_all_extras() -> None:
    """Verifies the CUDA describe() lists device, per-gpu processes, precision, worker cap, and each accelerator."""
    profile = _profile(
        device="cuda",
        gpus=(0, 1),
        gpu_processes=2,
        amp_dtype=torch.bfloat16,
        tf32=True,
        cudnn_benchmark=True,
        channels_last=True,
        torch_compile=True,
    )
    assert profile.describe() == (
        "CUDA [0, 1] x2/gpu | bfloat16 | workers=4, tf32+cudnn.benchmark+channels_last+compile"
    )


def test_describe_cpu_without_extras() -> None:
    """Verifies that the CPU branch shows workers x threads and omits the accelerator suffix when nothing is enabled."""
    profile = _profile(
        device="cpu",
        gpus=(),
        cpu_workers=3,
        cpu_threads_per_worker=8,
        amp_dtype=None,
        tf32=False,
        cudnn_benchmark=False,
        channels_last=False,
        torch_compile=False,
    )
    assert profile.describe() == "CPU 3x8t | fp32 | workers=3"


def test_describe_mps_uppercases_device() -> None:
    """Verifies that the fallback branch uppercases the raw device string (MPS) and reports a single worker."""
    profile = _profile(
        device="mps",
        gpus=(),
        cpu_workers=0,
        amp_dtype=None,
        tf32=False,
        cudnn_benchmark=False,
        channels_last=False,
        torch_compile=False,
    )
    assert profile.describe() == "MPS | fp32 | workers=1"


# resolve_inference_profile


def test_resolve_profile_cuda_auto_defaults(monkeypatch) -> None:
    """Verifies the Ampere CUDA auto defaults enable bf16, tf32, and channels-last, one process per GPU."""
    _patch_cuda(monkeypatch, count=2, capability=(8, 0))
    profile = resolve_inference_profile(device=DeviceType.CUDA, fixed_input_size=True)
    assert profile.device == "cuda"
    assert profile.gpus == (0, 1)
    assert profile.gpu_processes == 1
    assert profile.cpu_workers == 0
    assert profile.cpu_threads_per_worker is None
    assert profile.amp_dtype is torch.bfloat16
    assert profile.tf32 is True
    assert profile.cudnn_benchmark is True  # auto follows fixed_input_size=True
    assert profile.channels_last is True
    assert profile.torch_compile is False  # auto default is off (warm-up cost)


def test_resolve_profile_cuda_pre_ampere_disables_tf32_and_amp(monkeypatch) -> None:
    """Verifies a pre-Ampere GPU leaves tf32 and auto-AMP off, and the varying-input default leaves autotuner off."""
    _patch_cuda(monkeypatch, count=1, capability=(7, 5))
    profile = resolve_inference_profile(device=DeviceType.CUDA)
    assert profile.gpus == (0,)
    assert profile.tf32 is False
    assert profile.amp_dtype is None
    assert profile.cudnn_benchmark is False  # fixed_input_size defaults False -> auto off


def test_resolve_profile_warns_on_forced_benchmark_without_fixed_size(monkeypatch, capsys) -> None:
    """Verifies that forcing the cuDNN autotuner on without a fixed input size honors the flag but emits a warning."""
    _patch_cuda(monkeypatch, count=1, capability=(8, 0))
    profile = resolve_inference_profile(device=DeviceType.CUDA, cudnn_benchmark=Toggle.ON, fixed_input_size=False)
    assert profile.cudnn_benchmark is True
    assert "cuDNN benchmark was forced on" in capsys.readouterr().err


def test_resolve_profile_cpu_forces_cuda_flags_off_and_sizes_workers(monkeypatch) -> None:
    """Verifies the CPU path forces CUDA-only flags off and sizes the worker/thread budget from the physical cores."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(psutil, "cpu_count", lambda **_kwargs: 32)
    profile = resolve_inference_profile(device=DeviceType.CPU)
    assert profile.device == "cpu"
    assert profile.gpus == ()
    assert profile.gpu_processes == 0
    assert profile.cpu_workers == 3  # 30 usable // 8 threads
    assert profile.cpu_threads_per_worker == 8
    assert profile.tf32 is False
    assert profile.cudnn_benchmark is False
    assert profile.channels_last is False
    assert profile.amp_dtype is None


def test_resolve_profile_mps(monkeypatch) -> None:
    """Verifies that an MPS request yields no GPU/CPU parallelism fields and a single-worker profile."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    profile = resolve_inference_profile(device=DeviceType.MPS)
    assert profile.device == "mps"
    assert profile.gpus == ()
    assert profile.gpu_processes == 0
    assert profile.cpu_workers == 0
    assert profile.cpu_threads_per_worker is None
    assert profile.amp_dtype is None
    assert profile.total_workers == 1
    assert profile.amp_device_type == "mps"


# apply_runtime_optimizations


def test_apply_runtime_optimizations_cpu_restores_thread_count(monkeypatch) -> None:
    """Verifies that a CPU profile forwards its backend flags and restores the worker's intra-op thread count."""
    backend_calls = {}
    monkeypatch.setattr(optimization, "apply_backend_flags", lambda **kwargs: backend_calls.update(kwargs))
    thread_calls = []
    monkeypatch.setattr(torch, "set_num_threads", thread_calls.append)
    profile = _profile(
        device="cpu",
        gpus=(),
        gpu_processes=0,
        cpu_workers=3,
        cpu_threads_per_worker=6,
        amp_dtype=None,
        tf32=False,
        cudnn_benchmark=False,
        channels_last=False,
        torch_compile=False,
    )
    apply_runtime_optimizations(profile)
    assert backend_calls == {"device": "cpu", "tf32": False, "cudnn_benchmark": False}
    assert thread_calls == [6]


def test_apply_runtime_optimizations_skips_thread_count_when_none(monkeypatch) -> None:
    """Verifies a profile without a CPU thread budget (e.g. CUDA) applies backend flags but never sets thread count."""
    backend_calls = {}
    monkeypatch.setattr(optimization, "apply_backend_flags", lambda **kwargs: backend_calls.update(kwargs))
    thread_calls = []
    monkeypatch.setattr(torch, "set_num_threads", thread_calls.append)
    profile = _profile(device="cuda", cpu_threads_per_worker=None, tf32=True, cudnn_benchmark=True)
    apply_runtime_optimizations(profile)
    assert backend_calls == {"device": "cuda", "tf32": True, "cudnn_benchmark": True}
    assert thread_calls == []


# _resolve_gpu_processes


def test_resolve_gpu_processes_honors_explicit_value() -> None:
    """Verifies that an explicit per-device process count of one or more is returned unchanged."""
    assert _resolve_gpu_processes(gpu_processes=4) == 4
    assert _resolve_gpu_processes(gpu_processes=1) == 1


def test_resolve_gpu_processes_defaults_for_non_positive() -> None:
    """Verifies that a non-positive request (-1 or 0) collapses to the one-video-per-GPU default."""
    assert _resolve_gpu_processes(gpu_processes=-1) == _DEFAULT_GPU_PROCESSES
    assert _resolve_gpu_processes(gpu_processes=0) == _DEFAULT_GPU_PROCESSES


# _resolve_cpu_parallelism


def test_cpu_parallelism_explicit_threads_derive_workers(monkeypatch) -> None:
    """Verifies an explicit thread budget fixes the thread count and derives the worker count from usable cores."""
    monkeypatch.setattr(psutil, "cpu_count", lambda **_kwargs: 32)
    # 32 physical - 2 reserved = 30 usable; 30 // 4 = 7 workers of 4 threads each.
    workers, threads = _resolve_cpu_parallelism(cpu_workers=-1, cpu_threads_per_worker=4)
    assert (workers, threads) == (7, 4)


def test_cpu_parallelism_explicit_workers_split_usable_cores(monkeypatch) -> None:
    """Verifies an explicit worker count with an automatic thread budget splits the usable cores across workers."""
    monkeypatch.setattr(psutil, "cpu_count", lambda **_kwargs: 32)
    # 30 usable // 5 requested workers = 6 threads per worker.
    workers, threads = _resolve_cpu_parallelism(cpu_workers=5, cpu_threads_per_worker=-1)
    assert (workers, threads) == (5, 6)


def test_cpu_parallelism_full_auto_uses_default_thread_block(monkeypatch) -> None:
    """Verifies fully automatic sizing uses the default per-worker thread block and fills usable cores with workers."""
    monkeypatch.setattr(psutil, "cpu_count", lambda **_kwargs: 32)
    # threads = min(8 default, 30 usable) = 8; workers = 30 // 8 = 3.
    workers, threads = _resolve_cpu_parallelism(cpu_workers=-1, cpu_threads_per_worker=-1)
    assert threads == _DEFAULT_CPU_THREADS_PER_WORKER
    assert (workers, threads) == (3, 8)


def test_cpu_parallelism_tiny_machine_clamps_to_single_worker(monkeypatch) -> None:
    """Verifies a machine with fewer usable cores than the default block clamps thread and worker counts to min one."""
    monkeypatch.setattr(psutil, "cpu_count", lambda **_kwargs: 4)
    # 4 physical - 2 reserved = 2 usable; threads = min(8, 2) = 2; workers = 2 // 2 = 1.
    workers, threads = _resolve_cpu_parallelism(cpu_workers=-1, cpu_threads_per_worker=-1)
    assert (workers, threads) == (1, 2)


def test_cpu_parallelism_falls_back_to_os_cpu_count(monkeypatch) -> None:
    """Verifies that when psutil cannot report physical cores the code falls back to os.cpu_count for the topology."""
    monkeypatch.setattr(psutil, "cpu_count", lambda **_kwargs: None)
    monkeypatch.setattr(os, "cpu_count", lambda: 10)
    # os.cpu_count() 10 - 2 reserved = 8 usable; threads = min(8, 8) = 8; workers = 8 // 8 = 1.
    workers, threads = _resolve_cpu_parallelism(cpu_workers=-1, cpu_threads_per_worker=-1)
    assert (workers, threads) == (1, 8)
