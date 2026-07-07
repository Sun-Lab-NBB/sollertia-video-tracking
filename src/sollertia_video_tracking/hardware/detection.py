"""Provides the device, capability, and mixed-precision detection shared by the training and inference optimizers."""

import sys
from typing import Literal

import torch

type Toggle = Literal["auto", "on", "off"]
"""The tri-state control for one optimization: use the capability-detected default, force it on, or force it off."""

type AmpMode = Literal["auto", "off", "bf16", "fp16"]
"""The automatic-mixed-precision selection: capability-detected default, disabled, or a forced compute dtype."""

DEFAULT_RESERVED_CPU_THREADS: int = 2
"""The number of CPU cores held back from the automatic worker and thread budgets so other work stays responsive."""

_AMPERE_CAPABILITY: tuple[int, int] = (8, 0)
"""The minimum CUDA compute capability (Ampere) that provides TF32 and native bfloat16 tensor-core acceleration."""


def warn(message: str) -> None:
    """Writes a non-fatal warning to the standard error stream.

    Args:
        message: The warning text to emit, without the ``WARNING:`` prefix or trailing newline.
    """
    sys.stderr.write(f"WARNING: {message}\n")
    sys.stderr.flush()


def cuda_device_count() -> int:
    """Returns the number of visible CUDA devices, or zero when CUDA is unavailable.

    Returns:
        The count of CUDA devices reported by the active PyTorch build.
    """
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


def supports_ampere(gpus: tuple[int, ...]) -> bool:
    """Determines whether every listed CUDA device reaches Ampere compute capability or newer.

    Ampere is the threshold for both TF32 matmul/convolution acceleration and native bfloat16 tensor cores, so the
    same check gates both optimizations.

    Args:
        gpus: The CUDA device indices to check.

    Returns:
        True when all listed devices report at least Ampere compute capability, False otherwise.
    """
    return len(gpus) > 0 and all(torch.cuda.get_device_capability(device=index) >= _AMPERE_CAPABILITY for index in gpus)


def resolve_toggle(value: Toggle, *, auto: bool) -> bool:
    """Resolves a tri-state toggle to a boolean, using the given capability-detected default for ``"auto"``.

    Args:
        value: The requested tri-state value.
        auto: The capability-detected default applied when the value is ``"auto"``.

    Returns:
        The resolved boolean decision.
    """
    if value == "on":
        return True
    if value == "off":
        return False
    return auto


def resolve_target_device(
    device: str | None, gpus: tuple[int, ...] | None, *, role: str
) -> tuple[str, tuple[int, ...]]:
    """Reconciles the requested device and GPU indices with the available hardware.

    The device selection cascades ``cuda`` -> ``cpu`` when no CUDA device is visible, so the same call works unchanged
    on a GPU server or a CPU-only server.

    Args:
        device: The requested device (``"auto"``, ``"cpu"``, ``"mps"``, ``"cuda"``, or ``"cuda:N"``), or None for
            automatic selection.
        gpus: The explicitly requested CUDA device indices, or None to select them automatically.
        role: The run role named in error messages, for example ``"training"`` or ``"inference"``.

    Returns:
        A tuple of the resolved base device type and the tuple of CUDA indices to use.

    Raises:
        ValueError: When an explicitly requested CUDA index is not present on the machine, or when the requested device
            is not one of ``"auto"``, ``"cpu"``, ``"mps"``, ``"cuda"``, or ``"cuda:N"``.
    """
    request = (device or "auto").lower()
    available = cuda_device_count()

    if request == "cpu":
        return "cpu", ()
    if request == "mps":
        return "mps", ()

    if request.startswith("cuda") or request == "auto":
        if available == 0:
            if request != "auto":
                warn(f"Requested device '{request}' but no CUDA device is available. Falling back to CPU.")
            return "cpu", ()
        if gpus is not None:
            for index in gpus:
                if index < 0 or index >= available:
                    message = (
                        f"Unable to select GPUs using the requested indices. Expected each index below the visible "
                        f"device count {available}, but got {index}."
                    )
                    raise ValueError(message)
            return "cuda", tuple(gpus)
        if ":" in request:
            index = int(request.split(":", 1)[1])
            if index < 0 or index >= available:
                message = (
                    f"Unable to select a GPU using device '{request}'. Expected an index below the visible device "
                    f"count {available}, but got {index}."
                )
                raise ValueError(message)
            return "cuda", (index,)
        return "cuda", tuple(range(available))

    message = (
        f"Unable to resolve the {role} device. Expected 'auto', 'cpu', 'mps', 'cuda', or 'cuda:N', but got '{request}'."
    )
    raise ValueError(message)


def resolve_amp_dtype(amp: AmpMode, device: str, gpus: tuple[int, ...]) -> torch.dtype | None:
    """Reconciles the requested mixed-precision mode with the device and its capabilities into an autocast dtype.

    A forced dtype the device cannot support (bfloat16 on MPS, float16 off CUDA) is disabled with a warning rather
    than a silent refusal. The ``"auto"`` default enables bfloat16 only where it is natively fast, so it stays close
    to stock float32 behavior; on CPU the benefit is chip-dependent and left as an explicit opt-in.

    Args:
        amp: The requested mixed-precision mode.
        device: The resolved base device type.
        gpus: The resolved CUDA device indices.

    Returns:
        The autocast dtype to use, or None when mixed precision is disabled. Callers that train derive their own
        gradient-scaler requirement from a ``torch.float16`` result.
    """
    if amp == "off":
        return None
    if amp == "auto":
        if device == "cuda" and supports_ampere(gpus):
            return torch.bfloat16
        return None
    if amp == "bf16":
        if device == "mps":
            warn("bfloat16 autocast is unreliable on MPS. Disabling mixed precision.")
            return None
        if device == "cuda" and not supports_ampere(gpus):
            warn(
                "bfloat16 was requested but the selected GPU lacks native bfloat16 support (pre-Ampere); it may run "
                "slowly. Consider '--amp fp16' instead."
            )
        return torch.bfloat16
    # The only remaining mode is float16, which is a CUDA-only precision.
    if device != "cuda":
        warn(f"float16 autocast is only supported on CUDA, not '{device}'. Disabling mixed precision.")
        return None
    return torch.float16


def apply_backend_flags(*, device: str, tf32: bool, cudnn_benchmark: bool) -> None:
    """Applies the process-global CUDA backend flags shared by training and inference workers.

    TF32 and the cuDNN autotuner are process-global CUDA backend flags, so each worker process must apply them itself
    before its first forward pass. This is a no-op on non-CUDA devices.

    Args:
        device: The resolved base device type the worker runs on.
        tf32: Whether TF32 acceleration is enabled for float32 matmuls and convolutions.
        cudnn_benchmark: Whether the cuDNN convolution autotuner is enabled.
    """
    if device != "cuda":
        return
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    if tf32:
        torch.set_float32_matmul_precision("high")
    if cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def precision_label(amp_dtype: torch.dtype | None) -> str:
    """Returns the human-readable precision label for an autocast dtype.

    Args:
        amp_dtype: The autocast compute dtype, or None for full float32 precision.

    Returns:
        The precision label (``"bfloat16"``, ``"float16"``, or ``"fp32"``).
    """
    return "fp32" if amp_dtype is None else str(amp_dtype).removeprefix("torch.")
