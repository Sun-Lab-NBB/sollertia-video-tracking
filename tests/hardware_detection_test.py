"""Tests for the device, capability, and mixed-precision detection helpers shared by the optimizers."""

import torch
import pytest

from sollertia_video_tracking.hardware import detection
from sollertia_video_tracking.hardware.detection import (
    Toggle,
    AmpMode,
    DeviceType,
    warn,
    resolve_toggle,
    precision_label,
    supports_ampere,
    resolve_amp_dtype,
    apply_backend_flags,
    resolve_target_device,
)


def _set_cuda(monkeypatch: pytest.MonkeyPatch, count: int) -> None:
    """Fakes the visible CUDA device count so the resolvers see a deterministic machine."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: count > 0)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: count)


def _set_capabilities(monkeypatch: pytest.MonkeyPatch, caps: dict[int, tuple[int, int]]) -> None:
    """Fakes per-index CUDA compute capabilities for the Ampere check."""
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: caps[device])


# --------------------------------------------------------------------------------------------------------------------
# Enums
# --------------------------------------------------------------------------------------------------------------------
def test_enum_string_values() -> None:
    """The tri-state and selection enums are string enums with the documented literal values."""
    assert Toggle.AUTO == "auto"
    assert Toggle.ON == "on"
    assert Toggle.OFF == "off"
    assert AmpMode.AUTO == "auto"
    assert AmpMode.OFF == "off"
    assert AmpMode.BF16 == "bf16"
    assert AmpMode.FP16 == "fp16"
    assert DeviceType.AUTO == "auto"
    assert DeviceType.CPU == "cpu"
    assert DeviceType.MPS == "mps"
    assert DeviceType.CUDA == "cuda"
    assert detection.DEFAULT_RESERVED_CPU_THREADS == 2


# --------------------------------------------------------------------------------------------------------------------
# warn
# --------------------------------------------------------------------------------------------------------------------
def test_warn_writes_prefixed_line_to_stderr(capsys: pytest.CaptureFixture[str]) -> None:
    """The warning helper prepends the WARNING prefix and a trailing newline on stderr."""
    warn("something went sideways")
    captured = capsys.readouterr()
    assert captured.err == "WARNING: something went sideways\n"
    assert captured.out == ""


# --------------------------------------------------------------------------------------------------------------------
# _cuda_device_count
# --------------------------------------------------------------------------------------------------------------------
def test_cuda_device_count_reports_visible_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    """When CUDA is available the raw device count is returned."""
    _set_cuda(monkeypatch, 3)
    assert detection._cuda_device_count() == 3


def test_cuda_device_count_zero_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """When CUDA is unavailable the count is zero and device_count is never consulted."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: (_ for _ in ()).throw(AssertionError("must not call")))
    assert detection._cuda_device_count() == 0


# --------------------------------------------------------------------------------------------------------------------
# supports_ampere
# --------------------------------------------------------------------------------------------------------------------
def test_supports_ampere_empty_list_is_false() -> None:
    """An empty device list short-circuits to False without probing any capability."""
    assert supports_ampere(()) is False


def test_supports_ampere_all_devices_ampere_or_newer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every device at or above the Ampere threshold yields True."""
    _set_capabilities(monkeypatch, {0: (8, 0), 1: (9, 0)})
    assert supports_ampere((0, 1)) is True


def test_supports_ampere_one_pre_ampere_device_is_false(monkeypatch: pytest.MonkeyPatch) -> None:
    """A single pre-Ampere device drops the whole set below the threshold."""
    _set_capabilities(monkeypatch, {0: (8, 0), 1: (7, 5)})
    assert supports_ampere((0, 1)) is False


# --------------------------------------------------------------------------------------------------------------------
# resolve_toggle
# --------------------------------------------------------------------------------------------------------------------
def test_resolve_toggle_on_and_off_ignore_auto_default() -> None:
    """Explicit on/off forces the decision regardless of the capability-detected default."""
    assert resolve_toggle(Toggle.ON, auto=False) is True
    assert resolve_toggle(Toggle.OFF, auto=True) is False


def test_resolve_toggle_auto_uses_detected_default() -> None:
    """The auto value defers to the supplied capability default."""
    assert resolve_toggle(Toggle.AUTO, auto=True) is True
    assert resolve_toggle(Toggle.AUTO, auto=False) is False


# --------------------------------------------------------------------------------------------------------------------
# resolve_target_device
# --------------------------------------------------------------------------------------------------------------------
def test_resolve_device_explicit_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit CPU request returns CPU with no GPU indices."""
    _set_cuda(monkeypatch, 2)  # CUDA present but not requested.
    assert resolve_target_device("cpu", None, role="training") == ("cpu", ())


def test_resolve_device_explicit_mps(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit MPS request returns MPS with no GPU indices."""
    _set_cuda(monkeypatch, 2)
    assert resolve_target_device("MPS", None, role="training") == ("mps", ())


def test_resolve_device_cuda_requested_but_absent_warns_and_falls_back(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An explicit CUDA request on a CPU-only machine warns once and falls back to CPU."""
    _set_cuda(monkeypatch, 0)
    result = resolve_target_device("cuda", None, role="inference")
    assert result == ("cpu", ())
    assert "Falling back to CPU" in capsys.readouterr().err


def test_resolve_device_auto_absent_is_silent_cpu(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Automatic selection (None -> auto) on a CPU-only machine falls back silently."""
    _set_cuda(monkeypatch, 0)
    result = resolve_target_device(None, None, role="training")
    assert result == ("cpu", ())
    assert capsys.readouterr().err == ""


def test_resolve_device_explicit_gpu_indices(monkeypatch: pytest.MonkeyPatch) -> None:
    """Valid explicit GPU indices are honored exactly on the CUDA path."""
    _set_cuda(monkeypatch, 4)
    assert resolve_target_device("cuda", (0, 2), role="inference") == ("cuda", (0, 2))


def test_resolve_device_explicit_index_out_of_range_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit GPU index at or above the device count raises."""
    _set_cuda(monkeypatch, 2)
    with pytest.raises(ValueError, match="visible device count 2"):
        resolve_target_device("cuda", (5,), role="training")


def test_resolve_device_explicit_negative_index_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A negative explicit GPU index hits the other half of the range check and raises, naming the offending index."""
    _set_cuda(monkeypatch, 2)
    with pytest.raises(ValueError, match="but got -1"):
        resolve_target_device("cuda", (-1,), role="training")


def test_resolve_device_cuda_colon_index(monkeypatch: pytest.MonkeyPatch) -> None:
    """A cuda:N request selects that single GPU index."""
    _set_cuda(monkeypatch, 3)
    assert resolve_target_device("cuda:1", None, role="inference") == ("cuda", (1,))


def test_resolve_device_cuda_colon_index_out_of_range_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A cuda:N request beyond the device count raises."""
    _set_cuda(monkeypatch, 2)
    with pytest.raises(ValueError, match="device 'cuda:5'"):
        resolve_target_device("cuda:5", None, role="training")


def test_resolve_device_cuda_colon_negative_index_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A negative cuda:N index exercises the other half of the range check and raises, echoing the request."""
    _set_cuda(monkeypatch, 2)
    with pytest.raises(ValueError, match="device 'cuda:-1'"):
        resolve_target_device("cuda:-1", None, role="training")


def test_resolve_device_auto_all_gpus(monkeypatch: pytest.MonkeyPatch) -> None:
    """With default_all_gpus the bare cuda path spreads across every visible GPU."""
    _set_cuda(monkeypatch, 3)
    assert resolve_target_device("cuda", None, role="inference", default_all_gpus=True) == ("cuda", (0, 1, 2))


def test_resolve_device_auto_first_gpu_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without default_all_gpus the bare cuda path selects only the first GPU."""
    _set_cuda(monkeypatch, 3)
    assert resolve_target_device("cuda", None, role="training", default_all_gpus=False) == ("cuda", (0,))


def test_resolve_device_unknown_device_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A device string outside the known set raises a descriptive error naming the role."""
    _set_cuda(monkeypatch, 1)
    with pytest.raises(ValueError, match="Unable to resolve the training device"):
        resolve_target_device("tpu", None, role="training")


# --------------------------------------------------------------------------------------------------------------------
# resolve_amp_dtype
# --------------------------------------------------------------------------------------------------------------------
def test_amp_off_disables_precision() -> None:
    """The off mode returns None irrespective of the device."""
    assert resolve_amp_dtype(AmpMode.OFF, "cuda", (0,)) is None


def test_amp_auto_cuda_ampere_uses_bfloat16(monkeypatch: pytest.MonkeyPatch) -> None:
    """Auto enables bfloat16 on an Ampere-or-newer CUDA device."""
    _set_capabilities(monkeypatch, {0: (8, 0)})
    assert resolve_amp_dtype(AmpMode.AUTO, "cuda", (0,)) is torch.bfloat16


def test_amp_auto_cpu_stays_fp32() -> None:
    """Auto leaves the CPU at full float32 (no capability probe on the non-CUDA path)."""
    assert resolve_amp_dtype(AmpMode.AUTO, "cpu", ()) is None


def test_amp_bf16_on_mps_warns_and_disables(capsys: pytest.CaptureFixture[str]) -> None:
    """Forced bfloat16 on MPS is disabled with a warning."""
    assert resolve_amp_dtype(AmpMode.BF16, "mps", ()) is None
    assert "unreliable on MPS" in capsys.readouterr().err


def test_amp_bf16_on_pre_ampere_cuda_warns_but_keeps_bfloat16(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Forced bfloat16 on a pre-Ampere GPU warns yet still returns bfloat16."""
    _set_capabilities(monkeypatch, {0: (7, 5)})
    assert resolve_amp_dtype(AmpMode.BF16, "cuda", (0,)) is torch.bfloat16
    assert "lacks native bfloat16 support" in capsys.readouterr().err


def test_amp_bf16_on_cpu_returns_bfloat16_without_warning(capsys: pytest.CaptureFixture[str]) -> None:
    """Forced bfloat16 on a non-MPS, non-CUDA device returns bfloat16 without a capability probe."""
    assert resolve_amp_dtype(AmpMode.BF16, "cpu", ()) is torch.bfloat16
    assert capsys.readouterr().err == ""


def test_amp_fp16_off_cuda_warns_and_disables(capsys: pytest.CaptureFixture[str]) -> None:
    """Forced float16 off CUDA is disabled with a warning."""
    assert resolve_amp_dtype(AmpMode.FP16, "cpu", ()) is None
    assert "only supported on CUDA" in capsys.readouterr().err


def test_amp_fp16_on_cuda_uses_float16() -> None:
    """Forced float16 on CUDA returns float16."""
    assert resolve_amp_dtype(AmpMode.FP16, "cuda", (0,)) is torch.float16


# --------------------------------------------------------------------------------------------------------------------
# apply_backend_flags
# --------------------------------------------------------------------------------------------------------------------
def _snapshot_backend_flags() -> tuple[bool, bool, bool, bool]:
    """Captures the process-global CUDA backend flags so a test can restore them afterward."""
    return (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
    )


def _restore_backend_flags(snapshot: tuple[bool, bool, bool, bool]) -> None:
    """Restores previously captured CUDA backend flags."""
    torch.backends.cuda.matmul.allow_tf32 = snapshot[0]
    torch.backends.cudnn.allow_tf32 = snapshot[1]
    torch.backends.cudnn.benchmark = snapshot[2]
    torch.backends.cudnn.deterministic = snapshot[3]


def test_apply_backend_flags_non_cuda_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    """On a non-CUDA device the function returns immediately without touching any global flag."""
    recorded: list[str] = []
    monkeypatch.setattr(torch, "set_float32_matmul_precision", recorded.append)
    snapshot = _snapshot_backend_flags()
    try:
        apply_backend_flags(device="cpu", tf32=True, cudnn_benchmark=True)
        # The matmul-precision setter is the only observable side effect on the CUDA path; it must stay untouched.
        assert recorded == []
        assert _snapshot_backend_flags() == snapshot
    finally:
        _restore_backend_flags(snapshot)


def test_apply_backend_flags_cuda_enables_all(monkeypatch: pytest.MonkeyPatch) -> None:
    """On CUDA with both toggles on, TF32, the matmul-precision hint, and the cuDNN autotuner are all enabled."""
    recorded: list[str] = []
    monkeypatch.setattr(torch, "set_float32_matmul_precision", recorded.append)
    snapshot = _snapshot_backend_flags()
    try:
        apply_backend_flags(device="cuda", tf32=True, cudnn_benchmark=True)
        assert torch.backends.cuda.matmul.allow_tf32 is True
        assert torch.backends.cudnn.allow_tf32 is True
        assert recorded == ["high"]
        assert torch.backends.cudnn.benchmark is True
        assert torch.backends.cudnn.deterministic is False
    finally:
        _restore_backend_flags(snapshot)


def test_apply_backend_flags_cuda_both_toggles_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """On CUDA with both toggles off, TF32 is disabled and neither the precision hint nor the autotuner is set."""
    recorded: list[str] = []
    monkeypatch.setattr(torch, "set_float32_matmul_precision", recorded.append)
    snapshot = _snapshot_backend_flags()
    try:
        # Seed the autotuner flags to the OPPOSITE of what the enabled path would set them to (it sets benchmark
        # True and deterministic False). If the off path wrongly ran that block, benchmark would flip to True and
        # deterministic to False, so the flags surviving as seeded is the only way these assertions can pass -
        # seeding them to the enabled values would make the assertions vacuously true.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        apply_backend_flags(device="cuda", tf32=False, cudnn_benchmark=False)
        assert torch.backends.cuda.matmul.allow_tf32 is False
        assert torch.backends.cudnn.allow_tf32 is False
        assert recorded == []
        # cudnn_benchmark is False, so the autotuner block is skipped and the seeded flags survive untouched.
        assert torch.backends.cudnn.benchmark is False
        assert torch.backends.cudnn.deterministic is True
    finally:
        _restore_backend_flags(snapshot)


# --------------------------------------------------------------------------------------------------------------------
# precision_label
# --------------------------------------------------------------------------------------------------------------------
def test_precision_label_fp32_for_none() -> None:
    """A None dtype maps to the fp32 label."""
    assert precision_label(None) == "fp32"


def test_precision_label_strips_torch_prefix() -> None:
    """A concrete autocast dtype is labeled by its bare dtype name."""
    assert precision_label(torch.bfloat16) == "bfloat16"
    assert precision_label(torch.float16) == "float16"
