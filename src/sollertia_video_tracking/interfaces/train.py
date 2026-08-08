"""Provides the ``slvt train`` command that trains a DeepLabCut shuffle with mixed precision, DDP, and a monitor."""

from pathlib import Path

import click

from ..hardware import warn
from ..training import (
    Toggle,
    AmpMode,
    DeviceType,
    MultiGpuStrategy,
    TrainingFailedError,
    TrainingInterruptedError,
    train_model,
    detect_fixed_input_size,
    resolve_optimization_profile,
)

_CONTEXT_SETTINGS: dict[str, int] = {"max_content_width": 120}
"""Widens displayed Click help messages to 120 columns so option descriptions wrap consistently."""


@click.command("train", context_settings=_CONTEXT_SETTINGS)
@click.option(
    "-cfg",
    "--config-path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="The path to the DeepLabCut project's config.yaml that owns the shuffle to train.",
)
@click.option(
    "-s",
    "--shuffle",
    type=click.IntRange(min=1),
    default=1,
    show_default=True,
    help="The index of the shuffle to train. A shuffle pairs a train/test split with a model architecture, both fixed "
    "when it was created by 'slvt prepare'. Change it to train a different prepared shuffle.",
)
@click.option(
    "-e",
    "--epochs",
    type=click.IntRange(min=0),
    default=None,
    help="The maximum number of passes over the training set for the pose model. Higher values train longer, but may "
    "be necessary to achieve model convergence on complex datasets. Set to 0 to skip pose training and fit only a "
    "top-down shuffle's detector. Omit to use the model's default value.",
)
@click.option(
    "-b",
    "--batch-size",
    type=click.IntRange(min=1),
    default=None,
    help="The number of frames the pose model processes per optimization step. Larger batches use more GPU memory and "
    "can speed up training. Omit to use the model's default value.",
)
@click.option(
    "-se",
    "--save-epochs",
    type=click.IntRange(min=1),
    default=None,
    help="The number of epochs between saved pose-model snapshots. Smaller values checkpoint more often at the cost of "
    "disk space. Omit to use the model's default value.",
)
@click.option(
    "-di",
    "--display-iterations",
    type=click.IntRange(min=1),
    default=None,
    help="The number of training iterations between loss readouts within each epoch. Omit to use the model's default "
    "value.",
)
@click.option(
    "-ms",
    "--maximum-snapshots",
    type=click.IntRange(min=1),
    default=None,
    help="The maximum number of recent snapshots to retain per model. Older snapshots beyond this count are deleted. "
    "Omit to use the model's default value.",
)
@click.option(
    "-sp",
    "--snapshot-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="The path to a pose-model snapshot to resume training from, instead of starting from the shuffle's initial "
    "weights.",
)
@click.option(
    "-dp",
    "--detector-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="The path to a detector snapshot to resume training from. Applies to top-down models only.",
)
@click.option(
    "-dbs",
    "--detector-batch-size",
    type=click.IntRange(min=1),
    default=None,
    help="The number of frames the object detector processes per step, for top-down models. Omit to use the model's "
    "default value.",
)
@click.option(
    "-de",
    "--detector-epochs",
    type=click.IntRange(min=0),
    default=None,
    help="The maximum number of training passes for the object detector, for top-down models. Set to 0 to skip "
    "detector training and fit only the pose model.",
)
@click.option(
    "-dse",
    "--detector-save-epochs",
    type=click.IntRange(min=1),
    default=None,
    help="The number of epochs between saved detector snapshots, for top-down models. Omit to use the model's default "
    "value.",
)
@click.option(
    "-lhw",
    "--load-head-weights/--no-load-head-weights",
    default=True,
    show_default=True,
    help="Determines whether the pose model's head weights are restored when resuming from a snapshot. Disable this "
    "when the bodypart set has changed, since the snapshot's head no longer matches.",
)
@click.option(
    "-d",
    "--device",
    type=click.Choice([device.value for device in DeviceType]),
    default=DeviceType.AUTO.value,
    show_default=True,
    help="The base device to train on. 'auto' selects a CUDA GPU when one is visible and otherwise the CPU. 'cuda' "
    "does the same but warns before falling back to the CPU when none is present. 'cpu' and 'mps' (Apple Metal) force "
    "those devices. Choose specific GPUs with --gpus.",
)
@click.option(
    "-g",
    "--gpus",
    default=None,
    metavar="INDICES",
    help="The comma-separated CUDA device indices to train on. Omit to train on GPU 0. List two or more indices (e.g. "
    "'0,1') to train across them, with the strategy chosen by --multi-gpu. Only applicable when training device is a "
    "CUDA-compatible GPU.",
)
@click.option(
    "-mg",
    "--multi-gpu",
    type=click.Choice([strategy.value for strategy in MultiGpuStrategy if strategy is not MultiGpuStrategy.SINGLE]),
    default=MultiGpuStrategy.AUTO.value,
    show_default=True,
    help="The strategy for training across the GPUs selected with --gpus. 'auto' uses DistributedDataParallel when two "
    "or more GPUs are selected. 'ddp' forces it. 'dp' uses the slower DataParallel. Ignored when a single GPU is used.",
)
@click.option(
    "-a",
    "--amp",
    type=click.Choice([mode.value for mode in AmpMode]),
    default=AmpMode.AUTO.value,
    show_default=True,
    help="The mixed-precision compute mode, which trades some numerical precision for speed and lower memory use. "
    "'auto' enables bfloat16 on Ampere or newer GPUs and stays in float32 elsewhere. 'off' forces float32. 'bf16' "
    "forces bfloat16 (disabled with a warning on MPS) and 'fp16' forces float16 on CUDA (disabled with a warning "
    "elsewhere).",
)
@click.option(
    "-t",
    "--tf32",
    type=click.Choice([toggle.value for toggle in Toggle]),
    default=Toggle.AUTO.value,
    show_default=True,
    help="TF32 acceleration for float32 matmuls and convolutions on CUDA, which speeds up float32 math at slightly "
    "reduced precision. 'auto' enables it on Ampere or newer GPUs. 'on' and 'off' force it. It is a no-op off CUDA.",
)
@click.option(
    "-cb",
    "--cudnn-benchmark",
    type=click.Choice([toggle.value for toggle in Toggle]),
    default=Toggle.AUTO.value,
    show_default=True,
    help="The cuDNN convolution autotuner on CUDA. 'auto' enables it only when the shuffle's training transform is "
    "detected to feed one fixed input size, where it speeds up convolutions. It disables deterministic training and "
    "can slow variable-size augmentation, so it stays off otherwise. 'on' and 'off' force it.",
)
@click.option(
    "-cm",
    "--compile-model",
    type=click.Choice([toggle.value for toggle in Toggle]),
    default=Toggle.AUTO.value,
    show_default=True,
    help="Determines whether the model is compiled with torch.compile for faster steps. 'auto' leaves it off because "
    "its one-time warm-up cost may not amortize. 'on' and 'off' force it.",
)
@click.option(
    "-dw",
    "--dataloader-workers",
    type=click.IntRange(min=-1),
    default=-1,
    show_default=True,
    help="The number of worker processes each training process uses to load and augment data. More workers feed the "
    "GPU faster until the CPU saturates. Set to -1 to derive it from the CPU count, capped at 8 per process, when "
    "training on a GPU. CPU training instead defaults to 0 workers, because the main process performs the compute.",
)
@click.option(
    "-pm",
    "--pin-memory",
    type=click.Choice([toggle.value for toggle in Toggle]),
    default=Toggle.AUTO.value,
    show_default=True,
    help="Determines whether dataloaders pin host memory to speed up host-to-device transfers on CUDA. 'auto' enables "
    "it on CUDA. 'on' and 'off' force it. It has no effect off CUDA.",
)
@click.option(
    "-ev",
    "--evaluate/--no-evaluate",
    default=True,
    show_default=True,
    help="Determines whether the trained snapshot is scored against the labeled frames as a final step, writing the "
    "evaluation feather and its provenance sidecar. Disable to finish at the last snapshot without evaluating.",
)
@click.option(
    "-ebs",
    "--evaluation-batch-size",
    type=click.IntRange(min=1),
    default=1,
    show_default=True,
    help="The number of frames scored per forward pass during the post-training evaluation, which runs in float32 on "
    "a single device. Larger values evaluate faster but use more GPU memory. Raise it on a capable GPU.",
)
@click.option(
    "-ecc",
    "--evaluation-confidence-cutoff",
    type=click.FloatRange(min=0.0, max=1.0),
    default=None,
    help="The confidence cutoff for the evaluation's cutoff-filtered metrics. Predictions the model makes below this "
    "confidence are excluded from the cutoff-filtered error, so it reflects accuracy on only the keypoints the model "
    "is confident about. Omit to fall back to the project config's p-cutoff (0.6 when unset).",
)
@click.option(
    "-p",
    "--progress/--no-progress",
    default=True,
    show_default=True,
    help="Determines whether the aggregate progress bar is shown during training.",
)
def train_command(
    config_path: Path,
    shuffle: int,
    epochs: int | None,
    batch_size: int | None,
    save_epochs: int | None,
    display_iterations: int | None,
    maximum_snapshots: int | None,
    snapshot_path: Path | None,
    detector_path: Path | None,
    detector_batch_size: int | None,
    detector_epochs: int | None,
    detector_save_epochs: int | None,
    device: str,
    gpus: str | None,
    multi_gpu: str,
    amp: str,
    tf32: str,
    cudnn_benchmark: str,
    compile_model: str,
    dataloader_workers: int,
    pin_memory: str,
    evaluation_batch_size: int,
    evaluation_confidence_cutoff: float | None,
    *,
    load_head_weights: bool,
    evaluate: bool,
    progress: bool,
) -> None:
    """Trains a DeepLabCut shuffle with hardware optimizations and a clean progress monitor.

    ``--config-path`` names the DeepLabCut project's config.yaml. The shuffle's model architecture and train/test split
    are fixed when the shuffle is created (see ``slvt prepare``) and this command fits that shuffle. Every optimization
    is exposed as a flag: automatic defaults are chosen for the detected hardware and never run slower than stock
    DeepLabCut, while explicit flags allow further tuning for known hardware. Training runs on a single GPU (index 0) by
    default, since multi-GPU training is often slower for DeepLabCut workloads. Select two or more GPUs with ``--gpus``
    to train across them, as a DistributedDataParallel process group (``--multi-gpu ddp``, the default for multiple
    GPUs) or the slower DataParallel (``--multi-gpu dp``). Single-process training also covers the CPU and Apple MPS.

    Training always runs in worker processes. A worker that fails, crashes, or is killed by the out-of-memory killer
    ends the command with an error naming the cause, the model folder, and the training log. A worker that raised
    quotes its traceback, and one that died without unwinding quotes the log's tail. An interrupted run exits with
    status 130 and leaves its snapshots intact.
    """
    try:
        gpu_indices = tuple(int(part) for part in gpus.split(",")) if gpus else None
    except ValueError as error:
        message = (
            f"Unable to parse the --gpus value. Expected comma-separated GPU indices such as '0,1', but got '{gpus}'."
        )
        raise click.ClickException(message=message) from error

    # Detects whether the shuffle's training transform feeds the network one fixed input size so the cuDNN autotuner's
    # 'auto' default can enable itself only when it pays off.
    fixed_input_size = detect_fixed_input_size(config=config_path, shuffle=shuffle)

    try:
        profile = resolve_optimization_profile(
            device=DeviceType(device),
            gpus=gpu_indices,
            multi_gpu=MultiGpuStrategy(multi_gpu),
            amp=AmpMode(amp),
            tf32=Toggle(tf32),
            cudnn_benchmark=Toggle(cudnn_benchmark),
            torch_compile=Toggle(compile_model),
            dataloader_workers=dataloader_workers,
            pin_memory=Toggle(pin_memory),
            fixed_input_size=fixed_input_size,
        )
        summary = train_model(
            config=config_path,
            profile=profile,
            shuffle=shuffle,
            epochs=epochs,
            batch_size=batch_size,
            save_epochs=save_epochs,
            display_iterations=display_iterations,
            snapshot_path=snapshot_path,
            detector_path=detector_path,
            detector_batch_size=detector_batch_size,
            detector_epochs=detector_epochs,
            detector_save_epochs=detector_save_epochs,
            maximum_snapshots_to_keep=maximum_snapshots,
            load_head_weights=load_head_weights,
            evaluate=evaluate,
            evaluation_batch_size=evaluation_batch_size,
            evaluation_confidence_cutoff=evaluation_confidence_cutoff,
            display_progress=progress,
        )
    except TrainingInterruptedError as error:
        warn(str(error))
        click.get_current_context().exit(130)
    # EOFError is funneled here because Click reduces it to a bare 'Aborted!' that reads exactly like an operator
    # interrupt, which hides a manager process that failed to start.
    except (TrainingFailedError, ValueError, FileNotFoundError, EOFError) as error:
        raise click.ClickException(message=str(error)) from error

    click.echo(message=summary.describe())
    if summary.evaluation_error is not None:
        message = (
            f"Training completed and its snapshots are intact, but the post-training evaluation failed, so the "
            f"evaluation feather and its provenance sidecar were not written. The full traceback was appended to "
            f"{summary.model_folder / 'train.txt'}. Lower --evaluation-batch-size and re-run, or pass --no-evaluate "
            f"to accept the run without an evaluation."
        )
        raise click.ClickException(message=message)
