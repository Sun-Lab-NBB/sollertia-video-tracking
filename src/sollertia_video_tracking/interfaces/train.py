"""Provides the ``slvt train`` command that trains a DeepLabCut shuffle with mixed precision, DDP, and a monitor."""

from typing import Literal
from pathlib import Path

import click

from ..training import Toggle, AmpMode, train_model, detect_fixed_input_size, resolve_optimization_profile

_CONTEXT_SETTINGS: dict[str, int] = {"max_content_width": 120}
"""Ensures that displayed Click help messages are formatted according to the lab standard."""


@click.command("train", context_settings=_CONTEXT_SETTINGS)
@click.option(
    "-cfg",
    "--config-path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="The path to the DeepLabCut project's config.yaml whose shuffle to train.",
)
@click.option("-s", "--shuffle", default=1, show_default=True, help="The shuffle index to train.")
@click.option(
    "-mp",
    "--model-prefix",
    default="",
    help="The model subdirectory prefix, matching the one chosen when the shuffle was created.",
)
@click.option(
    "-e",
    "--epochs",
    type=int,
    default=None,
    help="The maximum number of pose-model training epochs. Omit to use the shuffle's configured value.",
)
@click.option(
    "-b",
    "--batch-size",
    type=int,
    default=None,
    help="The pose-model batch size. Omit to use the configured value; larger batches use more GPU memory.",
)
@click.option("-se", "--save-epochs", type=int, default=None, help="The number of epochs between saved snapshots.")
@click.option(
    "-di",
    "--display-iterations",
    type=int,
    default=None,
    help="The number of iterations between loss updates shown during each epoch.",
)
@click.option(
    "-ms",
    "--maximum-snapshots",
    type=int,
    default=None,
    help="The maximum number of snapshots to keep per model. Omit to use the configured value.",
)
@click.option(
    "-sp",
    "--snapshot-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="The pose snapshot to resume training from.",
)
@click.option(
    "-dp",
    "--detector-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="The detector snapshot to resume training from, for top-down models.",
)
@click.option(
    "-dbs",
    "--detector-batch-size",
    type=int,
    default=None,
    help="The detector batch size, for top-down models. Omit to use the configured value.",
)
@click.option(
    "-de",
    "--detector-epochs",
    type=int,
    default=None,
    help="The maximum number of detector epochs, for top-down models. Set to 0 to skip detector training.",
)
@click.option(
    "-dse",
    "--detector-save-epochs",
    type=int,
    default=None,
    help="The number of epochs between detector snapshots, for top-down models.",
)
@click.option(
    "-lhw",
    "--load-head-weights/--no-load-head-weights",
    default=True,
    show_default=True,
    help="Determines whether to restore head weights when resuming a pose model. Disable when changing bodyparts.",
)
@click.option(
    "-d",
    "--device",
    default="auto",
    show_default=True,
    help="The device to train on: 'auto', 'cpu', 'mps', 'cuda', or 'cuda:N'. 'auto' selects every visible GPU.",
)
@click.option(
    "-g",
    "--gpus",
    default=None,
    metavar="INDICES",
    help="The comma-separated CUDA device indices to use (e.g. '0,1'). Omit to use every visible GPU.",
)
@click.option(
    "-mg",
    "--multi-gpu",
    type=click.Choice(["auto", "ddp", "dp", "single"]),
    default="auto",
    show_default=True,
    help="The multi-GPU strategy: DistributedDataParallel, the slower DataParallel, or a single device. 'auto' uses "
    "DDP when two or more GPUs are selected.",
)
@click.option(
    "-a",
    "--amp",
    type=click.Choice(["auto", "off", "bf16", "fp16"]),
    default="auto",
    show_default=True,
    help="The mixed-precision mode. 'auto' enables bfloat16 on Ampere or newer GPUs. Force 'fp16' for pre-Ampere "
    "cards with float16 tensor cores.",
)
@click.option(
    "-t",
    "--tf32",
    type=click.Choice(["auto", "on", "off"]),
    default="auto",
    show_default=True,
    help="TF32 acceleration for float32 matmuls and convolutions (CUDA only). 'auto' enables it on Ampere or newer.",
)
@click.option(
    "-cb",
    "--cudnn-benchmark",
    type=click.Choice(["auto", "on", "off"]),
    default="auto",
    show_default=True,
    help="The cuDNN convolution autotuner. 'auto' enables it when the shuffle's training transform is detected to use "
    "a single fixed input size, where it speeds up convolutions; it disables deterministic training and can slow "
    "variable-size augmentation, so it stays off otherwise.",
)
@click.option(
    "-cm",
    "--compile-model",
    type=click.Choice(["auto", "on", "off"]),
    default="auto",
    show_default=True,
    help="Whether to compile the model for faster steps. Off by default because of its warm-up cost.",
)
@click.option(
    "-dw",
    "--dataloader-workers",
    type=int,
    default=-1,
    show_default=True,
    help="The number of dataloader workers per process. Set to -1 to choose automatically from the CPU count.",
)
@click.option(
    "-pm",
    "--pin-memory",
    type=click.Choice(["auto", "on", "off"]),
    default="auto",
    show_default=True,
    help="Whether dataloaders pin host memory for faster transfers (CUDA only). 'auto' enables it on CUDA.",
)
@click.option(
    "-ev",
    "--evaluate/--no-evaluate",
    default=True,
    show_default=True,
    help="Determines whether to score the trained snapshot against the labeled frames as a final step and write the "
    "evaluation results.",
)
@click.option(
    "-ebs",
    "--evaluation-batch-size",
    type=int,
    default=16,
    show_default=True,
    help="The number of frames scored per forward pass during the post-training evaluation.",
)
@click.option(
    "-ecc",
    "--evaluation-confidence-cutoff",
    type=float,
    default=None,
    help="The confidence cutoff for the evaluation's cutoff-filtered metrics. Omit to use the default of 0.6.",
)
@click.option(
    "-p",
    "--progress/--no-progress",
    default=True,
    show_default=True,
    help="Determines whether to render the live progress monitor and keep the underlying training logs off the "
    "console.",
)
def train_command(
    config_path: Path,
    shuffle: int,
    model_prefix: str,
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
    multi_gpu: Literal["auto", "ddp", "dp", "single"],
    amp: AmpMode,
    tf32: Toggle,
    cudnn_benchmark: Toggle,
    compile_model: Toggle,
    dataloader_workers: int,
    pin_memory: Toggle,
    evaluation_batch_size: int,
    evaluation_confidence_cutoff: float | None,
    *,
    load_head_weights: bool,
    evaluate: bool,
    progress: bool,
) -> None:
    """Trains a DeepLabCut shuffle with hardware optimizations and a clean progress monitor.

    ``--config-path`` names the DeepLabCut project's config.yaml. The shuffle's model architecture and train/test split
    are fixed when the shuffle is created (see ``slvt prepare``); this command fits that shuffle. Every
    optimization is exposed as a flag: automatic defaults are chosen for the detected hardware and never run slower
    than stock DeepLabCut, while explicit flags allow tuning for known hardware. Training runs as a
    DistributedDataParallel process group across multiple GPUs, or a single process on one GPU, the CPU, or
    DataParallel across multiple GPUs via ``--multi-gpu dp``.
    """
    try:
        gpu_indices = tuple(int(part) for part in gpus.split(",")) if gpus else None
    except ValueError as error:
        message = (
            f"Unable to parse the --gpus value. Expected comma-separated GPU indices such as '0,1', but got '{gpus}'."
        )
        raise click.ClickException(message=message) from error

    # Detect whether the shuffle's training transform feeds the network one fixed input size so the cuDNN autotuner's
    # 'auto' default can enable itself only when it pays off, replacing the operator-declared flag this used to require.
    fixed_input_size = detect_fixed_input_size(config=config_path, shuffle=shuffle, model_prefix=model_prefix)

    try:
        profile = resolve_optimization_profile(
            device=device,
            gpus=gpu_indices,
            multi_gpu=multi_gpu,
            amp=amp,
            tf32=tf32,
            cudnn_benchmark=cudnn_benchmark,
            torch_compile=compile_model,
            dataloader_workers=dataloader_workers,
            pin_memory=pin_memory,
            fixed_input_size=fixed_input_size,
        )
        summary = train_model(
            config=config_path,
            profile=profile,
            shuffle=shuffle,
            model_prefix=model_prefix,
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
    except (ValueError, FileNotFoundError) as error:
        raise click.ClickException(message=str(error)) from error

    click.echo(message=summary.describe())
