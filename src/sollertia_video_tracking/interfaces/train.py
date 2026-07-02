"""Provides the ``slvt train`` command that trains a DeepLabCut shuffle with mixed precision, DDP, and a monitor."""

from typing import Literal
from pathlib import Path

import click

from ..training import Toggle, AmpMode, train_model, resolve_optimization_profile

_CONTEXT_SETTINGS: dict[str, int] = {"max_content_width": 120}
"""Ensures that displayed Click help messages are formatted according to the lab standard."""


@click.command("train", context_settings=_CONTEXT_SETTINGS)
@click.argument("config", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("-s", "--shuffle", default=1, show_default=True, help="The shuffle index to train.")
@click.option(
    "-i",
    "--training-set-index",
    "training_set_index",
    default=0,
    show_default=True,
    help="The index of the TrainingsetFraction to train, selecting a row of the config's TrainingFraction list.",
)
@click.option("--modelprefix", default="", help="The model subdirectory prefix, matching the create step.")
@click.option(
    "-e",
    "--epochs",
    type=int,
    default=None,
    help="The maximum number of pose-model epochs. Omit to use the value stored in the model configuration.",
)
@click.option(
    "-b",
    "--batch-size",
    "batch_size",
    type=int,
    default=None,
    help="The pose-model batch size. Omit to use the configured value; larger batches use more GPU memory.",
)
@click.option("--save-epochs", "save_epochs", type=int, default=None, help="The number of epochs between snapshots.")
@click.option(
    "--display-iterations",
    "display_iterations",
    type=int,
    default=None,
    help="The number of iterations between intra-epoch loss lines written to the training log.",
)
@click.option(
    "--maximum-snapshots",
    "maximum_snapshots",
    type=int,
    default=None,
    help="The maximum number of snapshots to keep per model. Omit to use the configured value.",
)
@click.option(
    "--snapshot-path",
    "snapshot_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="The pose snapshot to resume training from.",
)
@click.option(
    "--detector-path",
    "detector_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="The detector snapshot to resume training from, for top-down models.",
)
@click.option(
    "--detector-batch-size",
    "detector_batch_size",
    type=int,
    default=None,
    help="The detector batch size, for top-down models. Omit to use the configured value.",
)
@click.option(
    "--detector-epochs",
    "detector_epochs",
    type=int,
    default=None,
    help="The maximum number of detector epochs, for top-down models. Set to 0 to skip detector training.",
)
@click.option(
    "--detector-save-epochs",
    "detector_save_epochs",
    type=int,
    default=None,
    help="The number of epochs between detector snapshots, for top-down models.",
)
@click.option(
    "--load-head-weights/--no-load-head-weights",
    "load_head_weights",
    default=True,
    show_default=True,
    help="Determines whether to restore head weights when resuming a pose model. Disable when changing bodyparts.",
)
@click.option(
    "--device",
    default="auto",
    show_default=True,
    help="The device to train on: 'auto', 'cpu', 'mps', 'cuda', or 'cuda:N'. 'auto' selects every visible GPU.",
)
@click.option(
    "--gpus",
    default=None,
    metavar="INDICES",
    help="The comma-separated CUDA device indices to use (e.g. '0,1'). Omit to use every visible GPU.",
)
@click.option(
    "--multi-gpu",
    "multi_gpu",
    type=click.Choice(["auto", "ddp", "dp", "single"]),
    default="auto",
    show_default=True,
    help="The multi-GPU strategy: DistributedDataParallel, the slower DataParallel, or a single device. 'auto' uses "
    "DDP when two or more GPUs are selected.",
)
@click.option(
    "--amp",
    type=click.Choice(["auto", "off", "bf16", "fp16"]),
    default="auto",
    show_default=True,
    help="The mixed-precision mode. 'auto' enables bfloat16 on Ampere or newer GPUs. Force 'fp16' for pre-Ampere "
    "cards with float16 tensor cores.",
)
@click.option(
    "--tf32",
    type=click.Choice(["auto", "on", "off"]),
    default="auto",
    show_default=True,
    help="TF32 acceleration for float32 matmuls and convolutions (CUDA only). 'auto' enables it on Ampere or newer.",
)
@click.option(
    "--cudnn-benchmark",
    "cudnn_benchmark",
    type=click.Choice(["auto", "on", "off"]),
    default="auto",
    show_default=True,
    help="The cuDNN convolution autotuner. Only enable it with --fixed-input-size, since it disables deterministic "
    "training and can slow variable-size augmentation.",
)
@click.option(
    "--compile",
    "torch_compile",
    type=click.Choice(["auto", "on", "off"]),
    default="auto",
    show_default=True,
    help="Whether to compile the model with torch.compile. Off by default because of its warm-up cost.",
)
@click.option(
    "-w",
    "--dataloader-workers",
    "dataloader_workers",
    type=int,
    default=-1,
    show_default=True,
    help="The number of dataloader workers per process. Set to -1 to choose automatically from the CPU count.",
)
@click.option(
    "--pin-memory",
    "pin_memory",
    type=click.Choice(["auto", "on", "off"]),
    default="auto",
    show_default=True,
    help="Whether dataloaders pin host memory for faster transfers (CUDA only). 'auto' enables it on CUDA.",
)
@click.option(
    "--fixed-input-size",
    "fixed_input_size",
    is_flag=True,
    help="Declares that the training transform produces a single fixed input resolution, which makes the cuDNN "
    "autotuner safe to enable.",
)
@click.option(
    "--evaluate/--no-evaluate",
    "evaluate",
    default=True,
    show_default=True,
    help="Whether to score the trained snapshot against the labeled frames as a final step and write the evaluation "
    "feather and provenance sidecar.",
)
@click.option(
    "--evaluation-batch-size",
    "evaluation_batch_size",
    type=int,
    default=16,
    show_default=True,
    help="The number of frames scored per forward pass during the post-training evaluation.",
)
@click.option(
    "--evaluation-pcutoff",
    "evaluation_pcutoff",
    type=float,
    default=None,
    help="The confidence cutoff for the evaluation's cutoff-filtered metrics. Omit to use the default of 0.6.",
)
@click.option(
    "--heartbeat",
    default=30.0,
    show_default=True,
    metavar="SECONDS",
    help="The minimum interval, in seconds, between progress lines when the output is not a TTY.",
)
@click.option(
    "--progress/--no-progress",
    "display_progress",
    default=True,
    show_default=True,
    help="Determines whether to render the live progress monitor and route DeepLabCut's own logs off the console.",
)
def train_command(
    config: Path,
    shuffle: int,
    training_set_index: int,
    modelprefix: str,
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
    torch_compile: Toggle,
    dataloader_workers: int,
    pin_memory: Toggle,
    evaluation_batch_size: int,
    evaluation_pcutoff: float | None,
    heartbeat: float,
    *,
    load_head_weights: bool,
    evaluate: bool,
    fixed_input_size: bool,
    display_progress: bool,
) -> None:
    """Trains a DeepLabCut shuffle with hardware optimizations and a clean progress monitor.

    CONFIG is the path to the DeepLabCut project's config.yaml. The shuffle's model architecture and train/test split
    are fixed when the shuffle is created (see ``slvt create-training-dataset``); this command fits that shuffle. Every
    optimization is exposed as a flag: automatic defaults are chosen for the detected hardware and never run slower
    than stock DeepLabCut, while explicit flags let you tune for silicon you know. Training runs as a
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

    try:
        profile = resolve_optimization_profile(
            device=device,
            gpus=gpu_indices,
            multi_gpu=multi_gpu,
            amp=amp,
            tf32=tf32,
            cudnn_benchmark=cudnn_benchmark,
            torch_compile=torch_compile,
            dataloader_workers=dataloader_workers,
            pin_memory=pin_memory,
            fixed_input_size=fixed_input_size,
        )
        summary = train_model(
            config=config,
            profile=profile,
            shuffle=shuffle,
            training_set_index=training_set_index,
            modelprefix=modelprefix,
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
            evaluation_pcutoff=evaluation_pcutoff,
            heartbeat=heartbeat,
            display_progress=display_progress,
        )
    except (ValueError, FileNotFoundError) as error:
        raise click.ClickException(str(error)) from error

    click.echo(message=summary.describe())
