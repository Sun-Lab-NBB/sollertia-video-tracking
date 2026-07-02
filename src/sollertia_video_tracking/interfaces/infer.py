"""Provides the ``slvt infer`` command that analyzes videos with a trained DeepLabCut model across GPU or CPU slots."""

from pathlib import Path

import click

from ..inference import Toggle, AmpMode, run_inference, resolve_inference_profile

_CONTEXT_SETTINGS: dict[str, int] = {"max_content_width": 120}
"""Ensures that displayed Click help messages are formatted according to the lab standard."""


@click.command("infer", context_settings=_CONTEXT_SETTINGS)
@click.argument("config", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("videos", nargs=-1, required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "-d",
    "--dest",
    "dest",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path(),
    show_default=True,
    help="The directory prediction files are written to.",
)
@click.option("-s", "--shuffle", default=1, show_default=True, help="The shuffle index whose trained model is used.")
@click.option(
    "--snapshot-index",
    "snapshot_index",
    type=int,
    default=None,
    help="The pose snapshot index to use. Omit to use the snapshot selected by the model configuration.",
)
@click.option(
    "--detector-snapshot-index",
    "detector_snapshot_index",
    type=int,
    default=None,
    help="The detector snapshot index to use, for top-down models. Omit to use the configured default.",
)
@click.option(
    "-b",
    "--batch-size",
    "batch_size",
    type=int,
    default=None,
    help="The pose-model inference batch size. Omit to use the configured value.",
)
@click.option(
    "--detector-batch-size",
    "detector_batch_size",
    type=int,
    default=None,
    help="The detector inference batch size, for top-down models. Omit to use the configured value.",
)
@click.option(
    "--to-polars/--no-to-polars",
    "to_polars",
    default=True,
    show_default=True,
    help="Whether to convert each video's predictions in-flight to a wide polars feather file.",
)
@click.option(
    "--likelihood-threshold",
    "likelihood_threshold",
    type=float,
    default=0.0,
    show_default=True,
    help="The likelihood below which keypoint positions are masked to NaN in the feather output.",
)
@click.option(
    "--save-csv",
    "save_as_csv",
    is_flag=True,
    help="Also write a DeepLabCut CSV alongside each prediction file.",
)
@click.option(
    "--keep-dlc-outputs/--no-keep-dlc-outputs",
    "keep_dlc_outputs",
    default=False,
    show_default=True,
    help="Whether to keep DeepLabCut's own prediction files (HDF5, pickles, and tracker files) in the destination. By "
    "default, once predictions are converted to feather, deployment leaves only the feather and its provenance "
    "sidecar.",
)
@click.option(
    "--device",
    default="auto",
    show_default=True,
    help="The device to run on: 'auto', 'cpu', 'mps', 'cuda', or 'cuda:N'. 'auto' uses every visible GPU, else "
    "the CPU.",
)
@click.option(
    "--gpus",
    default=None,
    metavar="INDICES",
    help="The comma-separated CUDA device indices to use (e.g. '0,1'). Omit to use every visible GPU.",
)
@click.option(
    "--gpu-processes",
    "gpu_processes",
    type=int,
    default=-1,
    show_default=True,
    help="The number of worker processes per GPU. 1 runs one video per GPU; raise it to oversubscribe a GPU and fill "
    "decode gaps. Set to -1 for the default of one video per GPU.",
)
@click.option(
    "--cpu-workers",
    "cpu_workers",
    type=int,
    default=-1,
    show_default=True,
    help="The number of CPU worker processes, each pinned to a disjoint core block. Set to -1 to choose automatically.",
)
@click.option(
    "--cpu-threads-per-worker",
    "cpu_threads_per_worker",
    type=int,
    default=-1,
    show_default=True,
    help="The number of intra-op threads (and cores) per CPU worker. Set to -1 to choose automatically.",
)
@click.option(
    "--amp",
    type=click.Choice(["auto", "off", "bf16", "fp16"]),
    default="auto",
    show_default=True,
    help="The mixed-precision mode. 'auto' enables bfloat16 on Ampere or newer GPUs; force 'bf16' to also use it on a "
    "capable CPU.",
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
    help="The cuDNN convolution autotuner (CUDA only). Best with --fixed-input-size, since videos of differing "
    "resolutions re-tune it.",
)
@click.option(
    "--channels-last",
    "channels_last",
    type=click.Choice(["auto", "on", "off"]),
    default="auto",
    show_default=True,
    help="The channels-last memory format, which accelerates convolutions. 'auto' enables it on CUDA.",
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
    "--pin-memory",
    "pin_memory",
    type=click.Choice(["auto", "on", "off"]),
    default="auto",
    show_default=True,
    help="Whether to use a non-blocking host-to-device transfer (CUDA only). 'auto' enables it on CUDA.",
)
@click.option(
    "--fixed-input-size",
    "fixed_input_size",
    is_flag=True,
    help="Declares that every video is analyzed at a single fixed resolution, which makes the cuDNN autotuner safe.",
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
    help="Determines whether to render the live aggregate progress bar and route DeepLabCut's own output off the "
    "console.",
)
def infer_command(
    config: Path,
    videos: tuple[Path, ...],
    dest: Path,
    shuffle: int,
    snapshot_index: int | None,
    detector_snapshot_index: int | None,
    batch_size: int | None,
    detector_batch_size: int | None,
    likelihood_threshold: float,
    device: str,
    gpus: str | None,
    gpu_processes: int,
    cpu_workers: int,
    cpu_threads_per_worker: int,
    amp: AmpMode,
    tf32: Toggle,
    cudnn_benchmark: Toggle,
    channels_last: Toggle,
    torch_compile: Toggle,
    pin_memory: Toggle,
    heartbeat: float,
    *,
    to_polars: bool,
    save_as_csv: bool,
    keep_dlc_outputs: bool,
    fixed_input_size: bool,
    display_progress: bool,
) -> None:
    """Analyzes videos with a trained DeepLabCut model, distributing whole videos across GPU or CPU worker slots.

    CONFIG is the path to the DeepLabCut project's config.yaml, and VIDEOS are the video files to analyze. Each worker
    analyzes whole videos pulled from a shared queue, so the work is balanced without splitting any video, and every
    forward pass runs with the mixed precision and memory format chosen for the detected hardware. Predictions are
    written to the destination directory and, by default, converted in-flight to wide polars feather files for the rest
    of the Sollertia stack. The same command runs on multiple GPUs, one GPU, or a CPU-only machine.
    """
    try:
        gpu_indices = tuple(int(part) for part in gpus.split(",")) if gpus else None
    except ValueError as error:
        message = (
            f"Unable to parse the --gpus value. Expected comma-separated GPU indices such as '0,1', but got '{gpus}'."
        )
        raise click.ClickException(message=message) from error

    try:
        profile = resolve_inference_profile(
            device=device,
            gpus=gpu_indices,
            amp=amp,
            tf32=tf32,
            cudnn_benchmark=cudnn_benchmark,
            channels_last=channels_last,
            torch_compile=torch_compile,
            gpu_processes=gpu_processes,
            cpu_workers=cpu_workers,
            cpu_threads_per_worker=cpu_threads_per_worker,
            pin_memory=pin_memory,
            fixed_input_size=fixed_input_size,
        )
        summary = run_inference(
            config=config,
            videos=list(videos),
            destination=dest,
            profile=profile,
            shuffle=shuffle,
            snapshot_index=snapshot_index,
            detector_snapshot_index=detector_snapshot_index,
            batch_size=batch_size,
            detector_batch_size=detector_batch_size,
            to_polars=to_polars,
            likelihood_threshold=likelihood_threshold,
            save_as_csv=save_as_csv,
            keep_dlc_outputs=keep_dlc_outputs,
            heartbeat=heartbeat,
            display_progress=display_progress,
        )
    except (ValueError, FileNotFoundError) as error:
        raise click.ClickException(str(error)) from error

    click.echo(message=summary.describe())
