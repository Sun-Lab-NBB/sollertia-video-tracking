"""Provides the ``slvt infer`` command that analyzes videos with a trained DeepLabCut model across GPU or CPU slots."""

from pathlib import Path

import click

from ..inference import Toggle, AmpMode, run_inference, resolve_project_videos, resolve_inference_profile

_CONTEXT_SETTINGS: dict[str, int] = {"max_content_width": 120}
"""Ensures that displayed Click help messages are formatted according to the lab standard."""


@click.command("infer", context_settings=_CONTEXT_SETTINGS)
@click.argument("config", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("videos", nargs=-1, required=False, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--project-videos",
    "project_videos",
    is_flag=True,
    help="Analyze every video registered in the project configuration's video_sets, in addition to any VIDEOS given "
    "explicitly. Registered videos that no longer exist on disk are skipped.",
)
@click.option(
    "-d",
    "--dest",
    "dest",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="The directory prediction files are written to. Omit to write each video's predictions beside the video "
    "itself, matching DeepLabCut and the location the 'extract outliers' step reads.",
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
    default=False,
    show_default=True,
    help="Whether to convert each video's predictions in-flight to a wide polars feather file. Off by default so the "
    "command behaves like DeepLabCut's own analyze and preserves the native prediction files the refinement loop "
    "(evaluation, outlier extraction) reads; the feather output is deferred to the deployment path.",
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
    default=True,
    show_default=True,
    help="Whether to keep DeepLabCut's own prediction files (HDF5, pickles, and tracker files) in the destination. "
    "Kept by default so all DeepLabCut data survives; only takes effect with --to-polars, since conversion is what "
    "would otherwise remove them.",
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
    dest: Path | None,
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
    *,
    project_videos: bool,
    to_polars: bool,
    save_as_csv: bool,
    keep_dlc_outputs: bool,
    fixed_input_size: bool,
    display_progress: bool,
) -> None:
    """Analyzes videos with a trained DeepLabCut model, distributing whole videos across GPU or CPU worker slots.

    CONFIG is the path to the DeepLabCut project's config.yaml, and VIDEOS are the video files to analyze. VIDEOS may be
    omitted when ``--project-videos`` is passed, which analyzes every existing video registered in the project
    configuration. Each worker analyzes whole videos pulled from a shared queue, so the work is balanced without
    splitting any video, and every forward pass runs with the mixed precision and memory format chosen for the detected
    hardware. By default it writes DeepLabCut's native prediction files (preserving them for the evaluation and
    outlier-extraction steps of the refinement loop); pass ``--to-polars`` to also convert them to wide polars feather
    files. The same command runs on multiple GPUs, one GPU, or a CPU-only machine.
    """
    try:
        gpu_indices = tuple(int(part) for part in gpus.split(",")) if gpus else None
    except ValueError as error:
        message = (
            f"Unable to parse the --gpus value. Expected comma-separated GPU indices such as '0,1', but got '{gpus}'."
        )
        raise click.ClickException(message=message) from error

    resolved_videos = list(videos)
    if project_videos:
        resolved_videos.extend(resolve_project_videos(config))
    seen: set[str] = set()
    unique_videos: list[str | Path] = []
    for video in resolved_videos:
        key = str(video.resolve())
        if key not in seen:
            seen.add(key)
            unique_videos.append(video)
    if not unique_videos:
        message = (
            "No videos to analyze. Provide one or more VIDEOS, or pass --project-videos to analyze the videos "
            "registered in the project configuration."
        )
        raise click.UsageError(message=message)
    if project_videos:
        click.echo(message=f"analyzing {len(unique_videos)} videos from the project configuration")

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
            videos=unique_videos,
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
            display_progress=display_progress,
        )
    except (ValueError, FileNotFoundError) as error:
        raise click.ClickException(str(error)) from error

    click.echo(message=summary.describe())
