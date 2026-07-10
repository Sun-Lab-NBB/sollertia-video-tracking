"""Provides the ``slvt infer`` command that analyzes videos with a trained DeepLabCut model across GPU or CPU slots."""

from pathlib import Path

import click

from ..inference import (
    Toggle,
    AmpMode,
    DeviceType,
    run_inference,
    resolve_project_videos,
    detect_fixed_input_size,
    resolve_inference_profile,
)

_CONTEXT_SETTINGS: dict[str, int] = {"max_content_width": 120}
"""Widens displayed Click help messages to 120 columns so option descriptions wrap consistently."""


@click.command("infer", context_settings=_CONTEXT_SETTINGS)
@click.argument("config", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("videos", nargs=-1, required=False, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "-pv",
    "--project-videos",
    is_flag=True,
    help="Also analyze every video registered in the project, in addition to any VIDEOS given explicitly. Registered "
    "videos that no longer exist on disk are skipped.",
)
@click.option(
    "-d",
    "--destination",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="The directory prediction files are written to. Omit to write each video's predictions beside the video "
    "itself, where the 'extract outliers' step reads them.",
)
@click.option("-s", "--shuffle", default=1, show_default=True, help="The shuffle index whose trained model is used.")
@click.option(
    "-si",
    "--snapshot-index",
    type=int,
    default=None,
    help="The trained pose snapshot to use. Omit to use the configured default.",
)
@click.option(
    "-dsi",
    "--detector-snapshot-index",
    type=int,
    default=None,
    help="The detector snapshot to use, for top-down models. Omit to use the configured default.",
)
@click.option(
    "-b",
    "--batch-size",
    type=int,
    default=None,
    help="The pose model batch size. Omit to use the model's default value.",
)
@click.option(
    "-dbs",
    "--detector-batch-size",
    type=int,
    default=None,
    help="The detector batch size, for top-down models. Omit to use the model's default value.",
)
@click.option(
    "-tp",
    "--to-polars/--no-to-polars",
    default=False,
    show_default=True,
    help="Also convert each video's predictions to a single wide table file. Off by default so the native prediction "
    "files that the refinement loop (evaluation and outlier extraction) reads are preserved.",
)
@click.option(
    "-lt",
    "--likelihood-threshold",
    type=float,
    default=0.0,
    show_default=True,
    help="The confidence below which keypoint positions are cleared in the converted table output.",
)
@click.option(
    "-sc",
    "--save-csv",
    is_flag=True,
    help="Also write a CSV copy of the predictions alongside each prediction file.",
)
@click.option(
    "-kdo",
    "--keep-dlc-outputs/--no-keep-dlc-outputs",
    default=True,
    show_default=True,
    help="Keep the original per-video prediction files in the destination. Kept by default so no prediction data is "
    "lost; only takes effect with --to-polars, since conversion is what would otherwise remove them.",
)
@click.option(
    "-dv",
    "--device",
    type=click.Choice([device.value for device in DeviceType]),
    default=DeviceType.AUTO.value,
    show_default=True,
    help="The base device to run on. 'auto' uses every visible CUDA GPU when present and otherwise the CPU. 'cpu' and "
    "'mps' (Apple Metal) force those devices. Choose specific GPUs with --gpus.",
)
@click.option(
    "-g",
    "--gpus",
    default=None,
    metavar="INDICES",
    help="The comma-separated CUDA device indices to use (e.g. '0,1'). Omit to use every visible GPU.",
)
@click.option(
    "-gp",
    "--gpu-processes",
    type=int,
    default=-1,
    show_default=True,
    help="The number of worker processes per GPU. 1 runs one video per GPU. Raise it to oversubscribe a GPU and fill "
    "decode gaps. Set to -1 for the default of one video per GPU.",
)
@click.option(
    "-cw",
    "--cpu-workers",
    type=int,
    default=-1,
    show_default=True,
    help="The number of CPU worker processes, each pinned to a disjoint core block. Set to -1 to choose automatically.",
)
@click.option(
    "-ctpw",
    "--cpu-threads-per-worker",
    type=int,
    default=-1,
    show_default=True,
    help="The number of CPU threads (and cores) per worker. Set to -1 to choose automatically.",
)
@click.option(
    "-a",
    "--amp",
    type=click.Choice([mode.value for mode in AmpMode]),
    default=AmpMode.AUTO.value,
    show_default=True,
    help="The mixed-precision mode. 'auto' enables bfloat16 on Ampere or newer GPUs. Force 'bf16' to also use it on a "
    "capable CPU.",
)
@click.option(
    "-t",
    "--tf32",
    type=click.Choice([toggle.value for toggle in Toggle]),
    default=Toggle.AUTO.value,
    show_default=True,
    help="TF32 acceleration for float32 matmuls and convolutions (CUDA only). 'auto' enables it on Ampere or newer.",
)
@click.option(
    "-cb",
    "--cudnn-benchmark",
    type=click.Choice([toggle.value for toggle in Toggle]),
    default=Toggle.AUTO.value,
    show_default=True,
    help="The convolution autotuner (CUDA only). 'auto' enables it when the run is detected to use a single fixed "
    "input size (a shared project crop, or one shared video resolution), where it pays off rather than re-tuning per "
    "size.",
)
@click.option(
    "-cl",
    "--channels-last",
    type=click.Choice([toggle.value for toggle in Toggle]),
    default=Toggle.AUTO.value,
    show_default=True,
    help="The channels-last memory format, which accelerates convolutions. 'auto' enables it on CUDA.",
)
@click.option(
    "-cm",
    "--compile-model",
    type=click.Choice([toggle.value for toggle in Toggle]),
    default=Toggle.AUTO.value,
    show_default=True,
    help="Whether to compile the model for faster inference. Off by default because of its warm-up cost.",
)
@click.option(
    "-pm",
    "--pin-memory",
    type=click.Choice([toggle.value for toggle in Toggle]),
    default=Toggle.AUTO.value,
    show_default=True,
    help="Whether to use a non-blocking host-to-device transfer (CUDA only). 'auto' enables it on CUDA.",
)
@click.option(
    "-p",
    "--progress/--no-progress",
    default=True,
    show_default=True,
    help="Determines whether the aggregate progress bar is shown during analysis.",
)
def infer_command(
    config: Path,
    videos: tuple[Path, ...],
    destination: Path | None,
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
    amp: str,
    tf32: str,
    cudnn_benchmark: str,
    channels_last: str,
    compile_model: str,
    pin_memory: str,
    *,
    project_videos: bool,
    to_polars: bool,
    save_csv: bool,
    keep_dlc_outputs: bool,
    progress: bool,
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

    # Detect whether the run feeds the network one fixed input size so the cuDNN autotuner's 'auto' default can enable
    # itself only when it pays off, replacing the operator-declared flag this used to require.
    fixed_input_size = detect_fixed_input_size(config, unique_videos)

    try:
        profile = resolve_inference_profile(
            device=DeviceType(device),
            gpus=gpu_indices,
            amp=AmpMode(amp),
            tf32=Toggle(tf32),
            cudnn_benchmark=Toggle(cudnn_benchmark),
            channels_last=Toggle(channels_last),
            torch_compile=Toggle(compile_model),
            gpu_processes=gpu_processes,
            cpu_workers=cpu_workers,
            cpu_threads_per_worker=cpu_threads_per_worker,
            pin_memory=Toggle(pin_memory),
            fixed_input_size=fixed_input_size,
        )
        summary = run_inference(
            config=config,
            videos=unique_videos,
            destination=destination,
            profile=profile,
            shuffle=shuffle,
            snapshot_index=snapshot_index,
            detector_snapshot_index=detector_snapshot_index,
            batch_size=batch_size,
            detector_batch_size=detector_batch_size,
            to_polars=to_polars,
            likelihood_threshold=likelihood_threshold,
            save_as_csv=save_csv,
            keep_dlc_outputs=keep_dlc_outputs,
            display_progress=progress,
        )
    except (ValueError, FileNotFoundError) as error:
        raise click.ClickException(str(error)) from error

    click.echo(message=summary.describe())
