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

_CROP_FIELD_COUNT: int = 4
"""The number of comma-separated integers, ``x1,x2,y1,y2``, in a ``--crop`` rectangle."""


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
    "-cr",
    "--crop",
    multiple=True,
    metavar="X1,X2,Y1,Y2",
    help="A crop rectangle 'x1,x2,y1,y2' to analyze instead of the project's configured crop, decoupling cropping "
    "from the project configuration so de-novo videos that are not registered in the project can be analyzed at a "
    "chosen region. Pass once to apply one rectangle to every video, or once per video (matching the video count, and "
    "not with --project-videos) for per-video crops in the order the videos are given.",
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
    crop: tuple[str, ...],
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
    progress: bool,
) -> None:
    """Analyzes videos with a trained DeepLabCut model, distributing whole videos across GPU or CPU worker slots.

    CONFIG is the path to the DeepLabCut project's config.yaml, and VIDEOS are the video files to analyze. VIDEOS may be
    omitted when ``--project-videos`` is passed, which analyzes every existing video registered in the project
    configuration. Each worker analyzes whole videos pulled from a shared queue, so the work is balanced without
    splitting any video, and every forward pass runs with the mixed precision and memory format chosen for the detected
    hardware. Each video's predictions are written as DeepLabCut's native ``.h5`` prediction file, beside the video or
    into ``--destination``, which is exactly what the evaluation and outlier-extraction steps of the refinement loop
    read. Pass ``--crop`` to analyze a chosen region rather than the project's configured crop, which lets de-novo
    videos that are not registered in the project be analyzed. The same command runs on multiple GPUs, one GPU, or a
    CPU-only machine.
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

    crop_override = _resolve_crop_override(crop=crop, video_count=len(unique_videos), project_videos=project_videos)

    # Detect whether the run feeds the network one fixed input size so the cuDNN autotuner's 'auto' default can enable
    # itself only when it pays off, replacing the operator-declared flag this used to require.
    fixed_input_size = detect_fixed_input_size(config, unique_videos, crop_override)

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
            crop_override=crop_override,
            display_progress=progress,
        )
    except (ValueError, FileNotFoundError) as error:
        raise click.ClickException(str(error)) from error

    click.echo(message=summary.describe())


def _parse_crop_option(value: str) -> tuple[int, int, int, int]:
    """Parses an ``x1,x2,y1,y2`` crop rectangle from a single ``--crop`` option value.

    Args:
        value: The comma-separated crop rectangle as four integers.

    Returns:
        The parsed ``(x1, x2, y1, y2)`` rectangle.

    Raises:
        click.UsageError: When the value is not four integers or does not describe a positive-area rectangle.
    """
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != _CROP_FIELD_COUNT:
        message = f"Unable to parse the --crop value '{value}'. Expected four comma-separated integers 'x1,x2,y1,y2'."
        raise click.UsageError(message=message)
    try:
        corners = [int(part) for part in parts]
    except ValueError as error:
        message = f"Unable to parse the --crop value '{value}'. Expected four integers 'x1,x2,y1,y2'."
        raise click.UsageError(message=message) from error
    x1, x2, y1, y2 = corners
    if x2 <= x1 or y2 <= y1:
        message = (
            f"Unable to use the --crop value '{value}'. Expected x1 < x2 and y1 < y2 to describe a positive-area "
            f"rectangle."
        )
        raise click.UsageError(message=message)
    return x1, x2, y1, y2


def _resolve_crop_override(
    crop: tuple[str, ...], video_count: int, *, project_videos: bool
) -> list[tuple[int, int, int, int]] | None:
    """Resolves the ``--crop`` option values into one crop rectangle per analyzed video, or None when unset.

    A single ``--crop`` is applied uniformly to every video. Several ``--crop`` rectangles are matched to the videos in
    order and must equal the video count; because registered project videos carry their own configured crops and have
    no stable position, per-video crops cannot be combined with ``--project-videos``.

    Args:
        crop: The raw ``--crop`` option values, each a ``x1,x2,y1,y2`` string.
        video_count: The number of videos the run will analyze.
        project_videos: Determines whether the run also analyzes the project's registered videos.

    Returns:
        One ``(x1, x2, y1, y2)`` rectangle per video, or None when no ``--crop`` was given.

    Raises:
        click.UsageError: When a rectangle is malformed, when several rectangles are combined with ``--project-videos``,
            or when the rectangle count matches neither one nor the video count.
    """
    if not crop:
        return None
    rectangles = [_parse_crop_option(value) for value in crop]
    if len(rectangles) == 1:
        return rectangles * video_count
    if project_videos:
        message = (
            "Per-video --crop rectangles cannot be combined with --project-videos. Pass a single --crop to apply one "
            "rectangle to every video, or list the videos explicitly with one --crop each."
        )
        raise click.UsageError(message=message)
    if len(rectangles) != video_count:
        message = (
            f"Got {len(rectangles)} --crop rectangles for {video_count} videos. Pass a single --crop to apply it to "
            f"every video, or exactly one --crop per video."
        )
        raise click.UsageError(message=message)
    return rectangles
