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
@click.option(
    "-cfg",
    "--config-path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="The path to the DeepLabCut project's config.yaml whose trained model analyzes the videos.",
)
@click.option(
    "-v",
    "--videos",
    multiple=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    metavar="PATH",
    help="A video file to analyze. Provide the option several times for several videos. These need not be registered "
    "in the project, so de-novo videos can be analyzed, optionally at a chosen region with --crop. Omit --videos to "
    "analyze every video registered in the project's config.yaml.",
)
@click.option(
    "-o",
    "--output",
    multiple=True,
    type=click.Path(file_okay=False, path_type=Path),
    metavar="DIRECTORY",
    help="A directory the prediction files are written to. Pass once to collect every video's predictions in one "
    "directory, or once per --videos file (in order, matching the video count) to bundle each video's predictions "
    "with its own directory. Per-video outputs require explicit --videos. Omit to write each video's predictions "
    "beside the video itself, where 'slvt extract outliers' reads them.",
)
@click.option(
    "-s",
    "--shuffle",
    default=1,
    show_default=True,
    help="The shuffle index whose trained model analyzes the videos.",
)
@click.option(
    "-si",
    "--snapshot-index",
    type=int,
    default=None,
    help="The index of the trained pose-model snapshot to run. Omit to use the project's configured default.",
)
@click.option(
    "-dsi",
    "--detector-snapshot-index",
    type=int,
    default=None,
    help="The index of the detector snapshot to run, for top-down models. Omit to use the project's configured "
    "default.",
)
@click.option(
    "-b",
    "--batch-size",
    type=int,
    default=None,
    help="The number of frames the pose model processes per forward pass. Larger batches use more GPU memory and can "
    "speed up analysis. Omit to use the model's default value.",
)
@click.option(
    "-dbs",
    "--detector-batch-size",
    type=int,
    default=None,
    help="The number of frames the object detector processes per step, for top-down models. Omit to use the model's "
    "default value.",
)
@click.option(
    "-cr",
    "--crop",
    multiple=True,
    metavar="X1,X2,Y1,Y2",
    help="A crop rectangle 'x1,x2,y1,y2' to analyze instead of the project's configured crop, so de-novo videos can be "
    "analyzed at a chosen region. Pass once to apply one rectangle to every video, or once per --videos file (in "
    "order, matching the video count) for per-video crops. Per-video crops require explicit --videos.",
)
@click.option(
    "-d",
    "--device",
    type=click.Choice([device.value for device in DeviceType]),
    default=DeviceType.AUTO.value,
    show_default=True,
    help="The base device to run on. 'auto' uses every visible CUDA GPU when present and otherwise the CPU. 'cuda' "
    "targets every visible GPU but warns before falling back to the CPU when none is present. 'cpu' and 'mps' (Apple "
    "Metal) force those devices. Choose specific GPUs with --gpus.",
)
@click.option(
    "-g",
    "--gpus",
    default=None,
    metavar="INDICES",
    help="The comma-separated CUDA device indices to run on (e.g. '0,1'). Omit to use every visible GPU. Applies only "
    "when the device is a CUDA GPU.",
)
@click.option(
    "-gp",
    "--gpu-processes",
    type=int,
    default=-1,
    show_default=True,
    help="The number of inference worker processes per GPU, each analyzing one video at a time. Raise it to "
    "oversubscribe a GPU and fill decode gaps. Most GPUs fully saturate with 1 or 2 workers. Set to -1 for the "
    "default of one process (one video) per GPU.",
)
@click.option(
    "-ch",
    "--chunks",
    type=int,
    default=1,
    show_default=True,
    help="The number of contiguous frame-range pieces each running video is split into, all analyzed concurrently. "
    "Raise it to run several frame ranges of one video at once, filling decode gaps within a single video, so total "
    "per-GPU concurrency becomes gpu-processes x chunks. Set to 1 to analyze each video as a single unbroken frame "
    "range.",
)
@click.option(
    "-cw",
    "--cpu-workers",
    type=int,
    default=-1,
    show_default=True,
    help="The number of CPU inference worker processes, each pinned to a disjoint block of CPU cores. Set to -1 to "
    "choose automatically from the core count.",
)
@click.option(
    "-ctpw",
    "--cpu-threads-per-worker",
    type=int,
    default=-1,
    show_default=True,
    help="The number of CPU threads (and cores) each CPU worker uses. Set to -1 to choose automatically.",
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
    help="The cuDNN convolution autotuner on CUDA. 'auto' enables it only when the run is detected to feed one fixed "
    "input size (a shared --crop rectangle, a shared project crop, or one shared video resolution), where it speeds "
    "up convolutions rather than re-tuning per size. 'on' and 'off' force it.",
)
@click.option(
    "-cl",
    "--channels-last",
    type=click.Choice([toggle.value for toggle in Toggle]),
    default=Toggle.AUTO.value,
    show_default=True,
    help="The channels-last memory format, which speeds up convolutions on tensor-core GPUs. 'auto' enables it on "
    "CUDA. 'on' and 'off' force it.",
)
@click.option(
    "-cm",
    "--compile-model",
    type=click.Choice([toggle.value for toggle in Toggle]),
    default=Toggle.AUTO.value,
    show_default=True,
    help="Determines whether the model is compiled with torch.compile for faster inference. 'auto' leaves it off "
    "because its one-time warm-up cost may not amortize. 'on' and 'off' force it.",
)
@click.option(
    "-p",
    "--progress/--no-progress",
    default=True,
    show_default=True,
    help="Determines whether the aggregate progress bar is shown during analysis.",
)
def infer_command(
    config_path: Path,
    videos: tuple[Path, ...],
    output: tuple[Path, ...],
    shuffle: int,
    snapshot_index: int | None,
    detector_snapshot_index: int | None,
    batch_size: int | None,
    detector_batch_size: int | None,
    crop: tuple[str, ...],
    device: str,
    gpus: str | None,
    gpu_processes: int,
    chunks: int,
    cpu_workers: int,
    cpu_threads_per_worker: int,
    amp: str,
    tf32: str,
    cudnn_benchmark: str,
    channels_last: str,
    compile_model: str,
    *,
    progress: bool,
) -> None:
    """Analyzes videos with a trained DeepLabCut model, distributing whole videos across GPU or CPU worker slots.

    ``--config-path`` names the DeepLabCut project's config.yaml whose trained model runs. Provide the videos to analyze
    with ``--videos`` (given several times for several files), or omit ``--videos`` to analyze every existing video
    registered in the project configuration. Each worker pulls work from a shared queue, so the work is balanced across
    slots. By default a worker analyzes a whole video, and ``--chunks`` instead splits each running video into that many
    contiguous frame ranges analyzed concurrently, with the parent stitching each video's ranges back into one
    prediction file. Each forward pass runs with the mixed precision and memory format chosen for the detected
    hardware, except conditional-top-down models, which run at stock precision. Each video's
    predictions are written as DeepLabCut's native ``.h5`` prediction file, beside the video or into an ``--output``
    directory (one shared directory, or one per video). Pass ``--crop`` to analyze a chosen region rather than the
    project's configured crop, which lets de-novo videos that are not registered in the project be analyzed. The same
    command runs on multiple GPUs, one GPU, or a CPU-only machine.
    """
    try:
        gpu_indices = tuple(int(part) for part in gpus.split(",")) if gpus else None
    except ValueError as error:
        message = (
            f"Unable to parse the --gpus value. Expected comma-separated GPU indices such as '0,1', but got '{gpus}'."
        )
        raise click.ClickException(message=message) from error

    if chunks < 1:
        message = (
            f"Unable to use the --chunks value '{chunks}'. Expected a positive whole number of frame-range pieces."
        )
        raise click.UsageError(message=message)

    whole_project = not videos
    resolved_videos: list[Path] = list(resolve_project_videos(config_path)) if whole_project else list(videos)
    seen: set[str] = set()
    unique_videos: list[str | Path] = []
    for video in resolved_videos:
        key = str(video.resolve())
        if key not in seen:
            seen.add(key)
            unique_videos.append(video)
    if not unique_videos:
        message = (
            "Unable to run inference without videos. Provide one or more videos with --videos, or register videos "
            "in the project's config.yaml to analyze the whole project."
        )
        raise click.UsageError(message=message)
    if whole_project:
        click.echo(message=f"analyzing {len(unique_videos)} videos from the project configuration")

    crop_override = _resolve_crop_override(crop=crop, video_count=len(unique_videos), whole_project=whole_project)
    destination_override = _resolve_output_override(
        output=output, video_count=len(unique_videos), whole_project=whole_project
    )

    # Detects whether the run feeds the network one fixed input size so the cuDNN autotuner's 'auto' default can enable
    # itself only when it pays off.
    fixed_input_size = detect_fixed_input_size(config=config_path, videos=unique_videos, crop_override=crop_override)

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
            chunks=chunks,
            cpu_workers=cpu_workers,
            cpu_threads_per_worker=cpu_threads_per_worker,
            fixed_input_size=fixed_input_size,
        )
        summary = run_inference(
            config=config_path,
            videos=unique_videos,
            destination_override=destination_override,
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
        raise click.ClickException(message=str(error)) from error

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
    crop: tuple[str, ...], video_count: int, *, whole_project: bool
) -> list[tuple[int, int, int, int]] | None:
    """Resolves the ``--crop`` option values into one crop rectangle per analyzed video, or None when unset.

    A single ``--crop`` is applied uniformly to every video. Several ``--crop`` rectangles are matched to the videos in
    order and must equal the video count; because the whole-project video order is not user-controlled, per-video crops
    cannot be combined with the whole-project default and require explicit ``--videos``.

    Args:
        crop: The raw ``--crop`` option values, each a ``x1,x2,y1,y2`` string.
        video_count: The number of videos the run will analyze.
        whole_project: Determines whether the run defaults to every registered project video.

    Returns:
        One ``(x1, x2, y1, y2)`` rectangle per video, or None when no ``--crop`` was given.

    Raises:
        click.UsageError: When a rectangle is malformed, when several rectangles are given while defaulting to the whole
            project, or when the rectangle count matches neither one nor the video count.
    """
    if not crop:
        return None
    rectangles = [_parse_crop_option(value) for value in crop]
    if len(rectangles) == 1:
        return rectangles * video_count
    if whole_project:
        message = (
            "Unable to apply per-video --crop rectangles when analyzing the whole project. Pass a single --crop to "
            "apply one rectangle to every video, or list the videos explicitly with --videos and one --crop each."
        )
        raise click.UsageError(message=message)
    if len(rectangles) != video_count:
        message = (
            f"Unable to match the --crop rectangles to the videos. Expected a single --crop or exactly one per video, "
            f"but got {len(rectangles)} --crop rectangles for {video_count} videos."
        )
        raise click.UsageError(message=message)
    return rectangles


def _resolve_output_override(output: tuple[Path, ...], video_count: int, *, whole_project: bool) -> list[Path] | None:
    """Resolves the ``--output`` option values into one output directory per analyzed video, or None when unset.

    A single ``--output`` collects every video's predictions in one directory. Several ``--output`` directories are
    matched to the videos in order and must equal the video count; because the whole-project video order is not
    user-controlled, per-video directories cannot be combined with the whole-project default and require explicit
    ``--videos``.

    Args:
        output: The raw ``--output`` option values, each a directory path.
        video_count: The number of videos the run will analyze.
        whole_project: Determines whether the run defaults to every registered project video.

    Returns:
        One output directory per video, or None when no ``--output`` was given.

    Raises:
        click.UsageError: When several directories are given while defaulting to the whole project, or when the
            directory count matches neither one nor the video count.
    """
    if not output:
        return None
    directories = list(output)
    if len(directories) == 1:
        return directories * video_count
    if whole_project:
        message = (
            "Unable to apply per-video --output directories when analyzing the whole project. Pass a single --output "
            "to collect every video's predictions in one directory, or list the videos explicitly with --videos and "
            "one --output each."
        )
        raise click.UsageError(message=message)
    if len(directories) != video_count:
        message = (
            f"Unable to match the --output directories to the videos. Expected a single --output or exactly one per "
            f"video, but got {len(directories)} --output directories for {video_count} videos."
        )
        raise click.UsageError(message=message)
    return directories
