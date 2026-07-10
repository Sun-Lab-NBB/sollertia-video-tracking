"""Provides the ``slvt predict`` command that runs a portable model asset over videos into caller-chosen feathers."""

from pathlib import Path

import click

from ..deploy import PredictionJob, run_predictions
from ..inference import Toggle, AmpMode, DeviceType, resolve_inference_profile

_CONTEXT_SETTINGS: dict[str, int] = {"max_content_width": 120}
"""Ensures that displayed Click help messages are formatted according to the lab standard."""


@click.command("predict", context_settings=_CONTEXT_SETTINGS)
@click.argument("asset", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "-j",
    "--job",
    "jobs",
    multiple=True,
    required=True,
    type=(click.Path(exists=True, dir_okay=False, path_type=Path), click.Path(dir_okay=False, path_type=Path)),
    metavar="VIDEO OUTPUT",
    help="A video to analyze paired with the feather path its predictions are written to. Provide the option several "
    "times to analyze several videos in one run, which shares the model load across them.",
)
@click.option(
    "-lt",
    "--likelihood-threshold",
    type=float,
    default=-1.0,
    show_default=True,
    help="The confidence below which keypoint positions are cleared in the output. A negative value uses the default "
    "baked into the asset at export time.",
)
@click.option(
    "-sd",
    "--scratch-directory",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="A directory the temporary model extraction and prediction files are placed under. Omit to use the system "
    "temporary directory; point it at fast node-local storage on a cluster.",
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
    help="The convolution autotuner (CUDA only). 'auto' leaves it off, since a portable asset may analyze videos of "
    "differing resolutions; force it on when every analyzed video shares one resolution.",
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
def predict_command(
    asset: Path,
    jobs: tuple[tuple[Path, Path], ...],
    likelihood_threshold: float,
    scratch_directory: Path | None,
    batch_size: int | None,
    detector_batch_size: int | None,
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
    progress: bool,
) -> None:
    """Runs a portable model asset over videos, writing each video's predictions to the feather path paired with it.

    ASSET is a model asset written by ``slvt export``. Each ``--job VIDEO OUTPUT`` pair analyzes VIDEO and writes its
    predictions to OUTPUT as a single wide table file. The asset is extracted, inference runs with the mixed precision
    and memory format chosen for the detected hardware, and every intermediary, including the extracted model and
    DeepLabCut's own prediction files, is removed before the command returns, so each OUTPUT path holds only its
    prediction file. The same command runs on multiple GPUs, one GPU, or a CPU-only machine.
    """
    try:
        gpu_indices = tuple(int(part) for part in gpus.split(",")) if gpus else None
    except ValueError as error:
        message = (
            f"Unable to parse the --gpus value. Expected comma-separated GPU indices such as '0,1', but got '{gpus}'."
        )
        raise click.ClickException(message=message) from error

    prediction_jobs = [PredictionJob(video=video, output=output) for video, output in jobs]
    threshold = None if likelihood_threshold < 0 else likelihood_threshold

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
        )
        summary = run_predictions(
            asset=asset,
            jobs=prediction_jobs,
            profile=profile,
            likelihood_threshold=threshold,
            batch_size=batch_size,
            detector_batch_size=detector_batch_size,
            scratch_directory=scratch_directory,
            display_progress=progress,
        )
    except (ValueError, FileNotFoundError) as error:
        raise click.ClickException(message=str(error)) from error

    for result in summary.results:
        if not result.succeeded:
            click.echo(message=f"failed: {result.video} -> {result.error}", err=True)
    click.echo(message=summary.describe())
    if not summary.successful:
        raise SystemExit(1)
