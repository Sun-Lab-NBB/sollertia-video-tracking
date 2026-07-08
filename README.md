# sollertia-video-tracking

Provides assets for designing and deploying DeepLabCut video tracking pipelines within the Sollertia platform.

![PyPI - Version](https://img.shields.io/pypi/v/sollertia-video-tracking)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/sollertia-video-tracking)
[![uv](https://tinyurl.com/uvbadge)](https://github.com/astral-sh/uv)
[![Ruff](https://tinyurl.com/ruffbadge)](https://github.com/astral-sh/ruff)
![type-checked: mypy](https://img.shields.io/badge/type--checked-mypy-blue?style=flat-square&logo=python)
![PyPI - License](https://img.shields.io/pypi/l/sollertia-video-tracking)
![PyPI - Status](https://img.shields.io/pypi/status/sollertia-video-tracking)
![PyPI - Wheel](https://img.shields.io/pypi/wheel/sollertia-video-tracking)

___

## Detailed Description

This library packages the DeepLabCut-side tooling used to build animal pose-tracking pipelines for the recordings
produced by the Sollertia data acquisition platform. It exposes a unified `slvt` command-line interface backed by a
reusable logic core that expands on the base DeepLabCut functionality to scale it to high-performance compute clusters
and unique constraints of working with large data projects. Because DeepLabCut requires the numpy 1.x series and 
Python 3.12 or earlier, this library runs in its own environment and is driven by the rest of the Sollertia stack 
across the command-line boundary rather than by direct import.

___

## Features

- Supports Linux.
- Provides a unified `slvt` command-line interface backed by a reusable, import-friendly logic core.
- Extracts DeepLabCut training frames in parallel, decoding one video per worker process pinned to a disjoint block of
  CPU cores.
- Trains DeepLabCut models with mixed precision and multi-GPU DistributedDataParallel, falling back cleanly to a single
  GPU or the CPU.
- Runs inference over many videos across multiple GPUs, a single GPU, or the CPU, analyzing one whole video per worker
  and optionally converting predictions in-flight to polars feather files for the rest of the Sollertia stack.
- Renders a single aggregate progress bar across all workers, falling back to periodic greppable progress lines when
  its output is redirected to a log.
- Apache 2.0 License.

___

## Table of Contents

- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
  - [CLI Commands](#cli-commands)
  - [Python API](#python-api)
- [API Documentation](#api-documentation)
- [Developers](#developers)
- [Versioning](#versioning)
- [Authors](#authors)
- [License](#license)
- [Acknowledgments](#acknowledgments)

___

## Dependencies

This library depends on [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut) 3.x and its
[PyTorch](https://pytorch.org/) backend. Because DeepLabCut pins the numpy 1.x series and supports only Python
3.10-3.12, the library must be installed into a dedicated environment, separate from the numpy-2 / Python-3.14
environment used by the rest of the Sollertia stack. A CUDA-capable GPU is recommended for training and deploying
pose-estimation networks.

For users, all other library dependencies are installed automatically by all supported installation methods. For
developers, see the [Developers](#developers) section for information on installing additional development
dependencies.

___

## Installation

### Source

***Note,*** installation from source is ***highly discouraged*** for anyone who is not an active project developer.

1. Download this repository to the local machine using the preferred method, such as git-cloning. Use one of the
   [stable releases](https://github.com/Sun-Lab-NBB/sollertia-video-tracking/tags) that include precompiled binary and
   source code distribution (sdist) wheels.
2. If the downloaded distribution is stored as a compressed archive, unpack it using the appropriate decompression
   tool.
3. `cd` to the root directory of the prepared project distribution.
4. Run `pip install .` to install the project and its dependencies.

### pip

Use the following command to install the library and all of its dependencies via
[pip](https://pip.pypa.io/en/stable/): `pip install sollertia-video-tracking`

___

## Usage

### CLI Commands

This library provides the `slvt` CLI that exposes the following commands:

| Command                   | Description                                                                              |
|---------------------------|------------------------------------------------------------------------------------------|
| `extract frames`          | Selects DeepLabCut training frames by clustering every video in a project in parallel    |
| `extract outliers`        | Extracts a trained model's likely-wrong frames from analyzed videos to refine the model  |
| `extract purge`           | Deletes targeted videos' entire labeled-data folders, labels included, after a dry-run preview |
| `create-training-dataset` | Creates a training-dataset shuffle with a chosen network architecture and train/test split |
| `train`                   | Trains a shuffle with mixed precision and multi-GPU DistributedDataParallel               |
| `infer`                   | Analyzes videos across multiple GPUs, a single GPU, or the CPU, with in-flight polars output |
| `export`                  | Serializes a trained model into a portable, self-contained asset that runs on another machine |
| `predict`                 | Runs an exported model asset over videos into caller-chosen prediction files, cleaning up after itself |

The `extract` group owns the project config.yaml every subcommand operates on, alongside the parallelism and
frame-selection options the `frames` and `outliers` subcommands share; these must be given before the subcommand name,
which then carries its own parameters. The `frames` subcommand grows the project toward a `--total-frames` budget,
always including any videos named with `--videos` and filling the remaining budget from the project's other videos,
optionally balanced across groups of related videos with `--balance-groups`; the `outliers` subcommand extracts
`--frames-per-video` outlier frames from each target video given with `--videos`. Both extract additively and re-roll
their unlabeled frames on `--overwrite` or `--reset` while preserving labeled frames; the `purge` subcommand instead
deletes a video's whole labeled-data folder, labels included, when a clean start is needed. Use `slvt --help`,
`slvt extract --help`, or `slvt COMMAND --help` for detailed usage information.

For example, the following command extracts training frames from every video referenced by a project's config.yaml,
sampling every 500th frame for clustering:
`slvt extract --config-path /path/to/project/config.yaml --clustering-stride 500 frames`

The following command grows the project toward a two-thousand-frame training set while ensuring every group is
represented, balancing the sampled videos across the groups inferred from the components their file names share:
`slvt extract --config-path /path/to/project/config.yaml frames --total-frames 2000 --balance-groups`

The following command extracts frames from two specific videos only, taking `--frames-per-video` frames from each and
ignoring the project-wide budget and group balancing:
`slvt extract --config-path /path/to/project/config.yaml frames --videos video1.mp4 --videos video2.mp4 --exclusive`

The following command extracts outlier frames from two analyzed videos to refine a trained model, adding
`--frames-per-video` corrected frames to each:
`slvt extract --config-path /path/to/project/config.yaml outliers --videos video1.mp4 --videos video2.mp4`

The following command analyzes two videos with a project's trained model, writing a polars feather of predictions per
video into an output directory: `slvt infer /path/to/project/config.yaml video1.mp4 video2.mp4 --dest /path/to/output`

While `infer` operates inside a live project as part of the refinement loop, `export` and `predict` cover deployment.
`export` packages a trained shuffle into a single portable asset that carries everything needed to run the model, with
no dependence on the original project's labeled data, so the asset can be moved to any machine. `predict` then runs that
asset over videos, writing each video's predictions to the exact file path paired with it and removing every
intermediary once it finishes.

The following command exports a project's trained shuffle into a portable model asset:
`slvt export /path/to/project/config.yaml --destination /path/to/model.slvtmodel`

The following command runs that asset over two videos, writing each video's predictions to its own file:
`slvt predict /path/to/model.slvtmodel --job video1.mp4 out1.feather --job video2.mp4 out2.feather --device cuda`

### Python API

The frame-extraction pipeline is also available as a function for programmatic use. It reads every run parameter from
the project's config.yaml, clusters the videos in parallel, and returns a summary of the run:

```python
from pathlib import Path

from sollertia_video_tracking import extract_frames_kmeans

# Clusters every video referenced by the DeepLabCut project's config.yaml, writing the selected frames into each
# video's labeled-data directory. Extraction is additive; pass overwrite=True or reset=True to re-roll unlabeled frames.
summary = extract_frames_kmeans(config_path=Path("/path/to/project/config.yaml"), clustering_stride=500)

print(
    f"{summary.extracted_video_count} extracted, {summary.cleared_frame_count} frames cleared, "
    f"{summary.failed_video_count} failed of {summary.total_video_count}"
)
```

Inference is likewise available as a function. It resolves the optimizations for the detected hardware, distributes
whole videos across worker slots, and returns a summary of the run:

```python
from pathlib import Path

from sollertia_video_tracking import run_inference, resolve_inference_profile

# Resolves the device, precision, and parallelism for the detected hardware (multiple GPUs, a single GPU, or the CPU),
# then analyzes the videos, writing a wide polars feather of predictions per video into the destination directory.
profile = resolve_inference_profile()
summary = run_inference(
    config=Path("/path/to/project/config.yaml"),
    videos=[Path("video1.mp4"), Path("video2.mp4")],
    destination=Path("/path/to/output"),
    profile=profile,
)

print(summary.describe())
```

Model export and deployment are also available as functions. `export_model` writes a portable asset from a trained
shuffle, and `run_predictions` runs that asset over videos into caller-chosen prediction files:

```python
from pathlib import Path

from sollertia_video_tracking import export_model, run_predictions, PredictionJob, resolve_inference_profile

# Packages a trained shuffle into a single portable asset that runs on any machine.
export_model(config=Path("/path/to/project/config.yaml"), destination=Path("/path/to/model.slvtmodel"))

# Runs the asset over videos, writing each video's predictions to the file paired with it and cleaning up every
# intermediary, including the extracted model and DeepLabCut's own prediction files.
profile = resolve_inference_profile()
summary = run_predictions(
    asset=Path("/path/to/model.slvtmodel"),
    jobs=[
        PredictionJob(video=Path("video1.mp4"), output=Path("out1.feather")),
        PredictionJob(video=Path("video2.mp4"), output=Path("out2.feather")),
    ],
    profile=profile,
)

print(summary.describe())
```

___

## API Documentation

See the [API documentation](https://sollertia-video-tracking-api-docs.netlify.app/) for the detailed description of
the methods and classes exposed by components of this library.

***Note,*** the API documentation also includes the details about the `slvt` CLI interface exposed by this library.

___

## Developers

This section provides installation, dependency, and build-system instructions for the developers that want to modify
the source code of this library.

### Installing the Project

***Note,*** this installation method requires **mamba version 2.3.2 or above**. Currently, all automation pipelines
require that mamba is installed through the [miniforge3](https://github.com/conda-forge/miniforge) installer.

1. Download this repository to the local machine using the preferred method, such as git-cloning.
2. If the downloaded distribution is stored as a compressed archive, unpack it using the appropriate decompression
   tool.
3. `cd` to the root directory of the prepared project distribution.
4. Install the core development dependencies into the ***base*** mamba environment via the `mamba install tox uv
   tox-uv` command.
5. Use the `tox -e create` command to create the project-specific development environment followed by `tox -e install`
   command to install the project into that environment as a library.

### Additional Dependencies

In addition to installing the project and all user dependencies, install the following dependencies:

1. A [Python](https://www.python.org/downloads/) 3.12 distribution. DeepLabCut does not support newer Python versions,
   so this library targets Python 3.12 exclusively. It is recommended to use a tool like
   [pyenv](https://github.com/pyenv/pyenv) to install and manage the required version.

### Development Automation

This project uses `tox` for development automation. The following tox environments are available:

| Environment | Description                                                |
|-------------|------------------------------------------------------------|
| `lint`      | Runs ruff formatting, ruff linting, and mypy type checking |
| `stubs`     | Generates py.typed marker and .pyi stub files              |
| `docs`      | Builds the API documentation via Sphinx                    |
| `build`     | Builds sdist and wheel distributions                       |
| `upload`    | Uploads distributions to PyPI via twine                    |
| `install`   | Builds and installs the project into its mamba environment |
| `uninstall` | Uninstalls the project from its mamba environment          |
| `create`    | Creates the project's mamba development environment        |
| `remove`    | Removes the project's mamba development environment        |
| `provision` | Recreates the mamba environment from scratch               |
| `export`    | Exports the mamba environment as .yml and spec.txt files   |
| `import`    | Creates or updates the mamba environment from a .yml file  |

Run any environment using `tox -e ENVIRONMENT`. For example, `tox -e lint`.

***Note,*** all pull requests for this project have to successfully complete the `tox` task before being merged. To
expedite the task's runtime, use the `tox --parallel` command to run some tasks in parallel.

### Automation Troubleshooting

Many packages used in `tox` automation pipelines (uv, mypy, ruff) and `tox` itself may experience runtime failures. In
most cases, this is related to their caching behavior. If an unintelligible error is encountered with any of the
automation components, deleting the corresponding cache directories (`.tox`, `.ruff_cache`, `.mypy_cache`, etc.)
manually or via a CLI command typically resolves the issue.

___

## Versioning

This project uses [semantic versioning](https://semver.org/). See the
[tags on this repository](https://github.com/Sun-Lab-NBB/sollertia-video-tracking/tags) for the available project
releases.

___

## Authors

- Ivan Kondratyev ([Inkaros](https://github.com/Inkaros))

___

## License

This project is licensed under the Apache 2.0 License: see the [LICENSE](LICENSE) file for details.

___

## Acknowledgments

- All Sun lab [members](https://neuroai.github.io/sunlab/people) for providing the inspiration and comments during the
  development of this library.
- The creators of [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut) and all other dependencies and projects listed
  in the [pyproject.toml](pyproject.toml) file.
