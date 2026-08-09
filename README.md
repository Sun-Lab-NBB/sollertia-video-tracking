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

This library packages the DeepLabCut-side tooling used to build, refine, and deploy animal pose-tracking pipelines for
the recordings produced by the Sollertia data acquisition platform. It exposes a unified `slvt` command-line interface
that covers the entire model lifecycle: selecting the initial training frames, preparing and training a shuffle,
analyzing videos, extracting the trained model's likely-wrong frames, and tracking their refinement into the next
iteration. Project creation, manual labeling, and the dataset merge fall outside the CLI, and they run in the
DeepLabCut GUI. The same interface deploys a finished model over new recordings, on hardware ranging from multi-GPU
servers to CPU-only machines. Throughout, it expands on the base DeepLabCut functionality to scale it to
high-performance compute clusters and the unique constraints of working with large data projects.

Because DeepLabCut requires the numpy 1.x series and Python 3.12 or earlier, this library runs in its own environment
and is driven by the rest of the Sollertia stack across the command-line boundary rather than by direct import. The
`slvt` CLI is therefore the library's only supported interface: the Python API exists to serve that CLI and is not
intended to be imported by end users. This library is part of the
[Sollertia](https://github.com/Sun-Lab-NBB/sollertia) AI-assisted scientific data acquisition and processing platform,
built on the [Ataraxis](https://github.com/Sun-Lab-NBB/ataraxis) framework.

___

## Features

- Supports Windows, Linux, and macOS.
- Targets DeepLabCut's PyTorch engine exclusively, tracking its deprecation of the legacy TensorFlow engine.
- Covers the entire DeepLabCut refinement loop through a single `slvt` CLI: frame extraction, shuffle preparation,
  training, inference, outlier extraction, and refinement tracking. Defers to the DeepLabCut GUI for project creation,
  the manual labeling this library does not implement, and DeepLabCut's dataset merge.
- Extracts training and outlier frames in parallel, decoding one video per worker process pinned to a disjoint block of
  CPU cores on Linux and Windows, where the operating system exposes a CPU affinity API.
- Grows a project toward a project-wide frame budget, optionally balanced across groups of related videos.
- Trains shuffles with mixed precision, TF32, cuDNN autotuning, and multi-GPU DistributedDataParallel, exposing every
  optimization as an explicit flag whose automatic default never runs slower than stock DeepLabCut.
- Analyzes videos across multiple GPUs, a single GPU, or the CPU, distributing whole videos across worker slots and
  writing DeepLabCut's native predictions, with crop overrides that analyze de-novo videos.
- Reports the resolved state of every optimization before a training or inference run takes the terminal.
- Renders a single aggregate progress bar across all workers, falling back to periodic greppable progress lines when its
  output is redirected to a log.
- Apache 2.0 License.

___

## Table of Contents

- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
  - [CLI Commands](#cli-commands)
  - [Failure Reporting](#failure-reporting)
  - [Enabling CUDA Support](#enabling-cuda-support)
  - [Workflow Overview](#workflow-overview)
  - [Extracting Initial Frames](#extracting-initial-frames)
  - [Labeling Frames](#labeling-frames)
  - [Preparing a Training Shuffle](#preparing-a-training-shuffle)
  - [Training a Model](#training-a-model)
  - [Analyzing Videos](#analyzing-videos)
  - [Extracting Outlier Frames](#extracting-outlier-frames)
  - [Refining Machine Labels](#refining-machine-labels)
  - [Resetting a Project](#resetting-a-project)
  - [Deployment](#deployment)
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
3.10-3.12, the library targets Python 3.12, the latest DeepLabCut-supported version. It must be installed into a
dedicated Python 3.12 environment, separate from the numpy-2 / Python-3.14 environment used by the rest of the Sollertia
stack. A CUDA-capable GPU is recommended for training and deploying pose-estimation networks, and the `gui` command
additionally requires a graphical session.

***Note,*** this library supports the PyTorch engine only. DeepLabCut still ships its legacy TensorFlow engine and is
deprecating it, so every `slvt` command targets the PyTorch engine exclusively and options that only the TensorFlow
engine would honor are not exposed. A project whose shuffles were created for the TensorFlow engine is not supported.

***Note,*** the DeepLabCut dependency is pinned to an exact version. The training subpackage subclasses DeepLabCut's
training runners, the inference subpackage patches its inference-runner builders, and the frame-extraction subpackage
replaces its k-means frame selector and its video-registration helper. Those internals carry no stability guarantee
within the 3.x series, so bumping the pin is a deliberate, tested action rather than a routine upgrade.

For users, all library dependencies are installed automatically by all supported installation methods. For
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

Use the following command to install the library and all of its dependencies via [pip](https://pip.pypa.io/en/stable/):
`pip install sollertia-video-tracking`

***Note,*** the torch distribution published for Windows carries no CUDA support, so a stock install there runs on the
CPU. Run `slvt cuda` after installing to replace it with the build the local NVIDIA driver runs. See
[Enabling CUDA Support](#enabling-cuda-support) for details.

___

## Usage

### CLI Commands

This library provides the `slvt` CLI that exposes the following commands:

| Command            | Description                                                                                     |
|--------------------|-------------------------------------------------------------------------------------------------|
| `extract frames`   | Selects initial training frames from the project's videos by clustering them in parallel        |
| `extract outliers` | Extracts a trained model's likely-wrong frames from the analyzed videos to refine the model     |
| `extract pending`  | Lists the video directories that still hold machine-labeled frames awaiting refinement          |
| `extract purge`    | Deletes targeted videos' entire labeled-data directories, labels included, after a dry-run      |
| `gui`              | Launches the standard DeepLabCut GUI, used for project creation and manual labeling             |
| `prepare`          | Creates a training-dataset shuffle, selecting the model, weights, augmentation, and split       |
| `train`            | Trains a shuffle with hardware optimizations and a clean progress monitor                       |
| `infer`            | Analyzes videos with a trained model, distributing whole videos across GPU or CPU worker slots  |
| `cuda`             | Installs the CUDA-enabled PyTorch build matching the local NVIDIA driver                        |

Every `slvt` input is an option: no command takes positional arguments. Options offer both a long form and a short
form, which may be multi-letter (`--config-path` and `-cfg`) or single-letter (`--videos` and `-v`), and every command
that operates on a project takes that project's config.yaml through `--config-path`. Options that take one value per
occurrence, such as `--videos`, are given once per value. `--gpus` instead takes all of its indices as a single
comma-separated value (`--gpus 0,1`).

The `extract` group is the one exception to the usual flat structure. It owns the config.yaml and the per-video frame
count `--frames-per-video`, which is a top-up ceiling for `frames` and the number of outlier frames to extract from each
video for `outliers`. It also owns the parallelism and clustering options, and the `--videos`, `--overwrite`, and
`--reset` options that its subcommands share. ***Note,*** these shared options must be given ***before*** the subcommand
name, which then carries its own parameters: `slvt extract --config-path PATH --workers 4 frames --total-frames 2000`.

Use `slvt --help`, `slvt extract --help`, or `slvt COMMAND --help` for detailed usage information. Every option's help
text documents its own defaults and interactions, so the sections below cover the workflow and the choices that matter
rather than restating each flag.

### Failure Reporting

Every command reports a failure rather than ending quietly. The work each one distributes runs in worker processes, so
the command process keeps its own console and stays able to speak for a worker that ends badly. A worker can raise,
crash inside a native backend, or be killed by the operating system's out-of-memory killer. Each case produces an error
naming the signal, the shell exit status, the affected video or shuffle, and the remedy that applies. Each worker also
installs the interpreter's fault handler, so a native crash leaves a readable dump on the standard error stream.

The batch commands report each unit separately. `slvt infer` and `slvt extract` print every failed video's reason
before their summary line, and finish with a non-zero exit status whenever any video failed or was skipped for an
unmet precondition, so a partially successful run is never mistaken for a complete one. An option value Click itself
rejects is refused at parse time with status 2, and one a command validates in its own body, such as `--gpus`, ends
the run with status 1. Stopping a run with Ctrl-C exits with status 130 and reports what was already written.

| Outcome                                                    | Exit status |
|------------------------------------------------------------|-------------|
| Every submitted unit succeeded                             | 0           |
| Some units succeeded and some failed                       | 1           |
| A named video was skipped for an unmet precondition        | 1           |
| The run itself could not start or complete                 | 1           |
| Training completed but its evaluation failed               | 1           |
| The operator interrupted the run                           | 130         |
| A Click-typed option value was rejected                    | 2           |
| A command-validated option value was rejected              | 1           |

### Enabling CUDA Support

The torch distribution published for Windows carries no CUDA support, so a stock install there runs training and
inference on the CPU. `slvt cuda` repairs that. It reads the CUDA version the local NVIDIA driver runs, resolves the
newest PyTorch build that version supports, and replaces the installed torch and torchvision distributions with it.
The replacement runs through uv where it is available and through pip otherwise, and it targets the environment this
library is installed into rather than whichever environment happens to be active.

The command reports the driver it found, the build it resolved, and the commands it would run, changing nothing until
`--yes` is given: `slvt cuda`

A build that already targets CUDA and reaches the local GPUs is left alone, which makes the command a no-op on a Linux
host, where the published torch distribution already carries CUDA support. Pass `--force` to replace such a build
anyway, `--cuda-version` to name the CUDA version instead of reading it from the driver, and `--torch-version` to pin an
exact torch version rather than taking the newest one the wheel index carries.

***Note,*** PyTorch publishes no CUDA build for macOS, so the command reports that none applies there. Apple hardware
runs through the Metal (MPS) backend, which the standard distribution already provides.

### Workflow Overview

The commands compose into DeepLabCut's model refinement loop, but not the project that hosts it. The project is created
outside `slvt`, from the DeepLabCut GUI's project-management window reached with `slvt gui`. That window writes the
config.yaml and registers the project's videos under its `video_sets`, which is what every `--config-path` below points
to. This library has no project-creation command, and `slvt extract frames` refuses to run against a config.yaml that
registers no videos. Once the project exists, it is bootstrapped once through the first two steps, then cycles through
the remaining ones, each pass adding human-corrected frames that train a more accurate model:

1. `slvt extract frames` selects the initial training frames from raw video.
2. `slvt gui` labels them by hand.
3. `slvt prepare` creates a training-dataset shuffle.
4. `slvt train` fits the shuffle and evaluates the result.
5. `slvt infer` analyzes videos with the trained model.
6. `slvt extract outliers` flags the model's likely-wrong frames as machine pre-labels.
7. `slvt gui` corrects those pre-labels, tracked by `slvt extract pending`, and advances the project's iteration.
8. The loop returns to step 3 to train the next iteration on the expanded label set.

Steps 3 through 8 repeat until the model's accuracy is sufficient. Once it is, deployment needs only step 5.

### Extracting Initial Frames

`slvt extract frames` bootstraps a project's training set by clustering raw video and keeping the frames that best
cover the visual variation. Each video is clustered in its own worker process, pinned on Linux and Windows to a
disjoint block of CPU cores, so extraction scales across the machine rather than decoding one video at a time. macOS
exposes no CPU affinity API, so its workers run in parallel but unpinned.

Frame selection is governed by two budgets. `--frames-per-video`, given before the subcommand, is a per-video ceiling,
and `--total-frames` (default 200), given after it, is the project-wide budget: videos are selected until the budget is
reached, preferring not-yet-extracted videos and falling back to below-ceiling ones. Videos named with `--videos` are
included first, except that a video already in outlier refinement is skipped with a warning. Naming a refined video
together with `--overwrite` instead aborts the run before any frames are extracted, because `extract frames` is the
pre-refinement bootstrap step and never disturbs a video that has entered refinement. Extraction is a top-up rather
than a re-roll, so a fresh video gains a full set while a partly-extracted one gains only the frames that reach the
ceiling. If topping every eligible video to the ceiling still cannot reach the total in one pass, the run reports the
shortfall and stops rather than silently under-delivering.

`--balance-groups` spreads the sample evenly across groups of related videos so every group is represented, inferring
the groups from the parts of the file names the videos share. `--group-regex` supplies an explicit pattern for naming
schemes the automatic grouping does not recognize. `--exclusive` restricts the run to exactly the `--videos` files,
ignoring the budget and group balancing entirely.

`--clustering-stride` controls how far apart, in frames, the run samples before clustering: for `frames` it strides the
whole video, and for `outliers` it strides the flagged candidate frames. It defaults to 1, which uses every frame.
Raising it clusters fewer frames, trading coverage for processing speed. `--workers` and `--cores` tune the parallelism,
and both default to deriving a saturating layout from the machine's core count.

The following command grows the project toward a two-thousand-frame training set while ensuring every group of related
videos is represented:
`slvt extract --config-path /path/to/project/config.yaml frames --total-frames 2000 --balance-groups`

The following command instead restricts the run to two specific videos, topping each up to `--frames-per-video` frames:
`slvt extract --config-path /path/to/project/config.yaml --videos video1.mp4 --videos video2.mp4 frames --exclusive`

Passing `--overwrite` clears the selected videos' unlabeled frames so they are re-rolled from scratch, and `--reset`
does the same across every not-yet-refined project video. Both preserve already-labeled and outlier frames, and both
refuse to disturb videos that have already entered outlier refinement. The two are mutually exclusive, and `--reset`
also cannot be combined with `--exclusive`, since one re-rolls the whole project while the other restricts the run to
the requested videos: use `--overwrite` to re-roll only those.

### Labeling Frames

`slvt gui` launches the standard DeepLabCut GUI: the same fully functional application reached by running `python -m
deeplabcut`. It takes no options. The project's config.yaml is created or opened from the application's own
project-management window, and the frame labeler, which opens in napari, is reached from there.

Manual labeling is the step of the refinement loop this library does not implement, because it is inherently
interactive. The GUI is therefore used for exactly the steps that have no `slvt` equivalent. Those are creating the
project and registering its videos, labeling the extracted frames, correcting the machine pre-labels that `slvt extract
outliers` produces, and merging the refined dataset to advance the project's iteration.

***Note,*** the GUI also offers tabs for frame extraction, training, evaluation, analysis, and outlier extraction, but
those tabs run the stock DeepLabCut implementations, which decode one video at a time and apply none of the training
and inference optimizations. Every step that has an `slvt` command is faster through that command, so the GUI is best
reserved for the work that only it can do.

The GUI needs a graphical session, so it runs on a workstation rather than a headless training or inference server. On
Linux it refuses to start with an explanatory error when neither `DISPLAY` nor `WAYLAND_DISPLAY` is set. The command
blocks until the window is closed.

### Preparing a Training Shuffle

`slvt prepare` creates the training-dataset shuffle that `slvt train` fits, mirroring the DeepLabCut GUI's
create-training-dataset tab for headless and scripted use. A shuffle bakes in the model architecture, the weight
initialization, the augmentation pipeline, and a train/test split, all of which are fixed once it is created.
`--shuffle` (default 1) is the index that identifies it everywhere downstream.

`--network` selects the pose-model architecture from DeepLabCut's full catalog, and omitting it uses the project
default. The augmentation pipeline is not selected separately, since it follows from the architecture's top-down or
bottom-up task. A top-down architecture also trains an object detector, chosen with `--detector`, and naming a
conditional top-down (`ctd_*`) architecture with `--network` additionally requires `--conditional-top-down-conditions`:
the predictions file (`.h5` or `.json`) or model snapshot (`.pt`) it conditions on. `--weight-initialization` controls
the starting weights: `imagenet` (the default) needs nothing further, while `transfer` and `fine-tune` do SuperAnimal
transfer learning or fine-tuning and both require a `--super-animal` dataset and an explicit `--network`.
`--from-shuffle` reuses an existing shuffle's train/test split instead of drawing a fresh one, which isolates an
architecture comparison from split variance. `--overwrite` replaces an existing shuffle's training-dataset files.

The following command creates shuffle 3 with an HRNet-W32 architecture:
`slvt prepare --config-path /path/to/project/config.yaml --shuffle 3 --network hrnet_w32`

***Note,*** `--memory-replay` is only valid with `--weight-initialization fine-tune`, and such shuffles can be created
but cannot be trained by `slvt train`.

### Training a Model

`slvt train` fits a prepared shuffle. Every optimization is exposed as a flag, and every flag's automatic default is
chosen for the detected hardware such that it never runs slower than stock DeepLabCut. Explicit values allow further
tuning for known hardware.

Training runs on a single GPU (index 0) by default, because multi-GPU training is often slower for DeepLabCut
workloads. Multi-GPU is an explicit opt-in: `--gpus 0,1` selects the devices and `--multi-gpu` picks the strategy,
defaulting to DistributedDataParallel for two or more GPUs, with the slower DataParallel available as `dp`. 
Single-process training also covers the CPU and Apple MPS through `--device`.

`--amp` sets the mixed-precision mode, enabling bfloat16 on Ampere or newer GPUs automatically and staying in float32
elsewhere. ***Note,*** DataParallel cannot combine with mixed precision, since autocast does not reach its per-GPU
replica threads: selecting `dp` disables `--amp` with a warning and trains in float32, so DistributedDataParallel is
required to pair mixed precision with multi-GPU training.

`--tf32`, `--cudnn-benchmark`, `--compile-model`, `--pin-memory`, and `--dataloader-workers` tune the remaining
accelerations. Two automatic defaults are deliberately conservative. `--cudnn-benchmark` engages only when the shuffle's
training transform is detected to feed one fixed input size, since it disables deterministic training and can slow
variable-size augmentation. `--compile-model` stays off because its one-time warm-up cost may not amortize.

`slvt train` and `slvt infer` both write the resolved state of every optimization to the standard error stream before
the progress display takes the terminal, so a run's configuration is visible while there is still time to stop it and
start over:

```text
-- training optimizations -------
  device              cuda [0, 1]
  strategy            ddp
  processes           2
  precision           bfloat16
  tf32                on
  cudnn.benchmark     off
  torch.compile       off
  dataloader workers  8
  pin_memory          on
---------------------------------
```

`slvt infer` writes the same block under an `inference optimizations` title, carrying `channels_last`, the per-GPU
process count, `chunks`, and the worker count the run actually spawns in place of the training-only rows. The report is
written whether or not `--no-progress` is passed, so a run redirected to a log opens with the configuration it ran
under.

The run's length and checkpointing are set by `--epochs`, `--batch-size`, `--save-epochs`, and `--maximum-snapshots`.
Omitting each uses the model's default. `--snapshot-path` resumes from an existing snapshot, and
`--no-load-head-weights` pairs with it when the project's bodypart set has changed and the snapshot's head no longer
matches. Top-down models expose the detector's own `--detector-epochs`, `--detector-batch-size`,
`--detector-save-epochs`, and `--detector-path`. Setting `--detector-epochs 0` skips detector training and fits only the
pose model.

By default, training ends by scoring the trained snapshot against the labeled frames and reporting the headline train
and test error. It writes two files into the shuffle's evaluation-results directory: `<snapshot>_evaluation.feather`,
the per-frame, per-keypoint comparison against the human labels, and `<snapshot>_evaluation.yaml`, the full metric set
and the run's provenance. Together they answer whether the model is accurate enough to leave the refinement loop.
`--evaluation-confidence-cutoff` sets the confidence below which predictions are excluded from the cutoff-filtered
error, falling back to the project's p-cutoff. `--evaluation-batch-size` (default 1) is worth raising on a capable GPU.
`--no-evaluate` finishes at the last snapshot instead.

A failed run additionally carries the cause, not only its location. A worker that raised has its traceback reproduced
in the report, and a worker that died without unwinding instead has the tail of the shuffle's `train.txt` quoted,
where the workers' diverted output is captured. A run whose training completes but whose
evaluation fails keeps its snapshots, reports them on standard output, appends the evaluation traceback to `train.txt`,
and still exits non-zero, because the evaluation files a later refinement decision reads were not written.

The following command trains shuffle 1 across two GPUs for 200 epochs:
`slvt train --config-path /path/to/project/config.yaml --gpus 0,1 --epochs 200`

### Analyzing Videos

`slvt infer` analyzes videos with a shuffle's trained model. Each worker pulls work from a shared queue, so the load is
balanced across the run's slots, and the same command runs on multiple GPUs, one GPU, or a CPU-only machine.

Providing `--videos` once per file analyzes those videos. `--videos-directory` instead analyzes every video stored
directly inside one directory, scanned a single level deep for the extensions DeepLabCut recognizes, leaving
subdirectories and the `_labeled` and `_full` companion files DeepLabCut writes beside an analyzed video out of the
selection. Providing neither analyzes every video registered in the project's config.yaml that still exists on disk,
silently skipping registered paths whose files are missing. The videos need not be registered in the project at all,
which is what allows de-novo recordings to be analyzed. `--crop` takes an `x1,x2,y1,y2` rectangle that replaces the
project's configured crop, given once to apply one rectangle to every video or once per `--videos` file for per-video
crops. A directory run and a whole-project run both discover their video order rather than taking it from the operator,
so each accepts a single `--crop` and a single `--output`.

Predictions are written as DeepLabCut's native `.h5` files. By default, each lands beside its own video, which is where
`slvt extract outliers` reads them, so the default keeps the refinement loop wired together. `--output` redirects them,
taking either one shared directory or one directory per `--videos` file. A prediction file is named after its video's
stem, so a run whose videos share a stem and resolve to one output directory is rejected before any analysis starts.

`--device` defaults to using every visible GPU, with `--gpus` narrowing the selection. `--gpu-processes` sets the worker
processes per GPU, defaulting to one process (one video) per GPU. Most GPUs saturate with 1 or 2 workers: raising it
oversubscribes a GPU so one worker's forward pass fills the gaps another leaves, and the useful factor is
workload-dependent and best found by measurement. `--chunks` splits each running video into that many contiguous frame
ranges analyzed at once, making total per-GPU concurrency `--gpu-processes` times `--chunks`. Unlike raising
`--gpu-processes`, it fills the decode and preprocessing gaps within a single long video, and the parent stitches each
video's ranges back into the one `.h5` a whole-video run produces, so it defaults to one. `--batch-size` sets the frames
the pose model processes per forward pass, where larger batches use more GPU memory and can speed up analysis, and
top-down models expose the detector's own `--detector-batch-size`. Omitting each uses the model's default. On CPU-only
machines, `--cpu-workers` and `--cpu-threads-per-worker` divide the cores into disjoint per-worker blocks. `--amp`,
`--tf32`, `--cudnn-benchmark`, and `--compile-model` carry the same meaning they do for `slvt train`, and
`--channels-last` additionally speeds up convolutions on tensor-core GPUs.

The following command analyzes two de-novo videos at a chosen crop rectangle, writing each video's predictions beside
it:
`slvt infer --config-path /path/to/project/config.yaml --videos video1.mp4 --videos video2.mp4 --crop 0,550,0,550`

The following command analyzes every video held in one directory at that same rectangle:
`slvt infer --config-path /path/to/project/config.yaml --videos-directory /path/to/videos --crop 0,550,0,550`

***Note,*** conditional-top-down models run at stock precision, as the acceleration path does not apply to them.

***Note,*** `--chunks` above one applies to single-animal bottom-up models only, as stitching per-frame predictions
does not reproduce multi-animal tracking. A multi-animal project, or a top-down or conditional-top-down shuffle, must
run with `--chunks 1`.

### Extracting Outlier Frames

`slvt extract outliers` closes the loop by finding the frames the trained model most likely got wrong and adding them to
the training set. It refines the videos given with `--videos`, or every registered video the current model has already
analyzed when `--videos` is omitted. Each target must be registered in the project's config.yaml ***and*** already
analyzed by `slvt infer`, because the detectors read the model's stored predictions rather than re-running the model.
Requested paths that are not registered project videos are skipped with a warning.

`--outlier-algorithm` chooses how likely-wrong frames are identified: `uncertain` (the default) flags low-confidence
predictions, `jump` flags large frame-to-frame motion, `fitting` flags departures from a fitted motion trajectory, and
`list` takes explicit `--frame-index` values. `--pixel-distance-threshold` sets how far a bodypart may move or depart
from its trajectory before its frame is flagged, and `--minimum-confidence` sets the confidence below which a
prediction is treated as unreliable. `--comparison-bodyparts` restricts the detectors to specific bodyparts, which
matters when only some are of interest. `--extraction-algorithm` then chooses which of the flagged candidates to keep.

The flagged frames are clustered in parallel, one video per worker, and added to each video's labeled-data directory
alongside the model's predictions as machine pre-labels. `--frames-per-video`, given before the subcommand, sets how
many outlier frames each refined video contributes per pass. Unlike the ceiling it imposes on `frames`, here it is a
per-pass count. `--save-labeled` additionally saves a copy of each extracted frame with the predictions drawn on it,
which is useful for eyeballing what the model is getting wrong.

The following command extracts the current model's least-confident frames from two analyzed videos:
`slvt extract --config-path /path/to/project/config.yaml --videos video1.mp4 --videos video2.mp4 outliers`

***Note,*** the `fitting` detector fits a SARIMAX trajectory per keypoint and is by far the most expensive option.
`--fit-workers` parallelizes those fits and by default uses every usable core, leaving a small reserve free.

Outlier extraction is additive, so repeated passes grow the refinement set. `--overwrite` replaces the refined videos'
outlier frames for the current iteration and `--reset` clears the whole iteration's outlier frames before
re-extracting. Both preserve every frame carrying a finite human coordinate, and both clear a frame that was opened in
the labeler but never annotated.

### Refining Machine Labels

Each extracted outlier frame is saved as a machine pre-label that a human corrects in `slvt gui`. The corrections are
saved into the directory's human label table, and a machine frame counts as refined only once it carries a finite human
coordinate. An all-NaN placeholder row, which the GUI writes for a frame that was opened but never touched, does not
count.

`slvt extract pending` reports which video directories still hold unrefined machine frames for the current iteration
and how many each has, so the next directories to open are obvious. It only reads the project, changing nothing:
`slvt extract --config-path /path/to/project/config.yaml pending`

***Note,*** this library does not wrap DeepLabCut's dataset merge, the step that advances the project's `iteration`
counter and folds the refined frames into the training set. That step runs from the DeepLabCut GUI, and it refuses to
advance until every labeled-data directory holds either a human label table or a refinement table. `slvt extract
pending` reports the stricter, per-frame view of the same workflow: which of the current iteration's machine frames
still lack a human correction. Clearing `pending` is therefore the labeling goal, not the merge's own gate, which only
probes for the presence of a label table. Once the iteration has advanced, `slvt prepare` creates a fresh shuffle for
the expanded label set and the loop returns to `slvt train`.

### Resetting a Project

`slvt extract purge` deletes each targeted video's entire `labeled-data` directory, human labels included. It is the
wholesale reset that the frame and outlier re-extraction options deliberately avoid: where `--overwrite` and `--reset`
clear only unlabeled or single-iteration frames and always keep the human labels, `purge` removes everything. It exists
for the rare start-completely-over case, such as changing the project's crop, that the label-preserving options cannot
serve.

The command purges the videos given with `--videos`, each of which must be registered in the project's config.yaml.
Requested paths that match no registered project video are skipped with a warning. ***Omitting `--videos` purges the
whole project***, removing every video directory in the project's `labeled-data` tree, including directories left behind
by videos no longer registered in config.yaml.

***Warning!*** Purging destroys human labeling work that cannot be recovered by re-running any other command. The
command previews what it would delete and removes nothing until `--yes` is given, so the preview is the way to confirm
the scope before committing to it:
`slvt extract --config-path /path/to/project/config.yaml --videos video1.mp4 purge`

### Deployment

Deploying a finished model needs only `slvt infer`. The command's inputs are the project's config.yaml, the videos, and
optionally a `--crop`, so a deployment host needs no more than a DeepLabCut project whose shuffle holds a trained
snapshot. Because the analyzed videos need not be registered in the project, deployment analyzes de-novo recordings
without adding videos or labels to the project.

The full project directory copied from the training machine works as-is. Analysis reads only the project configuration
and the trained shuffle, so a project truncated to those parts serves equally well and avoids copying the labeled data
and training videos, which are typically the bulk of a mature project. The minimum tree is:

```text
project-root/
├── config.yaml
└── dlc-models-pytorch/
    └── iteration-N/                                    # must match config.yaml's `iteration`
        └── TaskDate-trainsetFFshuffleS/                # `FF` is the training fraction as a percentage
            ├── train/
            │   ├── pytorch_config.yaml
            │   └── snapshot-best-NNN.pt                # plus snapshot-detector-NNN.pt for top-down models
            └── test/
                └── pose_cfg.yaml
```

The following command deploys such a project over a new recording, collecting the predictions in a chosen directory:
`slvt infer --config-path /path/to/deployed/config.yaml --videos session.mp4 --output /path/to/output`

***Note,*** the `test/pose_cfg.yaml` file is read during analysis and is easy to miss when assembling a truncated
project. The `--shuffle` index given to `slvt infer` must match the copied shuffle's directory, and `config.yaml`
should be copied verbatim rather than handwritten, since DeepLabCut rewrites an incomplete configuration through its
own template. The configuration's internal `project_path` does not need to be corrected: DeepLabCut resolves the
project from the config.yaml's own location and self-corrects the stale value, rewriting config.yaml in place.

Within the Sollertia platform, this deployment step is driven by the acquisition stack rather than by hand. The
experiment preprocessing pipeline launches `slvt infer` over each session's camera recordings and writes the predictions
beside the video, where they are shipped as part of the session's raw data.

___

## API Documentation

See the [API documentation](https://sollertia-video-tracking-api-docs.netlify.app/) for the detailed description of
the methods and classes exposed by components of this library.

***Note,*** the API documentation also includes the details about the `slvt` CLI interface exposed by this library.
The library's Python API backs that CLI and is not intended to be called directly by end users.

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

| Environment    | Description                                                 |
|----------------|-------------------------------------------------------------|
| `lint`         | Runs ruff formatting, ruff linting, and mypy type checking  |
| `stubs`        | Generates py.typed marker and .pyi stub files               |
| `{py312}-test` | Runs the test suite via pytest and aggregates coverage data |
| `coverage`     | Aggregates test coverage and applies the 100% coverage gate |
| `docs`         | Builds the API documentation via Sphinx                     |
| `build`        | Builds sdist and wheel distributions                        |
| `upload`       | Uploads distributions to PyPI via twine                     |
| `deploy`       | Uploads the built documentation to the Netlify site         |
| `install`      | Builds and installs the project into its mamba environment  |
| `uninstall`    | Uninstalls the project from its mamba environment           |
| `create`       | Creates the project's mamba development environment         |
| `remove`       | Removes the project's mamba development environment         |
| `provision`    | Recreates the mamba environment from scratch                |
| `export`       | Exports the mamba environment as a .yml file                |
| `import`       | Creates or updates the mamba environment from a .yml file   |

Run any environment using `tox -e ENVIRONMENT`. For example, `tox -e lint`.

***Note,*** all pull requests for this project have to successfully complete the `tox` task before being merged. To
expedite the task's runtime, use the `tox --parallel` command to run some tasks in parallel.

### AI-Assisted Development

Claude Code skills and other AI development assets for this project are distributed through the
[ataraxis](https://github.com/Sun-Lab-NBB/ataraxis) marketplace as part of the **automation** plugin. Install the
plugin from the marketplace to make all associated skills and development tools available to compatible AI coding
agents.

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
- The creators of [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut), on whose pose-estimation framework this
  library builds.
- The creators of all other dependencies and projects listed in the [pyproject.toml](pyproject.toml) file.
