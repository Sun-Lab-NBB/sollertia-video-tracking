# Claude Code Instructions

## Session start behavior

At the beginning of each coding session, before making any code changes, you should build a comprehensive
understanding of the codebase by invoking the `automation:explore-codebase` skill. This library overrides DeepLabCut
internals at an exactly pinned version, so exploring first prevents changes that silently diverge from them.

## Style guide compliance

You MUST invoke the appropriate style skill before performing ANY of the following tasks:

| Task                                          | Skill to invoke              |
|-----------------------------------------------|------------------------------|
| Writing or modifying Python code              | `automation:python-style`    |
| Writing or modifying README files             | `automation:readme-style`    |
| Writing git commit messages                   | `automation:commit`          |
| Writing or modifying skill files or this file | `automation:skill-design`    |
| Writing or modifying pyproject.toml           | `automation:pyproject-style` |
| Writing or modifying tox.ini                  | `automation:tox-config`      |
| Writing or modifying Sphinx docs files        | `automation:api-docs`        |

Each skill contains a verification checklist that you MUST complete before submitting any work.

Four project-specific deviations are deliberate. You MUST NOT report them as violations or "fix" them:

- **Short option flags may be multi-letter** (`-cfg`, `-ctdc`, `-ctpw`, `-crw`, `-mad`). This is a local CLI contract.
- **`deeplabcut[gui]==3.0.1` is an exact pin**, not a major-version range. The rationale is in `pyproject.toml`.
- **`tox.ini` uses classic `deps`**, not `dependency_groups`. DeepLabCut caps tox at 4.20, and tox below 4.22 ignores
  `dependency_groups` silently. The `tox.ini` comment explains this.
- **Errors use standard `raise`, `click.ClickException`, `click.UsageError`, and the local `warn()` helper** in
  `hardware/detection.py`. This project does not depend on `ataraxis-base-utilities`, so there is no `console`.

Deferred or inline imports are forbidden without exception. Never add a `# noqa: PLC0415`. CLI startup cost is not a
justification, since `__init__.py` already imports DeepLabCut eagerly.

## Cross-referenced library verification

This project has no `ataraxis-*` or `sollertia-*` runtime dependencies. `ataraxis-automation` supplies the
development toolchain through the `dev` dependency group and `tox.ini`. The only substantial runtime dependency is
DeepLabCut, and the other Sollertia libraries consume this library exclusively through the `slvt` CLI.

| Dependency               | Role                                                                         |
|--------------------------|------------------------------------------------------------------------------|
| `deeplabcut[gui]`        | Pinned to exactly 3.0.1. Provides the pose models, project format, and GUI   |
| `numpy`                  | Pinned to the 1.x series required by DeepLabCut                              |
| `torch`                  | Training and inference backend. DDP from `training/`, AMP from `hardware/`   |
| `triton-windows`         | `torch.compile` inductor kernels, Windows only                               |
| `opencv-python-headless` | Video decode (headless variant only, never add `opencv-python` alongside)    |
| `polars`, `psutil`       | Evaluation feather output, and cross-platform worker core affinity           |
| `click`                  | The `slvt` command tree, where every command and option is a Click decorator |
| `ruamel.yaml`            | Reads and round-trips `config.yaml`, writes `<snapshot>_evaluation.yaml`     |

**The environment is isolated by design.** `pyproject.toml` requires `>=3.12,<3.13` because DeepLabCut 3.x supports
only Python 3.10-3.12 and numpy 1.x. The rest of the Sollertia stack runs Python 3.14 and numpy 2, so it **cannot
import this library at all** and drives `slvt` as a subprocess instead. You MUST NOT propose importing this library
from another Sollertia project, nor relax the Python or numpy bounds to "align" it with the stack.

**Before writing code that touches a DeepLabCut override site, you MUST** read the corresponding source in the
installed DeepLabCut 3.0.1 and confirm the override still matches it. The overrides depend on private internals with
no stability guarantee: `training/runners.py` subclasses DeepLabCut's training runners and relies on their MRO,
`inference/runners.py` patches `get_pose_inference_runner` and `get_detector_inference_runner`, and `frame_extraction/`
replaces `KmeansbasedFrameselectioncv2` and `attempt_to_add_video` inside its workers.

## Available skills

This project ships no skills of its own and has no Claude Code plugin. The skills below come from the external
ataraxis **automation** plugin and are the only ones that apply here.

| Skill                             | Description                                                      |
|-----------------------------------|------------------------------------------------------------------|
| `automation:explore-codebase`     | Perform in-depth codebase exploration at session start           |
| `automation:explore-dependencies` | Explore installed dependency source to build a live API snapshot |
| `automation:python-style`         | Apply Python coding conventions (REQUIRED for Python work)       |
| `automation:readme-style`         | Apply README conventions (REQUIRED for README work)              |
| `automation:pyproject-style`      | Apply pyproject.toml conventions (REQUIRED for pyproject.toml)   |
| `automation:tox-config`           | Apply tox.ini conventions (REQUIRED for tox.ini work)            |
| `automation:api-docs`             | Apply Sphinx documentation conventions (REQUIRED for docs work)  |
| `automation:project-layout`       | Apply project directory structure conventions                    |
| `automation:skill-design`         | Apply skill and CLAUDE.md conventions (REQUIRED for this file)   |
| `automation:audit-correctness`    | Audit source for active and latent bugs                          |
| `automation:audit-facts`          | Fact-check documentation against authoritative source            |
| `automation:audit-performance`    | Audit source for algorithmic, allocation, and dtype costs        |
| `automation:audit-project`        | Orchestrate all four audits and merge their findings             |
| `automation:audit-style`          | Audit files against the applicable style checklists              |
| `automation:commit`               | Stage changes and create a style-compliant commit                |
| `automation:pr`                   | Draft a style-compliant pull request summary                     |
| `automation:release`              | Draft style-compliant release notes                              |

## Driving the slvt CLI

This library ships no MCP server and no project-specific plugin, so the `slvt` CLI is the only agent-facing surface it
provides, and the Python API under `src/sollertia_video_tracking/` exists solely to back that CLI. Unless the user
explicitly directs otherwise, you should resolve every request through a `slvt` invocation and MUST NOT reach for an
import or a handwritten driver script on your own initiative. An explicit user request overrides this default: when
the user directly asks for a driver script or to use the Python API, provide it.

The CLI is defined in `src/sollertia_video_tracking/interfaces/`: `entry_points.py` registers the root group, and
`extract.py`, `gui.py`, `prepare.py`, `train.py`, `infer.py`, and `cuda.py` own one command each. Read the module, or
run `slvt COMMAND --help`, before constructing a non-trivial invocation, and verify any recalled recipe against
`--help` before trusting it. `README.md` is authoritative on user-facing behavior.

### Request-to-command mapping

| The user asks for                                                      | Command to run                            |
|------------------------------------------------------------------------|-------------------------------------------|
| "get frames to label", "bootstrap the training set", "grow the set"    | `slvt extract --config-path CFG frames`   |
| "label frames", "create the project", "correct these", "merge/advance" | `slvt gui` (workstation only)             |
| "make a shuffle", "try a different architecture", "new split"          | `slvt prepare --config-path CFG`          |
| "train the model", "fit the shuffle", "resume from a snapshot"         | `slvt train --config-path CFG`            |
| "how accurate is it?", "evaluate the model"                            | `slvt train` (evaluates by default)       |
| "analyze the videos", "get predictions", "deploy the model"            | `slvt infer --config-path CFG --videos V` |
| "find what the model got wrong", "next refinement round"               | `slvt extract --config-path CFG outliers` |
| "what still needs labeling?", "how much is left?"                      | `slvt extract --config-path CFG pending`  |
| "start completely over", "I changed the crop"                          | `slvt extract --config-path CFG purge`    |
| "torch runs on the CPU", "enable CUDA", "the GPU is not being used"    | `slvt cuda`                               |

Two requests have no `slvt` command and MUST be routed to `slvt gui`: **project creation with its `video_sets`
registration**, and **DeepLabCut's dataset merge** that advances the project's `iteration`. There is no other way to
do either. There is also no separate evaluate command: `slvt train` evaluates the trained snapshot unless
`--no-evaluate` is given, writing `<snapshot>_evaluation.feather` and `<snapshot>_evaluation.yaml` into the shuffle's
evaluation-results directory. Evaluating an already-trained shuffle means reading those two files, not re-training.

### Refinement loop order

Each step's precondition is a hard gate. When a request names a step whose precondition is unmet, run or surface the
missing step first rather than letting the command fail.

| Step | Command                                       | Precondition                                            |
|------|-----------------------------------------------|---------------------------------------------------------|
| 0    | `slvt gui`                                    | Nothing. Creates config.yaml and registers videos       |
| 1    | `slvt extract ... frames`                     | config.yaml registers at least one video                |
| 2    | `slvt gui`                                    | Frames extracted                                        |
| 3    | `slvt prepare`                                | Labeled frames exist                                    |
| 4    | `slvt train`                                  | The `--shuffle` index exists (step 3 created it)        |
| 5    | `slvt infer`                                  | The shuffle holds a trained snapshot                    |
| 6    | `slvt extract ... outliers`                   | Registered in config.yaml AND already analyzed by infer |
| 7    | `slvt gui`, tracked by `slvt extract pending` | Outlier frames extracted                                |
| 8    | Back to step 3                                | The GUI's merge advanced the project's `iteration`      |

Steps 0-2 bootstrap once, and steps 3-8 cycle. Deploying a finished model needs only step 5.

Step 6 goes wrong most often: the outlier detectors read the model's **stored predictions** and do not re-run the
model, so `slvt extract outliers` on a video never passed through `slvt infer` finds nothing. Outlier extraction is
additive, so successive passes grow the same refinement set. That is how multiple detector rounds compose, and it is
why `--overwrite` and `--reset` are rarely what a refinement request means. `slvt extract pending` is stricter than
the GUI merge's own gate: a frame counts as refined only once it carries a finite human coordinate, so an all-NaN
placeholder row the GUI wrote for an opened-but-untouched frame still reads as pending.

### Invocation rules

- **Every input is an option. No command takes positional arguments.** `slvt infer CFG video.mp4` is always wrong. It
  is `slvt infer --config-path CFG --videos video.mp4`.
- **`extract` group options go BEFORE the subcommand name.** The group owns `--config-path`, `--workers`, `--cores`,
  `--frames-per-video`, `--clustering-stride`, `--clustering-resize-width`, `--color/--grayscale`,
  `--progress/--no-progress`, `--videos`, `--overwrite`, and `--reset`. The subcommand carries its own. Correct:
  `slvt extract --config-path CFG --workers 4 --videos a.mp4 outliers --outlier-algorithm uncertain`. Every other
  command is flat.
- **`--gpus` is one comma-separated value, and `--videos` repeats.** Use `--gpus 0,1`, never `--gpus 0 --gpus 1`.
  Conversely `--videos a.mp4 --videos b.mp4`, never `--videos a.mp4,b.mp4`. The `infer` command's `--output` and
  `--crop` follow `--videos`: one value applies to every video, or give one per `--videos` file. Multiple values
  without an explicit `--videos` is a `UsageError`, because whole-project video order is not user-controlled.
- **Omitting `--videos` scopes the run to the whole project, which is not the same as processing every video.** For
  `purge` and `pending` it is every project video. For `frames` the project is only the candidate pool that the
  default `--total-frames 200` budget samples from, so pass `--total-frames -1` to top up every eligible video
  instead. For `outliers` it is every registered video the current model has analyzed, and for `infer` every
  registered video still present on disk. There is no additive "do everything" flag, and an omitted `--videos` on
  `purge` is a project-wide deletion.
- **Prefer long forms in every command shown to the user.** Short forms may be multi-letter (`-cfg`, `-ctpw`), and the
  same short flag means different things across commands. `-o` is `--overwrite` on `extract`/`prepare` but `--output`
  on `infer`. `-cb` is `--comparison-bodyparts` on `outliers` but `--cudnn-benchmark` on `train`/`infer`. `-d` is
  `--device` on `train`/`infer` but `--detector` on `prepare`. `-e` is `--epochs` on `train` but `--exclusive` on
  `extract frames`. `-v` is `--videos` on `extract`/`infer` but `--version` on `cuda`.
- `slvt infer` accepts videos that are **not** registered in the project, which is what makes de-novo analysis and
  deployment work. `extract outliers` and `extract purge` do not: unregistered paths are skipped with a warning.

### When not to act autonomously

**You MUST NOT run `slvt extract purge` with `--yes` unless the user has seen the preview and explicitly approved
it.** The `purge` command deletes each targeted video's entire `labeled-data` directory, human labels included, which
is work no other command can regenerate. Without `--yes` it only previews, so run the preview, show the user its
output (it marks which directories `[has labels]`), and stop.

The `--overwrite` and `--reset` options on the `extract` group re-roll rather than top up. They always preserve human
labels, so they are recoverable, but they still throw away extraction work. They diverge in scope on both subcommands:
on `frames` `--reset` clears every non-refined project video before the selection runs, while `--overwrite` clears only
the videos that selection picked. On `outliers` `--reset` clears the current iteration's machine frames for every
project video, while `--overwrite` clears them only for the videos the run re-extracts. Neither re-extracts a video
already in refinement, and they part ways on a refined video named explicitly through `--videos`: `--reset` skips it
with a warning, while `--overwrite` aborts the whole run. The two are mutually exclusive, and `--reset` cannot combine
with `--exclusive`. Confirm before using either, and before `slvt prepare --overwrite`, which replaces an existing
shuffle's training-dataset files.

Do not launch `slvt train` or `slvt infer` speculatively to check something. They occupy the machine's GPUs for tens
of minutes to hours and collide with any run the user already has going. Confirm which videos, which shuffle, and
which GPUs before starting one.

**You MUST NOT run `slvt cuda --yes` unless the user has seen the preview and explicitly approved it.** It uninstalls
and reinstalls the environment's torch distributions, which downloads gigabytes and breaks every concurrent run in that
environment. Without `--yes` it only reports what it would run, so run the preview, show the user its output, and stop.

### Running long jobs

Measured on the eye-tracking project's ~252k-frame (70-minute, 60 fps) videos:

| Command                 | Cost                            | The lever that actually helps                     |
|-------------------------|---------------------------------|---------------------------------------------------|
| `slvt infer`            | ~30 min per video, decode-bound | `--chunks 4` on a long video, `-gp 2` for many    |
| `slvt extract outliers` | ~7 min per video, decode-bound  | `--workers` close to the video count              |
| `slvt train`            | Tens of minutes to hours        | `--gpus 0,1` with `--multi-gpu ddp`, as an opt-in |

Those levers are measured, and the obvious alternatives do not work. On a single long video the GPU sits idle waiting on
decode, so `--chunks` splits that one video into concurrent frame ranges to fill the gap and roughly triples throughput.
`--gpu-processes` only helps when several videos run at once, and stops helping past 2 per GPU once the device is
compute-bound. A moderately larger `--clustering-stride` does not cut `extract outliers` wall-clock. The reader makes
one sequential pass over the first-to-last candidate span whenever the mean candidate gap stays at or below 200 frames,
so a wider stride traverses that same span and barely moves the wall-clock, though it decodes fewer candidates. Only a
stride large enough to push the mean gap past 200 returns the reader to per-candidate seeking. Training defaults to one
GPU because multi-GPU is often slower for DeepLabCut workloads, and `dp` cannot combine with `--amp`.

Run these in the background with output redirected to a log, then poll the log. Never block a session on one:

```bash
slvt infer --config-path CFG --videos a.mp4 --gpus 0 --batch-size 32 --chunks 4 > /tmp/infer.log 2>&1 &
```

Redirecting is not merely tidy. `LiveBar` detects a non-TTY stream and **appends a whole greppable progress line every
30 seconds** instead of redrawing in place, which is what makes the log pollable. Use `--no-progress` only when
nothing will read the log. A single-video `extract outliers` run that appears frozen for ~7 minutes is normal decode.

The `--gpu-processes` option applies one value to every `--gpus` index, so asymmetric loading across GPUs requires
**two concurrent `slvt infer` invocations** with disjoint `--videos` lists. Predictions land beside each video, so the
runs do not conflict.

### The GUI

The `slvt gui` command takes no options, needs a graphical session, and **blocks until the window is closed**. Do not
launch it on a headless host: on Linux it raises an explanatory error when neither `DISPLAY` nor `WAYLAND_DISPLAY` is
set. When a request needs the GUI, tell the user to run `slvt gui` on their workstation rather than running it
yourself. Its extraction, training, evaluation, and analysis tabs run stock DeepLabCut and are slower than the
equivalent `slvt` command, so route those requests to the CLI.

## Project context

This is **sollertia-video-tracking**, a bridge library that designs and deploys DeepLabCut video-tracking pipelines
for the Sollertia platform through the single `slvt` command-line interface.

### Key areas

| Directory                                        | Purpose                                                           |
|--------------------------------------------------|-------------------------------------------------------------------|
| `src/sollertia_video_tracking/interfaces/`       | The `slvt` Click command tree. Parses options, calls one function |
| `src/sollertia_video_tracking/frame_extraction/` | Parallel k-means and outlier frame extraction, core planning      |
| `src/sollertia_video_tracking/training/`         | Shuffle creation, the DDP/AMP training pipeline, evaluation       |
| `src/sollertia_video_tracking/inference/`        | Multi-device analysis orchestration and the runner-builder patch  |
| `src/sollertia_video_tracking/hardware/`         | Shared device and AMP detection, and the CUDA torch installer     |
| `src/sollertia_video_tracking/reporting/`        | `LiveBar`, the progress-bar base every bar subclasses             |
| `tests/`                                         | Pytest suite, one `<subpackage>_<module>_test.py` per module      |
| `docs/`                                          | Sphinx API documentation sources built by `tox -e docs`           |

### Architecture

Each interface module resolves its options into a profile or parameters dataclass and calls exactly one domain
function re-exported from the package root. The `extract` command is the only nested group: shared options live on the
group and per-subcommand options on `frames`, `outliers`, `purge`, and `pending`. The `--config-path` option is
deliberately not `required=True` on the group, because that would break `slvt extract SUBCOMMAND --help`, so it is
validated per-subcommand through `require_config_path()`.

The package `__init__.py` is a side-effecting preamble, not just re-exports. Its first act is the
`multiprocessing.set_start_method("spawn", force=True)` call, which must precede any process creation. It then sets the
`*_NUM_THREADS` variables, `VECLIB_MAXIMUM_THREADS`, `OPENCV_LOG_LEVEL`, `OPENCV_FFMPEG_LOGLEVEL`, and `MPLBACKEND=Agg`
**before** the domain imports, which carry `# noqa: E402`. These must precede any numpy, OpenCV, or DeepLabCut import,
and spawned workers inherit the environment.

Extraction and inference both use `multiprocessing.get_context("spawn")` pools. Extraction pins one video per worker
to a disjoint core block from `plan_core_allocation`. Inference spawns processes that drain a shared video queue one
whole video at a time, so work balances without ever splitting a video. Workers never render: they push throttled
progress messages through a queue to a single parent-side `LiveBar` subclass, and dropped messages are non-fatal.

### Key patterns

- **Behavior-preserving edits in the outlier path.** The outlier code in `frame_extraction/` faithfully reimplements
  DeepLabCut 3.0.1's `outlier_frames.py`, including its quirks, such as the `jump`/`uncertain` label-index versus
  `fitting` positional-index mix. Separate an upstream-faithful quirk from a new bug before "fixing" one.
- **Frame budgeting belongs only to `extract frames`.** The `--total-frames`, `--balance-groups`, `--group-regex`, and
  `--exclusive` options are `frames`-only. The `outliers` subcommand is stock DeepLabCut N-per-video, with no budget,
  tiers, or group balancing.
- **`Path` until the DeepLabCut and cv2 boundary.** Call `str()` only at the call site. Give numpy explicit dtypes
  (`to_numpy(dtype=np.float64)`, `NDArray[np.float64]`) and use `math.floor`/`math.ceil` on Python scalars.
- **No Sphinx cross-reference roles in docstrings.** Reference symbols as double-backtick literals.
- **Deployment is driven by the acquisition stack**, not by this library. Experiment preprocessing launches
  `slvt infer` over each session's camera recordings. Omitting `--output` writes the `.h5` beside the video, where the
  session ships it as raw data and `slvt extract outliers` later reads it. Preserve that default.

### Code standards

- Python 3.12 and numpy 1.x only. Both bounds are DeepLabCut constraints and are not negotiable.
- The `interfaces/` layer is omitted from coverage and verified manually.
- See `automation:python-style` for complete conventions, and the deviations under Style guide compliance above.

### Development commands

```bash
tox -e lint       # ruff format, ruff check, and mypy (with pandas-stubs and types-psutil)
tox -e stubs      # regenerate type stubs
tox -e py312-test # run the pytest suite
tox -e coverage   # combine coverage from the parallel test run
tox -e docs       # build the Sphinx documentation
tox               # full envlist: uninstall, export, lint, stubs, py312-test, coverage, docs, build, install
```

Coverage MUST use the whole-package scope `--cov=sollertia_video_tracking`, never a dotted submodule target.
Resolving a submodule source makes coverage import it during its own startup, which runs the DeepLabCut import chain
in a context where OpenCV's `cv2.dnn.DictValue` bootstrap fails.

### Workflow guidance

**Adding or changing a CLI option:**

1. Decide whether it belongs on the `extract` group (shared across subcommands) or on a single command.
2. Add it as a Click option, never a positional, with a long form and a short form, and a `--help` string that states
   the default and any implication (for example, `--group-regex` implies `--balance-groups`).
3. Honor the selection contract: if the option is video-scoped, omitting `--videos` MUST scope the run to the whole
   project rather than erroring, and any per-video list MUST require an explicit `--videos`.
4. Keep the interface module thin. Resolve the option into the parameters dataclass or profile and let the domain
   function own the behavior. Validation that needs domain state belongs one layer down.
5. Update `README.md`, which is authoritative on user-facing behavior, invoking `automation:readme-style` to do so.

**Touching a DeepLabCut override:**

1. Read the installed DeepLabCut 3.x source for the symbol being subclassed, patched, or mirrored.
2. Confirm the override still matches its MRO, signature, and config-dict shape. These are private APIs with no
   stability guarantee within the 3.x series.
3. Keep monkeypatches scoped to the worker process and restored in a `finally` block.
4. Never bump the `deeplabcut[gui]==3.0.1` pin as a side effect. Bumping it is its own tested task.
