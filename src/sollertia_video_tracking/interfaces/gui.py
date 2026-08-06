"""Provides the ``slvt gui`` command that launches the DeepLabCut project-management GUI."""

import os
import sys

import click
import matplotlib as mpl
from deeplabcut.gui.launch_script import launch_dlc

_CONTEXT_SETTINGS: dict[str, int] = {"max_content_width": 120}
"""Widens displayed Click help messages to 120 columns so option descriptions wrap consistently."""


@click.command("gui", context_settings=_CONTEXT_SETTINGS)
def gui_command() -> None:
    """Launches the standard DeepLabCut GUI, used to create a project and manually label its frames.

    This opens the same fully functional DeepLabCut application reached elsewhere by running ``python -m deeplabcut``.
    From its project-management window, a project's config.yaml is created or opened to label the frames that
    ``slvt extract`` selects, correct the outlier frames it flags, and merge the refined data into the next iteration.
    The frame labeler itself opens in napari, reached from that window. Manual labeling is the only part of the
    refinement loop this library does not implement, so it is the reason to open the GUI. The extraction, training,
    evaluation, and analysis tabs it also offers run stock DeepLabCut and are slower than the equivalent ``slvt``
    commands. The GUI needs a graphical session, so it is run on a workstation rather than a headless training or
    inference server, and the command blocks until the window is closed.
    """
    if sys.platform == "linux" and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        message = (
            "Unable to launch the DeepLabCut GUI. No graphical display was detected (neither DISPLAY nor "
            "WAYLAND_DISPLAY is set), which typically means this is a headless server. Run 'slvt gui' from a "
            "workstation with a graphical session."
        )
        raise click.ClickException(message=message)

    # The GUI hosts interactive matplotlib panels (tracklet refinement, the skeleton builder, and the crop-region
    # selector) that need an interactive backend. The package forces the headless 'Agg' backend at import for its
    # compute-server commands, so an interactive Qt backend is restored here, for this GUI session only, before the
    # window opens. PySide6 is already imported at this point, so matplotlib binds to the same Qt binding as DeepLabCut.
    mpl.use("QtAgg")

    click.echo(message="Starting the DeepLabCut GUI...")
    launch_dlc()
