"""Provides shared pytest configuration and fixtures for the sollertia-video-tracking test suite.

Coverage must be measured against the whole ``sollertia_video_tracking`` package (``--cov=sollertia_video_tracking``),
never a dotted submodule target. Resolving a submodule source forces coverage to import it during its own startup, which
runs the DeepLabCut import chain in a context where OpenCV's ``cv2.dnn.DictValue`` bootstrap fails. The tox ``-test``
environment already uses the whole-package form.
"""

import pytest

_DEPENDENCY_WARNING_FILTERS: tuple[str, ...] = (
    "ignore::pyparsing.PyparsingDeprecationWarning:matplotlib._fontconfig_pattern",
    "ignore::pyparsing.PyparsingDeprecationWarning:matplotlib._mathtext",
    "ignore:invalid escape sequence:SyntaxWarning",
)
"""The filters that silence the warnings the DeepLabCut dependency tree raises as every xdist worker imports it.

Matplotlib 3.8 calls the pyparsing method names that pyparsing 3.3 deprecates. DeepLabCut caps matplotlib below 3.9,
which is the release that drops the pyparsing dependency, so the calls stand until that cap moves. Filterpy 1.4.5 writes
an invalid escape sequence in a plain docstring, which the byte-compiler reports the first time a worker compiles the
module. That entry names no module, because a warning the compiler raises matches no module pattern. Ruff W605 covers
this project's own sources and its tests, so the unscoped entry hides nothing here."""


def pytest_configure(config: pytest.Config) -> None:
    """Registers the dependency warning filters on the session, ahead of the collection that triggers them.

    The filters live here rather than in the pyproject.toml ``filterwarnings`` option, which the shared ataraxis
    project template owns.
    """
    for warning_filter in _DEPENDENCY_WARNING_FILTERS:
        config.addinivalue_line(name="filterwarnings", line=warning_filter)
