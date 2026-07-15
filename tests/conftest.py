"""Shared pytest configuration and fixtures for the sollertia-video-tracking test suite.

Coverage must be measured against the whole ``sollertia_video_tracking`` package (``--cov=sollertia_video_tracking``),
never a dotted submodule target. Resolving a submodule source forces coverage to import it during its own startup, which
runs the DeepLabCut import chain in a context where OpenCV's ``cv2.dnn.DictValue`` bootstrap fails. The tox ``-test``
environment already uses the whole-package form.
"""
