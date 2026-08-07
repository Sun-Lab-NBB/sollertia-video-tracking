.. This file provides the instructions for how to display the API documentation generated using sphinx autodoc
   extension. Use it to declare Python documentation sub-directories via appropriate modules (automodule, etc.).

Frame Extraction
================

.. automodule:: sollertia_video_tracking.frame_extraction
   :members:
   :undoc-members:
   :show-inheritance:

Model Training
==============

.. automodule:: sollertia_video_tracking.training
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: Toggle, AmpMode, DeviceType

Video Inference
===============

.. automodule:: sollertia_video_tracking.inference
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: Toggle, AmpMode, DeviceType

Command-Line Interface
======================

.. click:: sollertia_video_tracking.interfaces.entry_points:slvt_cli
   :prog: slvt
   :nested: full

Hardware Detection and Setup
============================

.. automodule:: sollertia_video_tracking.hardware
   :members:
   :undoc-members:
   :show-inheritance:

Progress Reporting
==================

.. automodule:: sollertia_video_tracking.reporting
   :members:
   :undoc-members:
   :show-inheritance:
