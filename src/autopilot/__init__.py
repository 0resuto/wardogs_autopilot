"""Autopilot for WARDOGS supply runs: player localization on the full map
via the minimap.

Package layout
--------------
- ``autopilot.common``     crash/segfault logging, configuration, logging
- ``autopilot.hardware``   screen capture (mss) and Arduino key input
- ``autopilot.vision``     map matching, tracker, and feature indexing
- ``autopilot.navigation`` route-following autopilot (FollowDriver)
- ``autopilot.ui``         Tkinter "studio" GUI
"""

import os

__version__ = "0.1.0"

# Absolute path to the repository root containing data/, output/, etc.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .common import crashlog  # noqa: E402

__all__ = ["PROJECT_ROOT", "crashlog"]
