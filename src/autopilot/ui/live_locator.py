"""Backward-compatibility facade for LiveLocator.

LiveLocator has moved to autopilot.vision.tracker.
"""

from ..vision.tracker import LiveLocator, _ang_diff, _vote_decide

__all__ = ["LiveLocator", "_ang_diff", "_vote_decide"]
