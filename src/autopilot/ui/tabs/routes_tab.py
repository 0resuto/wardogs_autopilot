"""Backward-compatibility shim aliasing RoutesTab to the unified MapTab."""

from .map_tab import MapTab as RoutesTab

__all__ = ["RoutesTab"]
