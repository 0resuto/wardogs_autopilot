"""Regression tests for the Qt map scene/view.

Covers the real production widgets (MapGraphicsScene, MapGraphicsView,
InteractiveMapWidget): route editing, waypoint clamping, vehicle marker,
route path geometry, fit/zoom/center transforms, and route/pan mode toggling.
"""

import os
import sys
import unittest

import numpy as np
from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QApplication

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from autopilot.ui.map_scene import MapGraphicsScene  # noqa: E402
from autopilot.ui.map_view import InteractiveMapWidget  # noqa: E402

_QT_APP = QApplication.instance() or QApplication([])


class TestMapScene(unittest.TestCase):
    def test_add_move_delete_waypoints(self):
        scene = MapGraphicsScene(map_size=1000, thumb=8)
        changes: list[int] = []
        scene.on_route_changed = lambda: changes.append(len(scene.route_pts))

        scene.add_waypoint(100.0, 200.0)
        scene.add_waypoint(300.0, 400.0)
        self.assertEqual(scene.route_pts, [[100.0, 200.0], [300.0, 400.0]])
        self.assertEqual(len(scene.waypoint_items), 2)
        self.assertEqual(changes, [1, 2])

        scene._on_waypoint_moved(0, 150.0, 250.0)
        self.assertEqual(scene.route_pts[0], [150.0, 250.0])
        self.assertEqual(changes[-1], 2)

        scene._on_waypoint_deleted(0)
        self.assertEqual(scene.route_pts, [[300.0, 400.0]])
        self.assertEqual(len(scene.waypoint_items), 1)
        self.assertEqual(changes[-1], 1)

    def test_add_waypoint_clamps_to_map_bounds(self):
        scene = MapGraphicsScene(map_size=1000, thumb=8)
        scene.add_waypoint(-10.0, 5000.0)
        self.assertEqual(scene.route_pts, [[0.0, 1000.0]])

    def test_route_mode_switches_between_endpoints_and_waypoints(self):
        scene = MapGraphicsScene(map_size=1000, thumb=8)
        scene.set_route([[0.0, 0.0], [100.0, 100.0], [200.0, 0.0]])
        # bare scenes default to the editor view with numbered waypoints
        self.assertEqual(len(scene.waypoint_items), 3)
        self.assertEqual(len(scene._endpoint_items), 0)

        scene.set_edit_mode(False)
        self.assertEqual(len(scene.waypoint_items), 0)
        self.assertEqual(len(scene._endpoint_items), 2)

        scene.set_edit_mode(True)
        self.assertEqual(len(scene.waypoint_items), 3)
        self.assertEqual(len(scene._endpoint_items), 0)

    def test_route_line_follows_waypoints(self):
        scene = MapGraphicsScene(map_size=1000, thumb=8)
        scene.set_route([[0.0, 0.0], [100.0, 100.0], [200.0, 0.0]])
        self.assertEqual(scene.route_path_item.path().elementCount(), 3)

        scene.set_route([[0.0, 0.0]])
        self.assertEqual(scene.route_path_item.path().elementCount(), 0)

    def test_set_map_updates_scene_rect(self):
        scene = MapGraphicsScene(map_size=1000, thumb=8)
        scene.set_map(np.zeros((32, 32, 3), np.uint8), None, map_size=2048, thumb=8)
        self.assertEqual(scene.sceneRect().width(), 2048.0)
        self.assertEqual(scene.sceneRect().height(), 2048.0)

    def test_vehicle_marker_pose_and_visibility(self):
        scene = MapGraphicsScene(map_size=1000, thumb=8)
        self.assertFalse(scene.vehicle_marker.isVisible())

        scene.update_vehicle(321.0, 654.0, 90.0)
        self.assertTrue(scene.vehicle_marker.isVisible())
        self.assertEqual(scene.vehicle_marker.pos(), QPointF(321.0, 654.0))
        self.assertEqual(scene.vehicle_marker.heading, 90.0)

        scene.hide_vehicle()
        self.assertFalse(scene.vehicle_marker.isVisible())


class TestMapView(unittest.TestCase):
    def _shown_widget(self, map_size: int = 1000) -> InteractiveMapWidget:
        widget = InteractiveMapWidget(map_size=map_size, thumb=8)
        widget.resize(500, 400)
        widget.show()
        _QT_APP.processEvents()
        self.addCleanup(widget.close)
        return widget

    def test_fit_view_scales_scene_to_viewport(self):
        widget = self._shown_widget()
        widget.view.fit_view()

        viewport = widget.view.viewport().size()
        expected = min(viewport.width(), viewport.height()) / 1000.0
        self.assertAlmostEqual(widget.view.transform().m11(), expected, delta=0.01)

    def test_zoom_increases_scale(self):
        widget = self._shown_widget()
        widget.view.fit_view()
        scale_before = widget.view.transform().m11()

        widget.view.zoom_in(1.5)
        self.assertAlmostEqual(widget.view.transform().m11(), scale_before * 1.5, delta=0.01)

        widget.view.zoom_out(3.0)
        self.assertAlmostEqual(widget.view.transform().m11(), scale_before * 0.5, delta=0.01)

    def test_center_on_coords_centers_viewport(self):
        widget = self._shown_widget()
        widget.view.zoom_in(4.0)
        widget.view.center_on_coords(500.0, 500.0)

        center = widget.view.mapToScene(widget.view.viewport().rect().center())
        self.assertAlmostEqual(center.x(), 500.0, delta=2.0)
        self.assertAlmostEqual(center.y(), 500.0, delta=2.0)

    def test_center_button_toggles_follow_mode(self):
        widget = InteractiveMapWidget(map_size=1000, thumb=8)
        calls: list[int] = []
        widget.set_on_center(lambda: calls.append(1))
        self.assertFalse(widget.is_follow_centered())

        widget.btn_center.setChecked(True)
        self.assertTrue(widget.is_follow_centered())
        self.assertEqual(calls, [1])
        self.assertEqual(widget.btn_center.text(), "Following")

        widget.btn_center.setChecked(False)
        self.assertFalse(widget.is_follow_centered())
        self.assertEqual(calls, [1])
        self.assertEqual(widget.btn_center.text(), "Center")

    def test_route_mode_toggle(self):
        widget = InteractiveMapWidget(map_size=1000, thumb=8, enable_route_editing=True)
        self.assertTrue(widget.view._route_edit_mode)

        widget.btn_mode.setChecked(False)
        self.assertFalse(widget.view._route_edit_mode)
        self.assertEqual(widget.btn_mode.text(), "Pan Mode")

        widget.btn_mode.setChecked(True)
        self.assertTrue(widget.view._route_edit_mode)
        self.assertEqual(widget.btn_mode.text(), "Route Mode")


if __name__ == "__main__":
    unittest.main()
