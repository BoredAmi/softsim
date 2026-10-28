"""Conversion tests. Run with ROS sourced and the venv active:
    python -m pytest ros2/softsim_ros/test -p no:launch_testing -p no:launch_ros
"""

import numpy as np
import pytest

pytest.importorskip("rclpy")

from softsim.backends.base import ContactWrench, RigidState  # noqa: E402
from softsim.geometry.rigid import Part  # noqa: E402
from softsim.scene import RigidBodySpec  # noqa: E402
from softsim_ros import conversions as conv  # noqa: E402
from visualization_msgs.msg import Marker  # noqa: E402


def test_stamp():
    t = conv.stamp(1.2345)
    assert (t.sec, t.nanosec) == (1, 234500000)
    assert conv.stamp(0.0).nanosec == 0


def test_pose_quaternion_order():
    s = RigidState(np.array([1.0, 2, 3]), np.array([0.5, 0.5, 0.5, 0.5]), np.zeros(3), np.zeros(3))
    p = conv.pose_stamped(s, "world", 0.1).pose
    assert (p.position.x, p.position.z) == (1.0, 3.0)
    assert (p.orientation.w, p.orientation.x) == (0.5, 0.5)
    tf = conv.transform("weight", s, "world", 0.1)
    assert tf.child_frame_id == "weight" and tf.header.frame_id == "world"


def test_ground_reaction_sums_ground_contacts_only():
    contacts = [
        ContactWrench("ground", "specimen", np.array([0, 0, 2.0]), np.zeros(3), np.array([0.01, 0, 0]), 10),
        ContactWrench("specimen", "ground", np.array([0, 0, -1.0]), np.zeros(3), np.zeros(3), 3),  # ground gets -1 => pushes +1
        ContactWrench("weight", "specimen", np.array([0, 0, -5.0]), np.zeros(3), np.zeros(3), 4),
    ]
    w = conv.ground_reaction(contacts, "world", 0.0).wrench
    assert w.force.z == pytest.approx(3.0)
    assert w.torque.y == pytest.approx(-0.02)  # r x F of the first contact


def test_markers_and_cloud():
    body = RigidBodySpec("weight", (Part("cylinder", (0.01, 0.003, 0), (0, 0, 0.003)),
                                    Part("box", (0.01, 0.02, 0.005), (0, 0, 0.011))), 0.02)
    arr = conv.rigid_markers([body])
    assert arr.markers[0].action == Marker.DELETEALL
    cyl, box = arr.markers[1:]
    assert cyl.type == Marker.CYLINDER and cyl.scale.z == pytest.approx(0.006) and cyl.header.frame_id == "weight"
    assert box.type == Marker.CUBE and box.scale.y == pytest.approx(0.04)
    verts = np.random.rand(10, 3)
    tris = np.array([[0, 1, 2], [2, 3, 4]])
    m = conv.soft_mesh_marker("specimen", 0, verts, tris, (1, 1, 1, 1), "world", 0.0)
    assert len(m.points) == 6 and m.points[3].x == pytest.approx(verts[2, 0])
    cloud = conv.point_cloud(verts, "world", 0.0)
    assert cloud.width * cloud.height == 10


def test_interface_messages():
    pytest.importorskip("softsim_interfaces")
    c = ContactWrench("weight", "specimen", np.array([0, 0, -1.0]), np.zeros(3), np.zeros(3), 2)
    msg = conv.contact_array([c], "world", 0.5)
    assert msg.contacts[0].force.z == -1.0 and msg.contacts[0].n_contacts == 2
    m = conv.scenario_metrics("drop_test", {"indentation": 0.002, "t": 0.5}, "world", 0.5)
    assert list(m.names) == ["indentation", "t"] and m.values[0] == 0.002
