"""softsim state -> ROS messages. Pure functions, no node required."""

from __future__ import annotations

import math
from typing import Iterable, Mapping

import numpy as np
from builtin_interfaces.msg import Time
from geometry_msgs.msg import (
    Point,
    Pose,
    PoseStamped,
    Quaternion,
    TransformStamped,
    TwistStamped,
    Vector3,
    WrenchStamped,
)
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import ColorRGBA, Header
from visualization_msgs.msg import Marker, MarkerArray

from softsim.backends.base import ContactWrench, RigidState
from softsim.scene import RigidBodySpec, Scene

GROUND = "ground"


def stamp(seconds: float) -> Time:
    """builtin_interfaces/Time from simulated seconds."""
    sec = int(math.floor(seconds))
    return Time(sec=sec, nanosec=int(round((seconds - sec) * 1e9)) % 1_000_000_000)


def header(frame_id: str, seconds: float) -> Header:
    return Header(frame_id=frame_id, stamp=stamp(seconds))


def _vec(v) -> Vector3:
    return Vector3(x=float(v[0]), y=float(v[1]), z=float(v[2]))


def _pt(v) -> Point:
    return Point(x=float(v[0]), y=float(v[1]), z=float(v[2]))


def _quat(q) -> Quaternion:  # softsim (w, x, y, z) -> ROS (x, y, z, w)
    return Quaternion(w=float(q[0]), x=float(q[1]), y=float(q[2]), z=float(q[3]))


def pose_stamped(state: RigidState, frame_id: str, seconds: float) -> PoseStamped:
    return PoseStamped(header=header(frame_id, seconds),
                       pose=Pose(position=_pt(state.position), orientation=_quat(state.quaternion)))


def twist_stamped(state: RigidState, frame_id: str, seconds: float) -> TwistStamped:
    """Twist of the body frame origin, expressed in ``frame_id`` (world)."""
    msg = TwistStamped(header=header(frame_id, seconds))
    msg.twist.linear = _vec(state.linear_velocity)
    msg.twist.angular = _vec(state.angular_velocity)
    return msg


def transform(name: str, state: RigidState, frame_id: str, seconds: float) -> TransformStamped:
    msg = TransformStamped(header=header(frame_id, seconds), child_frame_id=name)
    msg.transform.translation = _vec(state.position)
    msg.transform.rotation = _quat(state.quaternion)
    return msg


def point_cloud(vertices: np.ndarray, frame_id: str, seconds: float) -> PointCloud2:
    return point_cloud2.create_cloud_xyz32(header(frame_id, seconds), np.asarray(vertices, dtype=np.float32))


def _color(rgba) -> ColorRGBA:
    return ColorRGBA(r=float(rgba[0]), g=float(rgba[1]), b=float(rgba[2]), a=float(rgba[3]))


def soft_mesh_marker(name: str, marker_id: int, vertices: np.ndarray, triangles: np.ndarray,
                     rgba, frame_id: str, seconds: float) -> Marker:
    """Surface mesh as a TRIANGLE_LIST marker (vertices in ``frame_id``)."""
    m = Marker(header=header(frame_id, seconds), ns="soft/" + name, id=marker_id,
               type=Marker.TRIANGLE_LIST, action=Marker.ADD)
    m.pose.orientation.w = 1.0
    m.scale = Vector3(x=1.0, y=1.0, z=1.0)
    m.color = _color(rgba)
    m.points = [_pt(p) for p in np.asarray(vertices)[np.asarray(triangles).ravel()]]
    return m


def rigid_markers(bodies: Iterable[RigidBodySpec]) -> MarkerArray:
    """Primitive-part markers attached to each rigid body's TF frame (frame_locked)."""
    out = MarkerArray()
    out.markers.append(Marker(action=Marker.DELETEALL))
    for body in bodies:
        for i, part in enumerate(body.parts):
            m = Marker(ns="rigid/" + body.name, id=i, action=Marker.ADD, frame_locked=True)
            m.header.frame_id = body.name
            m.pose.position = _pt(part.center)
            m.pose.orientation.w = 1.0
            if part.kind == "cylinder":
                m.type = Marker.CYLINDER
                m.scale = Vector3(x=2 * part.size[0], y=2 * part.size[0], z=2 * part.size[1])
            else:
                m.type = Marker.CUBE
                m.scale = Vector3(x=2 * part.size[0], y=2 * part.size[1], z=2 * part.size[2])
            m.color = _color(body.rgba)
            out.markers.append(m)
    return out


def ground_reaction(contacts: Iterable[ContactWrench], frame_id: str, seconds: float) -> WrenchStamped:
    """Total wrench the ground exerts on everything, torque about the world origin."""
    force, torque = np.zeros(3), np.zeros(3)
    for c in contacts:
        if GROUND not in (c.body_a, c.body_b):
            continue
        other = c.body_b if c.body_a == GROUND else c.body_a
        f = c.on(other)
        t = c.torque if c.body_a == GROUND else -c.torque
        force += f
        torque += t + np.cross(c.point, f)
    msg = WrenchStamped(header=header(frame_id, seconds))
    msg.wrench.force = _vec(force)
    msg.wrench.torque = _vec(torque)
    return msg


def contact_array(contacts: Iterable[ContactWrench], frame_id: str, seconds: float):
    """softsim_interfaces/ContactWrenchArray."""
    from softsim_interfaces.msg import ContactWrench as ContactMsg, ContactWrenchArray

    msg = ContactWrenchArray(header=header(frame_id, seconds))
    for c in contacts:
        msg.contacts.append(ContactMsg(body_a=c.body_a, body_b=c.body_b, force=_vec(c.force), torque=_vec(c.torque),
                                       point=_pt(c.point), n_contacts=int(c.n_contacts)))
    return msg


def soft_body_state(name: str, vertices: np.ndarray, velocities: np.ndarray, frame_id: str, seconds: float):
    """softsim_interfaces/SoftBodyState."""
    from softsim_interfaces.msg import SoftBodyState

    return SoftBodyState(header=header(frame_id, seconds), name=name,
                         vertices=[_pt(v) for v in vertices], velocities=[_vec(v) for v in velocities])


def scenario_metrics(scenario: str, metrics: Mapping[str, float], frame_id: str, seconds: float):
    """softsim_interfaces/ScenarioMetrics."""
    from softsim_interfaces.msg import ScenarioMetrics

    return ScenarioMetrics(header=header(frame_id, seconds), scenario=scenario,
                           names=list(metrics), values=[float(v) for v in metrics.values()])


def surface(scene: Scene) -> dict[str, np.ndarray]:
    """Surface triangles of each soft body (computed once per build)."""
    return {b.name: b.mesh.surface_triangles() for b in scene.soft_bodies}
