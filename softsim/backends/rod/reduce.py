"""Reduce a slender tetrahedral body to a chain of beam frames (pure numpy; no SOFA).

The rod backend runs a scene's soft body as a beam along its z axis. This module does the geometry and
the section integrals, so it is testable without SOFA:

* **Frames.** ``n + 1`` frames at ``s_k`` (planes z = const); beam ``k`` joins frames ``k`` and ``k + 1``.
* **Section properties per beam** from the tetrahedra whose centroid lies in the slab: the extensional
  stiffness ``EA = sum(E_e V_e) / ds``, the bending stiffnesses ``EI_x = sum(E_e int (x - x_c)^2 dV) / ds``
  and ``EI_y`` likewise, about the stiffness-weighted centroid ``(x_c, y_c)`` (the neutral axis, which
  moves off the geometric centre in a layered section), and the mass per length. ``E_e`` is each
  element's small-strain Young's modulus, so material zones (stiff disks, layers) enter exactly. The
  second moment of a tetrahedron is the exact quadratic-integrand formula
  ``int x^2 dV = V/20 (sum x_i^2 + (sum x_i)^2)``.
* **Vertex map.** Every mesh vertex is tied to the frames by its axial position ``s`` (interpolation
  between the two frames, lerp for positions and slerp for rotations) and its in-plane offset from the
  centreline, so the deformed mesh, and therefore every region metric, is recovered from the frames.

BeamAdapter wants the beam axis along its x. The model therefore works in an *internal* frame
``(X, Y, Z) = (z, x, y)`` of the scene (a cyclic permutation, a proper rotation ``P``); frame poses are
internal, ``vertex_positions`` returns scene coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from softsim.errors import UnsupportedFeatureError
from softsim.scene import SoftBodySpec

# Scene (x, y, z) -> internal (X, Y, Z) = (z, x, y).
P = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

MIN_SLENDERNESS = 4.0  # length over the largest section size below which a beam is not a fair model


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Rotation matrices (..., 3, 3) of unit quaternions (..., 4) in SOFA's (x, y, z, w) order."""
    q = np.asarray(q, float)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    x, y, z, w = (q[..., i] for i in range(4))
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)


def slerp(a: np.ndarray, b: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Spherical interpolation of unit quaternions (N, 4) at fractions ``t`` (N,), shortest arc."""
    dot = np.sum(a * b, axis=-1, keepdims=True)
    b = np.where(dot < 0, -b, b)
    dot = np.abs(dot)
    theta = np.arccos(np.clip(dot, -1, 1))
    sin = np.sin(theta)
    t = t[:, None]
    near = sin < 1e-9
    wa = np.where(near, 1 - t, np.sin((1 - t) * theta) / np.where(near, 1, sin))
    wb = np.where(near, t, np.sin(t * theta) / np.where(near, 1, sin))
    out = wa * a + wb * b
    return out / np.linalg.norm(out, axis=-1, keepdims=True)


def tet_second_moments(points: np.ndarray, tets: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact per-tetrahedron integrals (V, int x dV, int x^2 dV) for each coordinate: shape (M, 3) each."""
    p = points[tets]
    vol = np.abs(np.einsum("ij,ij->i", np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), p[:, 3] - p[:, 0])) / 6.0
    first = vol[:, None] * p.mean(axis=1)
    second = vol[:, None] / 20.0 * ((p**2).sum(axis=1) + p.sum(axis=1) ** 2)
    return vol, first, second


@dataclass(frozen=True)
class RodModel:
    """A body reduced to beam frames; arrays are per beam (``n``) or per frame (``n + 1``) as noted.

    Attributes:
        s: Axial positions of the frames [m] (n + 1).
        centre: Rest in-plane position of each frame, the stiffness-weighted section centroid, scene x, y (n + 1, 2).
        ea, ei_x, ei_y: Extensional [N] and bending [N m^2] stiffnesses per beam; ``ei_x`` resists
            deflection along scene x, ``ei_y`` along scene y.
        mass_per_length: [kg/m] per beam.
        vertex_beam, vertex_t: Beam index and fraction along it of every mesh vertex (N,).
        vertex_offset: Rest in-plane offset of every vertex from the interpolated centreline (N, 2).
        rest_points: The rest mesh vertices (N, 3), scene frame.
    """

    s: np.ndarray
    centre: np.ndarray
    ea: np.ndarray
    ei_x: np.ndarray
    ei_y: np.ndarray
    mass_per_length: np.ndarray
    vertex_beam: np.ndarray
    vertex_t: np.ndarray
    vertex_offset: np.ndarray
    rest_points: np.ndarray

    @property
    def n_beams(self) -> int:
        return len(self.ea)

    def rest_frames(self) -> np.ndarray:
        """Internal-frame rest poses (n + 1, 7): position, then identity quaternion (x, y, z, w)."""
        pos = np.column_stack([self.s, self.centre[:, 0], self.centre[:, 1]])  # (X, Y, Z) = (z, x, y)
        return np.column_stack([pos, np.zeros((len(pos), 3)), np.ones(len(pos))])

    def equivalent_section(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(lengthY, lengthZ, E) per beam of the rectangle whose stiffnesses equal this rod's exactly.

        BeamAdapter takes a rectangle (lengthY along internal Y = scene x, lengthZ along internal Z = scene y) and
        a modulus per beam. With A = lY lZ: EA = E lY lZ, EI_x = E lZ lY^3 / 12 = EA lY^2 / 12 and
        EI_y = EA lZ^2 / 12, so three unknowns meet three stiffnesses: lY = sqrt(12 EI_x / EA),
        lZ = sqrt(12 EI_y / EA), E = EA / (lY lZ). (For a homogeneous rectangle this is the rectangle itself.)"""
        ly = np.sqrt(12.0 * self.ei_x / self.ea)
        lz = np.sqrt(12.0 * self.ei_y / self.ea)
        return ly, lz, self.ea / (ly * lz)

    # -- mesh from frames ------------------------------------------------------------------------------

    def _interpolate(self, pos: np.ndarray, quat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        k, t = self.vertex_beam, self.vertex_t
        centre = (1 - t)[:, None] * pos[k] + t[:, None] * pos[k + 1]
        rot = quat_to_matrix(slerp(quat[k], quat[k + 1], t))
        return centre, rot

    def vertex_positions(self, pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
        """Scene positions (N, 3) of the mesh vertices from internal frame positions (n + 1, 3) and quaternions
        (n + 1, 4, x y z w): the interpolated centreline point plus the rotated rest offset."""
        centre, rot = self._interpolate(pos, quat)
        local = np.column_stack([np.zeros(len(self.vertex_t)), self.vertex_offset])  # internal (0, x, y) offset
        return (centre + np.einsum("nij,nj->ni", rot, local)) @ P  # internal -> scene: P^T v = v @ P

    def vertex_velocities(self, pos: np.ndarray, quat: np.ndarray, vel: np.ndarray, omega: np.ndarray) -> np.ndarray:
        """Scene velocities (N, 3) of the vertices from the frames' linear (n + 1, 3) and angular (n + 1, 3) velocities."""
        k, t = self.vertex_beam, self.vertex_t
        lin = (1 - t)[:, None] * vel[k] + t[:, None] * vel[k + 1]
        ang = (1 - t)[:, None] * omega[k] + t[:, None] * omega[k + 1]
        _, rot = self._interpolate(pos, quat)
        arm = np.einsum("nij,nj->ni", rot, np.column_stack([np.zeros(len(t)), self.vertex_offset]))
        return (lin + np.cross(ang, arm)) @ P

    def frame_wrenches(self, pos: np.ndarray, quat: np.ndarray, forces: np.ndarray) -> np.ndarray:
        """Frame loads (n + 1, 6) in the internal frame, [force, moment], equivalent to vertex ``forces`` (N, 3), scene frame.

        Virtual work through the vertex map: a vertex between frames k and k + 1 at fraction t moves like
        (1 - t) dp_k + t dp_k+1 + dtheta x arm, with dtheta interpolated the same way, so frame k receives
        (1 - t) f and the moment (1 - t) arm x f (frame k + 1: t). First order in the relative rotation of
        the two frames, which is small within one beam."""
        k, t = self.vertex_beam, self.vertex_t
        f = self.to_internal(forces)
        _, rot = self._interpolate(pos, quat)
        arm = np.einsum("nij,nj->ni", rot, np.column_stack([np.zeros(len(t)), self.vertex_offset]))
        m = np.cross(arm, f)
        out = np.zeros((len(self.s), 6))
        for idx, w in ((k, 1 - t), (k + 1, t)):
            np.add.at(out[:, :3], idx, w[:, None] * f)
            np.add.at(out[:, 3:], idx, w[:, None] * m)
        return out

    def to_scene(self, internal: np.ndarray) -> np.ndarray:
        """Internal-frame vectors (..., 3) as scene vectors."""
        return np.asarray(internal) @ P

    def to_internal(self, scene: np.ndarray) -> np.ndarray:
        """Scene vectors (..., 3) as internal-frame vectors."""
        return np.asarray(scene) @ P.T


def slender_problems(body: SoftBodySpec) -> list[str]:
    """Reasons ``body`` cannot be run as a rod (empty if it can): rotated, stubby, or not clamped at its z-min end."""
    pts = body.mesh.points + np.asarray(body.pose.position)
    out = []
    if not np.allclose(body.pose.quaternion, (1.0, 0.0, 0.0, 0.0)):
        out.append(f"body {body.name!r} is rotated; the rod axis is the body z axis")
    z0 = float(pts[:, 2].min())
    length = float(pts[:, 2].max()) - z0
    size = max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]))
    if length < MIN_SLENDERNESS * size:
        out.append(f"body {body.name!r} is {length / size:.1f} times longer than wide (a rod needs at least "
                   f"{MIN_SLENDERNESS:g}); a beam is not a fair model, use a 3D backend")
    if not body.pinned:
        out.append(f"body {body.name!r} has no fixed end; the rod is clamped at its z-min end")
    elif not np.allclose(pts[list(body.pinned), 2], z0, atol=1e-9 * max(length, 1e-12)):
        out.append(f"the fixed vertices of {body.name!r} must be its z-min end face")
    return out


def reduce_body(body: SoftBodySpec, segments: int | None = None) -> RodModel:
    """Reduce ``body`` (slender along z, clamped at its z-min end) to a :class:`RodModel`.

    ``segments`` beams; by default one per mesh layer (length / the median tetrahedron's z extent).
    Raises :class:`UnsupportedFeatureError` for a body that is not a slender, axis-aligned column clamped
    at one end (see :func:`slender_problems`)."""
    problems = slender_problems(body)
    if problems:
        raise UnsupportedFeatureError("rod", problems)
    pts, tets = body.mesh.points + np.asarray(body.pose.position), body.mesh.tets
    z0, z1 = float(pts[:, 2].min()), float(pts[:, 2].max())
    length = z1 - z0
    vol, first, second = tet_second_moments(pts, tets)
    modulus = np.empty(len(tets))
    for material, elements in body.material_groups():
        modulus[elements] = material.elastic.initial_youngs_modulus()
    rho = body.element_densities()
    if segments is None:
        segments = max(1, int(round(length / float(np.median(np.ptp(pts[tets][:, :, 2], axis=1))))))
    s = z0 + length * np.arange(segments + 1) / segments
    ds = length / segments
    slab = np.clip(((pts[tets].mean(axis=1)[:, 2] - z0) / ds).astype(int), 0, segments - 1)
    ea, ei_x, ei_y, mass = (np.zeros(segments) for _ in range(4))
    centre_beam = np.zeros((segments, 2))
    for k in range(segments):
        m = slab == k
        if not m.any():
            raise UnsupportedFeatureError(f"rod: no elements between z = {s[k]:.4g} and {s[k + 1]:.4g}")
        ev = modulus[m] * vol[m]
        stiff = ev.sum()
        c = (modulus[m, None] * first[m, :2]).sum(axis=0) / stiff
        centre_beam[k] = c
        ea[k] = stiff / ds
        ei_x[k] = (modulus[m] * second[m, 0]).sum() - 2 * c[0] * (modulus[m] * first[m, 0]).sum() + c[0] ** 2 * stiff
        ei_y[k] = (modulus[m] * second[m, 1]).sum() - 2 * c[1] * (modulus[m] * first[m, 1]).sum() + c[1] ** 2 * stiff
        ei_x[k] /= ds
        ei_y[k] /= ds
        mass[k] = (rho[m] * vol[m]).sum() / ds
    centre = np.empty((segments + 1, 2))
    centre[0], centre[-1] = centre_beam[0], centre_beam[-1]
    centre[1:-1] = 0.5 * (centre_beam[:-1] + centre_beam[1:])
    beam = np.clip(((pts[:, 2] - z0) / ds).astype(int), 0, segments - 1)
    frac = (pts[:, 2] - s[beam]) / ds
    line = (1 - frac)[:, None] * centre[beam] + frac[:, None] * centre[beam + 1]
    return RodModel(s=s, centre=centre, ea=ea, ei_x=ei_x, ei_y=ei_y, mass_per_length=mass, vertex_beam=beam,
                    vertex_t=frac, vertex_offset=pts[:, :2] - line, rest_points=pts)
