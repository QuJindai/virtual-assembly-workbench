"""Explicit datum alignment. Returned transforms are current-world deltas.

RPS is a local, bounded scalar-normal least-squares solve. Its convergence flag
indicates numerical convergence, not satisfaction of a manufacturing tolerance.
"""
from __future__ import annotations

import json
import numpy as np
from scipy.optimize import least_squares

from .core import points_array


def _finite_result(result):
    try:
        json.dumps(result, allow_nan=False)
    except (ValueError, TypeError) as exc:
        raise ValueError('计算结果超出有限数值范围，请检查坐标尺度。') from exc
    return result


def _metrics(residuals):
    magnitude = np.max(np.abs(residuals))
    rms = magnitude * np.sqrt(np.mean((residuals/magnitude)**2)) if magnitude else 0.
    return dict(residuals_mm=residuals.tolist(), rms_mm=float(rms), max_mm=float(magnitude))


def _datum_frame(points):
    if points.shape != (6, 3):
        raise ValueError('3-2-1 基准需要依次提供 A1,A2,A3,B1,B2,C1 六点。')
    # Work relative to A1 to avoid solving three large absolute dot products.
    relative = points - points[0]
    a, b = relative[1], relative[2]
    al, bl = np.linalg.norm(a), np.linalg.norm(b)
    if min(al, bl) <= 1e-12:
        raise ValueError('主基准三点重复，无法确定平面。')
    z = np.cross(a/al, b/bl)
    zl = np.linalg.norm(z)
    if zl < 1e-8:
        raise ValueError('主基准三点共线或近共线，无法确定平面。')
    z /= zl
    tangent = relative[4] - relative[3]
    tl = np.linalg.norm(tangent)
    x = tangent - np.dot(tangent, z)*z
    xl = np.linalg.norm(x)
    if tl <= 1e-12 or xl <= 1e-8*tl:
        raise ValueError('次基准两点重复或方向平行于主基准法向。')
    x /= xl
    y = np.cross(z, x)
    rotation = np.column_stack((x, y, z))
    origin = points[0] + x*np.dot(x, relative[5]) + y*np.dot(y, relative[3])
    return rotation, origin


def align_321(source, target):
    """Align hierarchical planes: A normal Z, B tangent X, B normal Y.

    Point order controls orientation. The tertiary point supplies the X
    coordinate. Six residuals are A1/A2/A3 along target Z, B1/B2 along target Y,
    C1 along target X. Tangential mismatch is separately reported.
    """
    s, t = points_array(source), points_array(target)
    sr, so = _datum_frame(s)
    tr, to = _datum_frame(t)
    rotation = tr @ sr.T
    delta = np.eye(4)
    delta[:3, :3] = rotation
    delta[:3, 3] = to - rotation @ so
    difference = (s-so) @ rotation.T - (t-to)
    directions = tr[:, [2, 2, 2, 1, 1, 0]].T
    residuals = np.einsum('ij,ij->i', difference, directions)
    return _finite_result(dict(method='3-2-1', transform=delta.tolist(),
                               **_metrics(residuals), converged=True,
                               point_deviations_mm=np.linalg.norm(difference, axis=1).tolist(),
                               residual_directions=directions.tolist()))


def _rotation_derivatives(angles_deg):
    a, b, c = np.deg2rad(angles_deg)
    sa, sb, sc = np.sin([a, b, c])
    ca, cb, cc = np.cos([a, b, c])
    rx = np.array([[1., 0, 0], [0, ca, -sa], [0, sa, ca]])
    ry = np.array([[cb, 0, sb], [0, 1., 0], [-sb, 0, cb]])
    rz = np.array([[cc, -sc, 0], [sc, cc, 0], [0, 0, 1.]])
    dx = np.array([[0., 0, 0], [0, -sa, -ca], [0, ca, -sa]])
    dy = np.array([[-sb, 0, cb], [0, 0, 0], [-cb, 0, -sb]])
    dz = np.array([[-sc, -cc, 0], [cc, -sc, 0], [0, 0, 0]])
    factor = np.pi/180
    return rz@ry@rx, (rz@ry@dx*factor, rz@dy@rx*factor, dz@ry@rx*factor)


def _rank_condition(jac):
    lengths = np.linalg.norm(jac, axis=0)
    if np.any(lengths < 1e-12) or not np.isfinite(lengths).all():
        raise ValueError('RPS 活动自由度缺少独立约束，Jacobian 秩不足。')
    singular = np.linalg.svd(jac/lengths, compute_uv=False)
    rank = int(np.count_nonzero(singular > singular[0]*1e-8))
    if rank < jac.shape[1]:
        raise ValueError('RPS 活动自由度约束相关，Jacobian 秩不足。')
    return rank, float(singular[0]/singular[-1])


def align_rps(source, target, normals, *, weights=None,
              dofs=(True, True, True, True, True, True), pivot=None,
              translation_limit_mm=100.0, rotation_limit_deg=30.0):
    """Minimize sum(weights * scalar_normal_residual**2) within travel bounds.

    Parameters are TX,TY,TZ in mm and RX,RY,RZ in degrees, with Rz Ry Rx
    rotation about pivot. Locked parameters stay exactly zero. Scalar residuals
    retain the sign dot(moved_source - target, normal), and are unweighted in
    the output. Condition is computed on column-normalized weighted Jacobian.
    """
    s, t, n = points_array(source), points_array(target), points_array(normals)
    if s.shape != t.shape or s.shape != n.shape:
        raise ValueError('RPS 源点、目标点和法向必须逐点对应且形状相同。')
    if not np.allclose(np.linalg.norm(n, axis=1), 1., rtol=0, atol=1e-7):
        raise ValueError('每个 RPS 约束法向必须为单位向量。')
    raw_dofs = np.asarray(dofs)
    if raw_dofs.shape != (6,) or raw_dofs.dtype.kind != 'b' or not raw_dofs.any():
        raise ValueError('自由度必须是六个布尔值，并至少开放一个自由度。')
    active = np.flatnonzero(raw_dofs)
    limits = np.asarray([translation_limit_mm, rotation_limit_deg], dtype=float)
    if limits.shape != (2,) or not np.isfinite(limits).all() or np.any(limits <= 0):
        raise ValueError('平移和旋转行程限值必须是有限正标量。')
    if rotation_limit_deg > 180:
        raise ValueError('局部 RPS 每轴旋转行程不得超过 180 度。')
    w = np.ones(len(s)) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != (len(s),) or not np.isfinite(w).all() or np.any(w <= 0):
        raise ValueError('每个 RPS 权重必须为有限正数。')
    # Normalize globally: this preserves the minimizer without weight overflow.
    root_w = np.sqrt(w/w.max())
    p = s[0]+np.mean(s-s[0], axis=0) if pivot is None else np.asarray(pivot, dtype=float)
    if p.shape != (3,) or not np.isfinite(p).all():
        raise ValueError('旋转中心必须为有限 XYZ 三维坐标。')
    local_source, local_target = s-p, t-p
    bounds = np.r_[np.full(3, limits[0]), np.full(3, limits[1])][active]

    def unpack(values):
        parameters = np.zeros(6)
        parameters[active] = values
        return parameters

    def residual(values):
        parameters = unpack(values)
        r, _ = _rotation_derivatives(parameters[3:])
        difference = local_source @ r.T + parameters[:3] - local_target
        return np.einsum('ij,ij->i', difference, n)

    def jacobian(values):
        parameters = unpack(values)
        _, derivatives = _rotation_derivatives(parameters[3:])
        j = np.column_stack([n, *[np.einsum('ij,ij->i', local_source@d.T, n) for d in derivatives]])
        return j[:, active]*root_w[:, None]

    _rank_condition(jacobian(np.zeros(len(active))))
    before = residual(np.zeros(len(active)))
    fit = least_squares(lambda x: residual(x)*root_w, np.zeros(len(active)),
                        jac=jacobian, bounds=(-bounds, bounds), x_scale='jac',
                        ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=500)
    rank, condition = _rank_condition(jacobian(fit.x))
    parameters = unpack(fit.x)
    r, _ = _rotation_derivatives(parameters[3:])
    delta = np.eye(4)
    delta[:3, :3] = r
    delta[:3, 3] = p-r@p+parameters[:3]
    at_bounds = np.zeros(6, dtype=bool)
    # Compare fractions of travel: an absolute tolerance can span the entire
    # feasible interval for valid tiny limits and mislabel zero as saturated.
    at_bounds[active] = np.isclose(np.abs(fit.x)/bounds, 1., rtol=1e-7, atol=0.)
    return _finite_result(dict(method='rps', transform=delta.tolist(),
                               **_metrics(residual(fit.x)), converged=bool(fit.success),
                               parameters=parameters.tolist(), pivot=p.tolist(),
                               before_residuals_mm=before.tolist(), rank=rank,
                               condition=condition, condition_method='column-normalized weighted Jacobian',
                               at_bounds=at_bounds.tolist(), dofs=raw_dofs.tolist(),
                               iterations=int(fit.nfev), solver_message=str(fit.message)))
