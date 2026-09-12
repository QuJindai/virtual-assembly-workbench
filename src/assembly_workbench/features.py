"""Orthogonal geometric fits and bounded, seeded plane/circle proposals.

These are least-squares associated features, not certified GD&T evaluations.
All points use mm. Every output is finite JSON data; inputs are never changed.
"""
from __future__ import annotations

import json
import numpy as np
from scipy.optimize import least_squares

from .core import points_array

MAX_CONSENSUS_POINTS = 200_000
_MINIMUM = dict(line=2, plane=3, circle=3, sphere=4, cylinder=6)


def _points(points):
    p = points_array(points)
    if np.max(np.abs(p)) > 1e100:
        raise ValueError('坐标尺度过大，无法可靠拟合，请转换局部坐标。')
    return p


def _center_scale(p):
    origin = p[0] + (p-p[0]).mean(0)
    centered = p-origin
    scale = float(np.sqrt(np.mean(np.sum(centered*centered, axis=1))))
    if not np.isfinite(scale) or scale <= 1e-12:
        raise ValueError('点集重复或尺寸为零，无法拟合特征。')
    return origin, scale, centered/scale


def _oriented(direction):
    direction = direction/np.linalg.norm(direction)
    if direction[np.argmax(np.abs(direction))] < 0:
        direction = -direction
    return direction


def _basis(axis):
    axis = axis/np.linalg.norm(axis)
    other = np.eye(3)[np.argmin(np.abs(axis))]
    first = np.cross(axis, other)
    first /= np.linalg.norm(first)
    return first, np.cross(axis, first)


def _axis(base, first, second, slopes):
    a = base + slopes[0]*first + slopes[1]*second
    return a/np.linalg.norm(a)


def _conditioning(jac, count):
    singular = np.linalg.svd(jac, compute_uv=False)
    if (len(singular) < count or singular[0] <= 0 or
            singular[count-1] <= singular[0]*1e-8):
        raise ValueError('特征局部 Jacobian 秩不足，点集无法独立约束全部参数。')
    return float(singular[0]/singular[count-1])


def _coverage(points, center, axis):
    e, f = _basis(axis)
    d = points-center
    angles = np.sort(np.mod(np.arctan2(d@f, d@e), 2*np.pi))
    gaps = np.diff(np.r_[angles, angles[0]+2*np.pi])
    return float(np.rad2deg(2*np.pi-gaps.max()))


def _algebraic_circle(xy):
    a = np.column_stack((2*xy, np.ones(len(xy))))
    values, _, rank, _ = np.linalg.lstsq(a, np.sum(xy*xy, axis=1), rcond=None)
    radius2 = values[2]+np.dot(values[:2], values[:2])
    if rank < 3 or radius2 <= 1e-16:
        raise ValueError('圆点集共线或半径为零。')
    return values[:2], float(np.sqrt(radius2))


def _circle_fit(q, vt):
    base = vt[2]
    e, f = vt[:2]
    center2, radius = _algebraic_circle(q @ np.column_stack((e, f)))
    initial = np.r_[center2[0]*e+center2[1]*f, 0., 0., radius]

    def residual(parameters):
        center = parameters[:3]
        axis = _axis(base, e, f, parameters[3:5])
        d = q-center
        axial = d@axis
        radial = np.linalg.norm(d-axial[:, None]*axis, axis=1)-parameters[5]
        return np.r_[axial, radial]

    lower = [-1e4]*3+[-10.]*2+[1e-8]
    upper = [1e4]*3+[10.]*2+[1e4]
    if not np.all(np.asarray(lower) < initial) or not np.all(initial < np.asarray(upper)):
        raise ValueError('圆弧覆盖不足，圆心或半径超出可靠范围。')
    fit = least_squares(residual, initial, bounds=(lower, upper),
                        ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=300)
    if not fit.success or np.any(fit.active_mask):
        raise ValueError('圆正交拟合未收敛或触及数值边界，请扩大点集覆盖。')
    condition = _conditioning(fit.jac, 6)
    axis = _axis(base, e, f, fit.x[3:5])
    residuals = residual(fit.x).reshape(2, len(q))
    return fit.x[:3], axis, float(fit.x[5]), np.linalg.norm(residuals, axis=0), condition


def _sphere_fit(q):
    a = np.column_stack((2*q, np.ones(len(q))))
    values, _, rank, _ = np.linalg.lstsq(a, np.sum(q*q, axis=1), rcond=None)
    radius2 = values[3]+np.dot(values[:3], values[:3])
    if rank < 4 or radius2 <= 1e-16:
        raise ValueError('球点集空间支撑不足或半径为零。')
    initial = np.r_[values[:3], np.sqrt(radius2)]
    if np.any(np.abs(initial) >= 1e4):
        raise ValueError('球面覆盖不足，球心或半径超出可靠范围。')

    def residual(parameters):
        return np.linalg.norm(q-parameters[:3], axis=1)-parameters[3]

    fit = least_squares(residual, initial, bounds=([-1e4]*3+[1e-8], [1e4]*4),
                        ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=300)
    if not fit.success or np.any(fit.active_mask):
        raise ValueError('球正交拟合未收敛或触及数值边界。')
    return fit.x[:3], float(fit.x[3]), residual(fit.x), _conditioning(fit.jac, 4)


def _cylinder_fit(q, vt):
    # Each start has two transverse center coordinates, two axis slopes and
    # radius. Center constrained to the start's normal plane removes the
    # unobservable translation along the infinite cylinder axis.
    starts = [*vt, *np.eye(3)]
    candidates = []
    for base in starts:
        e, f = _basis(base)
        try:
            center2, radius = _algebraic_circle(q@np.column_stack((e, f)))
        except ValueError:
            continue
        initial = np.r_[center2, 0., 0., radius]
        if np.any(np.abs(initial) >= 1e4) or radius <= 1e-8:
            continue

        def residual(parameters):
            center = parameters[0]*e+parameters[1]*f
            axis = _axis(base, e, f, parameters[2:4])
            d = q-center
            radial = d-(d@axis)[:, None]*axis
            return np.linalg.norm(radial, axis=1)-parameters[4]

        fit = least_squares(residual, initial,
                            bounds=([-1e4]*2+[-10.]*2+[1e-8], [1e4]*2+[10.]*2+[1e4]),
                            ftol=1e-11, xtol=1e-11, gtol=1e-11, max_nfev=250)
        if not fit.success or np.any(fit.active_mask):
            continue
        try:
            condition = _conditioning(fit.jac, 5)
        except ValueError:
            continue
        center = fit.x[0]*e+fit.x[1]*f
        axis = _axis(base, e, f, fit.x[2:4])
        # Report the axis point nearest the data centroid, independent of chart.
        center -= np.dot(center, axis)*axis
        errors = residual(fit.x)
        candidates.append((float(np.dot(errors, errors)), center, axis, float(fit.x[4]), errors, condition))
    if not candidates:
        raise ValueError('圆柱多起点拟合未获得满秩收敛解，请增加轴向及周向点集覆盖。')
    _, center, axis, radius, errors, condition = min(candidates, key=lambda item: item[0])
    return center, axis, radius, errors, condition


def fit_feature(points, kind):
    """Fit line, plane, 3D circle, sphere or unknown-axis infinite cylinder.

    Circle residual is Euclidean distance to its 3D circumference (joint axial
    and radial residuals); sphere/cylinder use signed radial residuals. Line
    uses distance to its axis. Form error is explicitly defined in diagnostics,
    and is an associated-fit indicator, not a minimum-zone GD&T certification.
    """
    if kind not in _MINIMUM:
        raise ValueError('支持的特征类型为 line、plane、circle、sphere、cylinder。')
    p = _points(points)
    if len(p) < _MINIMUM[kind]:
        raise ValueError(f'{kind} 拟合至少需要 {_MINIMUM[kind]} 个点。')
    origin, scale, q = _center_scale(p)
    _, singular, vt = np.linalg.svd(q, full_matrices=False)
    required = dict(line=1, plane=2, circle=2, sphere=3, cylinder=3)[kind]
    if len(singular) < required or singular[required-1] <= singular[0]*1e-8:
        raise ValueError('点集共线、共面或维度支撑不足，无法唯一拟合指定特征。')
    if kind == 'line' and singular[0]-singular[1] <= singular[0]*1e-8:
        raise ValueError('主方向特征值重合，无法确定唯一直线方向。')
    if kind == 'plane' and singular[1]-singular[2] <= singular[0]*1e-8:
        raise ValueError('最小方向特征值重合，无法确定唯一平面法向。')
    warnings = []
    center, direction, radius = np.zeros(3), None, None
    diagnostics = dict(coordinate_scale_mm=scale, support_rank=required,
                       support_condition=float(singular[0]/singular[required-1]))
    if kind == 'line':
        direction = vt[0]
        residuals = np.linalg.norm(q-(q@direction)[:, None]*direction, axis=1)
        if len(singular) > 1 and singular[1] > singular[0]*.5:
            warnings.append('点集横向散布较大，直线方向识别较弱。')
    elif kind == 'plane':
        direction = vt[-1]
        residuals = q@direction
        if singular[-1] > singular[1]*.5:
            warnings.append('点集厚度较大，平面法向识别较弱。')
    elif kind == 'circle':
        center, direction, radius, residuals, condition = _circle_fit(q, vt)
        diagnostics['condition'] = condition
    elif kind == 'sphere':
        center, radius, residuals, condition = _sphere_fit(q)
        diagnostics['condition'] = condition
        offsets = q-center
        units = offsets/np.linalg.norm(offsets, axis=1)[:, None]
        coverage = float(np.linalg.norm(units.mean(0)))
        diagnostics['directional_imbalance'] = coverage
        if coverage > .7:
            warnings.append('球面覆盖集中在局部区域，球心与半径对噪声敏感。')
    else:
        center, direction, radius, residuals, condition = _cylinder_fit(q, vt)
        diagnostics['condition'] = condition
        axial_span = float(np.ptp((q-center)@direction))
        diagnostics['axial_span_mm'] = axial_span*scale
        diagnostics['axial_span_to_radius'] = axial_span/radius
        if axial_span < radius*1e-5:
            raise ValueError('圆柱轴向覆盖不足，无法可靠确定轴线。')
        if axial_span < radius*.5:
            warnings.append('圆柱轴向覆盖较短，轴线倾斜对噪声敏感。')
    if kind in ('circle', 'cylinder'):
        coverage = _coverage(q, center, direction)
        diagnostics['angular_coverage_deg'] = coverage
        if coverage < 1:
            raise ValueError('周向覆盖小于 1 度，半径估计不可靠，请扩大选区。')
        if coverage < 120:
            warnings.append('周向覆盖不足 120 度，圆心、轴线及半径对噪声敏感。')
    residuals = residuals*scale
    if kind in ('line', 'circle'):
        form_error = 2*float(np.max(np.abs(residuals)))
        diagnostics['form_error_definition'] = 'twice maximum orthogonal distance to associated feature'
    else:
        form_error = float(np.ptp(residuals))
        diagnostics['form_error_definition'] = 'peak-to-valley signed residual to associated feature'
    result = dict(kind=kind, method='orthogonal_least_squares', count=len(p),
                  center=(origin+center*scale).tolist(),
                  rms_mm=float(np.sqrt(np.mean(residuals*residuals))),
                  max_mm=float(np.max(np.abs(residuals))), form_error_mm=form_error,
                  diagnostics=diagnostics, warnings=warnings)
    if direction is not None:
        result['direction'] = _oriented(direction).tolist()
    if radius is not None:
        result['radius_mm'] = float(radius*scale)
    try:
        json.dumps(result, allow_nan=False)
    except (ValueError, TypeError) as exc:
        raise ValueError('特征计算超出有限数值范围。') from exc
    return result


def _proposal(sample, kind):
    a, b = sample[1]-sample[0], sample[2]-sample[0]
    normal = np.cross(a, b)
    nl = np.linalg.norm(normal)
    if nl < 1e-8*np.linalg.norm(a)*np.linalg.norm(b) or nl <= 1e-15:
        raise ValueError('共线候选点。')
    normal /= nl
    if kind == 'plane':
        return sample[0], normal, 0.
    e = a/np.linalg.norm(a)
    f = np.cross(normal, e)
    center2, radius = _algebraic_circle((sample-sample[0])@np.column_stack((e, f)))
    return sample[0]+center2[0]*e+center2[1]*f, normal, radius


def _distances(points, center, normal, radius, kind):
    difference = points-center
    axial = difference@normal
    if kind == 'plane':
        return np.abs(axial)
    radial = np.linalg.norm(difference-axial[:, None]*normal, axis=1)-radius
    return np.hypot(axial, radial)


def detect_features(points, kind='plane', *, threshold_mm=.1,
                    min_points=20, max_features=5, seed=0):
    """Seeded bounded consensus, returning original full-resolution indices.

    At most 200000 input points, 20 features, 256 proposals per feature, 4096
    scoring points per proposal and six full-resolution refits are allowed.
    Proposals are geometric candidates; they do not identify semantic holes.
    """
    if kind not in ('plane', 'circle'):
        raise ValueError('自动特征候选仅支持 plane 或 circle。')
    if not np.isscalar(threshold_mm) or not np.isfinite(threshold_mm) or threshold_mm <= 0:
        raise ValueError('候选距离阈值必须是有限正数。')
    for name, value, low, high in [('min_points', min_points, 3, MAX_CONSENSUS_POINTS),
                                  ('max_features', max_features, 1, 20),
                                  ('seed', seed, 0, 2**32-1)]:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
            raise ValueError(f'{name} 必须是 {low}–{high} 范围内的整数。')
    p = _points(points)
    if len(p) > MAX_CONSENSUS_POINTS:
        raise ValueError(f'自动特征候选最多支持 {MAX_CONSENSUS_POINTS} 点，请先明确选区。')
    if len(p) < min_points:
        return []
    origin, scale, q = _center_scale(p)
    threshold = threshold_mm/scale
    rng = np.random.default_rng(seed)
    remaining = np.arange(len(p))
    results = []
    for _ in range(max_features):
        if len(remaining) < min_points:
            break
        score_indices = rng.choice(remaining, size=min(4096, len(remaining)), replace=False)
        score_points = q[score_indices]
        best = None
        best_score = (-1, -np.inf)
        for _ in range(256):
            sample = q[rng.choice(remaining, size=3, replace=False)]
            try:
                center, axis, radius = _proposal(sample, kind)
            except ValueError:
                continue
            distances = _distances(score_points, center, axis, radius, kind)
            inside = distances <= threshold
            count = int(inside.sum())
            score = (count, -float(distances[inside].sum()))
            if score > best_score:
                best, best_score = (center, axis, radius), score
        if best is None:
            break
        chosen = remaining[_distances(q[remaining], *best, kind) <= threshold]
        if len(chosen) < min_points:
            break
        fitted = None
        stable = False
        for _ in range(6):
            try:
                fitted = fit_feature(p[chosen], kind)
            except ValueError:
                break
            center = (np.asarray(fitted['center'])-origin)/scale
            axis = np.asarray(fitted['direction'])
            radius = fitted.get('radius_mm', 0.)/scale
            updated = remaining[_distances(q[remaining], center, axis, radius, kind) <= threshold]
            if np.array_equal(chosen, updated):
                stable = True
                break
            if len(updated) < min_points:
                break
            chosen = updated
        if not stable:
            break
        fitted['indices'] = chosen.tolist()
        fitted['diagnostics'].update(consensus_seed=int(seed), threshold_mm=float(threshold_mm),
                                      proposal_limit=256, scoring_point_limit=4096)
        results.append(fitted)
        remaining = remaining[~np.isin(remaining, chosen)]
    return results
