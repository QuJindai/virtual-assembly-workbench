"""Independent analytic fixtures for current-world datum delta transforms."""
import json

import numpy as np
import pytest


def datum_points():
    return np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0],
                     [0, 0, 3], [10, 0, 3], [0, 4, 5]], float)


def test_321_recovers_known_pose():
    from assembly_workbench.datums import align_321
    from assembly_workbench.core import rigid_transform
    p = datum_points()
    t = rigid_transform(3, -2, 1, 5, -3, 7)
    q = p @ t[:3, :3].T + t[:3, 3]
    np.testing.assert_allclose(align_321(p, q)['transform'], t, atol=1e-8)


def test_321_hierarchy_uses_secondary_normal_and_tertiary_coordinate():
    from assembly_workbench.datums import align_321
    p = datum_points()
    q = p.copy()
    q[:3, 2] += 2
    q[3:5, 1] += 3
    q[5, 0] += 4
    q[3:5, 0] += [17, 28]  # tangential motion does not move secondary datum
    result = align_321(p, q)
    np.testing.assert_allclose(np.array(result['transform'])[:3, 3], [4, 3, 2])
    np.testing.assert_allclose(result['residuals_mm'], 0, atol=1e-10)
    assert result['converged']
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('change', ['collinear', 'secondary_parallel', 'nonfinite', 'count'])
def test_321_rejects_invalid_definitions(change):
    from assembly_workbench.datums import align_321
    p = datum_points()
    if change == 'collinear':
        p[2] = 2*p[1]
    elif change == 'secondary_parallel':
        p[4] = p[3] + [0, 0, 1]
    elif change == 'nonfinite':
        p[0, 0] = np.nan
    else:
        p = p[:5]
    with pytest.raises(ValueError):
        align_321(p, p)


def rps_fixture():
    rng = np.random.default_rng(91)
    p = rng.uniform(-20, 20, (40, 3)) + [1200, -800, 700]
    n = rng.normal(size=p.shape)
    n /= np.linalg.norm(n, axis=1)[:, None]
    return p, n


def test_rps_translation_only_scalar_residuals_and_locked_dofs():
    from assembly_workbench.datums import align_rps
    p, n = rps_fixture()
    original = p.copy()
    shift = np.array([3, -2, 1.5])
    result = align_rps(p, p+shift, n, weights=np.linspace(.5, 2, len(p)),
                       dofs=(True, True, True, False, False, False))
    np.testing.assert_allclose(result['parameters'], [3, -2, 1.5, 0, 0, 0], atol=1e-8)
    np.testing.assert_allclose(result['before_residuals_mm'], -n@shift, atol=1e-10)
    assert result['rank'] == 3 and result['converged']
    np.testing.assert_array_equal(p, original)
    json.dumps(result, allow_nan=False)


def test_rps_recovers_rotation_about_large_nonzero_pivot():
    from assembly_workbench.datums import align_rps
    from assembly_workbench.core import rigid_transform
    p, n = rps_fixture()
    p += [1e8, -2e8, 3e8]
    pivot = p.mean(0)
    pose = rigid_transform(.7, -.4, .3, 4, -3, 7)
    q = (p-pivot)@pose[:3, :3].T + pivot + pose[:3, 3]
    result = align_rps(p, q, n)
    np.testing.assert_allclose(result['parameters'], [.7, -.4, .3, 4, -3, 7], atol=2e-6)
    matrix = np.asarray(result['transform'])
    np.testing.assert_allclose(p@matrix[:3, :3].T+matrix[:3, 3], q, atol=1e-6)
    assert result['rms_mm'] < 1e-6 and result['rank'] == 6


def test_rps_fixed_translation_remains_zero_with_nonzero_pivot_rotation():
    from assembly_workbench.datums import align_rps
    from assembly_workbench.core import rigid_transform
    p, n = rps_fixture()
    pivot = np.array([1100, -700, 650.])
    rot = rigid_transform(rz=9)[:3, :3]
    q = (p-pivot)@rot.T + pivot
    result = align_rps(p, q, n, dofs=(False, False, False, False, False, True), pivot=pivot)
    np.testing.assert_allclose(result['parameters'], [0, 0, 0, 0, 0, 9], atol=1e-8)


def test_rps_bound_saturation_is_reported_with_remaining_error():
    from assembly_workbench.datums import align_rps
    p = np.array([[10, 20, 30], [11, 25, 32]], float)
    n = np.tile([1., 0, 0], (2, 1))
    result = align_rps(p, p+[8, 100, 200], n,
                       dofs=(True, False, False, False, False, False), translation_limit_mm=2)
    assert result['parameters'][0] == pytest.approx(2)
    assert result['at_bounds'][0]
    assert result['rms_mm'] == pytest.approx(6)
    np.testing.assert_allclose(result['residuals_mm'], [-6, -6])


@pytest.mark.parametrize('kwargs', [
    {'weights': [0]*40}, {'weights': [np.inf]*40}, {'weights': [-1]*40},
    {'dofs': [1]*6}, {'dofs': [False]*6}, {'dofs': [True]*5},
    {'translation_limit_mm': 0}, {'rotation_limit_deg': np.inf}, {'pivot': [0, 0]},
])
def test_rps_rejects_invalid_options(kwargs):
    from assembly_workbench.datums import align_rps
    p, n = rps_fixture()
    with pytest.raises(ValueError):
        align_rps(p, p, n, **kwargs)


def test_rps_rejects_unconstrained_axis_and_nonunit_normals():
    from assembly_workbench.datums import align_rps
    p, n = rps_fixture()
    with pytest.raises(ValueError, match='rank|秩|约束'):
        align_rps(p, p, np.tile([0., 0, 1], (len(p), 1)))
    with pytest.raises(ValueError):
        align_rps(p, p, n*2)


def test_rps_weights_change_the_physical_compromise():
    from assembly_workbench.datums import align_rps
    source = np.array([[0., 2, 3], [2, 4, 5]])
    target = source + np.array([[0., 900, 800], [10, -900, -800]])
    result = align_rps(source, target, [[1., 0, 0], [1., 0, 0]], weights=[1, 3],
                       dofs=(True, False, False, False, False, False))
    assert result['parameters'][0] == pytest.approx(7.5)
    np.testing.assert_allclose(result['residuals_mm'], [7.5, -2.5])
    assert result['rms_mm'] == pytest.approx(np.sqrt((7.5**2+2.5**2)/2))


def test_rps_rotation_bound_reports_active_rotation_axis():
    from assembly_workbench.datums import align_rps
    from assembly_workbench.core import rigid_transform
    source, normals = rps_fixture()
    pivot = source.mean(0)
    rotation = rigid_transform(rz=12)[:3, :3]
    target = (source-pivot)@rotation.T+pivot
    result = align_rps(source, target, normals, pivot=pivot, rotation_limit_deg=3,
                       dofs=(False, False, False, False, False, True))
    assert result['parameters'][5] == pytest.approx(3)
    assert result['at_bounds'] == [False, False, False, False, False, True]
    assert result['rms_mm'] > .1


def test_rps_zero_adjustment_is_not_at_tiny_positive_travel_bound():
    from assembly_workbench.datums import align_rps
    p = np.array([[0., 2, 3], [1., 4, 5]])
    result = align_rps(p, p, [[1., 0, 0], [1., 0, 0]],
                       dofs=(True, False, False, False, False, False),
                       translation_limit_mm=1e-12)
    assert result['parameters'] == [0., 0., 0., 0., 0., 0.]
    assert result['at_bounds'] == [False]*6
    assert result['converged']
