"""Geometric ground truth is generated independently of fitting routines."""
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation


ROT = Rotation.from_euler('xyz', [29, -36, 17], degrees=True).as_matrix()
CENTER = np.array([1600., -2700., 3300.])


def world(p, center=CENTER):
    return np.asarray(p)@ROT.T + center


def circle_points(count=100, radius=7., arc=2*np.pi):
    theta = np.linspace(0, arc, count, endpoint=False)
    return np.c_[radius*np.cos(theta), radius*np.sin(theta), np.zeros(count)]


def cylinder_points(radius=7., length=20., count=500):
    rng = np.random.default_rng(820)
    theta = rng.uniform(0, 2*np.pi, count)
    z = rng.uniform(-length/2, length/2, count)
    return np.c_[radius*np.cos(theta), radius*np.sin(theta), z]


@pytest.mark.parametrize('kind', ['line', 'plane', 'circle', 'sphere', 'cylinder'])
def test_tilted_noisy_feature_geometry(kind):
    from assembly_workbench.features import fit_feature
    rng = np.random.default_rng(930)
    if kind == 'line':
        p = np.c_[np.linspace(-20, 20, 200), np.zeros((200, 2))]
        expected_direction = ROT[:, 0]
    elif kind == 'plane':
        p = np.c_[rng.uniform(-20, 20, (300, 2)), np.zeros(300)]
        expected_direction = ROT[:, 2]
    elif kind == 'circle':
        p = circle_points(200)
        expected_direction = ROT[:, 2]
    elif kind == 'sphere':
        p = rng.normal(size=(500, 3))
        p = 7*p/np.linalg.norm(p, axis=1)[:, None]
    else:
        p = cylinder_points()
        expected_direction = ROT[:, 2]
    points = world(p) + rng.normal(0, .002, p.shape)
    original = points.copy()
    result = fit_feature(points, kind)
    assert result['kind'] == kind and result['count'] == len(p)
    assert result['rms_mm'] < .006 and result['max_mm'] < .02
    if kind != 'sphere':
        assert abs(np.dot(result['direction'], expected_direction)) > .99999
    if kind in ('circle', 'sphere', 'cylinder'):
        assert result['radius_mm'] == pytest.approx(7, abs=.005)
    if kind in ('circle', 'sphere'):
        np.testing.assert_allclose(result['center'], CENTER, atol=.005)
    assert result['form_error_mm'] >= 0
    assert isinstance(result['diagnostics'], dict)
    assert isinstance(result['warnings'], list)
    json.dumps(result, allow_nan=False)
    np.testing.assert_array_equal(points, original)


def test_circle_and_sphere_large_offsets():
    from assembly_workbench.features import fit_feature
    center = np.array([1e9, -2e9, 3e9])
    circle = fit_feature(world(circle_points(), center), 'circle')
    np.testing.assert_allclose(circle['center'], center, atol=1e-5, rtol=0)
    assert circle['radius_mm'] == pytest.approx(7, abs=1e-6)
    p = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])*7
    sphere = fit_feature(world(p, center), 'sphere')
    np.testing.assert_allclose(sphere['center'], center, atol=1e-5, rtol=0)
    assert sphere['radius_mm'] == pytest.approx(7, abs=1e-6)


@pytest.mark.parametrize('length', [1.5, 80.])
def test_cylinder_fits_unknown_axis_for_short_and_long_clouds(length):
    from assembly_workbench.features import fit_feature
    result = fit_feature(world(cylinder_points(length=length)), 'cylinder')
    assert abs(np.dot(result['direction'], ROT[:, 2])) > .999999
    assert result['radius_mm'] == pytest.approx(7, abs=1e-6)
    assert result['rms_mm'] < 1e-7
    if length < 2:
        assert result['warnings']


def test_partial_circle_reports_coverage_warning():
    from assembly_workbench.features import fit_feature
    result = fit_feature(world(circle_points(100, arc=np.pi/3)), 'circle')
    assert result['radius_mm'] == pytest.approx(7, abs=1e-5)
    assert result['warnings']
    assert result['diagnostics']['angular_coverage_deg'] < 90


@pytest.mark.parametrize('kind', ['line', 'plane', 'circle', 'sphere', 'cylinder'])
@pytest.mark.parametrize('bad', ['constant', 'nonfinite', 'undersampled'])
def test_bad_feature_inputs_raise(kind, bad):
    from assembly_workbench.features import fit_feature
    p = np.zeros((20, 3))
    if bad == 'nonfinite':
        p[0, 0] = np.inf
    elif bad == 'undersampled':
        p = p[:1]
    with pytest.raises(ValueError):
        fit_feature(p, kind)


@pytest.mark.parametrize('kind', ['plane', 'circle', 'sphere', 'cylinder'])
def test_collinear_feature_inputs_raise(kind):
    from assembly_workbench.features import fit_feature
    p = np.c_[np.linspace(-10, 10, 40), np.zeros((40, 2))]
    with pytest.raises(ValueError):
        fit_feature(world(p), kind)


def test_cylinder_requires_axial_support_and_sphere_requires_spatial_support():
    from assembly_workbench.features import fit_feature
    p = world(circle_points())
    for kind in ('sphere', 'cylinder'):
        with pytest.raises(ValueError):
            fit_feature(p, kind)


def test_tiny_arc_rejected_as_unreliable_radius():
    from assembly_workbench.features import fit_feature
    with pytest.raises(ValueError):
        fit_feature(world(circle_points(100, arc=.001)), 'circle')


@pytest.mark.parametrize('kind', ['plane', 'circle'])
def test_seeded_consensus_recovers_original_inlier_indices(kind):
    from assembly_workbench.features import detect_features
    rng = np.random.default_rng(630)
    if kind == 'plane':
        p = np.c_[rng.uniform(-10, 10, (180, 2)), np.zeros(180)]
    else:
        p = circle_points(180)
    inliers = world(p) + rng.normal(0, .001, p.shape)
    outliers = world(rng.uniform(-20, 20, (70, 3)))
    points = np.vstack([outliers[:35], inliers, outliers[35:]])
    result = detect_features(points, kind, threshold_mm=.02, min_points=100, max_features=2, seed=43)
    assert len(result) == 1
    assert set(range(35, 215)).issubset(result[0]['indices'])
    assert len(result[0]['indices']) <= 182
    assert result == detect_features(points, kind, threshold_mm=.02, min_points=100, max_features=2, seed=43)
    json.dumps(result, allow_nan=False)


def test_consensus_extracts_multiple_planes_without_losing_original_indices():
    from assembly_workbench.features import detect_features
    rng = np.random.default_rng(92)
    a = np.c_[rng.uniform(-10, 10, (100, 2)), np.zeros(100)]
    b = np.c_[rng.uniform(-10, 10, (80, 2)), np.full(80, 5.)]
    result = detect_features(world(np.vstack([a, b])), 'plane', min_points=50, threshold_mm=.01)
    assert len(result) == 2
    assert {frozenset(r['indices']) for r in result} == {frozenset(range(100)), frozenset(range(100, 180))}


@pytest.mark.parametrize('kwargs', [
    {'kind': 'sphere'}, {'threshold_mm': 0}, {'threshold_mm': np.nan},
    {'min_points': 2}, {'max_features': 0}, {'max_features': 1000}, {'seed': -1},
])
def test_consensus_options_are_bounded(kwargs):
    from assembly_workbench.features import detect_features
    with pytest.raises(ValueError):
        detect_features(circle_points(), **kwargs)


def test_consensus_rejects_over_budget_input():
    from assembly_workbench.features import detect_features
    with pytest.raises(ValueError):
        detect_features(np.zeros((200001, 3)))


@pytest.mark.parametrize('kind', ['line', 'plane'])
def test_isotropic_support_rejects_nonunique_principal_direction(kind):
    from assembly_workbench.features import fit_feature
    points = np.array([[1., 0, 0], [-1., 0, 0], [0, 1., 0],
                       [0, -1., 0], [0, 0, 1.], [0, 0, -1.]])
    with pytest.raises(ValueError, match='唯一|方向|法向'):
        fit_feature(world(points), kind)


def test_consensus_returns_full_resolution_indices_beyond_scoring_budget():
    from assembly_workbench.features import detect_features
    rng = np.random.default_rng(195)
    p = np.c_[rng.uniform(-30, 30, (5000, 2)), np.zeros(5000)]
    outliers = np.c_[rng.uniform(-30, 30, (300, 2)), np.full(300, 20.)]
    result = detect_features(world(np.vstack([outliers, p])), min_points=4000,
                             threshold_mm=.001, max_features=1)
    assert result[0]['count'] == 5000
    assert result[0]['indices'] == list(range(300, 5300))


def test_direct_fit_accepts_selection_above_consensus_budget():
    from assembly_workbench.features import fit_feature
    p = np.c_[np.linspace(-20, 20, 200001), np.zeros((200001, 2))]
    result = fit_feature(world(p), 'line')
    assert result['count'] == 200001
    assert result['rms_mm'] < 1e-9
    assert abs(np.dot(result['direction'], ROT[:, 0])) > .999999
