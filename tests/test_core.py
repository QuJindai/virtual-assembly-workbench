import numpy as np
import pytest

from assembly_workbench.core import (
    Dataset, apply_registration, fit_landmarks, make_demo,
    measure_deviation, register, rigid_transform,
)


@pytest.mark.parametrize('points', [[], [[1, 2]], [[1, 2, float('nan')]], [[1, float('inf'), 3]]])
def test_reject_invalid_points(points):
    with pytest.raises(ValueError):
        Dataset('invalid', points)


def test_reject_nonrigid_transform():
    matrix = np.diag([2., 1., 1., 1.])
    with pytest.raises(ValueError):
        Dataset('scaled', [[0, 0, 0]], transform=matrix)


def test_degree_rotation_and_translation_direction():
    part = Dataset('part', [[1, 0, 0]], transform=rigid_transform(tx=10, rz=90))
    np.testing.assert_allclose(part.world_points(), [[10, 1, 0]], atol=1e-12)


def test_landmark_fit_has_known_direction():
    source = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3.]])
    target = np.array([[4, 5, 6], [4, 6, 6], [2, 5, 6], [4, 5, 9.]])
    matrix = fit_landmarks(source, target)
    np.testing.assert_allclose(matrix, [[0, -1, 0, 4], [1, 0, 0, 5], [0, 0, 1, 6], [0, 0, 0, 1]], atol=1e-12)


def test_landmark_fit_rejects_collinear_points():
    with pytest.raises(ValueError):
        fit_landmarks([[0, 0, 0], [1, 0, 0], [2, 0, 0]], [[0, 1, 0], [1, 1, 0], [2, 1, 0]])


@pytest.mark.parametrize('method', ['icp', 'gicp'])
def test_real_registration_recovers_demo_transform_without_mutating(method):
    source, target, expected = make_demo()
    original = source.points.copy()
    result = register(source, target, method=method, voxel_mm=0.7, max_distance_mm=15)
    assert result.converged
    assert result.fitness > .99
    assert result.rmse_mm < .08
    np.testing.assert_allclose(result.transform, expected, atol=.05)
    np.testing.assert_array_equal(source.points, original)
    np.testing.assert_array_equal(source.transform, np.eye(4))
    apply_registration(source, result)
    assert measure_deviation(source, target).statistics['p95_mm'] < .08


def test_registration_composes_existing_world_transform():
    source, target, expected = make_demo()
    source.transform = rigid_transform(tx=1, tz=.5)
    result = register(source, target, voxel_mm=.7, max_distance_mm=15)
    np.testing.assert_allclose(result.transform, expected, atol=.05)


def test_nonoverlap_cannot_report_converged():
    source, target, _ = make_demo()
    source.transform = rigid_transform(tx=10000)
    result = register(source, target, max_distance_mm=1)
    assert not result.converged
    assert result.fitness == 0
    with pytest.raises(ValueError):
        apply_registration(source, result)


@pytest.mark.parametrize('args', [dict(voxel_mm=0), dict(max_distance_mm=-1), dict(threads=0), dict(max_iterations=0), dict(method='imaginary')])
def test_registration_rejects_invalid_parameters(args):
    source, target, _ = make_demo()
    with pytest.raises(ValueError):
        register(source, target, **args)


def test_stale_registration_cannot_be_applied():
    source, target, _ = make_demo()
    result = register(source, target)
    source.transform = rigid_transform(tx=2)
    with pytest.raises(ValueError):
        apply_registration(source, result)


def test_mesh_distance_is_to_triangle_interior_not_vertices():
    target = Dataset('triangle', [[0, 0, 0], [10, 0, 0], [0, 10, 0]], kind='mesh', triangles=[[0, 1, 2]])
    source = Dataset('samples', [[2, 2, 3], [2, 2, -4]])
    result = measure_deviation(source, target, tolerance_mm=3)
    np.testing.assert_allclose(result.distances_mm, [3, 4], atol=1e-10)
    assert result.method == 'mesh_surface'
    assert result.statistics['within_fraction'] == .5


def test_exact_cad_distance_measures_surface_including_interior():
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    target = Dataset('box', [[0, 0, 0], [10, 10, 10]], kind='cad', cad_shape=BRepPrimAPI_MakeBox(10., 10., 10.).Shape())
    source = Dataset('samples', [[5, 5, 5], [12, 5, 5], [5, 5, 10]])
    result = measure_deviation(source, target)
    np.testing.assert_allclose(result.distances_mm, [5, 2, 0], atol=1e-7)
    assert result.method == 'cad_exact'


def test_sampling_and_cloud_distance_statistics_are_explicit():
    source = Dataset('source', [[0, 0, 1], [0, 0, 2], [0, 0, 3], [0, 0, 4]])
    target = Dataset('origin', [[0, 0, 0]])
    result = measure_deviation(source, target, tolerance_mm=2, max_samples=2)
    np.testing.assert_allclose(result.distances_mm, [1, 4])
    assert result.statistics['sample_count'] == 2
    assert result.statistics['total_count'] == 4


def test_measurement_respects_both_world_transforms():
    source = Dataset('s', [[0, 0, 0]], transform=rigid_transform(tx=13))
    target = Dataset('t', [[0, 0, 0]], transform=rigid_transform(tx=10))
    np.testing.assert_allclose(measure_deviation(source, target).distances_mm, [3])
