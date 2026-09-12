"""Numerical contracts. All geometry and distances are in millimetres."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
import hashlib
import time
import uuid
import re

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

MAX_POINTS = 2_000_000


def points_array(value):
    a = np.asarray(value, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 3 or not 0 < len(a) <= MAX_POINTS:
        raise ValueError(f'点集必须为 1–{MAX_POINTS:,} 行的 XYZ 三列坐标。')
    if not np.isfinite(a).all():
        raise ValueError('坐标包含 NaN 或无穷值，请清理源数据。')
    return np.ascontiguousarray(a)


def validate_transform(value):
    t = np.asarray(value, dtype=np.float64)
    if t.shape != (4, 4) or not np.isfinite(t).all():
        raise ValueError('变换必须为有限数值的 4×4 矩阵。')
    if not np.allclose(t[3], [0, 0, 0, 1], atol=1e-9):
        raise ValueError('刚体变换最后一行必须为 [0,0,0,1]。')
    r = t[:3, :3]
    if not np.allclose(r.T @ r, np.eye(3), atol=1e-7) or not np.isclose(np.linalg.det(r), 1, atol=1e-7):
        raise ValueError('仅支持刚体旋转和平移，请先将单位转换为 mm。')
    return t.copy()


@dataclass
class Dataset:
    name: str
    points: np.ndarray
    kind: str = 'cloud'
    triangles: np.ndarray | None = None
    source_path: str | None = None
    cad_shape: object | None = None
    transform: np.ndarray = field(default_factory=lambda: np.eye(4))
    id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def __post_init__(self):
        self.points = points_array(self.points).copy()
        self.transform = validate_transform(self.transform)
        if self.kind not in ('cloud', 'mesh', 'cad'):
            raise ValueError('不支持的几何类型。')
        if not isinstance(self.name, str) or not self.name.strip() or len(self.name) > 512:
            raise ValueError('几何名称必须为 1–512 个字符。')
        if not isinstance(self.id, str) or not re.fullmatch(r'[a-f0-9]{32}', self.id):
            raise ValueError('无效的几何标识。')
        if self.triangles is not None:
            original = np.asarray(self.triangles)
            if original.ndim != 2 or original.shape[1] != 3 or not len(original):
                raise ValueError('网格必须包含三角形索引。')
            if not np.isfinite(original).all() or not np.equal(original, np.floor(original)).all():
                raise ValueError('网格索引必须为整数。')
            self.triangles = original.astype(np.int64)
            if self.triangles.min() < 0 or self.triangles.max() >= len(self.points):
                raise ValueError('网格索引超出顶点范围。')
        if self.kind == 'mesh' and self.triangles is None:
            raise ValueError('网格缺少三角形。')
        if self.kind == 'cad' and (self.cad_shape is None or self.cad_shape.IsNull()):
            raise ValueError('CAD形状为空。')

    def world_points(self):
        t = validate_transform(self.transform)
        return self.points @ t[:3, :3].T + t[:3, 3]


def signature(asset):
    h = hashlib.sha256()
    for a in (asset.points, asset.transform, asset.triangles):
        if a is not None:
            h.update(np.ascontiguousarray(a).tobytes())
    h.update(asset.id.encode('ascii'))
    return h.hexdigest()


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


@dataclass
class RegistrationResult:
    source_id: str
    target_id: str
    method: str
    transform: np.ndarray
    elapsed_s: float
    rmse_mm: float
    fitness: float
    converged: bool
    iterations: int
    parameters: dict
    source_transform: np.ndarray
    target_transform: np.ndarray
    source_fingerprint: str = ''
    target_fingerprint: str = ''

    def to_dict(self):
        return _jsonable(asdict(self))


@dataclass
class DeviationResult:
    source_id: str
    target_id: str
    indices: np.ndarray
    distances_mm: np.ndarray
    method: str
    tolerance_mm: float
    total_count: int
    elapsed_s: float
    source_transform: np.ndarray
    target_transform: np.ndarray
    source_fingerprint: str = ''
    target_fingerprint: str = ''

    @property
    def statistics(self):
        d = self.distances_mm
        return dict(rms_mm=float(np.sqrt(np.mean(d * d))), p95_mm=float(np.percentile(d, 95)),
                    max_mm=float(np.max(d)), mean_mm=float(np.mean(d)),
                    within_fraction=float(np.mean(d <= self.tolerance_mm)),
                    sample_count=len(d), total_count=self.total_count)

    def to_dict(self):
        result = _jsonable(asdict(self))
        result['statistics'] = self.statistics
        return result


def rigid_transform(tx=0, ty=0, tz=0, rx=0, ry=0, rz=0):
    values = np.asarray([tx, ty, tz, rx, ry, rz], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('平移和转角必须为有限数值。')
    t = np.eye(4)
    t[:3, :3] = Rotation.from_euler('xyz', [rx, ry, rz], degrees=True).as_matrix()
    t[:3, 3] = [tx, ty, tz]
    return t


def fit_landmarks(source_points, target_points):
    s, t = points_array(source_points), points_array(target_points)
    if s.shape != t.shape or len(s) < 3:
        raise ValueError('至少需要三组对应且不共线的特征点。')
    sc, tc = s - s.mean(axis=0), t - t.mean(axis=0)
    if np.linalg.matrix_rank(sc) < 2 or np.linalg.matrix_rank(tc) < 2:
        raise ValueError('特征点共线，无法确定唯一刚体变换。')
    u, _, vt = np.linalg.svd(sc.T @ tc)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1] *= -1
        r = vt.T @ u.T
    matrix = np.eye(4)
    matrix[:3, :3] = r
    matrix[:3, 3] = t.mean(axis=0) - r @ s.mean(axis=0)
    return validate_transform(matrix)


def make_demo():
    """Reproducible synthetic formed panel; not factory or metrology data."""
    rng = np.random.default_rng(240912)
    x, y = rng.uniform(-90, 90, 4200), rng.uniform(-55, 55, 4200)
    keep = np.ones(len(x), bool)
    for cx, cy in [(-65, -35), (65, -35), (-65, 35), (65, 35)]:
        keep &= (x-cx)**2 + (y-cy)**2 > 8**2
    x, y = x[keep], y[keep]
    panel = np.c_[x, y, .001*x*x + .002*y*y]
    parts = [panel]
    for cy in (-55, 55):
        x = rng.uniform(-90, 90, 450)
        parts.append(np.c_[x, np.full(len(x), cy), .001*x*x + .002*cy*cy + rng.uniform(0, 16, len(x))])
    for cx, cy in [(-65, -35), (65, -35), (-65, 35), (65, 35)]:
        angle, z = rng.uniform(0, 2*np.pi, 250), rng.uniform(0, 12, 250)
        parts.append(np.c_[cx+8*np.cos(angle), cy+8*np.sin(angle), .001*cx*cx+.002*cy*cy+z])
    target = np.vstack(parts)
    expected = rigid_transform(tx=2, ty=-1.3, tz=1.5, rx=1, ry=-2, rz=2)
    source = (target-expected[:3, 3]) @ expected[:3, :3]
    source += rng.normal(0, .008, source.shape)
    return (Dataset('合成扫描件 · 待配准', source),
            Dataset('合成基准件 · 固定', target), expected)


def _registration_points(asset):
    p = asset.world_points()
    if asset.triangles is None:
        return p
    # Native mesh vertices alone undersample planar CAD faces. Sample by area.
    tri = p[asset.triangles]
    areas = np.linalg.norm(np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]), axis=1)
    if areas.sum() <= 1e-15:
        raise ValueError('网格三角形面积为零。')
    rng = np.random.default_rng(240912)
    chosen = tri[rng.choice(len(tri), size=20000, p=areas/areas.sum())]
    uv = rng.random((len(chosen), 2))
    uv[uv.sum(axis=1) > 1] = 1-uv[uv.sum(axis=1) > 1]
    return chosen[:, 0] + uv[:, :1]*(chosen[:, 1]-chosen[:, 0]) + uv[:, 1:]*(chosen[:, 2]-chosen[:, 0])


def register(source, target, method='gicp', voxel_mm=2.0,
             max_distance_mm=20.0, threads=4, max_iterations=60):
    import small_gicp
    if source.id == target.id:
        raise ValueError('请选择不同的移动件和基准件。')
    if method not in ('icp', 'gicp'):
        raise ValueError('请选择 ICP 或 GICP。')
    if not np.isfinite([voxel_mm, max_distance_mm]).all() or voxel_mm <= 0 or max_distance_mm <= 0:
        raise ValueError('体素和对应点距离必须大于零。')
    if not isinstance(threads, (int, np.integer)) or not 1 <= threads <= 64:
        raise ValueError('线程数必须为 1–64。')
    if not isinstance(max_iterations, (int, np.integer)) or not 1 <= max_iterations <= 500:
        raise ValueError('迭代数必须为 1–500。')
    start = time.perf_counter()
    s, t = _registration_points(source), _registration_points(target)
    if len(s) < 20 or len(t) < 20 or np.linalg.matrix_rank(s-s.mean(0)) < 2 or np.linalg.matrix_rank(t-t.mean(0)) < 2:
        raise ValueError('配准需要至少20个有效且不共线的点。')
    origin = (s.mean(0)+t.mean(0))/2
    sl, tl = np.ascontiguousarray(s-origin), np.ascontiguousarray(t-origin)
    for p in (sl, tl):
        voxels = np.floor(p/voxel_mm)
        if np.max(np.abs(voxels)) >= 2**20:
            raise ValueError('坐标范围超出体素索引范围，请增大体素或调整原点。')
        if len(np.unique(voxels, axis=0)) < 20:
            raise ValueError('体素过大，下采样后不足20个点。')
    tree = cKDTree(t)
    initial_d, _ = tree.query(s, workers=threads)
    params = dict(voxel_mm=float(voxel_mm), max_distance_mm=float(max_distance_mm),
                  threads=threads, max_iterations=max_iterations,
                  backend='small_gicp', source_registration_points=len(s), target_registration_points=len(t))
    matrix, converged, iterations = source.transform.copy(), False, 0
    distances = initial_d
    if np.count_nonzero(initial_d <= max_distance_mm) >= 20:
        native = small_gicp.align(tl, sl, registration_type=method.upper(),
                                 downsampling_resolution=float(voxel_mm),
                                 max_correspondence_distance=float(max_distance_mm),
                                 num_threads=int(threads), max_iterations=int(max_iterations))
        delta = validate_transform(native.T_target_source)
        delta[:3, 3] += origin-delta[:3, :3]@origin
        matrix = validate_transform(delta @ source.transform)
        moved = s @ delta[:3, :3].T + delta[:3, 3]
        distances, _ = tree.query(moved, workers=threads)
        converged = bool(native.converged)
        iterations = int(native.iterations)
    inliers = distances <= max_distance_mm
    fitness = float(np.mean(inliers))
    rmse = float(np.sqrt(np.mean(distances[inliers]**2))) if inliers.any() else float(np.sqrt(np.mean(distances**2)))
    # Numerical convergence alone is insufficient if correspondence support collapsed.
    converged = converged and np.count_nonzero(inliers) >= 20 and fitness >= .1
    return RegistrationResult(source.id, target.id, method, matrix,
                              time.perf_counter()-start, rmse, fitness, converged, iterations, params,
                              source.transform.copy(), target.transform.copy(), signature(source), signature(target))


def apply_registration(source, result):
    if not result.converged:
        raise ValueError('配准未收敛或有效对应点不足，不能应用结果。')
    if source.id != result.source_id or signature(source) != result.source_fingerprint:
        raise ValueError('移动件已变化，配准结果过期，请重新计算。')
    source.transform = validate_transform(result.transform)


def polydata_from_arrays(points, triangles=None):
    from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
    from vtkmodules.vtkCommonCore import vtkPoints
    from vtkmodules.vtkCommonDataModel import vtkPolyData, vtkCellArray
    data = vtkPolyData()
    vp = vtkPoints()
    vp.SetData(numpy_to_vtk(np.ascontiguousarray(points), deep=True))
    data.SetPoints(vp)
    if triangles is not None:
        cells = vtkCellArray()
        packed = np.c_[np.full(len(triangles), 3), triangles].astype(np.int64).ravel()
        cells.SetCells(len(triangles), numpy_to_vtkIdTypeArray(packed, deep=True))
        data.SetPolys(cells)
    return data


def _mesh_distances(points, target):
    from vtkmodules.vtkCommonCore import reference
    from vtkmodules.vtkCommonDataModel import vtkStaticCellLocator
    mesh = polydata_from_arrays(target.world_points(), target.triangles)
    locator = vtkStaticCellLocator()
    locator.SetDataSet(mesh)
    locator.BuildLocator()
    distances = np.empty(len(points))
    closest = [0., 0., 0.]
    cell, sub, dist2 = reference(0), reference(0), reference(0.)
    for i, point in enumerate(points):
        locator.FindClosestPoint(point, closest, cell, sub, dist2)
        if int(cell) < 0:
            raise ValueError('无法计算网格表面最近点。')
        distances[i] = np.sqrt(max(float(dist2), 0))
    return distances


def _cad_distances(points, target):
    from OCP.BRep import BRep_Builder
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
    from OCP.BRepExtrema import BRepExtrema_DistShapeShape
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopoDS import TopoDS_Compound
    from OCP.gp import gp_Pnt
    # A solid-volume distance is zero for interior points. Compare against faces.
    faces, builder = TopoDS_Compound(), BRep_Builder()
    builder.MakeCompound(faces)
    explorer = TopExp_Explorer(target.cad_shape, TopAbs_FACE)
    count = 0
    while explorer.More():
        builder.Add(faces, explorer.Current())
        count += 1
        explorer.Next()
    if not count:
        raise ValueError('CAD没有可测量的表面。')
    matrix = validate_transform(target.transform)
    local = (points-matrix[:3, 3]) @ matrix[:3, :3]
    result = np.empty(len(local))
    for i, point in enumerate(local):
        vertex = BRepBuilderAPI_MakeVertex(gp_Pnt(*map(float, point))).Vertex()
        distance = BRepExtrema_DistShapeShape(vertex, faces)
        if not distance.IsDone() or distance.NbSolution() < 1:
            raise ValueError(f'第{i+1}个采样点的CAD距离求解失败。')
        result[i] = distance.Value()
    return result


def measure_deviation(source, target, tolerance_mm=.2, max_samples=5000):
    if not np.isfinite(tolerance_mm) or tolerance_mm < 0:
        raise ValueError('距离阈值必须为非负有限值。')
    if not isinstance(max_samples, (int, np.integer)) or not 1 <= max_samples <= MAX_POINTS:
        raise ValueError('采样数必须为有效正整数。')
    if source.id == target.id:
        raise ValueError('请选择不同的移动件和基准件。')
    start = time.perf_counter()
    world = source.world_points()
    indices = np.linspace(0, len(world)-1, min(len(world), max_samples), dtype=np.int64)
    sample = world[indices]
    if target.kind == 'cad':
        distances, method = _cad_distances(sample, target), 'cad_exact'
    elif target.triangles is not None:
        distances, method = _mesh_distances(sample, target), 'mesh_surface'
    else:
        distances, _ = cKDTree(target.world_points()).query(sample)
        method = 'cloud_nearest_point'
    if not np.isfinite(distances).all():
        raise ValueError('距离求解返回了无效结果。')
    return DeviationResult(source.id, target.id, indices, distances, method, float(tolerance_mm), len(world),
                           time.perf_counter()-start, source.transform.copy(), target.transform.copy(),
                           signature(source), signature(target))
