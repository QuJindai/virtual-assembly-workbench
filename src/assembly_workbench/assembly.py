"""Explicit edge measurements and native world-space assembly geometry (mm)."""
from __future__ import annotations

import numpy as np

from .core import MAX_POINTS, Dataset, points_array, validate_transform


def _vector(value, name, *, unit=False):
    try:
        vector = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name}必须为有限 XYZ 三维向量。') from exc
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError(f'{name}必须为有限 XYZ 三维向量。')
    if unit:
        scale = float(np.max(np.abs(vector)))
        if scale == 0:
            raise ValueError(f'{name}不能为零向量。')
        vector = vector / scale
        vector = vector / np.linalg.norm(vector)
    return vector


def _positive(value, name):
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name}必须为有限正数。') from exc
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f'{name}必须为有限正数。')
    return value


def _finite(points):
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    if not np.isfinite(points).all():
        raise ValueError('几何计算超出有限数值范围，请检查坐标和单位。')
    return points


def gap_flush(source_points, reference_points, gap_axis, flush_axis) -> dict:
    """Project paired edge displacements; seam sign follows gap_axis × flush_axis."""
    source, reference = points_array(source_points), points_array(reference_points)
    if source.shape != reference.shape:
        raise ValueError('间隙与面差需要数量相同、顺序对应的两组边缘点。')
    gap = _vector(gap_axis, '间隙方向', unit=True)
    flush = _vector(flush_axis, '面差方向', unit=True)
    if abs(float(gap @ flush)) > 1e-8:
        raise ValueError('间隙方向与面差方向必须正交。')
    seam = np.cross(gap, flush)
    with np.errstate(over='ignore', invalid='ignore'):
        projections = (source - reference) @ np.column_stack([gap, flush, seam])
    projections = _finite(projections)
    stats = {}
    for column, name in enumerate(('gap', 'flush')):
        values = projections[:, column]
        # Scale the mean to avoid overflow when all measurements are large.
        scale = float(np.abs(values).max())
        mean = float(np.mean(values / scale) * scale) if scale else 0.0
        stats.update({f'{name}_min_mm': float(values.min()),
                      f'{name}_max_mm': float(values.max()), f'{name}_mean_mm': mean})
    return dict(method='edge_projection', count=len(source),
                gap_mm=projections[:, 0].tolist(), flush_mm=projections[:, 1].tolist(),
                seam_offset_mm=projections[:, 2].tolist(), source_points=source.tolist(),
                reference_points=reference.tolist(), gap_axis=gap.tolist(), flush_axis=flush.tolist(),
                statistics=stats)


def _world_shape(asset):
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCP.gp import gp_Trsf
    if not isinstance(asset, Dataset) or asset.kind != 'cad' or asset.cad_shape is None or asset.cad_shape.IsNull():
        raise ValueError('此操作需要有效的原生 CAD 几何。')
    matrix = validate_transform(asset.transform)
    trsf = gp_Trsf()
    trsf.SetValues(*map(float, matrix[:3].ravel()))
    # Deep-copy geometry: boolean preparation must never update the imported BRep.
    operation = BRepBuilderAPI_Transform(asset.cad_shape, trsf, True, False)
    if not operation.IsDone() or operation.Shape().IsNull():
        raise ValueError('CAD 世界坐标变换失败。')
    return operation.Shape()


def _limit(points, max_points):
    points = _finite(points)
    if len(points) > max_points:
        return points[np.linspace(0, len(points)-1, max_points, dtype=int)], True
    return points, False


def _cad_section(shape, origin, normal, max_points):
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Section
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.GeomAbs import GeomAbs_Line
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_VERTEX
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS
    from OCP.BRep import BRep_Tool
    from OCP.gp import gp_Pln, gp_Pnt, gp_Dir
    if not BRepCheck_Analyzer(shape).IsValid():
        raise ValueError('CAD 几何无效，无法可靠提取截面。')
    operation = BRepAlgoAPI_Section(shape, gp_Pln(gp_Pnt(*origin), gp_Dir(*normal)), False)
    operation.SetNonDestructive(True)
    operation.Build()
    if not operation.IsDone() or operation.Shape().IsNull():
        raise ValueError('OCCT 截面求交失败。')
    curves = []
    explorer = TopExp_Explorer(operation.Shape(), TopAbs_EDGE)
    while explorer.More():
        curve = BRepAdaptor_Curve(TopoDS.Edge_s(explorer.Current()))
        first, last = curve.FirstParameter(), curve.LastParameter()
        if not np.isfinite([first, last]).all() or max(abs(first), abs(last)) >= 1e100 or last < first:
            raise ValueError('CAD 截面含无界或无效曲线，无法进行有限采样。')
        curves.append((curve, first, last))
        if len(curves) > 100_000:
            raise ValueError('CAD 截面边数超过 100,000，请缩小几何选择范围。')
        explorer.Next()
    points = []
    sampled = bool(curves)
    if curves:
        chosen = np.linspace(0, len(curves)-1, min(len(curves), max_points), dtype=int)
        budget, extra = divmod(max_points, len(chosen))
        for i, index in enumerate(chosen):
            curve, first, last = curves[index]
            desired = 2 if curve.GetType() == GeomAbs_Line else 64
            count = min(desired, budget + (i < extra))
            for parameter in np.linspace(first, last, count):
                point = curve.Value(float(parameter))
                points.append([point.X(), point.Y(), point.Z()])
    else:
        # Tangency can produce isolated vertices rather than section edges.
        explorer = TopExp_Explorer(operation.Shape(), TopAbs_VERTEX)
        while explorer.More() and len(points) < max_points:
            point = BRep_Tool.Pnt_s(TopoDS.Vertex_s(explorer.Current()))
            points.append([point.X(), point.Y(), point.Z()])
            explorer.Next()
        sampled = explorer.More()
    return np.unique(_finite(points), axis=0), sampled


def _plane_distances(points, origin, normal):
    """Signed distance and projected floating-point error, independently per point.

    Only coordinates contributing to the normal projection affect its bound;
    a large tangential translation cannot broaden a plane or cloud slab.
    """
    with np.errstate(over='ignore', invalid='ignore'):
        local = points - origin
        distance = local @ normal
        eps = np.finfo(float).eps
        coordinate_error = eps * np.abs(points) + eps * np.abs(origin) + eps * np.abs(local)
        error = 16 * (coordinate_error @ np.abs(normal))
    if not np.isfinite(distance).all() or not np.isfinite(error).all():
        raise ValueError('截面距离计算超出有限数值范围，请检查坐标。')
    return distance, error


def _mesh_section(asset, origin, normal, max_points):
    points = _finite(asset.world_points())
    triangles = np.asarray(asset.triangles)
    if (triangles.ndim != 2 or triangles.shape[1] != 3 or not len(triangles) or
            not np.issubdtype(triangles.dtype, np.integer) or triangles.min() < 0 or triangles.max() >= len(points)):
        raise ValueError('网格三角形索引无效。')
    result = np.empty((0, 3))
    sampled = False
    for start in range(0, len(triangles), 8192):
        tri = points[triangles[start:start+8192]]
        distances, epsilon = _plane_distances(tri, origin, normal)
        possible_section = ((distances - epsilon).min(axis=1) <= 0) & ((distances + epsilon).max(axis=1) >= 0)
        # A mesh section has no user thickness. Refuse an uncertain intersection
        # beyond 1e-7 mm numerical resolution instead of silently thickening it.
        if np.any(possible_section[:, None] & (epsilon > 1e-7)):
            raise ValueError('网格法向坐标的数值精度不足以可靠求交，请将几何和截面原点移近坐标原点。')
        on_plane = np.abs(distances) <= epsilon
        pieces = [tri[on_plane]]
        for a, b in ((0, 1), (1, 2), (2, 0)):
            crossing = ((distances[:, a] < -epsilon[:, a]) & (distances[:, b] > epsilon[:, b])) | ((distances[:, b] < -epsilon[:, b]) & (distances[:, a] > epsilon[:, a]))
            da, db = distances[crossing, a], distances[crossing, b]
            fraction = da / (da - db)
            pieces.append(tri[crossing, a] + fraction[:, None] * (tri[crossing, b] - tri[crossing, a]))
        merged = np.unique(_finite(np.concatenate([result, *pieces])), axis=0)
        result, reduced = _limit(merged, max_points)
        sampled = sampled or reduced
    return result, sampled


def section_geometry(asset, origin, normal, *, slab_mm=0.2, max_points=20000) -> dict:
    """Extract world-space section points; slab_mm is full cloud slab thickness.

    CAD curves use at most 64 samples per edge (lines: endpoints), bounded by
    max_points. Mesh returns intersection endpoints and coplanar vertices. A
    capped selection is deterministic, not a uniform arc-length sampling.
    """
    origin = _vector(origin, '截面原点')
    normal = _vector(normal, '截面法向', unit=True)
    slab_mm = _positive(slab_mm, '薄层厚度')
    if isinstance(max_points, bool) or not isinstance(max_points, (int, np.integer)) or not 1 <= max_points <= MAX_POINTS:
        raise ValueError(f'截面点数上限必须为 1–{MAX_POINTS:,} 的整数。')
    if not isinstance(asset, Dataset):
        raise ValueError('请选择有效的几何数据。')
    try:
        if asset.kind == 'cad':
            points, sampled = _cad_section(_world_shape(asset), origin, normal, max_points)
            method = 'cad_section'
        elif asset.kind == 'mesh':
            points, sampled = _mesh_section(asset, origin, normal, max_points)
            method = 'mesh_section'
        elif asset.kind == 'cloud':
            world = _finite(asset.world_points())
            distance, roundoff = _plane_distances(world, origin, normal)
            if np.any(roundoff > slab_mm * .01):
                raise ValueError('点云法向坐标的数值精度超过薄层厚度的 1%，请将几何和截面原点移近坐标原点。')
            # Only existing points; do not project or synthesize measured points.
            points, sampled = _limit(world[np.abs(distance) <= slab_mm / 2 + roundoff], max_points)
            method = 'cloud_slab'
        else:
            raise ValueError('不支持的截面几何类型。')
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f'截面计算失败：{exc}') from exc
    return dict(points=_finite(points).tolist(), method=method, source_id=asset.id,
                origin=origin.tolist(), normal=normal.tolist(), slab_mm=slab_mm, sampled=sampled)


def _mass(shape, *, surface=False):
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    properties = GProp_GProps()
    if surface:
        BRepGProp.SurfaceProperties_s(shape, properties)
    else:
        BRepGProp.VolumeProperties_s(shape, properties, True, False, False)
    mass = float(properties.Mass())
    if not np.isfinite(mass):
        raise ValueError('CAD 面积或体积超出有限数值范围。')
    return mass


def _closed_solids(shape):
    from OCP.BRep import BRep_Tool
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.TopAbs import TopAbs_COMPOUND, TopAbs_COMPSOLID, TopAbs_SOLID, TopAbs_SHELL
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS_Iterator
    if shape.IsNull() or not BRepCheck_Analyzer(shape).IsValid():
        raise ValueError('干涉判定需要通过 OCCT 有效性检查的闭合 CAD 实体。')
    solids = []
    pending = [shape]
    while pending:
        item = pending.pop()
        if item.ShapeType() in (TopAbs_COMPOUND, TopAbs_COMPSOLID):
            children = TopoDS_Iterator(item)
            count = 0
            while children.More():
                pending.append(children.Value())
                children.Next()
                count += 1
            if not count:
                raise ValueError('CAD 组合含空成分，无法进行实体判定。')
        elif item.ShapeType() == TopAbs_SOLID:
            explorer = TopExp_Explorer(item, TopAbs_SHELL)
            count = 0
            while explorer.More():
                if not BRep_Tool.IsClosed_s(explorer.Current()):
                    raise ValueError('CAD 实体含开放壳，请先修复闭合性。')
                count += 1
                explorer.Next()
            if not count or _mass(item) <= 0:
                raise ValueError('CAD 实体必须具有闭合壳和有限正体积，请检查方向。')
            solids.append(item)
        else:
            raise ValueError('干涉判定只支持闭合实体；组合中的面、线或点不受支持。')
    if not solids:
        raise ValueError('CAD 几何不含闭合实体。')
    return solids


def _common(shape_a, shape_b):
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Common
    from OCP.BRepCheck import BRepCheck_Analyzer
    operation = BRepAlgoAPI_Common()
    from OCP.TopTools import TopTools_ListOfShape
    arguments, tools = TopTools_ListOfShape(), TopTools_ListOfShape()
    arguments.Append(shape_a)
    tools.Append(shape_b)
    operation.SetArguments(arguments)
    operation.SetTools(tools)
    operation.SetNonDestructive(True)
    operation.Build()
    if not operation.IsDone() or operation.Shape().IsNull() or not BRepCheck_Analyzer(operation.Shape()).IsValid():
        raise ValueError('OCCT 实体布尔求交失败，不能给出分离或干涉结论。')
    volume = _mass(operation.Shape())
    if volume < 0:
        raise ValueError('OCCT 公共体积为负，无法可靠判定干涉。')
    return volume


def cad_contact(source, target, *, tolerance_mm=1e-5) -> dict:
    """Classify valid closed solids by native common volume and minimum distance.

    Volume tolerance is the smaller body's surface area × distance tolerance,
    with a floating-point floor scaled by input volume. It is a conservative
    geometric resolution threshold, not a manufacturing acceptance tolerance.
    """
    tolerance = _positive(tolerance_mm, '接触距离容差')
    try:
        a, b = _world_shape(source), _world_shape(target)
        solids_a, solids_b = _closed_solids(a), _closed_solids(b)
        volume_a, volume_b = _mass(a), _mass(b)
        area = min(_mass(a, surface=True), _mass(b, surface=True))
        volume_tolerance = max(tolerance ** 3, tolerance * area,
                               128 * np.finfo(float).eps * max(volume_a, volume_b))
        if not np.isfinite(volume_tolerance):
            raise ValueError('接触容差与几何尺度超出有限数值范围。')
        # A compound is a union of disjoint solids. Overlapping constituents
        # would double-count volume; require the importer/user to fuse them first.
        for solids in (solids_a, solids_b):
            if len(solids) > 128:
                raise ValueError('单次干涉查询最多支持 128 个实体，请拆分装配选择。')
            for i, first in enumerate(solids):
                for second in solids[i+1:]:
                    if first.IsSame(second) or _common(first, second) > 0:
                        raise ValueError('CAD 组合含重复或重叠实体，请先合并实体再判定干涉。')
        common_volume = _common(a, b)
        from OCP.BRepExtrema import BRepExtrema_DistShapeShape
        distance = BRepExtrema_DistShapeShape(a, b)
        distance.Perform()
        if not distance.IsDone() or distance.NbSolution() == 0:
            raise ValueError('OCCT 最小距离求解失败，不能给出分离结论。')
        minimum = float(distance.Value())
        if not np.isfinite(minimum) or minimum < 0:
            raise ValueError('OCCT 最小距离无效，不能给出分离结论。')
        state = 'interference' if common_volume > volume_tolerance else ('contact' if minimum <= tolerance else 'clear')
        return dict(method='occt_solid_common', state=state, distance_mm=minimum,
                    common_volume_mm3=common_volume, volume_tolerance_mm3=float(volume_tolerance),
                    tolerance_mm=tolerance)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f'CAD 接触计算失败：{exc}') from exc
