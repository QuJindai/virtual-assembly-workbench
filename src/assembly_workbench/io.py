"""Geometry import, self-contained projects and reproducible local reports."""
from __future__ import annotations

import csv
import html
import io
import json
import os
from pathlib import Path
import tempfile
import uuid
import zipfile
from datetime import datetime, timezone

import numpy as np

from . import __version__
from .core import Dataset, points_array, signature, MAX_POINTS

MAX_FILE_BYTES = 512 * 1024 * 1024
MAX_ARCHIVE_BYTES = 768 * 1024 * 1024
MAX_MANIFEST_BYTES = 8 * 1024 * 1024


def _json(text):
    def bad_constant(value):
        raise ValueError(f'JSON包含非有限数值: {value}')
    return json.loads(text, parse_constant=bad_constant)


def _text_points(path, delimiter=None):
    with path.open('r', encoding='utf-8-sig') as stream:
        lines = [line.strip() for line in stream if line.strip() and not line.lstrip().startswith('#')]
    if not lines:
        raise ValueError('文件没有点坐标。')
    split = lambda line: next(csv.reader([line])) if delimiter else line.split()
    first = split(lines[0])
    fields = [item.strip().lower() for item in first]
    if all(name in fields for name in ('x', 'y', 'z')):
        indices = [fields.index(name) for name in ('x', 'y', 'z')]
        lines = lines[1:]
    else:
        indices = [0, 1, 2]
    if len(lines) > MAX_POINTS:
        raise ValueError('点数超过当前工作台上限，请先分块。')
    try:
        return points_array([[float(split(line)[i]) for i in indices] for line in lines])
    except (IndexError, TypeError, ValueError) as error:
        raise ValueError('无法读取XYZ坐标；请使用三列数值或带x,y,z列名的文件。') from error


def _pcd_points(path):
    header, data = {}, []
    with path.open('r', encoding='utf-8') as stream:
        for line in stream:
            fields = line.strip().split()
            if not fields or fields[0].startswith('#'):
                continue
            header[fields[0].upper()] = fields[1:]
            if fields[0].upper() == 'DATA':
                if fields[1:] != ['ascii']:
                    raise ValueError('当前支持ASCII PCD，请将二进制PCD转换为ASCII或PLY。')
                data = [line.strip() for line in stream if line.strip()]
                break
    names = [n.lower() for n in header.get('FIELDS', [])]
    counts = header.get('COUNT', ['1'] * len(names))
    if len(counts) != len(names) or any(c != '1' for c in counts):
        raise ValueError('PCD仅支持COUNT=1的标量字段。')
    if not all(n in names for n in ('x', 'y', 'z')):
        raise ValueError('PCD缺少x/y/z字段。')
    expected = int(header.get('POINTS', [len(data)])[0])
    if expected != len(data) or len(data) > MAX_POINTS:
        raise ValueError('PCD点数与声明不符或超过上限。')
    try:
        indices = [names.index(n) for n in ('x', 'y', 'z')]
        return points_array([[float(line.split()[i]) for i in indices] for line in data])
    except (IndexError, ValueError) as error:
        raise ValueError('PCD坐标无效。') from error


def _polydata_arrays(data):
    from vtkmodules.util.numpy_support import vtk_to_numpy
    from vtkmodules.vtkFiltersCore import vtkTriangleFilter
    if data.GetNumberOfPoints() < 1 or data.GetNumberOfPoints() > MAX_POINTS:
        raise ValueError('几何文件为空或顶点数超过上限。')
    triangle = vtkTriangleFilter()
    triangle.SetInputData(data)
    triangle.Update()
    mesh = triangle.GetOutput()
    p = vtk_to_numpy(mesh.GetPoints().GetData()).astype(np.float64)
    triangles = []
    for i in range(mesh.GetNumberOfCells()):
        cell = mesh.GetCell(i)
        if cell.GetCellDimension() == 2 and cell.GetNumberOfPoints() == 3:
            triangles.append([cell.GetPointId(k) for k in range(3)])
    return p, np.asarray(triangles, dtype=np.int64) if triangles else None


def tessellate_shape(shape, deflection=.2):
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.BRep import BRep_Tool
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
    from OCP.TopoDS import TopoDS
    from OCP.TopLoc import TopLoc_Location
    mesher = BRepMesh_IncrementalMesh(shape, float(deflection), False, .35, True)
    if not mesher.IsDone():
        raise ValueError('CAD显示网格生成失败。')
    vertices, triangles = [], []
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = TopoDS.Face_s(explorer.Current())
        location = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation_s(face, location)
        if triangulation is not None:
            offset = len(vertices)
            for i in range(1, triangulation.NbNodes()+1):
                p = triangulation.Node(i).Transformed(location.Transformation())
                vertices.append([p.X(), p.Y(), p.Z()])
            for i in range(1, triangulation.NbTriangles()+1):
                a, b, c = triangulation.Triangle(i).Get()
                if face.Orientation() == TopAbs_REVERSED:
                    b, c = c, b
                triangles.append([offset+a-1, offset+b-1, offset+c-1])
        if len(vertices) > MAX_POINTS:
            raise ValueError('CAD显示网格过大，请简化模型。')
        explorer.Next()
    if not triangles:
        raise ValueError('CAD没有可显示的面。')
    return points_array(vertices), np.asarray(triangles, dtype=np.int64)


def load_dataset(path, unit='mm'):
    path = Path(path)
    if unit not in ('mm', 'm'):
        raise ValueError('导入单位仅支持 mm 或 m。')
    if not path.is_file() or not 0 < path.stat().st_size <= MAX_FILE_BYTES:
        raise ValueError('文件不存在、为空或超过512 MiB。')
    extension = path.suffix.lower()
    if extension in ('.catpart', '.catproduct', '.catdrawing', '.3dxml'):
        raise ValueError('当前不支持CATIA原生格式；请使用CATIA或合规转换器导出STEP/IGES。改文件扩展名不能转换格式。')
    scale = 1000. if unit == 'm' else 1.
    if extension in ('.csv', '.xyz', '.txt', '.pts'):
        points = _text_points(path, ',' if extension == '.csv' else None) * scale
        return Dataset(path.name, points, source_path=str(path.resolve()))
    if extension == '.pcd':
        return Dataset(path.name, _pcd_points(path)*scale, source_path=str(path.resolve()))
    if extension in ('.stl', '.obj', '.ply'):
        from vtkmodules.vtkIOGeometry import vtkSTLReader, vtkOBJReader
        from vtkmodules.vtkIOPLY import vtkPLYReader
        reader = {'.stl':vtkSTLReader, '.obj':vtkOBJReader, '.ply':vtkPLYReader}[extension]()
        reader.SetFileName(str(path.resolve()))
        reader.Update()
        points, triangles = _polydata_arrays(reader.GetOutput())
        return Dataset(path.name, points*scale, 'mesh' if triangles is not None else 'cloud',
                       triangles, str(path.resolve()))
    if extension in ('.step', '.stp', '.iges', '.igs'):
        from OCP.IFSelect import IFSelect_RetDone
        from OCP.STEPControl import STEPControl_Reader
        from OCP.IGESControl import IGESControl_Reader
        from OCP.Interface import Interface_Static
        # OCCT exchanges to millimetres regardless of source file units.
        if unit != 'mm':
            raise ValueError('STEP/IGES由OCCT读取文件单位并转换为mm，请选择mm。')
        reader = STEPControl_Reader() if extension in ('.step', '.stp') else IGESControl_Reader()
        Interface_Static.SetCVal_s('xstep.cascade.unit', 'MM')
        if reader.ReadFile(str(path.resolve())) != IFSelect_RetDone:
            raise ValueError('OCCT无法读取此CAD文件。')
        if not reader.TransferRoots():
            raise ValueError('CAD文件未包含可转换的实体或曲面。')
        shape = reader.OneShape()
        if shape.IsNull():
            raise ValueError('CAD形状为空。')
        points, triangles = tessellate_shape(shape)
        return Dataset(path.name, points, 'cad', triangles, str(path.resolve()), shape)
    raise ValueError('不支持的文件格式；请选择CSV/XYZ/PCD/PLY/STL/OBJ/STEP/IGES。')


def save_project(path, assets, history=None):
    path = Path(path)
    if not assets or len(assets) > 100 or len({a.id for a in assets}) != len(assets):
        raise ValueError('工程需包含1–100个不同的几何对象。')
    version = 2 if any(a.point_ids is not None for a in assets) else 1
    manifest = dict(format='assembly-workbench', version=version, units='mm', app_version=__version__, assets=[], history=history or [])
    # Validate JSON before opening/replacing any destination.
    json.dumps(manifest, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.assembly-', suffix='.vaw', dir=path.parent)
    os.close(fd)
    try:
        total = 0
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for asset in assets:
                asset.world_points()  # validate any edited matrix
                entry = dict(id=asset.id, name=asset.name, kind=asset.kind,
                             transform=asset.transform.tolist(), source_name=Path(asset.source_path).name if asset.source_path else None)
                data = io.BytesIO()
                arrays = dict(points=asset.points)
                if asset.point_ids is not None:
                    arrays['point_ids'] = np.asarray(asset.point_ids, dtype=str)
                    entry['point_scope'] = asset.point_scope
                if asset.triangles is not None:
                    arrays['triangles'] = asset.triangles
                np.savez(data, **arrays)
                content = data.getvalue()
                total += len(content)
                archive.writestr(f'{asset.id}.npz', content)
                if asset.kind == 'cad':
                    from OCP.BRepTools import BRepTools
                    data = io.BytesIO()
                    BRepTools.Write_s(asset.cad_shape, data)
                    content = data.getvalue()
                    total += len(content)
                    archive.writestr(f'{asset.id}.brep', content)
                if total > MAX_ARCHIVE_BYTES:
                    raise ValueError('工程展开后超过768 MiB，请拆分工程。')
                manifest['assets'].append(entry)
            manifest_bytes = json.dumps(manifest, ensure_ascii=False, allow_nan=False).encode('utf-8')
            if len(manifest_bytes) > MAX_MANIFEST_BYTES:
                raise ValueError('工程历史/清单超过8 MiB，请减少历史数据；原工程已保留。')
            if total + len(manifest_bytes) > MAX_ARCHIVE_BYTES:
                raise ValueError('工程展开后超过768 MiB，请拆分工程；原工程已保留。')
            archive.writestr('manifest.json', manifest_bytes)
        if os.path.getsize(temporary) > MAX_ARCHIVE_BYTES:
            raise ValueError('工程文件超过768 MiB，请拆分工程；原工程已保留。')
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_project(path):
    path = Path(path)
    if path.stat().st_size > MAX_ARCHIVE_BYTES:
        raise ValueError('工程文件过大。')
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            names = [i.filename for i in infos]
            if len(infos) > 201 or len(set(names)) != len(names) or sum(i.file_size for i in infos) > MAX_ARCHIVE_BYTES:
                raise ValueError('工程条目重复、过多或解压体积过大。')
            if any('/' in n or '\\' in n or n.startswith('.') for n in names):
                raise ValueError('工程包含非法路径。')
            if archive.getinfo('manifest.json').file_size > MAX_MANIFEST_BYTES:
                raise ValueError('工程清单过大。')
            manifest = _json(archive.read('manifest.json'))
            if manifest.get('format') != 'assembly-workbench' or manifest.get('version') not in (1, 2) or manifest.get('units') != 'mm':
                raise ValueError('不支持的工程版本或单位。')
            entries = manifest.get('assets')
            if not isinstance(entries, list) or not 1 <= len(entries) <= 100:
                raise ValueError('工程几何清单无效。')
            assets = []
            ids = set()
            for entry in entries:
                asset_id = entry['id']
                if not isinstance(asset_id, str) or len(asset_id) != 32 or any(c not in '0123456789abcdef' for c in asset_id) or asset_id in ids:
                    raise ValueError('工程几何标识无效或重复。')
                ids.add(asset_id)
                content = archive.read(f'{asset_id}.npz')
                with zipfile.ZipFile(io.BytesIO(content)) as inner:
                    if sum(i.file_size for i in inner.infolist()) > MAX_ARCHIVE_BYTES:
                        raise ValueError('数组数据展开体积过大。')
                with np.load(io.BytesIO(content), allow_pickle=False) as arrays:
                    points = arrays['points']
                    triangles = arrays['triangles'] if 'triangles' in arrays else None
                    point_ids = arrays['point_ids'].tolist() if 'point_ids' in arrays else None
                shape = None
                if entry['kind'] == 'cad':
                    from OCP.TopoDS import TopoDS_Shape
                    from OCP.BRep import BRep_Builder
                    from OCP.BRepTools import BRepTools
                    shape = TopoDS_Shape()
                    BRepTools.Read_s(shape, io.BytesIO(archive.read(f'{asset_id}.brep')), BRep_Builder())
                assets.append(Dataset(entry['name'], points, entry['kind'], triangles,
                                      entry.get('source_name'), shape, entry['transform'], asset_id,
                                      point_ids, entry.get('point_scope')))
            history = manifest.get('history', [])
            if not isinstance(history, list) or any(not isinstance(row, dict) for row in history):
                raise ValueError('工程历史记录无效。')
            return assets, history
    except (zipfile.BadZipFile, KeyError, TypeError, json.JSONDecodeError, OSError, RuntimeError) as error:
        raise ValueError(f'无法读取工程: {error}') from error


def export_report(folder, source, target, deviation, registrations=None):
    if (deviation.source_id != source.id or deviation.target_id != target.id or
            signature(source) != deviation.source_fingerprint or signature(target) != deviation.target_fingerprint):
        raise ValueError('分析结果已过期或几何不匹配，请重新计算后导出。')
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    names = ['report.json', 'deviations.csv', 'aligned.xyz', 'report.html']
    if any((folder / name).exists() for name in names):
        # Preserve imported geometry and all prior reports without an overwrite
        # dialog. Return the actual paths so callers can show the new location.
        folder = folder / f'report-{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:8]}'
        folder.mkdir(exist_ok=False)
    data = dict(schema='assembly-workbench-report/1', app_version=__version__, units='mm',
                generated_at=datetime.now(timezone.utc).isoformat(),
                source=dict(id=source.id, name=source.name, kind=source.kind, transform=source.transform.tolist()),
                target=dict(id=target.id, name=target.name, kind=target.kind, transform=target.transform.tolist()),
                deviation=deviation.to_dict(),
                registrations=[r.to_dict() if hasattr(r, 'to_dict') else r for r in (registrations or [])],
                interpretation='Unsigned geometric distance. Tolerance coverage is not a manufacturing acceptance verdict. '
                'feature_id matches labeled feature positions in the same scope; other clouds use nearest points, not a continuous surface.')
    if deviation.target_indices is not None:
        data['deviation']['feature_ids'] = [source.point_ids[i] for i in deviation.indices]
        source_keys, target_keys = set(source.point_ids), set(target.point_ids)
        data['deviation']['unmatched_source_ids'] = sorted(source_keys-target_keys)
        data['deviation']['unmatched_reference_ids'] = sorted(target_keys-source_keys)
        data['source']['point_scope'] = source.point_scope
        data['target']['point_scope'] = target.point_scope
    serialized = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)
    points = source.world_points()
    labeled = deviation.target_indices is not None
    target_points = target.world_points() if labeled else None
    paths = [folder / name for name in names]
    with paths[0].open('x', encoding='utf-8') as stream:
        stream.write(serialized)
    with paths[1].open('x', newline='', encoding='utf-8-sig') as stream:
        writer = csv.writer(stream)
        header = ['point_index', 'x_mm', 'y_mm', 'z_mm', 'unsigned_distance_mm', 'within_threshold']
        if labeled:
            header += ['feature_id', 'target_point_index', 'delta_x_mm', 'delta_y_mm', 'delta_z_mm']
        writer.writerow(header)
        for position, (index, d) in enumerate(zip(deviation.indices, deviation.distances_mm)):
            row = [int(index), *points[index], float(d), int(d <= deviation.tolerance_mm)]
            if labeled:
                target_index = int(deviation.target_indices[position])
                feature_id = source.point_ids[index]
                if feature_id.startswith(('=', '+', '-', '@', '\t', '\r')):
                    feature_id = "'" + feature_id
                row += [feature_id, target_index, *(points[index]-target_points[target_index])]
            writer.writerow(row)
    with paths[2].open('x', encoding='utf-8') as stream:
        np.savetxt(stream, points, fmt='%.9f')
    stats = deviation.statistics
    escape = html.escape
    rows = ''.join(f'<tr><th>{escape(k)}</th><td>{v:.6g}</td></tr>' for k, v in stats.items())
    counts, bins = np.histogram(deviation.distances_mm, bins=20)
    peak = max(int(counts.max()), 1)
    bars = ''.join(f'<rect x="{40+i*26}" y="{180-int(c)/peak*140:.1f}" width="22" height="{int(c)/peak*140:.1f}" fill="#0891b2"/>' for i,c in enumerate(counts))
    document = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>虚拟装配分析报告</title><style>body{{font:16px system-ui,sans-serif;max-width:1000px;margin:40px auto;padding:0 24px;color:#172b45}}h1{{color:#0e7490}}table{{border-collapse:collapse;width:100%}}th,td{{padding:10px;border-bottom:1px solid #d9e2ea;text-align:left}}pre{{white-space:pre-wrap;background:#edf2f7;padding:16px}}.muted{{color:#526779}}svg{{width:100%;max-width:620px}}</style>
<h1>虚拟装配 · 几何距离报告</h1><p>{escape(source.name)} → {escape(target.name)}</p>
<p class="muted">{escape(data['generated_at'])} · Assembly Workbench {__version__} · 单位 mm</p>
<p>方法：{escape(deviation.method)}；绝对距离阈值：{deviation.tolerance_mm:g} mm；采样 {len(deviation.indices):,} / {deviation.total_count:,} 点。</p>
<table>{rows}</table><h2>绝对距离分布</h2><svg viewBox="0 0 600 220" role="img" aria-label="距离直方图">{bars}<text x="40" y="210">{bins[0]:.4g} mm</text><text x="490" y="210">{bins[-1]:.4g} mm</text></svg>
<h2>移动件变换矩阵</h2><pre>{escape(np.array2string(source.transform, precision=9))}</pre>
<p class="muted">结果是无符号几何距离。feature_id按同一检测计划和零件的测点编号对应，CSV保留坐标差；reference_coverage为具有对应实测坐标的参考测点比例，未对应编号见JSON。其他点云采用最近点距离；网格采用三角面距离；CAD采用OCCT精确表面距离。阈值内比例仅描述本次采样结果，不代表工艺放行、干涉判断或计量认证。未测点不参与统计。</p></html>'''
    with paths[3].open('x', encoding='utf-8') as stream:
        stream.write(document)
    return paths
