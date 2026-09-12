"""Explicit coordinate controls, with missing/unjudged states kept visible."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np


def unit_vector(value, label='方向'):
    a=np.asarray(value,dtype=float)
    if a.shape!=(3,) or not np.isfinite(a).all() or np.max(np.abs(a))==0:
        raise ValueError(f'{label}必须为有限、非零的三维向量。')
    a=a/np.max(np.abs(a))
    return a/np.linalg.norm(a)


def load_emma_controls(path, scope, unit='mm'):
    """Import explicit nominal XYZ deviation bounds in the reference's frame.

    Empty bounds are absent, not zero. Other characteristic/GD&T columns have
    different semantics and are deliberately not interpreted here.
    """
    from .emma import _csv_rows
    from .core import MAX_POINTS
    path = Path(path)
    if unit not in ('mm', 'm'):
        raise ValueError('导入单位仅支持 mm 或 m。')
    if not path.is_file() or not 0 < path.stat().st_size <= 512 * 1024 * 1024:
        raise ValueError('控制文件不存在、为空或超过512 MiB。')
    if not isinstance(scope, str) or not scope:
        raise ValueError('请选择带检测范围的名义资源。')
    scale = 1000. if unit == 'm' else 1.
    identity = ('InspectionTask', 'InspectionPlan', 'PartSingle', 'IPE.Name',
                'IPE.Type', 'InspectionCategories', 'Component.ID')
    bound_keys = [f'TolAxisPos{axis}.{bound}ToleranceValue'
                  for axis in 'XYZ' for bound in ('Lower', 'Upper')]
    controls = {}; counts = Counter(total_rows=0, nominal_rows=0,
                                    empty_axis_limits=0, duplicate_controls=0)
    with path.open(encoding='utf-8-sig', newline='') as stream:
        records = _csv_rows(stream)
        header = [s.strip() for s in next(records, [])]
        if len(set(header)) != len(header) or any(k not in header for k in (*identity, *bound_keys)):
            raise ValueError('eMMA坐标公差表头缺失、重复或不完整。')
        for line, values in enumerate(records, 2):
            if not any(v.strip() for v in values):
                continue
            if len(values) != len(header):
                raise ValueError(f'eMMA第{line}条CSV记录列数错误，文件可能截断。')
            counts['total_rows'] += 1
            if counts['total_rows'] > MAX_POINTS:
                raise ValueError('eMMA记录数超过当前导入上限。')
            row = dict(zip(header, (v.strip() for v in values)))
            row_scope = json.dumps([row[k] for k in identity[:3]], ensure_ascii=False, separators=(',', ':'))
            if row['InspectionCategories'] != 'MPT' or row['Component.ID'] or row_scope != scope:
                continue
            counts['nominal_rows'] += 1
            if not row['IPE.Type'] or not row['IPE.Name']:
                raise ValueError(f'eMMA第{line}条记录缺少测点类型或名称。')
            pid = row['IPE.Type'] + ':' + row['IPE.Name']
            for index, axis in enumerate('XYZ'):
                raw = [row[f'TolAxisPos{axis}.{b}ToleranceValue'] for b in ('Lower', 'Upper')]
                if not any(raw):
                    counts['empty_axis_limits'] += 1
                    continue
                try:
                    lo, hi = [float(v) * scale if v else None for v in raw]
                    if any(v is not None and not np.isfinite(v) for v in (lo, hi)) or (lo is not None and hi is not None and lo > hi):
                        raise ValueError()
                except ValueError as error:
                    raise ValueError(f'eMMA第{line}条记录的{axis}坐标公差无效。') from error
                name = f'{pid} / {axis}'
                c = dict(name=name, point_id=pid, axis=np.eye(3)[index].tolist(), lower_mm=lo, upper_mm=hi)
                if name in controls:
                    if controls[name] != c:
                        raise ValueError(f'同一测点的{axis}坐标公差存在冲突。')
                    counts['duplicate_controls'] += 1
                else:
                    controls[name] = c
                if len(controls) > 50000:
                    raise ValueError('单个方案超过50000项控制，请拆分检测计划。')
    if not controls:
        raise ValueError('选定检测范围中没有明确的名义XYZ坐标公差。')
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(schema='emma-axis-controls/1', controls=list(controls.values()),
                scope=scope, axis_frame='reference_local', input_unit=unit, units='mm',
                source_sha256=digest, statistics=dict(counts),
                interpretation='Nominal TolAxisPos X/Y/Z deviation bounds only; empty axes excluded; no GD&T inference.')


def evaluate_points(source, target, controls, *, axis_frame='world'):
    """Screen signed world-coordinate deltas against explicit deviation limits.

    Bounds refer to the deviation, not absolute nominal coordinates. The caller
    chooses the coordinate system and axes; this does not establish a GD&T DRF.
    """
    source.validate_labels();target.validate_labels()
    if axis_frame not in ('world', 'reference_local'):
        raise ValueError('控制方向坐标系必须为世界坐标或名义资源局部坐标。')
    if source.point_ids is None or target.point_ids is None or source.point_scope!=target.point_scope:
        raise ValueError('坐标公差筛查需要同一检测范围内、带编号的实测和名义数据。')
    if not isinstance(controls,list) or not 1<=len(controls)<=50000:
        raise ValueError('请定义1–50000项测点控制。')
    src=dict(zip(source.point_ids,source.world_points()))
    ref=dict(zip(target.point_ids,target.world_points()))
    rows=[];names=set()
    for c in controls:
        if not isinstance(c,dict) or set(c)-{'name','point_id','axis','lower_mm','upper_mm'}:
            raise ValueError('测点控制字段无效。')
        name=c.get('name');pid=c.get('point_id')
        if not isinstance(name,str) or not name.strip() or len(name)>1024 or name in names:
            raise ValueError('每项控制需要唯一、非空的名称。')
        if not isinstance(pid,str) or not pid or len(pid)>512:
            raise ValueError('测点编号无效。')
        names.add(name);axis=unit_vector(c.get('axis'),'控制方向')
        if axis_frame == 'reference_local':
            axis = target.transform[:3, :3] @ axis
        lo,hi=c.get('lower_mm'),c.get('upper_mm')
        for value in (lo,hi):
            if value is not None and (isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value)):
                raise ValueError('上下限必须为有限数值或留空。')
        if lo is not None and hi is not None and lo>hi:
            raise ValueError('控制下限不能大于上限。')
        value=None;status='missing'
        missing=[]
        if pid not in src:missing.append('source')
        if pid not in ref:missing.append('reference')
        if not missing:
            value=float((src[pid]-ref[pid])@axis)
            if not np.isfinite(value):raise ValueError('坐标差超出有限数值范围，请检查单位与坐标。')
            if lo is None and hi is None:status='unjudged'
            else:status='pass' if (lo is None or value>=lo) and (hi is None or value<=hi) else 'fail'
        rows.append(dict(name=name,point_id=pid,axis=axis.tolist(),value_mm=value,
                         lower_mm=lo,upper_mm=hi,status=status,missing=missing))
    counts=Counter(x['status'] for x in rows)
    overall='incomplete' if counts['missing'] else 'fail' if counts['fail'] else 'unjudged' if counts['unjudged'] else 'pass'
    return dict(method='point_directional_limits',units='mm',scope=source.point_scope,axis_frame=axis_frame,
                rows=rows,overall=overall,
                statistics=dict(total_count=len(rows),pass_count=counts['pass'],fail_count=counts['fail'],
                                missing_count=counts['missing'],unjudged_count=counts['unjudged'],
                                coverage=(len(rows)-counts['missing'])/len(rows)),
                interpretation='Explicit signed coordinate-deviation screening. Not a GD&T or manufacturing release verdict.')
