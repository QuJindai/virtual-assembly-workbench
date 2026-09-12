"""Read eMMA coordinate exports without mixing nominal values and specimens.

Only complete XYZ records become geometry. Original characteristic tolerances
and calculated dimensions are deliberately not interpreted as GD&T verdicts.
"""
from __future__ import annotations

from collections import Counter
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from .core import Dataset, MAX_POINTS

REQUIRED = ('InspectionTask', 'InspectionPlan', 'PartSingle', 'IPE.Name', 'IPE.Type',
            'IPE.Origin.X', 'IPE.Origin.Y', 'IPE.Origin.Z', 'InspectionCategories',
            'Component.ID', 'History.DateTime')


@dataclass
class EmmaImport:
    assets: list[Dataset]
    summary: dict


def _csv_rows(stream):
    reader = csv.reader(stream, skipinitialspace=True, strict=True)
    try:
        yield from reader
    except csv.Error as error:
        raise ValueError(f'第{reader.line_num}行CSV引号或分隔结构无效，文件可能截断：{error}') from error


def is_emma_csv(path):
    path = Path(path)
    if path.suffix.lower() != '.csv':
        return False
    with path.open(encoding='utf-8-sig', newline='') as stream:
        header = [s.strip() for s in next(_csv_rows(stream), [])]
    return 'IPE.Name' in header or 'IPE.Origin.X' in header


def load_emma(path, unit='mm'):
    path = Path(path)
    if unit not in ('mm', 'm'):
        raise ValueError('导入单位仅支持 mm 或 m。')
    if not path.is_file() or not 0 < path.stat().st_size <= 512 * 1024 * 1024:
        raise ValueError('文件不存在、为空或超过512 MiB。')
    scale = 1000. if unit == 'm' else 1.
    counts = Counter(total_rows=0, nominal_rows=0, actual_rows=0,
                     inferred_actual_rows=0, missing_xyz_rows=0, invalid_xyz_rows=0,
                     unknown_role_rows=0, conflicting_keys=0, duplicate_rows=0)
    groups, conflicts, issues = {}, set(), []

    def issue(line, reason):
        # Counts remain complete; bounded examples keep project history compact.
        if len(issues) < 100:
            issues.append({'line': line, 'reason': reason})

    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.reader(stream, skipinitialspace=True, strict=True)
        def checked_rows():
            try:
                yield from reader
            except csv.Error as error:
                raise ValueError(f'eMMA第{reader.line_num}行CSV引号或分隔结构无效，文件可能截断：{error}') from error
        records = checked_rows()
        header = [s.strip() for s in next(records, [])]
        if any(k not in header for k in REQUIRED) or len(set(header)) != len(header):
            raise ValueError('eMMA表头缺失、重复或不完整；请重新导出完整的CSV文件。')
        for values in records:
            line = reader.line_num
            if not any(v.strip() for v in values):
                continue
            if len(values) != len(header):
                raise ValueError(f'eMMA第{line}行列数为{len(values)}，表头为{len(header)}；文件可能截断，请重新导出。')
            counts['total_rows'] += 1
            if counts['total_rows'] > MAX_POINTS:
                raise ValueError('eMMA记录数超过当前导入上限，请分批导出。')
            row = dict(zip(header, (v.strip() for v in values)))
            category, component, timestamp = (row[k] for k in
                ('InspectionCategories', 'Component.ID', 'History.DateTime'))
            if category == 'MPT' and not component:
                role = 'nominal'
            elif category in ('A', '') and component and timestamp:
                role = 'actual'
                if not category:
                    counts['inferred_actual_rows'] += 1
            else:
                counts['unknown_role_rows'] += 1
                issue(line, '记录角色不明确，未导入')
                continue
            counts[role + '_rows'] += 1
            xyz = [row['IPE.Origin.'+a] for a in 'XYZ']
            if not all(xyz):
                counts['missing_xyz_rows'] += 1
                issue(line, 'XYZ不完整；可能为缺测或无坐标的计算特征，未补零')
                continue
            try:
                point = np.asarray(xyz, dtype=float) * scale
                if not np.isfinite(point).all():
                    raise ValueError()
            except ValueError:
                counts['invalid_xyz_rows'] += 1
                issue(line, 'XYZ非数值或非有限值，未导入')
                continue
            if not row['IPE.Name'] or not row['IPE.Type'] or not row['InspectionPlan']:
                counts['invalid_xyz_rows'] += 1
                issue(line, '测点名称、类型或检测计划缺失，未导入')
                continue
            scope_values = [row[k] for k in ('InspectionTask', 'InspectionPlan', 'PartSingle')]
            scope = json.dumps(scope_values, ensure_ascii=False, separators=(',', ':'))
            group_key = (scope, role, component if role == 'actual' else '',
                         timestamp if role == 'actual' else '')
            if group_key not in groups:
                if len(groups) >= 100:
                    raise ValueError('eMMA包含超过100个独立样本/计划，请分批导出。')
                groups[group_key] = {}
            point_id = row['IPE.Type'] + ':' + row['IPE.Name']
            key = (group_key, point_id)
            if key in conflicts:
                continue
            records = groups[group_key]
            if point_id in records:
                if np.array_equal(records[point_id], point):
                    counts['duplicate_rows'] += 1
                else:
                    counts['conflicting_keys'] += 1
                    conflicts.add(key)
                    del records[point_id]
                    issue(line, '相同测点编号存在不同坐标；该编号全部隔离')
            else:
                records[point_id] = point

    assets, descriptions = [], []
    # One measured sample followed by its nominal reference makes the common
    # single-sample file ready for the existing source/target selectors.
    for scope in dict.fromkeys(k[0] for k in groups):
        actual_keys = [k for k in groups if k[0] == scope and k[1] == 'actual']
        nominal_keys = [k for k in groups if k[0] == scope and k[1] == 'nominal']
        keys = actual_keys[:1] + nominal_keys + actual_keys[1:]
        for key in keys:
            records = groups[key]
            if not records:
                continue
            _, role, component, timestamp = key
            plan = json.loads(scope)[1]
            name = f'{component} · 实测 · {timestamp}' if role == 'actual' else f'名义 · {plan}'
            if len(name) > 512:
                name = name[:495] + '…' + hashlib.sha256(name.encode()).hexdigest()[:12]
            asset = Dataset(name, list(records.values()), source_path=str(path.resolve()),
                            point_ids=list(records), point_scope=scope)
            assets.append(asset)
            descriptions.append({'asset_id': asset.id, 'role': role, 'component': component,
                                 'timestamp': timestamp, 'point_count': len(asset.points)})
    if not assets:
        raise ValueError('eMMA没有可导入的完整XYZ坐标；缺失值不会自动补零。')
    summary = dict(counts)
    summary.update(schema='emma-import/1', source_file=path.name, input_unit=unit,
                   units='mm', nominal_assets=sum(d['role']=='nominal' for d in descriptions),
                   actual_assets=sum(d['role']=='actual' for d in descriptions),
                   groups=descriptions, issue_examples=issues,
                   interpretation='MPT=nominal; A with component and timestamp=actual. '
                   'Blank category with component and timestamp is explicitly inferred actual. '
                   'Coordinates are feature positions, not dense scan surfaces. '
                   'Calculated characteristics and GD&T are not evaluated.')
    return EmmaImport(assets, summary)
