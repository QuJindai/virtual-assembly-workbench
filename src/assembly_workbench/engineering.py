"""Replayable dimensional-engineering operations shared by desktop and CLI."""
from __future__ import annotations

from datetime import datetime,timezone
import csv
import hashlib
import html
import io
import json
from pathlib import Path
import re
import tempfile
import time

import numpy as np

from . import __version__
from .core import Dataset,signature,validate_transform
from .inspection import evaluate_points,unit_vector


TOOLS={
    'datum321':{'source_points','target_points'},
    'rps':{'source_points','target_points','normals','weights','dofs','pivot','translation_limit_mm','rotation_limit_deg'},
    'adjustment':{'source_points','target_points','normals','weights','dofs','pivot','translation_limit_mm','rotation_limit_deg','offsets_mm'},
    'feature':{'source_points','target_points','kind','compare'},
    'detect':{'source_points','kind','threshold_mm','min_points','max_features','seed'},
    'gap_flush':{'source_points','target_points','gap_axis','flush_axis'},
    'section':{'origin','normal','slab_mm','max_points'},
    'contact':{'tolerance_mm'},
    'inspection':{'controls','axis_frame'},
}
LABELS={'datum321':'3-2-1基准建系','rps':'RPS约束定位','adjustment':'刚体装调建议',
        'feature':'几何特征拟合','detect':'几何候选识别','gap_flush':'间隙与面差',
        'section':'截面提取','contact':'CAD实体干涉','inspection':'测点坐标公差'}


def fingerprint(asset):
    h=hashlib.sha256((signature(asset)+asset.kind).encode())
    if asset.kind=='cad':
        from OCP.BRepTools import BRepTools
        buf=io.BytesIO();BRepTools.Write_s(asset.cad_shape,buf);h.update(buf.getvalue())
    return h.hexdigest()


def _json_copy(value):
    try:
        return json.loads(json.dumps(value,ensure_ascii=False,allow_nan=False))
    except (TypeError,ValueError,OverflowError) as e:
        raise ValueError('方案与结果必须为有限数值的JSON结构。') from e


def _digest(receipt):
    data={k:v for k,v in receipt.items() if k!='receipt_sha256'}
    return hashlib.sha256(json.dumps(data,sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode()).hexdigest()


def select_points(asset, selection=None):
    points=asset.world_points()
    labels=asset.point_ids or [f'#{i+1}' for i in range(len(points))]
    index={label:i for i,label in enumerate(labels)}
    if selection is None:
        selected=list(range(len(points)))
    elif isinstance(selection,dict):
        if set(selection)!={'prefix'} or not isinstance(selection['prefix'],str) or not selection['prefix']:
            raise ValueError('选点前缀必须非空。')
        selected=[i for i,label in enumerate(labels) if label.startswith(selection['prefix'])]
    elif isinstance(selection,list):
        selected=[]
        for value in selection:
            if not isinstance(value,str):raise ValueError('选点请使用测点编号或#1起算的行号。')
            if value in index:i=index[value]
            elif re.fullmatch(r'#[1-9][0-9]*',value):i=int(value[1:])-1
            else:raise ValueError(f'找不到测点：{value[:80]}')
            if not 0<=i<len(points):raise ValueError('测点行号超出范围。')
            selected.append(i)
    else:raise ValueError('选点必须为编号列表或前缀。')
    if not selected or len(set(selected))!=len(selected):
        raise ValueError('选点为空或包含重复测点。')
    return points[selected], [labels[i] for i in selected]


def _paired(source,target,recipe):
    if not recipe.get('source_points') or not recipe.get('target_points'):
        raise ValueError('请明确选择两侧对应测点。')
    sp,sids=select_points(source,recipe['source_points'])
    tp,tids=select_points(target,recipe['target_points'])
    if len(sp)!=len(tp):raise ValueError('两侧对应测点数量必须相等。')
    return sp,tp,dict(source=sids,reference=tids)


def run_engineering(source,target,recipe):
    recipe=_json_copy(recipe)
    if not isinstance(recipe,dict) or recipe.get('tool') not in TOOLS:
        raise ValueError('不支持的尺寸工程工具。')
    tool=recipe['tool']
    if set(recipe)-TOOLS[tool]-{'tool','name'}:
        raise ValueError('方案包含该工具不支持的字段。')
    if source is None or target is None or source.id==target.id:
        raise ValueError('请选择不同的移动件和参考件。')
    source.world_points();target.world_points()
    started=time.perf_counter();selected={}
    source_hash,target_hash=fingerprint(source),fingerprint(target)
    if tool in ('datum321','rps','adjustment','gap_flush'):
        sp,tp,selected=_paired(source,target,recipe)
        if tool=='datum321':
            from .datums import align_321
            result=align_321(sp,tp)
        elif tool in ('rps','adjustment'):
            from .datums import align_rps
            normals=np.asarray(recipe.get('normals'),dtype=float)
            if normals.shape!=sp.shape:raise ValueError('每对测点必须定义一个约束方向。')
            if tool=='adjustment':
                offsets=np.asarray(recipe.get('offsets_mm',[0.]*len(sp)),dtype=float)
                if offsets.shape!=(len(sp),) or not np.isfinite(offsets).all():raise ValueError('装调目标偏置数量或数值无效。')
                normals=np.array([unit_vector(n) for n in normals])
                tp=tp+normals*offsets[:,None]
            kwargs={k:recipe[k] for k in ('weights','dofs','pivot','translation_limit_mm','rotation_limit_deg') if k in recipe}
            result=align_rps(sp,tp,normals,**kwargs)
            if tool=='adjustment':result['target_offsets_mm']=offsets.tolist()
        else:
            from .assembly import gap_flush
            result=gap_flush(sp,tp,recipe.get('gap_axis'),recipe.get('flush_axis'))
    elif tool in ('feature','detect'):
        from .features import fit_feature,detect_features
        sp,ids=select_points(source,recipe.get('source_points'));selected={'source':ids}
        if tool=='detect':
            args={k:recipe[k] for k in ('kind','threshold_mm','min_points','max_features','seed') if k in recipe}
            candidates=detect_features(sp,**args)
            result={'method':'geometric_candidates','candidates':candidates,'count':len(candidates)}
            for c in candidates:c['point_ids']=[ids[i] for i in c['indices']]
        else:
            result=fit_feature(sp,recipe.get('kind','plane'))
            if recipe.get('compare',False):
                tp,tids=select_points(target,recipe.get('target_points'));selected['reference']=tids
                nominal=fit_feature(tp,recipe.get('kind','plane'))
                delta=np.asarray(result['center'])-nominal['center']
                comparison={'reference':nominal,'interpretation':'Geometric fit comparison; not a GD&T position or orientation tolerance evaluation.'}
                kind=result['kind']
                if kind in ('circle','sphere'):
                    comparison.update(center_delta_mm=delta.tolist(),center_distance_mm=float(np.linalg.norm(delta)))
                elif kind=='plane':
                    comparison['normal_offset_at_source_center_mm']=float(delta@nominal['direction'])
                else:
                    # Line/cylinder centers are arbitrary along their axes. The
                    # shortest infinite-axis distance is independent of support.
                    axis=np.asarray(result['direction']);ref_axis=np.asarray(nominal['direction'])
                    cross=np.cross(axis,ref_axis);norm=np.linalg.norm(cross)
                    distance=abs(delta@cross)/norm if norm>1e-10 else np.linalg.norm(np.cross(delta,ref_axis))
                    comparison['infinite_axis_distance_mm']=float(distance)
                if 'radius_mm' in result:comparison['diameter_delta_mm']=2*(result['radius_mm']-nominal['radius_mm'])
                if 'direction' in result:
                    cosine=np.clip(abs(np.dot(result['direction'],nominal['direction'])),0.,1.)
                    comparison['axis_angle_deg']=float(np.degrees(np.arccos(cosine)))
                result['comparison']=comparison
    elif tool=='inspection':result=evaluate_points(source,target,recipe.get('controls'),axis_frame=recipe.get('axis_frame','world'))
    elif tool=='section':
        from .assembly import section_geometry
        kwargs={k:recipe[k] for k in ('slab_mm','max_points') if k in recipe}
        result=section_geometry(source,recipe.get('origin'),recipe.get('normal'),**kwargs)
    else:
        from .assembly import cad_contact
        result=cad_contact(source,target,tolerance_mm=recipe.get('tolerance_mm',1e-5))
    receipt=dict(schema='assembly-engineering/1',version=__version__,tool=tool,
                 source_id=source.id,target_id=target.id,source_name=source.name,target_name=target.name,
                 source_fingerprint=source_hash,target_fingerprint=target_hash,
                 source_transform=source.transform.tolist(),target_transform=target.transform.tolist(),
                 units='mm',created_at=datetime.now(timezone.utc).isoformat(),
                 recipe=recipe,selections=selected,result=result,elapsed_s=time.perf_counter()-started)
    receipt=_json_copy(receipt);receipt['receipt_sha256']=_digest(receipt)
    return receipt


def validate_receipt(source,target,receipt):
    if (not isinstance(receipt,dict) or receipt.get('schema')!='assembly-engineering/1' or
            receipt.get('receipt_sha256')!=_digest(receipt)):
        raise ValueError('计算记录结构或校验值无效，请重新计算。')
    for side,asset in (('source',source),('target',target)):
        if receipt.get(side+'_id')!=asset.id or receipt.get(side+'_fingerprint')!=fingerprint(asset):
            raise ValueError('几何、坐标或测点编号已改变；请重新计算，不能应用或导出旧结果。')


def apply_engineering(source,target,receipt):
    validate_receipt(source,target,receipt)
    if receipt['tool'] not in ('datum321','rps','adjustment') or receipt['result'].get('converged') is not True:
        raise ValueError('只有已收敛的基准/装调结果可以应用。')
    delta=validate_transform(receipt['result'].get('transform'))
    source.transform=validate_transform(delta@source.transform)


def result_rows(receipt):
    """Flatten useful result values for the human table and CSV without data loss in JSON."""
    result=receipt['result'];tool=receipt['tool']
    if tool=='inspection':return result['rows']
    if tool=='gap_flush':
        return [dict(index=i+1,gap_mm=g,flush_mm=f,seam_offset_mm=d) for i,(g,f,d) in enumerate(zip(result['gap_mm'],result['flush_mm'],result['seam_offset_mm'],strict=True))]
    if tool=='detect':
        return [{k:v for k,v in x.items() if k not in ('indices','point_ids')} for x in result['candidates']]
    if tool=='section':
        return [dict(index=i+1,x_mm=p[0],y_mm=p[1],z_mm=p[2]) for i,p in enumerate(result['points'])]
    if tool in ('datum321','rps','adjustment'):
        before=result.get('before_residuals_mm',[])
        rows=[dict(index=i+1,before_mm=before[i] if i<len(before) else None,after_mm=x) for i,x in enumerate(result['residuals_mm'])]
        return rows
    return [result]


def export_engineering(folder,source,target,receipt):
    validate_receipt(source,target,receipt)
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    destination=Path(tempfile.mkdtemp(prefix='engineering-',dir=folder))
    json_file=destination/'engineering.json';html_file=destination/'engineering.html';csv_file=destination/'engineering.csv'
    rows=result_rows(receipt)
    columns=list(dict.fromkeys(k for row in rows for k in row))
    def display(value):
        return json.dumps(value,ensure_ascii=False,allow_nan=False) if isinstance(value,(list,dict)) else '' if value is None else str(value)
    def safe_cell(value):
        text=display(value)
        return "'"+text if isinstance(value,str) and text.startswith(('=','+','-','@','\t','\r')) else text
    json_file.write_text(json.dumps(receipt,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    with csv_file.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(columns)
        for row in rows:writer.writerow([safe_cell(row.get(c)) for c in columns])
    table='<tr>'+''.join('<th>'+html.escape(k)+'</th>' for k in columns)+'</tr>'
    table+=''.join('<tr>'+''.join('<td>'+html.escape(display(row.get(k)))+'</td>' for k in columns)+'</tr>' for row in rows)
    title=LABELS[receipt['tool']]
    summary={k:v for k,v in receipt['result'].items() if k not in ('rows','points','candidates')}
    html_file.write_text(f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title>
<style>body{{font:15px/1.6 system-ui,sans-serif;max-width:1200px;margin:auto;padding:28px;color:#163344}}h1{{color:#087e91}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #cbd8df;padding:8px;text-align:left;overflow-wrap:anywhere}}th{{background:#e8f2f6}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f2f6f8;padding:14px}}.scroll{{overflow:auto}}</style>
<h1>{title}</h1><p>{html.escape(source.name)} → {html.escape(target.name)} · 单位 mm · v{html.escape(receipt['version'])}</p>
<p>计算时间：{html.escape(receipt['created_at'])}。本报告是所列几何、坐标与方案的计算快照。坐标筛查、几何拟合和刚体装调不构成完整GD&T或工艺放行结论。</p>
<h2>结果</h2><pre>{html.escape(json.dumps(summary,ensure_ascii=False,indent=2))}</pre><div class="scroll"><table>{table}</table></div>
<h2>可复现方案</h2><pre>{html.escape(json.dumps(receipt['recipe'],ensure_ascii=False,indent=2))}</pre>
<p>计算记录SHA-256：{receipt['receipt_sha256']}。完整坐标选择、几何指纹及结果保存在同目录JSON中。</p></html>''',encoding='utf-8')
    return [html_file,json_file,csv_file]
