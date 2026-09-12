"""Replay a project recipe or privately validate explicit eMMA coordinate controls.

Outputs contain input names, coordinates and limits. Keep them outside public git.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from assembly_workbench import __version__
from assembly_workbench.emma import load_emma
from assembly_workbench.engineering import run_engineering,export_engineering,fingerprint
from assembly_workbench.inspection import load_emma_controls
from assembly_workbench.io import load_project,save_project


def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def write_json(path,data):
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def validate_folder(folder,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    files=[]
    for index,path in enumerate(sorted(Path(folder).glob('*.csv')),1):
        started=time.perf_counter();original=digest(path)
        record=dict(file=path.name,source_sha256=original,status='pending',comparisons=[])
        dest=output/f'case-{index:02d}';dest.mkdir()
        try:
            imported=load_emma(path)
        except ValueError as error:
            record.update(status='rejected',reason=str(error))
        else:
            descriptions={g['asset_id']:g for g in imported.summary['groups']}
            references={a.point_scope:a for a in imported.assets if descriptions[a.id]['role']=='nominal'}
            history=[dict(type='import',result=imported.summary)]
            independent={}
            with path.open(encoding='utf-8-sig',newline='') as f:
                for row in csv.DictReader(f,skipinitialspace=True,strict=True):
                    row={k:v.strip() for k,v in row.items()}
                    if row['InspectionCategories'] not in ('A','') or not row['Component.ID']:continue
                    scope=json.dumps([row[k] for k in ('InspectionTask','InspectionPlan','PartSingle')],ensure_ascii=False,separators=(',',':'))
                    key=(scope,row['Component.ID'],row['History.DateTime'],row['IPE.Type']+':'+row['IPE.Name'])
                    for axis in 'XYZ':
                        try:value=float(row.get('QC.DeviationPos.'+axis,''))
                        except ValueError:continue
                        if np.isfinite(value):independent.setdefault((*key,axis),[]).append(value)
            for source in imported.assets:
                detail=descriptions[source.id]
                if detail['role']!='actual':continue
                target=references.get(source.point_scope)
                if target is None:
                    record['comparisons'].append(dict(status='no_reference'));continue
                controls=load_emma_controls(path,target.point_scope)
                recipe=dict(tool='inspection',name='Original eMMA XYZ deviation bounds',
                            controls=controls['controls'],axis_frame='reference_local')
                before=[fingerprint(a) for a in (source,target)]
                receipt=run_engineering(source,target,recipe)
                checked=0;max_error=0.
                for row in receipt['result']['rows']:
                    if row['value_mm'] is None:continue
                    axis='XYZ'[int(np.argmax(row['axis']))]
                    key=(source.point_scope,detail['component'],detail['timestamp'],row['point_id'],axis)
                    for value in independent.get(key,[]):
                        error=abs(value-row['value_mm'])
                        if error>1e-6:raise RuntimeError('Exported QC deviation disagrees with current-coordinate evaluation')
                        max_error=max(max_error,error);checked+=1
                paths=export_engineering(dest,source,target,receipt)
                csv_path=next(p for p in paths if p.suffix=='.csv')
                with csv_path.open(encoding='utf-8-sig',newline='') as f:
                    exported=list(csv.DictReader(f))
                if len(exported)!=len(recipe['controls']):raise RuntimeError('Report export lost controls')
                if [fingerprint(a) for a in (source,target)]!=before:raise RuntimeError('Validation mutated source geometry')
                history.extend([dict(type='controls_import',result={k:v for k,v in controls.items() if k!='controls'}),dict(type='engineering',result=receipt)])
                record['comparisons'].append(dict(status='screened',source=source.name,target=target.name,
                    statistics=receipt['result']['statistics'],overall=receipt['result']['overall'],
                    original_coordinate_system=True,independent_qc_checks=checked,max_qc_error_mm=max_error,
                    controls_import={k:v for k,v in controls.items() if k!='controls'},
                    report=str(paths[0].relative_to(output))))
            project=dest/'engineering.vaw';save_project(project,imported.assets,history)
            restored,saved=load_project(project)
            if [fingerprint(a) for a in restored]!=[fingerprint(a) for a in imported.assets] or saved!=history:raise RuntimeError('Private project roundtrip failed')
            by_id={a.id:a for a in restored}
            replayed=0
            for row in saved:
                if row['type']!='engineering':continue
                receipt=row['result'];replay=run_engineering(by_id[receipt['source_id']],by_id[receipt['target_id']],receipt['recipe'])
                if replay['result']!=receipt['result']:raise RuntimeError('Private project replay failed')
                replayed+=1
            screened=sum(c['status']=='screened' for c in record['comparisons'])
            status=('screened' if screened==len(record['comparisons']) else 'partially_screened') if screened else 'no_reference' if imported.summary['actual_assets'] else 'nominal_only'
            record.update(status=status,project_roundtrip=True,recipe_replay=replayed>0,
                          replayed_recipes=replayed,screened_comparisons=screened)
        if digest(path)!=original:raise RuntimeError('Original file checksum changed')
        record.update(raw_file_unchanged=True,elapsed_s=time.perf_counter()-started)
        files.append(record);write_json(dest/'validation.json',record)
        print(json.dumps(dict(case=index,status=record['status'],elapsed_s=round(record['elapsed_s'],2))),flush=True)
    result=dict(schema='engineering-private-validation/1',app_version=__version__,generated_at=datetime.now(timezone.utc).isoformat(),files=files,
                interpretation='Explicit nominal eMMA XYZ deviation bounds in original reference coordinates. '
                'Empty bounds excluded, missing features retained, no automatic alignment. '
                'Not a full GD&T or manufacturing release verdict; sparse feature positions cannot validate hole-edge fitting or CAD contact.')
    write_json(output/'validation.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--emma-folder',type=Path)
    mode.add_argument('--project',type=Path)
    parser.add_argument('--recipe',type=Path)
    parser.add_argument('--source',type=int,default=1,help='One-based asset index')
    parser.add_argument('--target',type=int,default=2,help='One-based asset index')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.emma_folder:return validate_folder(args.emma_folder,args.output)
    if not args.recipe:parser.error('--project requires --recipe')
    if args.recipe.stat().st_size>16*1024*1024:parser.error('Recipe exceeds 16 MiB')
    assets,history=load_project(args.project)
    if not 1<=args.source<=len(assets) or not 1<=args.target<=len(assets):parser.error('Asset index out of bounds')
    source,target=assets[args.source-1],assets[args.target-1]
    raw=json.loads(args.recipe.read_text(encoding='utf-8-sig'));recipe=raw.get('recipe',raw)
    result=run_engineering(source,target,recipe)
    args.output.mkdir(parents=True,exist_ok=False)
    export_engineering(args.output,source,target,result)
    save_project(args.output/'replayed.vaw',assets,history+[dict(type='engineering',result=result)])
    print(json.dumps(dict(tool=result['tool'],receipt_sha256=result['receipt_sha256'])))


if __name__=='__main__':main()
