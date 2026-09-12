"""Private-data validation runner. Outputs contain user data; never commit them.

Usage: python scripts/validate_emma_folder.py INPUT_FOLDER --output OUTPUT_FOLDER
"""
from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from assembly_workbench import __version__
from assembly_workbench.core import apply_registration, measure_deviation, register, signature
from assembly_workbench.emma import load_emma
from assembly_workbench.io import export_report, load_project, save_project


def run(folder, output):
    folder, output = Path(folder), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    receipts = []
    for path in sorted(folder.glob('*.csv')):
        start = time.perf_counter()
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        receipt = {'file': path.name, 'source_sha256': before, 'size_bytes': path.stat().st_size}
        destination = output / path.stem
        destination.mkdir()
        try:
            batch = load_emma(path)
            receipt['import'] = batch.summary
        except ValueError as error:
            receipt.update(status='rejected', reason=str(error))
            assert hashlib.sha256(path.read_bytes()).hexdigest() == before
            receipt['raw_file_unchanged'] = True
            receipts.append(receipt)
            print(json.dumps({'file': path.name, 'status': 'rejected', 'reason': str(error)}, ensure_ascii=False), flush=True)
            continue

        history = [{'type': '导入', 'source': path.name, 'target': '',
                    'summary': 'eMMA原始坐标，未应用配准；详情见随包测试回执。',
                    'result': batch.summary}]
        project = destination/'original-coordinates.vaw'
        save_project(project, batch.assets, history)
        restored, restored_history = load_project(project)
        assert len(restored) == len(batch.assets)
        for old, new in zip(batch.assets, restored):
            np.testing.assert_array_equal(old.points, new.points)
            np.testing.assert_array_equal(old.transform, new.transform)
            assert old.point_ids == new.point_ids and old.point_scope == new.point_scope
        assert restored_history == history
        receipt['project_roundtrip'] = True
        receipt['comparisons'] = []
        nominal_ids = {g['asset_id'] for g in batch.summary['groups'] if g['role']=='nominal'}
        references = {a.point_scope:a for a in restored if a.id in nominal_ids}
        for source in (a for a in restored if a.id not in nominal_ids):
            target = references.get(source.point_scope)
            if target is None:
                receipt['comparisons'].append({'source':source.name,'status':'no_nominal_reference'})
                continue
            if not set(source.point_ids).intersection(target.point_ids):
                receipt['comparisons'].append({'source':source.name,'status':'no_matching_feature_ids'})
                continue
            deviation = measure_deviation(source, target, tolerance_mm=1., max_samples=len(source.points))
            assert deviation.method == 'feature_id'
            expected = np.linalg.norm(source.points[deviation.indices]-target.points[deviation.target_indices],axis=1)
            np.testing.assert_allclose(deviation.distances_mm, expected, rtol=1e-12, atol=1e-12)
            assert all(source.point_ids[i] == target.point_ids[j] for i,j in zip(deviation.indices,deviation.target_indices))
            comparison = {'source':source.name, 'target':target.name, 'status':'compared',
                          'statistics':deviation.statistics, 'reference_points':len(target.points),
                          'reference_coverage':len(deviation.indices)/len(target.points),
                          'original_coordinate_system':True, 'registration':[]}
            paths = export_report(destination/'measurement', source, target, deviation)
            with paths[1].open(encoding='utf-8-sig') as stream:
                lines = list(csv.DictReader(stream))
            assert len(lines) == len(deviation.indices)
            assert lines[0]['feature_id'].lstrip("'") == source.point_ids[deviation.indices[0]]
            receipt['report_export_verified'] = True
            original_signature = signature(source)
            # Registration runs only on a copy; the delivered project/report
            # keep the source coordinate system and unmodified measured values.
            for method in ('gicp', 'icp'):
                trial = copy.deepcopy(source)
                try:
                    result = register(trial, target, method=method, voxel_mm=2., max_distance_mm=20.,
                                      threads=2, max_iterations=60)
                    assert signature(trial) == original_signature
                    detail = {'method':method,'converged':result.converged,'fitness':result.fitness,
                              'rmse_mm':result.rmse_mm,'elapsed_s':result.elapsed_s,
                              'transform':result.transform.tolist()}
                    if result.converged:
                        apply_registration(trial,result)
                        after = measure_deviation(trial,target,max_samples=len(trial.points))
                        detail['feature_rms_after_mm'] = after.statistics['rms_mm']
                    comparison['registration'].append(detail)
                except ValueError as error:
                    comparison['registration'].append({'method':method,'converged':False,'reason':str(error)})
            assert signature(source) == original_signature
            receipt['comparisons'].append(comparison)
        receipt['status'] = ('compared' if any(c['status']=='compared' for c in receipt['comparisons'])
                             else 'no_reference' if batch.summary['actual_assets'] else 'nominal_only')
        assert hashlib.sha256(path.read_bytes()).hexdigest() == before
        receipt['raw_file_unchanged'] = True
        receipt['elapsed_s'] = time.perf_counter()-start
        (destination/'receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
        receipts.append(receipt)
        print(json.dumps({'file':path.name,'status':receipt['status'],
                          'samples':len(receipt['comparisons']),'elapsed_s':round(receipt['elapsed_s'],3)},ensure_ascii=False),flush=True)
    result = {'schema':'emma-real-data-validation/1', 'app_version':__version__,
              'generated_at':datetime.now(timezone.utc).isoformat(), 'files':receipts,
              'interpretation':'1 mm is an analytical distance threshold, not drawing tolerance or GD&T acceptance. '
              'Input unit assumed mm and explicitly passed. Registration is a computational exercise on copies; '
              'all delivered projects retain original measured coordinates. Rejected/truncated files are not repaired by guessing.'}
    (output/'validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder')
    parser.add_argument('--output', required=True)
    options = parser.parse_args()
    run(options.folder, options.output)
