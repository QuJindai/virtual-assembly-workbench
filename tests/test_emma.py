"""Synthetic eMMA fixtures only; no factory coordinates or identifiers."""
import csv
import json
import zipfile

import numpy as np
import pytest

from assembly_workbench.core import Dataset, measure_deviation, rigid_transform
from assembly_workbench.io import export_report, load_project, save_project
from assembly_workbench.emma import is_emma_csv, load_emma


HEADER = ['InspectionTask', 'InspectionPlan', 'PartSingle', 'IPE.Name', 'IPE.Type',
          'IPE.Origin.X', 'IPE.Origin.Y', 'IPE.Origin.Z', 'InspectionCategories',
          'Component.ID', 'History.DateTime']


def write_emma(path, records):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(reversed(HEADER)))
        writer.writeheader()
        for record in records:
            writer.writerow(record)
    return path


def row(name, xyz, category='MPT', component='', date='2026-01-01'):
    return dict(zip(HEADER, ['SYNTHETIC', 'PLAN', 'PART', name, 'FPT', *xyz,
                             category, component, date]))


def test_emma_separates_samples_and_keeps_missing_values_missing(tmp_path):
    path = write_emma(tmp_path/'measurements.csv', [
        row('P1', [0, 0, 0]), row('P2', [10, 0, 0]),
        row('P1', [.1, 0, 0], 'A', 'sample-1'),
        row('P2', ['', '', ''], 'A', 'sample-1'),
        row('P1', [.2, 0, 0], 'A', 'sample-2'),
        row('P2', [10.2, 0, 0], '', 'sample-2'),
    ])
    assert is_emma_csv(path)
    batch = load_emma(path)
    assert len(batch.assets) == 3
    assert batch.summary['missing_xyz_rows'] == 1
    assert batch.summary['inferred_actual_rows'] == 1
    actual = next(a for a in batch.assets if 'sample-1' in a.name)
    nominal = next(a for a in batch.assets if '名义' in a.name)
    assert len(actual.points) == 1
    assert actual.point_ids == ['FPT:P1']
    assert actual.point_scope == nominal.point_scope
    np.testing.assert_allclose(actual.points, [[.1, 0, 0]])


def test_emma_conflicting_duplicate_is_quarantined(tmp_path):
    path = write_emma(tmp_path/'conflict.csv', [row('P1', [0,0,0]),
        row('P1', [99,0,0]), row('P2', [1,2,3]),
        row('P1', [0,0,0], 'A', 'sample'), row('P2', [1,2,4], 'A', 'sample')])
    batch = load_emma(path)
    nominal = next(a for a in batch.assets if '名义' in a.name)
    assert nominal.point_ids == ['FPT:P2']
    assert batch.summary['conflicting_keys'] == 1


def test_emma_requires_header_and_rejects_truncated_rows(tmp_path):
    missing = tmp_path/'missing.csv'
    missing.write_text('part,point,1,2,3\n')
    with pytest.raises(ValueError, match='表头'):
        load_emma(missing)
    path = write_emma(tmp_path/'truncated.csv', [row('P1',[1,2,3])])
    with path.open('a') as stream:
        stream.write('part,point,1\n')
    with pytest.raises(ValueError, match='列数'):
        load_emma(path)


def test_emma_rejects_unterminated_final_quoted_field(tmp_path):
    path = tmp_path/'unterminated.csv'
    path.write_text(','.join(HEADER)+'\nSYNTHETIC,PLAN,PART,P1,FPT,1,2,3,A,sample,"2026-01',encoding='utf-8')
    with pytest.raises(ValueError, match='引号|CSV'):
        load_emma(path)


def test_emma_unknown_roles_and_nonfinite_coordinates_are_not_geometry(tmp_path):
    path = write_emma(tmp_path/'invalid.csv', [row('P1',[1,2,3]),
        row('P2',[1,2,float('nan')]), row('P3',[1,2,3],'UNKNOWN','sample'),
        row('P4',[1,2,3],'',''), row('P5',[1,'',3])])
    batch = load_emma(path)
    assert len(batch.assets) == 1
    assert len(batch.assets[0].points) == 1
    assert batch.summary['invalid_xyz_rows'] == 1
    assert batch.summary['unknown_role_rows'] == 2
    assert batch.summary['missing_xyz_rows'] == 1


def test_emma_converts_units_once_and_preserves_nominal_only_data(tmp_path):
    path = write_emma(tmp_path/'nominal.csv', [row('P1',[.001,.002,.003])])
    batch = load_emma(path, unit='m')
    np.testing.assert_allclose(batch.assets[0].points, [[1,2,3]])
    assert batch.summary['actual_assets'] == 0
    with pytest.raises(ValueError):
        load_emma(path, unit='inch')


def test_feature_correspondence_does_not_choose_a_nearer_wrong_point():
    source = Dataset('actual', [[99,0,0],[1,0,0]], point_ids=['A','B'], point_scope='plan')
    target = Dataset('nominal', [[0,0,0],[100,0,0]], point_ids=['A','B'], point_scope='plan')
    result = measure_deviation(source, target)
    assert result.method == 'feature_id'
    np.testing.assert_allclose(result.distances_mm, [99,99])
    np.testing.assert_array_equal(result.target_indices, [0,1])


def test_feature_scope_and_unmatched_points_are_explicit():
    source = Dataset('actual', [[1,0,0],[2,0,0]], point_ids=['A','X'], point_scope='plan')
    target = Dataset('nominal', [[0,0,0]], point_ids=['A'], point_scope='plan')
    result = measure_deviation(source,target)
    assert result.statistics['matched_count'] == 1
    assert result.statistics['unmatched_count'] == 1
    target.point_scope = 'other-plan'
    with pytest.raises(ValueError, match='测点范围'):
        measure_deviation(source,target)


def test_reference_coverage_and_missing_ids_survive_report_export(tmp_path):
    source = Dataset('actual',[[0,0,0]],point_ids=['P1'],point_scope='plan')
    target = Dataset('nominal',[[0,0,0],[10,0,0]],point_ids=['P1','P2'],point_scope='plan')
    result = measure_deviation(source,target)
    assert result.statistics['reference_coverage'] == .5
    assert result.statistics['unmatched_reference_count'] == 1
    paths = export_report(tmp_path/'coverage',source,target,result)
    data = json.loads(paths[0].read_text(encoding='utf-8'))
    assert data['deviation']['unmatched_reference_ids'] == ['P2']


def test_labeled_project_roundtrip_and_report_preserve_correspondence(tmp_path):
    source = Dataset('actual', [[0,0,2]], point_ids=['P1'], point_scope='plan')
    source.transform = rigid_transform(tx=1)
    target = Dataset('nominal', [[0,0,0]], point_ids=['P1'], point_scope='plan')
    path = tmp_path/'labeled.vaw'
    save_project(path,[source,target])
    with zipfile.ZipFile(path) as archive:
        assert json.loads(archive.read('manifest.json'))['version'] == 2
    restored,_ = load_project(path)
    assert restored[0].point_ids == source.point_ids
    assert restored[0].point_scope == 'plan'
    result = measure_deviation(*restored)
    paths = export_report(tmp_path/'report',*restored,result)
    with paths[1].open(encoding='utf-8-sig') as stream:
        entry = next(csv.DictReader(stream))
    assert entry['feature_id'] == 'P1'
    assert [float(entry['delta_'+a+'_mm']) for a in 'xyz'] == [1,0,2]
    restored[0].point_ids[0] = 'CHANGED'
    with pytest.raises(ValueError, match='过期'):
        export_report(tmp_path/'stale',*restored,result)


def test_feature_identifiers_must_be_unique_and_complete():
    with pytest.raises(ValueError):
        Dataset('bad',[[0,0,0],[1,0,0]],point_ids=['A','A'],point_scope='plan')
    with pytest.raises(ValueError):
        Dataset('bad',[[0,0,0]],point_ids=['A'])
