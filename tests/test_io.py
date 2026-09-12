import json
import zipfile
import numpy as np
import pytest

from assembly_workbench.core import Dataset, make_demo, measure_deviation, rigid_transform
from assembly_workbench.io import load_dataset, save_project, load_project, export_report


def test_csv_metres_convert_once(tmp_path):
    path = tmp_path/'part.csv'
    path.write_text('x,y,z\n.001,.002,.003\n.004,.005,.006\n')
    part = load_dataset(path, unit='m')
    np.testing.assert_allclose(part.points, [[1, 2, 3], [4, 5, 6]])


def test_invalid_unit_and_nan_rejected(tmp_path):
    path = tmp_path/'part.xyz'
    path.write_text('1 2 nan\n')
    with pytest.raises(ValueError):
        load_dataset(path)
    path.write_text('1 2 3\n')
    with pytest.raises(ValueError):
        load_dataset(path, unit='inch')


def test_ascii_pcd_respects_field_order(tmp_path):
    path = tmp_path/'part.pcd'
    path.write_text('VERSION .7\nFIELDS intensity z x y\nSIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1\nWIDTH 2\nHEIGHT 1\nPOINTS 2\nDATA ascii\n9 3 1 2\n8 6 4 5\n')
    np.testing.assert_allclose(load_dataset(path).points, [[1, 2, 3], [4, 5, 6]])


def test_project_roundtrip_is_self_contained(tmp_path):
    source, target, _ = make_demo()
    source.transform = rigid_transform(tx=2, rz=5)
    source.source_path = '/missing/original.xyz'
    path = tmp_path/'session.vaw'
    save_project(path, [source, target], [{'method':'gicp','note':'保存测试'}])
    assets, history = load_project(path)
    assert len(assets) == 2
    assert history[0]['method'] == 'gicp'
    np.testing.assert_array_equal(assets[0].points, source.points)
    np.testing.assert_array_equal(assets[0].transform, source.transform)
    assert assets[0].id == source.id


def test_cad_survives_project_move_without_original(tmp_path):
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    cad = Dataset('CAD', [[0, 0, 0], [10, 10, 10]], kind='cad', cad_shape=BRepPrimAPI_MakeBox(10., 10., 10.).Shape())
    path = tmp_path/'cad.vaw'
    save_project(path, [cad])
    restored, _ = load_project(path)
    result = measure_deviation(Dataset('s', [[12, 5, 5]]), restored[0])
    np.testing.assert_allclose(result.distances_mm, [2], atol=1e-7)


def test_malformed_project_does_not_extract_paths(tmp_path):
    path = tmp_path/'bad.vaw'
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr('../escape.txt', 'evil')
        z.writestr('manifest.json', json.dumps({'version':999,'assets':[]}))
    with pytest.raises(ValueError):
        load_project(path)
    assert not (tmp_path.parent/'escape.txt').exists()


def test_export_records_actual_counts_and_escapes_html(tmp_path):
    source = Dataset('<script>alert(1)</script>', [[0, 0, 1], [0, 0, 2]])
    target = Dataset('reference', [[0, 0, 0]])
    result = measure_deviation(source,target,tolerance_mm=1)
    paths = export_report(tmp_path/'report',source,target,result)
    assert all(p.exists() for p in paths)
    data = json.loads((tmp_path/'report/report.json').read_text())
    assert data['deviation']['statistics']['within_fraction'] == .5
    assert data['units'] == 'mm'
    assert '<script>alert(1)</script>' not in (tmp_path/'report/report.html').read_text()
    assert len((tmp_path/'report/deviations.csv').read_text().splitlines()) == 3


def test_stale_measurement_is_not_exported(tmp_path):
    source = Dataset('s', [[0, 0, 1]])
    target = Dataset('t', [[0, 0, 0]])
    result = measure_deviation(source, target)
    target.transform = rigid_transform(tz=1)
    with pytest.raises(ValueError):
        export_report(tmp_path, source, target, result)


def test_step_load_produces_native_shape_and_render_mesh(tmp_path):
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.STEPControl import STEPControl_Writer, STEPControl_AsIs
    path = tmp_path/'box.step'
    writer = STEPControl_Writer()
    writer.Transfer(BRepPrimAPI_MakeBox(10.,20.,30.).Shape(),STEPControl_AsIs)
    writer.Write(str(path))
    part = load_dataset(path)
    assert part.kind == 'cad'
    assert part.cad_shape is not None
    assert len(part.triangles) >= 12
    np.testing.assert_allclose(np.ptp(part.points,axis=0),[10,20,30],atol=1e-7)


def test_oversize_manifest_cannot_replace_a_readable_project(tmp_path):
    path = tmp_path / 'preserve.vaw'
    asset = Dataset('one', [[0, 0, 0]])
    save_project(path, [asset])
    before = path.read_bytes()
    with pytest.raises(ValueError, match='清单|历史'):
        save_project(path, [asset], [{'note': 'x' * (8 * 1024 * 1024)}])
    assert path.read_bytes() == before
    assert load_project(path)[0][0].id == asset.id


def test_export_preserves_input_and_existing_reports(tmp_path):
    original = tmp_path / 'aligned.xyz'
    original.write_text('0 0 1\n')
    source = load_dataset(original)
    source.transform = rigid_transform(tx=10)
    target = Dataset('target', [[10, 0, 0]])
    deviation = measure_deviation(source, target)
    first = export_report(tmp_path, source, target, deviation)
    assert original.read_text() == '0 0 1\n'
    before = {p: p.read_bytes() for p in first}
    second = export_report(tmp_path, source, target, deviation)
    assert all(p.read_bytes() == content for p, content in before.items())
    assert set(first).isdisjoint(second)
    assert all(p.exists() for p in second)
