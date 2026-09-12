"""Analytic tests for explicit edge gauges and native section/solid queries."""
import json

import numpy as np
import pytest

from assembly_workbench.core import Dataset, rigid_transform


def box(size=10, transform=None):
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    return Dataset('analytic box', [[0, 0, 0]], kind='cad',
                   cad_shape=BRepPrimAPI_MakeBox(float(size), float(size), float(size)).Shape(),
                   transform=np.eye(4) if transform is None else transform)


def test_signed_edge_gauge():
    from assembly_workbench.assembly import gap_flush
    r = gap_flush([[4, 2, 1], [-3, -2, -1]], [[0, 0, 0]] * 2, [1, 0, 0], [0, 0, 1])
    assert r['gap_mm'] == [4.0, -3.0]
    assert r['flush_mm'] == [1.0, -1.0]
    assert r['seam_offset_mm'] == [-2.0, 2.0]
    assert r['statistics']['gap_mean_mm'] == .5
    assert r['count'] == 2
    json.dumps(r, allow_nan=False)


def test_rotated_gauge_and_input_immutable():
    from assembly_workbench.assembly import gap_flush
    matrix = rigid_transform(rz=31, ry=16)
    rot = matrix[:3, :3]
    source = np.array([[4., 2., 1.]]) @ rot.T
    saved = source.copy()
    r = gap_flush(source, [[0, 0, 0]], 2*rot[:, 0], 3*rot[:, 2])
    assert r['gap_mm'] == pytest.approx([4])
    assert r['flush_mm'] == pytest.approx([1])
    np.testing.assert_array_equal(source, saved)


@pytest.mark.parametrize('source,reference,gap,flush', [
    ([[1, 2, 3]], [[0, 0, 0]] * 2, [1, 0, 0], [0, 0, 1]),
    ([[1, 2, 3]], [[0, 0, 0]], [0, 0, 0], [0, 0, 1]),
    ([[1, 2, 3]], [[0, 0, 0]], [1, 0, 0], [1, 1, 0]),
    ([[1, 2, np.nan]], [[0, 0, 0]], [1, 0, 0], [0, 0, 1]),
    ([[1, 2, 3]], [[0, 0, 0]], [1, 0], [0, 0, 1]),
])
def test_invalid_gauge(source, reference, gap, flush):
    from assembly_workbench.assembly import gap_flush
    with pytest.raises(ValueError):
        gap_flush(source, reference, gap, flush)


@pytest.mark.parametrize('translation,state,distance,volume', [
    ((12, 0, 0), 'clear', 2, 0),
    ((10, 0, 0), 'contact', 0, 0),
    ((8, 8, 8), 'interference', 0, 8),
])
def test_native_box_contact(translation, state, distance, volume):
    from assembly_workbench.assembly import cad_contact
    a = box(transform=rigid_transform(*translation))
    b = box()
    snapshot = a.transform.copy()
    result = cad_contact(a, b)
    assert result['state'] == state
    assert result['distance_mm'] == pytest.approx(distance, abs=1e-9)
    assert result['common_volume_mm3'] == pytest.approx(volume, abs=1e-8)
    assert 0 < result['volume_tolerance_mm3'] < .1
    np.testing.assert_array_equal(a.transform, snapshot)
    json.dumps(result, allow_nan=False)


def test_containment_and_joint_rotation():
    from assembly_workbench.assembly import cad_contact
    pose = rigid_transform(tx=101, ty=-20, tz=14, rx=13, ry=-19, rz=24)
    a = box(2, pose @ rigid_transform(tx=3, ty=3, tz=3))
    b = box(10, pose)
    result = cad_contact(a, b)
    assert result['state'] == 'interference'
    assert result['common_volume_mm3'] == pytest.approx(8, abs=1e-8)
    assert result['distance_mm'] == pytest.approx(0, abs=1e-9)
    a.transform = pose @ rigid_transform(tx=12)
    result = cad_contact(a, b)
    assert result['state'] == 'clear'
    assert result['distance_mm'] == pytest.approx(2, abs=1e-9)


def test_open_face_and_cloud_reject_contact():
    from assembly_workbench.assembly import cad_contact
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    from OCP.gp import gp_Pln, gp_Pnt, gp_Dir
    face = BRepBuilderAPI_MakeFace(gp_Pln(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)), 0., 10., 0., 10.).Face()
    for invalid in [Dataset('face', [[0, 0, 0]], kind='cad', cad_shape=face), Dataset('cloud', [[0, 0, 0]])]:
        with pytest.raises(ValueError):
            cad_contact(invalid, box())


def test_compound_mixed_contents_rejected():
    from assembly_workbench.assembly import cad_contact
    from OCP.BRep import BRep_Builder
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
    from OCP.TopoDS import TopoDS_Compound
    from OCP.gp import gp_Pnt
    compound = TopoDS_Compound()
    builder = BRep_Builder()
    builder.MakeCompound(compound)
    builder.Add(compound, box().cad_shape)
    builder.Add(compound, BRepBuilderAPI_MakeVertex(gp_Pnt(30, 30, 30)).Shape())
    with pytest.raises(ValueError):
        cad_contact(Dataset('mixed', [[0, 0, 0]], kind='cad', cad_shape=compound), box())


@pytest.mark.parametrize('normal', [[0, 0, 0], [0, 0, np.nan], [1, 2]])
def test_invalid_section_normal(normal):
    from assembly_workbench.assembly import section_geometry
    with pytest.raises(ValueError):
        section_geometry(box(), [0, 0, 0], normal)


def test_cloud_section_only_existing_points_in_inclusive_half_width():
    from assembly_workbench.assembly import section_geometry
    points = np.array([[0, 0, -.1], [1, 0, .1], [2, 0, .10001], [3, 0, 0.]])
    asset = Dataset('slab', points, transform=rigid_transform(tx=4, rz=90))
    result = section_geometry(asset, [0, 0, 0], [0, 0, 2], slab_mm=.2)
    assert result['method'] == 'cloud_slab'
    assert result['sampled'] is False
    np.testing.assert_allclose(result['points'], asset.world_points()[[0, 1, 3]])
    limited = section_geometry(asset, [0, 0, 0], [0, 0, 1], slab_mm=.2, max_points=2)
    assert len(limited['points']) == 2 and limited['sampled'] is True
    assert all(any(np.allclose(p, q) for q in result['points']) for p in limited['points'])
    assert section_geometry(asset, [0, 0, 10], [0, 0, 1])['points'] == []
    np.testing.assert_array_equal(asset.points, points)


def test_mesh_section_interpolates_actual_triangle_edges():
    from assembly_workbench.assembly import section_geometry
    pose = rigid_transform(tx=3, ty=8, rx=20, ry=10)
    asset = Dataset('triangle', [[0, 0, -1], [2, 0, 1], [0, 2, 1]],
                    kind='mesh', triangles=[[0, 1, 2]], transform=pose)
    result = section_geometry(asset, pose[:3, 3], pose[:3, 2])
    expected = np.array([[1, 0, 0], [0, 1, 0]]) @ pose[:3, :3].T + pose[:3, 3]
    assert result['method'] == 'mesh_section'
    assert len(result['points']) == 2
    assert all(any(np.allclose(p, q) for q in expected) for p in result['points'])


def test_cad_section_uses_native_shape_with_transformed_plane():
    from assembly_workbench.assembly import section_geometry
    pose = rigid_transform(tx=30, ty=-11, rx=25, ry=16, rz=3)
    asset = box(transform=pose)  # Deliberately only one display point.
    origin = np.array([0, 0, 5.]) @ pose[:3, :3].T + pose[:3, 3]
    result = section_geometry(asset, origin, pose[:3, 2], max_points=21)
    local = (np.array(result['points']) - pose[:3, 3]) @ pose[:3, :3]
    assert result['method'] == 'cad_section'
    assert 4 <= len(local) <= 21
    np.testing.assert_allclose(local[:, 2], 5, atol=1e-8)
    np.testing.assert_allclose(local[:, :2].min(0), [0, 0], atol=1e-8)
    np.testing.assert_allclose(local[:, :2].max(0), [10, 10], atol=1e-8)
    assert np.all(np.any(np.isclose(local[:, :2], 0) | np.isclose(local[:, :2], 10), axis=1))
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('kwargs', [{'slab_mm': 0}, {'slab_mm': np.nan}, {'max_points': 0}, {'max_points': 1.5}])
def test_section_invalid_sampling_settings(kwargs):
    from assembly_workbench.assembly import section_geometry
    with pytest.raises(ValueError):
        section_geometry(box(), [0, 0, 0], [0, 0, 1], **kwargs)


def compound_asset(*assets):
    from OCP.BRep import BRep_Builder
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCP.TopoDS import TopoDS_Compound
    from OCP.gp import gp_Trsf
    compound = TopoDS_Compound()
    builder = BRep_Builder()
    builder.MakeCompound(compound)
    for asset in assets:
        trsf = gp_Trsf()
        trsf.SetValues(*map(float, asset.transform[:3].ravel()))
        builder.Add(compound, BRepBuilderAPI_Transform(asset.cad_shape, trsf, True).Shape())
    return Dataset('compound', [[0, 0, 0]], kind='cad', cad_shape=compound)


def test_disjoint_compound_solids_and_overlap_rejection():
    from assembly_workbench.assembly import cad_contact
    valid = compound_asset(box(2), box(2, rigid_transform(tx=10)))
    result = cad_contact(valid, box(2, rigid_transform(tx=1)))
    assert result['common_volume_mm3'] == pytest.approx(4)
    assert result['state'] == 'interference'
    with pytest.raises(ValueError, match='重叠'):
        cad_contact(compound_asset(box(), box(transform=rigid_transform(tx=1))), box())


def test_native_queries_preserve_original_brep():
    import io
    from OCP.BRepTools import BRepTools
    from assembly_workbench.assembly import cad_contact, section_geometry
    asset = box(transform=rigid_transform(tx=8, ty=8, tz=8))
    def serialized():
        stream = io.BytesIO()
        BRepTools.Write_s(asset.cad_shape, stream)
        return stream.getvalue()
    before = serialized()
    cad_contact(asset, box())
    section_geometry(asset, [0, 0, 12], [0, 0, 1])
    assert serialized() == before


def test_curved_cad_section_is_bounded_and_on_exact_circle():
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder
    from assembly_workbench.assembly import section_geometry
    asset = Dataset('cylinder', [[0, 0, 0]], kind='cad', cad_shape=BRepPrimAPI_MakeCylinder(5., 10.).Shape())
    result = section_geometry(asset, [0, 0, 5], [0, 0, 1], max_points=17)
    points = np.array(result['points'])
    assert 10 <= len(points) <= 17
    assert result['sampled'] is True
    np.testing.assert_allclose(np.linalg.norm(points[:, :2], axis=1), 5, atol=1e-8)
    np.testing.assert_allclose(points[:, 2], 5, atol=1e-8)
    assert section_geometry(asset, [0, 0, 20], [0, 0, 1])['points'] == []


def test_coplanar_mesh_section_and_empty_section():
    from assembly_workbench.assembly import section_geometry
    asset = Dataset('flat', [[0, 0, 0], [2, 0, 0], [0, 2, 0]], kind='mesh', triangles=[[0, 1, 2]])
    result = section_geometry(asset, [0, 0, 0], [0, 0, 1])
    assert len(result['points']) == 3
    assert all(any(np.array_equal(p, q) for q in asset.points) for p in result['points'])
    assert section_geometry(asset, [0, 0, 1], [0, 0, 1])['points'] == []


@pytest.mark.parametrize('tolerance', [0, -1, np.nan, np.inf])
def test_bad_contact_tolerance(tolerance):
    from assembly_workbench.assembly import cad_contact
    with pytest.raises(ValueError):
        cad_contact(box(), box(), tolerance_mm=tolerance)


def test_boolean_failure_never_returns_clear(monkeypatch):
    from OCP import BRepAlgoAPI
    from assembly_workbench.assembly import cad_contact
    class FailedCommon:
        def SetArguments(self, _): pass
        def SetTools(self, _): pass
        def SetNonDestructive(self, _): pass
        def Build(self): pass
        def IsDone(self): return False
    monkeypatch.setattr(BRepAlgoAPI, 'BRepAlgoAPI_Common', FailedCommon)
    with pytest.raises(ValueError, match='布尔求交失败'):
        cad_contact(box(), box(transform=rigid_transform(tx=12)))


def test_distance_failure_never_returns_clear(monkeypatch):
    from OCP import BRepExtrema
    from assembly_workbench.assembly import cad_contact
    class FailedDistance:
        def __init__(self, *_): pass
        def Perform(self): pass
        def IsDone(self): return False
    monkeypatch.setattr(BRepExtrema, 'BRepExtrema_DistShapeShape', FailedDistance)
    with pytest.raises(ValueError, match='最小距离求解失败'):
        cad_contact(box(), box(transform=rigid_transform(tx=12)))


def test_section_failure_never_returns_fake_empty(monkeypatch):
    from OCP import BRepAlgoAPI
    from assembly_workbench.assembly import section_geometry
    class FailedSection:
        def __init__(self, *_): pass
        def SetNonDestructive(self, _): pass
        def Build(self): pass
        def IsDone(self): return False
    monkeypatch.setattr(BRepAlgoAPI, 'BRepAlgoAPI_Section', FailedSection)
    with pytest.raises(ValueError, match='截面求交失败'):
        section_geometry(box(), [0, 0, 5], [0, 0, 1])


def test_rotated_cloud_slab_keeps_boundary_points():
    from assembly_workbench.assembly import section_geometry
    pose = rigid_transform(tx=1000, ty=-20, tz=10, rx=17, ry=9, rz=23)
    asset = Dataset('rotated slab', [[0, 0, -.1], [0, 0, .1], [0, 0, .10001]], transform=pose)
    result = section_geometry(asset, pose[:3, 3], pose[:3, 2], slab_mm=.2)
    np.testing.assert_allclose(result['points'], asset.world_points()[:2])


@pytest.mark.parametrize('kind', ['cloud', 'mesh'])
def test_large_tangential_offset_does_not_broaden_section(kind):
    from assembly_workbench.assembly import section_geometry
    points = [[0, 0, .3], [2, 0, .3], [0, 2, .3]]
    asset = Dataset('tangential offset', points, kind=kind,
                    triangles=[[0, 1, 2]] if kind == 'mesh' else None,
                    transform=rigid_transform(tx=1e14))
    assert section_geometry(asset, [0, 0, 0], [0, 0, 1], slab_mm=.2)['points'] == []


@pytest.mark.parametrize('kind', ['cloud', 'mesh'])
def test_unresolvable_normal_offset_raises_instead_of_broadening(kind):
    from assembly_workbench.assembly import section_geometry
    asset = Dataset('normal uncertainty', [[0, 0, .3], [2, 0, .3], [0, 2, .3]], kind=kind,
                    triangles=[[0, 1, 2]] if kind == 'mesh' else None,
                    transform=rigid_transform(tz=1e14))
    with pytest.raises(ValueError, match='数值精度'):
        section_geometry(asset, [0, 0, 1e14], [0, 0, 1], slab_mm=.2)


def test_capped_tangent_only_cad_section_marks_sampling():
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeSphere
    from assembly_workbench.assembly import section_geometry
    spheres = [Dataset('sphere', [[0, 0, 0]], kind='cad',
                       cad_shape=BRepPrimAPI_MakeSphere(1.).Shape(),
                       transform=rigid_transform(tx=4*i)) for i in range(3)]
    asset = compound_asset(*spheres)
    full = section_geometry(asset, [0, 0, 1], [0, 0, 1])
    limited = section_geometry(asset, [0, 0, 1], [0, 0, 1], max_points=2)
    assert len(full['points']) == 3
    assert full['sampled'] is False
    assert len(limited['points']) == 2
    assert limited['sampled'] is True
    assert all(any(np.allclose(p, q) for q in full['points']) for p in limited['points'])
