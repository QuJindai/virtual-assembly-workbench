import json
import numpy as np
import pytest
from assembly_workbench.core import Dataset


def pair():
    return Dataset('actual',[[0,0,.3],[1,0,.5]],point_ids=['a','b'],point_scope='p'),Dataset('nominal',[[0,0,0],[1,0,0]],point_ids=['a','b'],point_scope='p')


def test_selection_resolves_labels_rows_prefix_and_rejects_duplicates():
    from assembly_workbench.engineering import select_points
    s,_=pair()
    p,ids=select_points(s,['b','#1'])
    np.testing.assert_array_equal(p,[[1,0,.5],[0,0,.3]])
    assert ids==['b','a']
    with pytest.raises(ValueError):select_points(s,['a','#1'])
    with pytest.raises(ValueError):select_points(s,['#0'])
    p,ids=select_points(s,{'prefix':'b'})
    assert ids==['b']


def test_receipt_replay_history_and_export(tmp_path):
    from assembly_workbench.engineering import run_engineering,export_engineering
    from assembly_workbench.io import save_project,load_project
    s,t=pair()
    recipe={'tool':'inspection','controls':[dict(name='<critical>',point_id='a',axis=[0,0,1],lower_mm=-.2,upper_mm=.2)]}
    r=run_engineering(s,t,recipe)
    assert r['result']['rows'][0]['status']=='fail'
    save_project(tmp_path/'p.vaw',[s,t],[dict(type='engineering',result=r)])
    assets,history=load_project(tmp_path/'p.vaw')
    restored=run_engineering(*assets,history[0]['result']['recipe'])
    assert restored['result']==r['result']
    files=export_engineering(tmp_path/'reports',s,t,r)
    assert len(files)==3
    html=next(x for x in files if x.suffix=='.html').read_text(encoding='utf-8')
    assert '&lt;critical&gt;' in html and '<critical>' not in html
    assert len(json.loads(next(x for x in files if x.suffix=='.json').read_text(encoding='utf-8'))['result']['rows'])==1
    second=export_engineering(tmp_path/'reports',s,t,r)
    assert second[0].parent!=files[0].parent
    s.points[0,2]=1
    with pytest.raises(ValueError):export_engineering(tmp_path/'reports',s,t,r)


def test_receipt_cannot_apply_a_measurement_or_edited_result():
    from assembly_workbench.engineering import run_engineering,apply_engineering
    s,t=pair();r=run_engineering(s,t,{'tool':'inspection','controls':[dict(name='a',point_id='a',axis=[1,0,0])]})
    with pytest.raises(ValueError):apply_engineering(s,t,r)


def test_unknown_recipe_keys_and_nonfinite_values_rejected():
    from assembly_workbench.engineering import run_engineering
    s,t=pair()
    with pytest.raises(ValueError):run_engineering(s,t,{'tool':'shell','command':'anything'})
    with pytest.raises(ValueError):run_engineering(s,t,{'tool':'inspection','controls':[],'unexpected':1})


def test_cad_fingerprint_detects_shape_change():
    from assembly_workbench.engineering import fingerprint
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    s=Dataset('CAD',[[0,0,0]],kind='cad',cad_shape=BRepPrimAPI_MakeBox(1.,1.,1.).Shape())
    first=fingerprint(s)
    s.cad_shape=BRepPrimAPI_MakeBox(2.,1.,1.).Shape()
    assert fingerprint(s)!=first


def test_plane_comparison_is_independent_of_sample_patch_center():
    from assembly_workbench.engineering import run_engineering
    s=Dataset('actual',[[100,0,1],[101,0,1],[100,1,1],[101,1,1]])
    t=Dataset('nominal',[[0,0,0],[1,0,0],[0,1,0],[1,1,0]])
    r=run_engineering(s,t,dict(tool='feature',kind='plane',compare=True))['result']['comparison']
    assert 'center_distance_mm' not in r
    assert abs(r['normal_offset_at_source_center_mm'])==pytest.approx(1)


def test_cad_receipt_fingerprint_survives_project_roundtrip(tmp_path):
    from assembly_workbench.engineering import fingerprint
    from assembly_workbench.engineering_examples import example
    from assembly_workbench.io import save_project,load_project
    s,t,_=example('contact');before=[fingerprint(a) for a in (s,t)]
    save_project(tmp_path/'cad.vaw',[s,t])
    restored,_=load_project(tmp_path/'cad.vaw')
    assert [fingerprint(a) for a in restored]==before


def test_large_engineering_history_is_lossless_and_bounded(tmp_path):
    from assembly_workbench.io import save_project,load_project
    import zipfile
    s,t=pair()
    history=[dict(type='engineering',result={'recipe':{'name':'x'*1000},'rows':[{'detail':'z'*1000} for _ in range(9000)]})]
    save_project(tmp_path/'large.vaw',[s,t],history)
    _,restored=load_project(tmp_path/'large.vaw')
    assert restored==history
    with zipfile.ZipFile(tmp_path/'large.vaw') as z:
        assert z.getinfo('manifest.json').file_size<10000
        assert json.loads(z.read('manifest.json'))['version']==3
