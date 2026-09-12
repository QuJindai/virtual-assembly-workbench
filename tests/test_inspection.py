import numpy as np
import pytest
from assembly_workbench.core import Dataset


def pair():
    return (Dataset('actual', [[1, 0, .2], [2, 0, -.3]], point_ids=['a', 'b'], point_scope='plan'),
            Dataset('nominal', [[1, 0, 0], [2, 0, 0], [3, 0, 0]], point_ids=['a', 'b', 'c'], point_scope='plan'))


def test_limits_keep_missing_and_unjudged_separate():
    from assembly_workbench.inspection import evaluate_points
    s,t=pair()
    controls=[dict(name=x,point_id=x,axis=[0,0,1],lower_mm=-.2,upper_mm=.2) for x in 'abc']
    controls.append(dict(name='no_limit',point_id='a',axis=[0,0,1]))
    r=evaluate_points(s,t,controls)
    assert [x['status'] for x in r['rows']]==['pass','fail','missing','unjudged']
    assert r['rows'][2]['value_mm'] is None
    assert r['statistics']['pass_count']==1
    assert r['statistics']['coverage']==.75
    assert r['overall']=='incomplete'


def test_inspection_validates_scope_limits_and_axes():
    from assembly_workbench.inspection import evaluate_points
    s,t=pair()
    c=dict(name='a',point_id='a',axis=[0,0,1],lower_mm=-1,upper_mm=1)
    for change in [dict(axis=[0,0,0]),dict(lower_mm=2),dict(upper_mm=float('nan'))]:
        with pytest.raises(ValueError):evaluate_points(s,t,[c|change])
    with pytest.raises(ValueError):evaluate_points(s,t,[c,c])
    t.point_scope='other'
    with pytest.raises(ValueError):evaluate_points(s,t,[c])


def test_inspection_uses_current_world_coordinates_without_mutation():
    from assembly_workbench.inspection import evaluate_points
    s,t=pair();before=s.points.copy();s.transform[2,3]=.3
    r=evaluate_points(s,t,[dict(name='a',point_id='a',axis=[0,0,2],lower_mm=.5,upper_mm=.5)])
    assert r['rows'][0]['value_mm']==pytest.approx(.5)
    assert r['rows'][0]['status']=='pass'
    np.testing.assert_array_equal(s.points,before)


def test_imported_axis_limits_keep_reference_coordinate_system(tmp_path):
    from assembly_workbench.inspection import load_emma_controls,evaluate_points
    from assembly_workbench.core import rigid_transform
    import csv,json
    fields=['InspectionTask','InspectionPlan','PartSingle','IPE.Name','IPE.Type','InspectionCategories','Component.ID']
    fields += [f'TolAxisPos{axis}.{bound}ToleranceValue' for axis in 'XYZ' for bound in ('Lower','Upper')]
    row=['task','plan','part','P1','PNT','MPT','','-.5','.5','','','0','.4']
    p=tmp_path/'limits.csv'
    with p.open('w',newline='') as f:
        w=csv.writer(f);w.writerow(fields);w.writerow(row)
    scope=json.dumps(['task','plan','part'],separators=(',',':'))
    imported=load_emma_controls(p,scope)
    assert len(imported['controls'])==2
    s=Dataset('a',[[0,0,.3]],point_ids=['PNT:P1'],point_scope=scope)
    t=Dataset('n',[[0,0,0]],point_ids=['PNT:P1'],point_scope=scope)
    s.transform=t.transform=rigid_transform(100,20,0,0,90,0)
    r=evaluate_points(s,t,imported['controls'],axis_frame='reference_local')
    assert r['rows'][1]['value_mm']==pytest.approx(.3)
    assert r['overall']=='pass'
    with p.open('a',newline='') as f:csv.writer(f).writerow(row[:-1])
    with pytest.raises(ValueError):load_emma_controls(p,scope)


def test_axis_normalization_cannot_overflow_to_zero_and_false_pass():
    from assembly_workbench.inspection import evaluate_points
    s,t=pair()
    r=evaluate_points(s,t,[dict(name='z',point_id='a',axis=[0,0,1e308],lower_mm=0,upper_mm=.1)])
    assert r['rows'][0]['value_mm']==pytest.approx(.2)
    assert r['rows'][0]['status']=='fail'


@pytest.mark.parametrize('actual,limit,status',[(10.3,.3,'unjudged'),(10.2,.2,'unjudged'),(10.30001,.3,'fail'),(10.29999,.3,'pass'),(10.25,.25,'pass')])
def test_floating_point_boundary_is_visible_without_relaxing_limits(actual,limit,status):
    from assembly_workbench.inspection import evaluate_points
    s=Dataset('a',[[actual,0,0]],point_ids=['p'],point_scope='p')
    t=Dataset('n',[[10,0,0]],point_ids=['p'],point_scope='p')
    r=evaluate_points(s,t,[dict(name='x',point_id='p',axis=[1,0,0],lower_mm=0,upper_mm=limit)])
    assert r['rows'][0]['status']==status
    assert r['rows'][0]['upper_mm']==limit
    assert r['statistics']['numerical_boundary_count']==int(status=='unjudged')
    if status=='unjudged':assert r['rows'][0]['unjudged_reason']=='floating_point_boundary'


@pytest.mark.parametrize('actual,nominal,lo,hi,status',[(.25,0,.25-np.spacing(.25),.25,'pass'),(1e-17,0,0,1e-18,'fail'),(1e308,1e308,0,1,'pass')])
def test_roundoff_estimate_has_no_fixed_length_floor_or_overflow(actual,nominal,lo,hi,status):
    import json
    from assembly_workbench.inspection import evaluate_points
    s=Dataset('a',[[actual,0,0]],point_ids=['p'],point_scope='p')
    t=Dataset('n',[[nominal,0,0]],point_ids=['p'],point_scope='p')
    r=evaluate_points(s,t,[dict(name='x',point_id='p',axis=[1,0,0],lower_mm=lo,upper_mm=hi)])
    assert r['rows'][0]['status']==status
    json.dumps(r,allow_nan=False)
