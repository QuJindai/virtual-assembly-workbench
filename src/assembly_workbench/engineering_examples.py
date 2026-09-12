"""Analytic examples; no production geometry or process datum assumptions."""
import numpy as np
from .core import Dataset,rigid_transform


def example(tool):
    scope='synthetic-engineering-example'
    def cloud(name,p,ids=None):
        return Dataset('合成示例 · '+name,p,point_ids=ids,point_scope=scope if ids else None)
    if tool in ('datum321','rps','adjustment'):
        p=np.array([[0,0,0],[100,0,0],[0,100,0],[0,0,30],[100,0,30],[0,40,50]],float)
        ids=['A1','A2','A3','B1','B2','C1'];T=rigid_transform(1,-.5,.2,.2,-.1,.3)
        source=cloud('待定位测点',p@T[:3,:3].T+T[:3,3],ids)
        target=cloud('名义基准',p,ids)
        recipe=dict(tool=tool,source_points=ids,target_points=ids)
        if tool!='datum321':recipe.update(normals=[[0,0,1]]*3+[[0,1,0]]*2+[[1,0,0]],weights=[1.]*6,
                    dofs=[True]*6,pivot=None,translation_limit_mm=10.,rotation_limit_deg=5.)
        if tool=='adjustment':recipe['offsets_mm']=[0.]*6
    elif tool=='feature':
        angles=np.linspace(0,2*np.pi,96,endpoint=False)
        ids=[f'RING_{i:03}' for i in range(len(angles))]
        p=np.c_[5*np.cos(angles),5*np.sin(angles),np.zeros(len(angles))]
        q=np.c_[5.15*np.cos(angles)+.2,5.15*np.sin(angles)-.1,np.full(len(angles),.3)]
        source=cloud('孔缘实测',q,ids);target=cloud('孔缘名义',p,ids)
        recipe=dict(tool=tool,kind='circle',compare=True,source_points=None,target_points=None)
    elif tool=='detect':
        rng=np.random.default_rng(2);xy=rng.uniform(-10,10,(200,2));p=np.c_[xy,rng.normal(0,.01,200)]
        p=np.vstack([p,rng.uniform(-10,10,(30,3))]);source=cloud('平面含离群点',p);target=cloud('参考平面',np.c_[xy,np.zeros(200)])
        recipe=dict(tool=tool,kind='plane',source_points=None,threshold_mm=.05,min_points=50,max_features=3,seed=0)
    elif tool=='gap_flush':
        source=cloud('正向门边',[[4,0,1],[4,10,1.2]],['E1','E2'])
        target=cloud('参考门框边',[[0,0,0],[0,10,0]],['E1','E2'])
        recipe=dict(tool=tool,source_points=['E1','E2'],target_points=['E1','E2'],gap_axis=[1.,0.,0.],flush_axis=[0.,0.,1.])
    elif tool=='inspection':
        source=cloud('实测含缺点',[[0,0,.1],[10,0,.4]],['P1','P2'])
        target=cloud('名义控制点',[[0,0,0],[10,0,0],[20,0,0]],['P1','P2','P3'])
        recipe=dict(tool=tool,controls=[dict(name=p+' / Z',point_id=p,axis=[0.,0.,1.],lower_mm=-.2,upper_mm=.2) for p in target.point_ids])
    elif tool in ('section','contact'):
        from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
        from .io import tessellate_shape
        a=BRepPrimAPI_MakeBox(10.,10.,10.).Shape();b=BRepPrimAPI_MakeBox(10.,10.,10.).Shape()
        p,tri=tessellate_shape(a);source=Dataset('合成示例 · 移动实体',p,'cad',tri,cad_shape=a)
        p,tri=tessellate_shape(b);target=Dataset('合成示例 · 参考实体',p,'cad',tri,cad_shape=b)
        source.transform=rigid_transform(8,8,8)
        recipe=dict(tool=tool,tolerance_mm=1e-5) if tool=='contact' else dict(tool=tool,origin=[10.,10.,10.],normal=[0.,0.,1.],slab_mm=.2,max_points=20000)
    else:raise ValueError('没有该工具的合成示例。')
    return source,target,recipe


def self_test(directory):
    """Known-answer checks through the same facade used by the desktop."""
    from pathlib import Path
    from .engineering import LABELS,run_engineering,apply_engineering,export_engineering,fingerprint
    from .io import save_project,load_project
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    assets=[];history=[];answers={}
    for tool in LABELS:
        source,target,recipe=example(tool)
        before=[fingerprint(a) for a in (source,target)]
        receipt=run_engineering(source,target,recipe);r=receipt['result']
        if [fingerprint(a) for a in (source,target)]!=before:raise RuntimeError('Engineering computation mutated an asset')
        if tool in ('datum321','rps','adjustment'):
            if not r['converged'] or r['rms_mm']>1e-7:raise RuntimeError(f'{tool} known-pose check failed')
        elif tool=='feature':
            if not np.isclose(r['radius_mm'],5.15,atol=1e-7):raise RuntimeError('Circle diameter check failed')
            answers['circle_diameter_mm']=2*r['radius_mm']
        elif tool=='detect':
            if not r['candidates'] or r['candidates'][0]['count']<195:raise RuntimeError('Plane candidate check failed')
        elif tool=='gap_flush':
            np.testing.assert_allclose(r['gap_mm'],[4,4],atol=1e-10)
            np.testing.assert_allclose(r['flush_mm'],[1,1.2],atol=1e-10)
            answers['gap_mm']=r['gap_mm'];answers['flush_mm']=r['flush_mm']
        elif tool=='contact':
            if r['state']!='interference' or not np.isclose(r['common_volume_mm3'],8.,atol=1e-8):raise RuntimeError('CAD common-volume check failed')
            answers['contact_volume_mm3']=round(r['common_volume_mm3'],9)
        elif tool=='section':
            points=np.asarray(r['points'])
            if len(points)<4 or not np.allclose(points[:,2],10,atol=1e-8):raise RuntimeError('CAD plane-section check failed')
        elif tool=='inspection':
            if [x['status'] for x in r['rows']]!=['pass','fail','missing']:raise RuntimeError('Coordinate-control state check failed')
        export_engineering(directory/tool,source,target,receipt)
        assets.extend((source,target));history.append(dict(type='engineering',result=receipt))
    project=directory/'engineering.vaw';save_project(project,assets,history)
    restored,saved_history=load_project(project)
    if saved_history!=history or [fingerprint(a) for a in restored]!=[fingerprint(a) for a in assets]:raise RuntimeError('Engineering project restore failed')
    for i,row in enumerate(saved_history):
        source,target=restored[2*i:2*i+2];receipt=row['result']
        replay=run_engineering(source,target,receipt['recipe'])
        if replay['result']!=receipt['result']:raise RuntimeError('Engineering recipe replay changed the result')
        if receipt['tool'] in ('datum321','rps','adjustment'):
            apply_engineering(source,target,replay)
            if run_engineering(source,target,receipt['recipe'])['result']['rms_mm']>1e-7:raise RuntimeError('Applied datum adjustment did not satisfy constraints')
    return dict(tools_passed=len(history),project_replay_verified=True,known_answers=answers,
                interpretation='Deterministic geometric examples; not production or certified GD&T validation.')
