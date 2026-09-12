"""Real desktop interactions; these run on both display-enabled CI platforms."""
import os
import sys
import json
import pytest

pytestmark=[pytest.mark.gui,pytest.mark.skipif(sys.platform.startswith('linux') and not os.environ.get('DISPLAY'),reason='Requires real desktop display')]


@pytest.fixture
def workspace(qtbot,monkeypatch):
    from assembly_workbench.gui import MainWindow
    window=MainWindow();qtbot.addWidget(window);window.show()
    errors=[]
    monkeypatch.setattr(window,'_show_error',lambda title,detail:errors.append((title,detail)))
    window.open_engineering()
    dialog=window.engineering_dialog;qtbot.addWidget(dialog)
    yield window,dialog,errors
    qtbot.waitUntil(lambda:not window.busy,timeout=30000)
    dialog.close();window.close()


def test_datum_compute_apply_history_and_stale_protection(workspace,qtbot):
    from assembly_workbench.engineering import run_engineering,apply_engineering
    import numpy as np
    window,dialog,errors=workspace
    dialog.load_example('datum321')
    before=dialog.source_asset().transform.copy()
    dialog.run_current()
    qtbot.waitUntil(lambda:not window.busy,timeout=30000)
    assert not errors and dialog.receipt['result']['converged']
    np.testing.assert_array_equal(before,dialog.source_asset().transform)
    assert dialog.apply_button.isEnabled()
    dialog.apply_current()
    assert not np.array_equal(before,dialog.source_asset().transform)
    assert not dialog.apply_button.isEnabled()
    assert any(x['type']=='engineering' for x in window.history)


def test_feature_measurement_and_recipe_roundtrip(workspace,qtbot,tmp_path):
    from assembly_workbench.engineering import export_engineering
    window,dialog,errors=workspace
    dialog.load_example('feature');dialog.run_current()
    qtbot.waitUntil(lambda:not window.busy,timeout=30000)
    assert not errors
    assert dialog.receipt['result']['radius_mm']==pytest.approx(5.15,abs=1e-5)
    assert dialog.results_table.rowCount()>0
    receipt=dialog.receipt
    recipe=dialog.current_recipe()
    dialog.set_recipe(json.loads(json.dumps(recipe)))
    assert dialog.current_recipe()==recipe
    assert dialog.receipt is None
    files=export_engineering(tmp_path,dialog.source_asset(),dialog.target_asset(),receipt)
    assert all(x.exists() for x in files)


def test_gap_and_inspection_examples_show_measured_rows(workspace,qtbot):
    window,dialog,errors=workspace
    dialog.load_example('gap_flush');dialog.run_current()
    qtbot.waitUntil(lambda:not window.busy,timeout=30000)
    assert dialog.receipt['result']['gap_mm']==pytest.approx([4,4])
    dialog.load_example('inspection');dialog.run_current()
    qtbot.waitUntil(lambda:not window.busy,timeout=30000)
    assert not errors
    assert dialog.receipt['result']['statistics']['missing_count']==1
    assert 'missing' in [row['status'] for row in dialog.receipt['result']['rows']]


def test_busy_dialog_cannot_close_early(workspace,qtbot):
    import threading
    window,dialog,errors=workspace
    gate=threading.Event()
    window._start_operation('controlled job',lambda:gate.wait(3),lambda _:None)
    try:
        dialog.close()
        assert dialog.isVisible()
    finally:gate.set()
    qtbot.waitUntil(lambda:not window.busy,timeout=5000)


def test_layout_keeps_actions_visible_at_minimum_size(workspace,qtbot):
    window,dialog,_=workspace
    dialog.resize(1060,660);qtbot.wait(100)
    assert dialog.run_button.isVisible() and dialog.export_button.isVisible()
    assert dialog.rect().contains(dialog.run_button.mapTo(dialog,dialog.run_button.rect().center()))


def test_recipe_import_rejects_clamping_and_preserves_current_inputs(workspace):
    _,dialog,_=workspace
    dialog.load_example('section');before=dialog.current_recipe()
    with pytest.raises(ValueError):dialog.set_recipe(before|{'slab_mm':1e9})
    assert dialog.current_recipe()==before
    with pytest.raises(ValueError):dialog.set_recipe(before|{'slab_mm':.1234567890123456})
    assert dialog.current_recipe()==before


def test_integer_and_table_structure_edits_invalidate_receipts(workspace,qtbot):
    window,dialog,errors=workspace
    dialog.load_example('detect');dialog.run_current()
    qtbot.waitUntil(lambda:not window.busy,timeout=30000)
    assert not errors and dialog.receipt
    dialog.detect_seed.setValue(11)
    assert dialog.receipt is None and not dialog.export_button.isEnabled()
    dialog.load_example('gap_flush');dialog.run_current()
    qtbot.waitUntil(lambda:not window.busy,timeout=30000)
    assert dialog.receipt
    dialog.gap_table.removeRow(0)
    assert dialog.receipt is None


def test_emma_control_import_preserves_meter_input_units(workspace,qtbot,monkeypatch,tmp_path):
    import csv
    from assembly_workbench.emma import REQUIRED,load_emma
    from PySide6.QtWidgets import QFileDialog
    window,dialog,errors=workspace
    fields=list(REQUIRED)+[f'TolAxisPos{a}.{b}ToleranceValue' for a in 'XYZ' for b in ('Lower','Upper')]
    nominal=dict(zip(REQUIRED,['task','plan','part','P1','PNT',0,0,0,'MPT','','']))
    nominal.update({'TolAxisPosZ.LowerToleranceValue':-.0005,'TolAxisPosZ.UpperToleranceValue':.0005})
    actual=nominal|{'IPE.Origin.Z':.0002,'InspectionCategories':'A','Component.ID':'sample','History.DateTime':'2026-01-01'}
    path=tmp_path/'meter.csv'
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows([nominal,actual])
    batch=load_emma(path,unit='m');window._finish_import(batch)
    source=next(a for a in batch.assets if '实测' in a.name);target=next(a for a in batch.assets if '名义' in a.name)
    dialog.refresh_assets(source.id,target.id);dialog.tool_combo.setCurrentIndex(dialog.tool_combo.findData('inspection'))
    assert dialog.control_unit.currentData()=='m'
    monkeypatch.setattr(QFileDialog,'getOpenFileName',lambda *a,**k:(str(path),''))
    dialog.import_controls();qtbot.waitUntil(lambda:not window.busy,timeout=30000)
    assert dialog.current_recipe()['controls'][0]['upper_mm']==.5
    dialog.run_current();qtbot.waitUntil(lambda:not window.busy,timeout=30000)
    assert not errors and dialog.receipt['result']['overall']=='pass'


def test_capture_failure_waits_for_native_worker_before_exit(workspace,qtbot):
    import threading
    from assembly_workbench.gui import _exit_after_work
    window,_,_=workspace;gate=threading.Event();exits=[]
    class AppProbe:
        def quitOnLastWindowClosed(self):return True
        def setQuitOnLastWindowClosed(self,value):pass
        def exit(self,code):exits.append(code)
    window._start_operation('capture-timeout probe',lambda:gate.wait(3),lambda _:None)
    try:
        _exit_after_work(window,AppProbe(),1)
        qtbot.wait(80);assert not exits and window.busy
    finally:gate.set()
    qtbot.waitUntil(lambda:exits==[1],timeout=5000)
