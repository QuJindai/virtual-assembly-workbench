"""Chinese dimensional-engineering workspace over the shared numerical facade."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from PySide6.QtCore import Qt,QTimer
from PySide6.QtWidgets import (QCheckBox,QComboBox,QDialog,QDialogButtonBox,QDoubleSpinBox,
    QFileDialog,QFormLayout,QHBoxLayout,QHeaderView,QLabel,QLineEdit,QListWidget,
    QMessageBox,QPushButton,QScrollArea,QSpinBox,QSplitter,QStackedWidget,QTableWidget,
    QTableWidgetItem,QVBoxLayout,QWidget)

from .core import Dataset
from .engineering import LABELS,TOOLS,run_engineering,apply_engineering,export_engineering,result_rows
from .engineering_examples import example
from .viewport import AssemblyViewport


class _NumberSpin(QDoubleSpinBox):
    def textFromValue(self,value):
        text=f'{value:.{self.decimals()}f}'.rstrip('0').rstrip('.')
        return text if text not in ('','-0') else '0'


def _spin(value=0.,lo=-1e7,hi=1e7,decimals=12):
    s=_NumberSpin();s.setRange(lo,hi);s.setDecimals(decimals);s.setValue(value);return s


def _vector(values):
    widget=QWidget();layout=QHBoxLayout(widget);layout.setContentsMargins(0,0,0,0)
    spins=[_spin(v) for v in values]
    for axis,spin in zip('XYZ',spins):layout.addWidget(QLabel(axis));layout.addWidget(spin)
    return widget,spins


def _get_vector(spins):return [s.value() for s in spins]


def _set_vector(spins,values):
    for s,v in zip(spins,values,strict=True):s.setValue(float(v))


def _selection(text):
    text=text.strip()
    if not text:return None
    if text.endswith('*') and ',' not in text:return {'prefix':text[:-1]}
    return [s.strip() for s in text.replace('，',',').replace('\n',',').split(',') if s.strip()]


def _selection_text(value):
    return '' if value is None else value['prefix']+'*' if isinstance(value,dict) else ','.join(value)


def _table(headers,rows=0):
    t=QTableWidget(rows,len(headers));t.setHorizontalHeaderLabels(headers)
    t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
    t.horizontalHeader().setStretchLastSection(True);t.setMinimumHeight(175)
    return t


def _write_table(table,rows):
    table.setRowCount(len(rows))
    for i,row in enumerate(rows):
        for j,value in enumerate(row):table.setItem(i,j,QTableWidgetItem('' if value is None else str(value)))


def _cell(table,row,col,default=''):
    item=table.item(row,col);return item.text().strip() if item else default


class EngineeringDialog(QDialog):
    def __init__(self,window):
        super().__init__(window)
        self.window=window;self.receipt=None;self._setting=False;self.recipe_name=None;self._view_key=None
        self.setWindowTitle('尺寸工程 · 基准 / 特征 / 装调')
        self.setWindowModality(Qt.WindowModality.WindowModal)
        self.resize(1320,840);self.setMinimumSize(1000,640)
        layout=QVBoxLayout(self)
        heading=QLabel('尺寸工程  |  基准定义 → 测量判定 → 装调建议 → 复测')
        heading.setObjectName('sectionTitle');layout.addWidget(heading)
        selectors=QHBoxLayout();self.source_combo=QComboBox();self.target_combo=QComboBox()
        for label,combo in [('移动 / 实测',self.source_combo),('参考 / 名义',self.target_combo)]:
            selectors.addWidget(QLabel(label));selectors.addWidget(combo,1)
            combo.setMinimumContentsLength(12);combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            combo.currentIndexChanged.connect(self.invalidate)
        layout.addLayout(selectors)
        bar=QHBoxLayout();self.tool_combo=QComboBox()
        for key,label in LABELS.items():self.tool_combo.addItem(label,key)
        self.tool_combo.currentIndexChanged.connect(self._tool_changed)
        bar.addWidget(QLabel('工具'));bar.addWidget(self.tool_combo,1)
        self.example_button=QPushButton('新增合成示例');self.example_button.clicked.connect(lambda:self.load_example(self.tool_combo.currentData()))
        bar.addWidget(self.example_button)
        for text,callback in [('载入方案',self.import_recipe),('保存方案',self.save_recipe)]:
            button=QPushButton(text);button.clicked.connect(callback);bar.addWidget(button)
        layout.addLayout(bar)
        splitter=QSplitter(Qt.Orientation.Horizontal);layout.addWidget(splitter,1)
        self.input_scroll=QScrollArea();self.input_scroll.setWidgetResizable(True);self.input_scroll.setMinimumWidth(460)
        self.inputs=QWidget();inputs_layout=QVBoxLayout(self.inputs)
        self.hint=QLabel();self.hint.setWordWrap(True);self.hint.setObjectName('hint');inputs_layout.addWidget(self.hint)
        self.pages=QStackedWidget();inputs_layout.addWidget(self.pages)
        self.pages.addWidget(self._build_pairs());self.pages.addWidget(self._build_features())
        self.pages.addWidget(self._build_gap());self.pages.addWidget(self._build_geometry());self.pages.addWidget(self._build_inspection())
        inputs_layout.addStretch();self.input_scroll.setWidget(self.inputs);splitter.addWidget(self.input_scroll)
        right=QSplitter(Qt.Orientation.Vertical)
        self.viewport=AssemblyViewport();right.addWidget(self.viewport)
        results=QWidget();results_layout=QVBoxLayout(results)
        self.summary=QLabel('选择工具和数据后运行。');self.summary.setObjectName('resultCard');self.summary.setWordWrap(True);results_layout.addWidget(self.summary)
        self.results_table=_table([],0);self.results_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.results_table.setAlternatingRowColors(True);results_layout.addWidget(self.results_table,1)
        right.addWidget(results);right.setSizes([440,240]);splitter.addWidget(right);splitter.setSizes([500,790]);splitter.setStretchFactor(1,1)
        history=QHBoxLayout();self.history_combo=QComboBox();history.addWidget(QLabel('历史方案'));history.addWidget(self.history_combo,1)
        replay=QPushButton('载入并重算');replay.clicked.connect(self.replay_history);history.addWidget(replay);layout.addLayout(history)
        actions=QHBoxLayout();self.run_button=QPushButton('运行计算');self.run_button.setObjectName('primaryButton');self.run_button.clicked.connect(self.run_current)
        self.apply_button=QPushButton('应用装调 / 基准变换');self.apply_button.clicked.connect(self.apply_current)
        self.export_button=QPushButton('导出工程报告');self.export_button.clicked.connect(self.export_current)
        self.section_button=QPushButton('截面加入资源');self.section_button.clicked.connect(self.add_section)
        self.save_project_button=QPushButton('保存工程');self.save_project_button.clicked.connect(window.save_project)
        self.close_button=QPushButton('关闭');self.close_button.clicked.connect(self.close)
        for b in (self.run_button,self.apply_button,self.export_button,self.section_button,self.save_project_button,self.close_button):actions.addWidget(b)
        layout.addLayout(actions)
        self.pair_table.itemChanged.connect(self.invalidate);self.gap_table.itemChanged.connect(self.invalidate);self.control_table.itemChanged.connect(self.invalidate)
        for table in (self.pair_table,self.gap_table,self.control_table):
            table.model().rowsRemoved.connect(self.invalidate);table.model().rowsInserted.connect(self.invalidate)
        for widget in self.inputs.findChildren(QDoubleSpinBox):widget.valueChanged.connect(self.invalidate)
        for widget in self.inputs.findChildren(QSpinBox):widget.valueChanged.connect(self.invalidate)
        for widget in self.inputs.findChildren(QLineEdit):widget.textEdited.connect(self.invalidate)
        for widget in self.inputs.findChildren(QComboBox):widget.currentIndexChanged.connect(self.invalidate)
        for widget in self.inputs.findChildren(QCheckBox):widget.toggled.connect(self.invalidate)
        self.refresh_assets();self.refresh_history();self._tool_changed()
        self.target_combo.currentIndexChanged.connect(self._sync_control_unit)

    def _build_pairs(self):
        page=QWidget();layout=QVBoxLayout(page);layout.setContentsMargins(0,0,0,0)
        self.pair_table=_table(['移动测点','参考测点','Nx','Ny','Nz','权重','目标偏置 mm'],6)
        _write_table(self.pair_table,[['','',*n,1,0] for n in ([[0,0,1]]*3+[[0,1,0]]*2+[[1,0,0]])])
        layout.addWidget(self.pair_table)
        buttons=QHBoxLayout()
        for text,callback in [('增加行',lambda:self._append_pair()),('删除行',lambda:self.pair_table.removeRow(max(self.pair_table.currentRow(),0))),
                              ('选移动点',lambda:self.pick_point('source')),('选参考点',lambda:self.pick_point('target'))]:
            b=QPushButton(text);b.clicked.connect(callback);buttons.addWidget(b)
        layout.addLayout(buttons)
        self.constraint_panel=QWidget();form=QFormLayout(self.constraint_panel)
        degrees=QWidget();d_layout=QHBoxLayout(degrees);d_layout.setContentsMargins(0,0,0,0);self.dofs=[]
        for name in ('TX','TY','TZ','RX','RY','RZ'):
            box=QCheckBox(name);box.setChecked(True);d_layout.addWidget(box);self.dofs.append(box)
        form.addRow('允许调整',degrees)
        self.translation_limit=_spin(10,.00001,1e6);self.rotation_limit=_spin(5,.00001,180)
        form.addRow('各轴平移上限 mm',self.translation_limit);form.addRow('各轴转角上限 °',self.rotation_limit)
        self.auto_pivot=QCheckBox('以所选移动测点质心为转动中心');self.auto_pivot.setChecked(True);form.addRow(self.auto_pivot)
        w,self.pivot_spins=_vector([0,0,0]);form.addRow('指定转动中心 mm',w)
        layout.addWidget(self.constraint_panel);return page

    def _append_pair(self):
        row=self.pair_table.rowCount();self.pair_table.insertRow(row)
        for i,v in enumerate(['','',0,0,1,1,0]):self.pair_table.setItem(row,i,QTableWidgetItem(str(v)))

    def _build_features(self):
        page=QWidget();form=QFormLayout(page);form.setContentsMargins(0,0,0,0)
        self.feature_kind=QComboBox()
        for label,key in [('平面','plane'),('直线','line'),('圆 / 孔缘','circle'),('球','sphere'),('圆柱','cylinder')]:self.feature_kind.addItem(label,key)
        form.addRow('特征类型',self.feature_kind)
        self.feature_source=QLineEdit();self.feature_source.setPlaceholderText('留空=全部；例如 RING_* 或 #1,#2,#3')
        self.feature_target=QLineEdit();self.feature_target.setPlaceholderText('参考侧选点，同上')
        form.addRow('移动侧选点',self.feature_source);form.addRow('参考侧选点',self.feature_target)
        self.feature_compare=QCheckBox('同时拟合参考件并比较尺寸');self.feature_compare.setChecked(True);form.addRow(self.feature_compare)
        self.detect_panel=QWidget();f=QFormLayout(self.detect_panel);self.detect_threshold=_spin(.1,.000001,10000)
        self.detect_min=QSpinBox();self.detect_min.setRange(3,200000);self.detect_min.setValue(20)
        self.detect_max=QSpinBox();self.detect_max.setRange(1,20);self.detect_max.setValue(5)
        self.detect_seed=QSpinBox();self.detect_seed.setRange(0,2**31-1)
        f.addRow('候选内点阈值 mm',self.detect_threshold);f.addRow('最少内点',self.detect_min);f.addRow('最多候选',self.detect_max);f.addRow('可复现随机种子',self.detect_seed)
        form.addRow(self.detect_panel);return page

    def _build_gap(self):
        page=QWidget();layout=QVBoxLayout(page);layout.setContentsMargins(0,0,0,0)
        self.gap_table=_table(['移动边缘测点','参考边缘测点'],2);layout.addWidget(self.gap_table)
        row=QHBoxLayout()
        for label,cb in [('增加行',lambda:self.gap_table.insertRow(self.gap_table.rowCount())),('删除行',lambda:self.gap_table.removeRow(max(self.gap_table.currentRow(),0))),('选移动点',lambda:self.pick_point('source')),('选参考点',lambda:self.pick_point('target'))]:
            b=QPushButton(label);b.clicked.connect(cb);row.addWidget(b)
        layout.addLayout(row);form=QFormLayout();w,self.gap_axis=_vector([1,0,0]);form.addRow('正向间隙方向',w)
        w,self.flush_axis=_vector([0,0,1]);form.addRow('正向面差方向',w);layout.addLayout(form);return page

    def _build_geometry(self):
        page=QWidget();form=QFormLayout(page);form.setContentsMargins(0,0,0,0)
        w,self.section_origin=_vector([0,0,0]);form.addRow('截面经过点 mm',w)
        w,self.section_normal=_vector([0,0,1]);form.addRow('截面法向',w)
        self.section_slab=_spin(.2,.000001,10000);form.addRow('点云薄层全厚度 mm',self.section_slab)
        self.section_max=QSpinBox();self.section_max.setRange(3,200000);self.section_max.setValue(20000);form.addRow('截面点上限',self.section_max)
        self.contact_tolerance=_spin(.00001,.0000001,1,7);form.addRow('实体接触容差 mm',self.contact_tolerance)
        return page

    def _build_inspection(self):
        page=QWidget();layout=QVBoxLayout(page);layout.setContentsMargins(0,0,0,0)
        self.control_table=_table(['控制名称','测点编号','Nx','Ny','Nz','下限 mm','上限 mm'],1)
        layout.addWidget(self.control_table);row=QHBoxLayout()
        for label,cb in [('增加行',lambda:self.control_table.insertRow(self.control_table.rowCount())),('删除行',lambda:self.control_table.removeRow(max(self.control_table.currentRow(),0))),('导入控制CSV',self.import_controls)]:
            b=QPushButton(label);b.clicked.connect(cb);row.addWidget(b)
        layout.addLayout(row)
        self.reference_axes=QCheckBox('方向随名义资源坐标系转动（eMMA 导入自动启用）');layout.addWidget(self.reference_axes)
        unit_row=QHBoxLayout();unit_row.addWidget(QLabel('待导入 eMMA 原文件单位'))
        self.control_unit=QComboBox();self.control_unit.addItem('毫米 (mm)','mm');self.control_unit.addItem('米 (m)','m');unit_row.addWidget(self.control_unit);layout.addLayout(unit_row)
        note=QLabel('上下限针对“实测−名义”的方向偏差；单侧留空为单侧限值，双侧留空不判定。支持原始 eMMA CSV，或表头为 name,point_id,nx,ny,nz,lower_mm,upper_mm 的控制表。')
        note.setWordWrap(True);layout.addWidget(note);return page

    def source_asset(self):return self.window._asset_by_id(self.source_combo.currentData())
    def target_asset(self):return self.window._asset_by_id(self.target_combo.currentData())

    def refresh_assets(self,source_id=None,target_id=None):
        self._setting=True
        for combo,chosen,fallback in [(self.source_combo,source_id,self.window.source_combo.currentData()),(self.target_combo,target_id,self.window.target_combo.currentData())]:
            combo.clear()
            for a in self.window.assets:combo.addItem(a.name,a.id)
            index=combo.findData(chosen or fallback);combo.setCurrentIndex(max(0,index))
        self._setting=False;self._sync_control_unit();self.invalidate()

    def _sync_control_unit(self):
        target=self.target_asset()
        if target is None or not hasattr(self,'control_unit'):return
        for row in reversed(self.window.history):
            result=row.get('result',{})
            if result.get('schema')=='emma-import/1' and any(g.get('asset_id')==target.id for g in result.get('groups',[])):
                self.control_unit.setCurrentIndex(self.control_unit.findData(result.get('input_unit','mm')))
                return

    def refresh_history(self):
        self.history_combo.clear()
        for i,row in enumerate(self.window.history):
            receipt=row.get('result',{})
            if row.get('type')=='engineering' and 'recipe' in receipt:
                self.history_combo.addItem(f"{LABELS.get(receipt.get('tool'),'工程分析')} · {row.get('source','')} · {receipt.get('created_at','')[:19]}",i)

    def _tool_changed(self):
        tool=self.tool_combo.currentData()
        page=0 if tool in ('datum321','rps','adjustment') else 1 if tool in ('feature','detect') else 2 if tool=='gap_flush' else 3 if tool in ('section','contact') else 4
        self.pages.setCurrentIndex(page);self.constraint_panel.setVisible(tool in ('rps','adjustment'))
        for col in range(2,7):self.pair_table.setColumnHidden(col,tool=='datum321' or (col==6 and tool!='adjustment'))
        self.detect_panel.setVisible(tool=='detect');self.feature_compare.setVisible(tool=='feature');self.feature_target.setEnabled(tool=='feature')
        hints={
            'datum321':'按 A1、A2、A3、B1、B2、C1 顺序选择6对基准点。主基准约束Z，次基准约束Y，第三基准约束X。点序定义方向，不会自动指定生产基准。',
            'rps':'按每个测点的约束方向进行加权定位。仅勾选的自由度可变化；约束不足时拒绝计算。方向使用当前世界坐标系。',
            'adjustment':'定义门盖/零件的调整自由度、转动中心和行程上限。目标偏置为相对参考点沿该行方向的期望值。结果仅预测刚体移动后的坐标。',
            'feature':'选择同一几何特征的表面或边缘点再拟合。孔中心测点不能用来拟合孔径。角度、中心差和直径差是几何量，尚未执行完整GD&T。',
            'detect':'自动提出平面或圆候选，保留内点与拟合误差；候选需按工艺确认，不会自动命名为真实零件孔。仅支持平面和圆。',
            'gap_flush':'选择同一测量截面中的两侧对应边缘点。间隙与面差方向必须垂直；数值为“移动边−参考边”的有符号投影。负间隙不等同实体干涉。',
            'section':'CAD/网格与指定平面求交；点云仅提取薄层内的已有测点。截面可加入资源继续选点分析；薄层点不伪装成精确曲线。',
            'contact':'仅支持有效闭合CAD实体。用OCCT最小距离和公共体积判断分离、接触和干涉；点云、开放曲面不作实体判定。',
            'inspection':'按同一范围中的测点编号筛查方向偏差。缺点与无公差分开标记；结果不替代工艺放行。'}
        self.hint.setText(hints[tool]);self.invalidate()

    def invalidate(self,*_):
        if self._setting:return
        self.receipt=None
        if hasattr(self,'summary'):self.summary.setText('输入或选择已改变，请运行计算。')
        if hasattr(self,'viewport'):
            source,target=self.source_asset(),self.target_asset()
            selected=[a for a in (source,target) if a is not None]
            key=tuple((a.id,tuple(a.transform.ravel())) for a in selected)
            if selected and key!=self._view_key:
                self._view_key=key
                if len(selected)==2 and selected[0].id==selected[1].id:selected=selected[:1]
                self.viewport.set_assets(selected,source.id if source else None,target.id if target else None,None)
                self.viewport.fit_view()
        self.refresh_busy()

    def refresh_busy(self):
        if not hasattr(self,'run_button'):return
        busy=self.window.busy;pair=self.source_asset() is not None and self.target_asset() is not None and self.source_combo.currentData()!=self.target_combo.currentData()
        for w in (self.inputs,self.source_combo,self.target_combo,self.tool_combo,self.example_button,self.history_combo):w.setEnabled(not busy)
        self.run_button.setEnabled(not busy and pair)
        self.export_button.setEnabled(not busy and self.receipt is not None)
        can_apply=self.receipt is not None and self.receipt['tool'] in ('datum321','rps','adjustment') and self.receipt['result'].get('converged') is True
        self.apply_button.setEnabled(not busy and can_apply)
        self.section_button.setEnabled(not busy and self.receipt is not None and self.receipt['tool']=='section')
        self.save_project_button.setEnabled(not busy and bool(self.window.assets));self.close_button.setEnabled(not busy)

    def current_recipe(self):
        tool=self.tool_combo.currentData();r={'tool':tool}
        if self.recipe_name is not None:r['name']=self.recipe_name
        if tool in ('datum321','rps','adjustment'):
            rows=self.pair_table.rowCount();r.update(source_points=[_cell(self.pair_table,i,0) for i in range(rows)],target_points=[_cell(self.pair_table,i,1) for i in range(rows)])
            if tool!='datum321':
                r.update(normals=[[float(_cell(self.pair_table,i,j)) for j in (2,3,4)] for i in range(rows)],weights=[float(_cell(self.pair_table,i,5,'1')) for i in range(rows)],
                         dofs=[x.isChecked() for x in self.dofs],pivot=None if self.auto_pivot.isChecked() else _get_vector(self.pivot_spins),translation_limit_mm=self.translation_limit.value(),rotation_limit_deg=self.rotation_limit.value())
            if tool=='adjustment':r['offsets_mm']=[float(_cell(self.pair_table,i,6,'0')) for i in range(rows)]
        elif tool in ('feature','detect'):
            r.update(kind=self.feature_kind.currentData(),source_points=_selection(self.feature_source.text()))
            if tool=='feature':r.update(target_points=_selection(self.feature_target.text()),compare=self.feature_compare.isChecked())
            else:r.update(threshold_mm=self.detect_threshold.value(),min_points=self.detect_min.value(),max_features=self.detect_max.value(),seed=self.detect_seed.value())
        elif tool=='gap_flush':
            rows=self.gap_table.rowCount();r.update(source_points=[_cell(self.gap_table,i,0) for i in range(rows)],target_points=[_cell(self.gap_table,i,1) for i in range(rows)],gap_axis=_get_vector(self.gap_axis),flush_axis=_get_vector(self.flush_axis))
        elif tool=='section':r.update(origin=_get_vector(self.section_origin),normal=_get_vector(self.section_normal),slab_mm=self.section_slab.value(),max_points=self.section_max.value())
        elif tool=='contact':r['tolerance_mm']=self.contact_tolerance.value()
        else:
            controls=[]
            for i in range(self.control_table.rowCount()):
                values=[_cell(self.control_table,i,j) for j in range(7)]
                controls.append(dict(name=values[0],point_id=values[1],axis=[float(x) for x in values[2:5]],lower_mm=float(values[5]) if values[5] else None,upper_mm=float(values[6]) if values[6] else None))
            r['controls']=controls;r['axis_frame']='reference_local' if self.reference_axes.isChecked() else 'world'
        return r

    def _validate_recipe_for_ui(self,r):
        if not isinstance(r,dict) or r.get('tool') not in LABELS:raise ValueError('无效的工程方案。')
        json.dumps(r,allow_nan=False)
        tool=r['tool']
        if set(r)-TOOLS[tool]-{'tool','name'}:raise ValueError('方案含有当前工具不支持的字段。')
        if 'name' in r and not isinstance(r['name'],str):raise ValueError('方案名称必须为文本。')
        def spin(w,v):
            if isinstance(v,bool) or not isinstance(v,(int,float)) or not w.minimum()<=v<=w.maximum():raise ValueError('方案数值超出界面支持范围。')
            digits=w.decimals() if isinstance(w,QDoubleSpinBox) else 0
            if round(v,digits)!=v:raise ValueError('方案精度超过界面支持范围；请通过命令行运行，避免数值被舍入。')
        def vector(ws,vs):
            for w,v in zip(ws,vs,strict=True):spin(w,v)
        for k in ('source_points','target_points'):
            value=r.get(k)
            if value is not None:
                if isinstance(value,dict):
                    if set(value)!={'prefix'} or not isinstance(value['prefix'],str) or not value['prefix']:raise ValueError('选区前缀无效。')
                elif not isinstance(value,list) or not value or any(not isinstance(x,str) or not x.strip() for x in value):raise ValueError('测点选择必须为非空编号列表。')
                if tool in ('feature','detect') and _selection(_selection_text(value))!=value:raise ValueError('选区编号无法无损表示，请使用命令行方案。')
        if tool in ('datum321','rps','adjustment','gap_flush'):
            if not isinstance(r.get('source_points'),list) or not isinstance(r.get('target_points'),list) or len(r['source_points'])!=len(r['target_points']):raise ValueError('必须逐行指定数量一致的配对测点。')
        if tool in ('rps','adjustment'):
            n=len(r['source_points'])
            normals=r.get('normals',[])
            if np.asarray(normals).shape!=(n,3):raise ValueError('每对测点必须有一个三维约束方向。')
            for k in ('weights','offsets_mm'):
                if k in r and len(r[k])!=n:raise ValueError('权重或偏置数量与测点不一致。')
            dofs=r.get('dofs',[True]*6)
            if not isinstance(dofs,list) or len(dofs)!=6 or any(type(x) is not bool for x in dofs):raise ValueError('自由度必须为6个布尔值。')
            spin(self.translation_limit,r.get('translation_limit_mm',100));spin(self.rotation_limit,r.get('rotation_limit_deg',30))
            if r.get('pivot') is not None:vector(self.pivot_spins,r['pivot'])
        if tool in ('feature','detect'):
            if self.feature_kind.findData(r.get('kind','plane'))<0:raise ValueError('未知特征类型。')
            if 'compare' in r and type(r['compare']) is not bool:raise ValueError('比较选项必须为布尔值。')
            if tool=='detect':
                for w,k,d in [(self.detect_threshold,'threshold_mm',.1),(self.detect_min,'min_points',20),(self.detect_max,'max_features',5),(self.detect_seed,'seed',0)]:spin(w,r.get(k,d))
        if tool=='gap_flush':vector(self.gap_axis,r['gap_axis']);vector(self.flush_axis,r['flush_axis'])
        if tool=='section':
            vector(self.section_origin,r['origin']);vector(self.section_normal,r['normal']);spin(self.section_slab,r.get('slab_mm',.2));spin(self.section_max,r.get('max_points',20000))
        if tool=='contact':spin(self.contact_tolerance,r.get('tolerance_mm',1e-5))
        if tool=='inspection':
            if r.get('axis_frame','world') not in ('world','reference_local'):raise ValueError('未知的控制方向坐标系。')
            if not isinstance(r.get('controls'),list) or not 1<=len(r['controls'])<=50000:raise ValueError('请定义1–50000项控制。')
            for c in r['controls']:
                if not isinstance(c,dict) or set(c)-{'name','point_id','axis','lower_mm','upper_mm'} or not {'name','point_id','axis'}<=set(c) or len(c['axis'])!=3:raise ValueError('控制字段无效。')

    def set_recipe(self,r):
        if self.window.busy:raise ValueError('正在计算，请等待完成。')
        self._validate_recipe_for_ui(r)
        self._setting=True
        try:
            self.recipe_name=r.get('name')
            self.tool_combo.setCurrentIndex(self.tool_combo.findData(r['tool']));tool=r['tool']
            if tool in ('datum321','rps','adjustment'):
                n=len(r['source_points']);normals=r.get('normals',[[0,0,1]]*n);weights=r.get('weights',[1]*n);offsets=r.get('offsets_mm',[0]*n)
                _write_table(self.pair_table,[[a,b,*axis,w,o] for a,b,axis,w,o in zip(r['source_points'],r['target_points'],normals,weights,offsets,strict=True)])
                for box,value in zip(self.dofs,r.get('dofs',[True]*6),strict=True):box.setChecked(bool(value))
                self.auto_pivot.setChecked(r.get('pivot') is None);_set_vector(self.pivot_spins,r.get('pivot') or [0,0,0])
                self.translation_limit.setValue(r.get('translation_limit_mm',100));self.rotation_limit.setValue(r.get('rotation_limit_deg',30))
            elif tool in ('feature','detect'):
                self.feature_kind.setCurrentIndex(self.feature_kind.findData(r.get('kind','plane')));self.feature_source.setText(_selection_text(r.get('source_points')));self.feature_target.setText(_selection_text(r.get('target_points')))
                self.feature_compare.setChecked(r.get('compare',False));self.detect_threshold.setValue(r.get('threshold_mm',.1));self.detect_min.setValue(r.get('min_points',20));self.detect_max.setValue(r.get('max_features',5));self.detect_seed.setValue(r.get('seed',0))
            elif tool=='gap_flush':
                _write_table(self.gap_table,list(zip(r['source_points'],r['target_points'],strict=True)));_set_vector(self.gap_axis,r['gap_axis']);_set_vector(self.flush_axis,r['flush_axis'])
            elif tool=='section':
                _set_vector(self.section_origin,r['origin']);_set_vector(self.section_normal,r['normal']);self.section_slab.setValue(r.get('slab_mm',.2));self.section_max.setValue(r.get('max_points',20000))
            elif tool=='contact':self.contact_tolerance.setValue(r.get('tolerance_mm',1e-5))
            else:
                _write_table(self.control_table,[[c['name'],c['point_id'],*c['axis'],c.get('lower_mm'),c.get('upper_mm')] for c in r['controls']])
                self.reference_axes.setChecked(r.get('axis_frame','world')=='reference_local')
        finally:self._setting=False
        self._tool_changed()

    def run_current(self):
        if self.window.busy:return
        try:
            recipe=self.current_recipe();source,target=self.source_asset(),self.target_asset()
            self.receipt=None;self.summary.setText('正在进行真实几何计算…')
            self.window._start_operation('尺寸工程计算中…',lambda:run_engineering(source,target,recipe),self._completed)
        except (ValueError,TypeError,KeyError) as e:self.window._show_error('输入无效',str(e))

    def _completed(self,receipt):
        self.receipt=receipt;r=receipt['result'];rows=result_rows(receipt)
        columns=list(dict.fromkeys(k for row in rows for k in row))
        self.results_table.setColumnCount(len(columns));self.results_table.setHorizontalHeaderLabels(columns)
        _write_table(self.results_table,[[json.dumps(row.get(k),ensure_ascii=False) if isinstance(row.get(k),(dict,list)) else row.get(k) for k in columns] for row in rows[:2000]])
        summary=LABELS[receipt['tool']]+'完成'
        if 'rms_mm' in r:summary+=f" · RMS {r['rms_mm']:.5f} mm"
        if 'radius_mm' in r:summary+=f" · 直径 {2*r['radius_mm']:.5f} mm"
        if 'rank' in r:summary+=f" · 约束秩 {r['rank']}"
        if any(r.get('at_bounds',[])):summary+=' · 部分调整量达到行程上限'
        if 'transform' in r:
            from scipy.spatial.transform import Rotation
            if 'parameters' in r:
                parameters=r['parameters'];origin='绕指定中心'
            else:
                transform=np.asarray(r['transform']);parameters=[*transform[:3,3],*Rotation.from_matrix(transform[:3,:3]).as_euler('xyz',degrees=True)];origin='绕世界原点'
            summary+='\n平移 XYZ：'+', '.join(f'{v:.5f}' for v in parameters[:3])+' mm；转角 XYZ：'+', '.join(f'{v:.5f}' for v in parameters[3:])+'°（'+origin+'）。'
        if 'state' in r:summary+=' · '+{'clear':'实体分离','contact':'实体接触','interference':'存在实体干涉'}.get(r['state'],r['state'])
        if receipt['tool']=='inspection':
            s=r['statistics'];summary+=f" · 满足 {s['pass_count']} / 超限 {s['fail_count']} / 缺测 {s['missing_count']} / 未判 {s['unjudged_count']}"
        if len(rows)>2000:summary+=f' · 表中显示前2000行，报告含全部{len(rows)}行'
        warnings=r.get('warnings',[])
        if warnings:summary+='\n'+'；'.join(str(x) for x in warnings)
        self.summary.setText(summary)
        self.window._append_history('engineering',receipt['source_name'],receipt['target_name'],summary,receipt)
        self.window.project_path=None;self.refresh_history();self.refresh_busy()

    def apply_current(self):
        if self.window.busy or self.receipt is None:return
        try:
            source,target=self.source_asset(),self.target_asset();receipt=self.receipt
            apply_engineering(source,target,receipt)
            self.window._append_history('transform',source.name,target.name,'已应用'+LABELS[receipt['tool']],dict(engineering_receipt_sha256=receipt['receipt_sha256'],transform=source.transform.tolist()))
            self.window._invalidate_results();self.window._update_asset_labels();self.window.project_path=None
            self.invalidate();self.summary.setText('变换已应用；请重新测量检查调整后的结果。')
        except ValueError as e:self.window._show_error('无法应用',str(e))

    def export_current(self):
        if self.window.busy or self.receipt is None:return
        folder=QFileDialog.getExistingDirectory(self,'选择工程报告目录')
        if folder:
            source,target,receipt=self.source_asset(),self.target_asset(),self.receipt
            self.window._start_operation('导出工程报告…',lambda:export_engineering(folder,source,target,receipt),lambda files:self.summary.setText('报告已保存：'+str(files[0].parent)))

    def add_section(self):
        if self.window.busy or self.receipt is None or self.receipt['tool']!='section':return
        from .engineering import validate_receipt
        try:
            validate_receipt(self.source_asset(),self.target_asset(),self.receipt)
            if len(self.window.assets)>=100:raise ValueError('工程已达到100个资源上限。')
            new=Dataset('截面 · '+self.source_asset().name,self.receipt['result']['points'])
            self.window.assets.append(new);self.window._refresh_asset_widgets();self.refresh_assets(new.id,self.target_combo.currentData())
            self.window.project_path=None
        except ValueError as e:self.window._show_error('截面无法加入',str(e))

    def load_example(self,tool):
        if self.window.busy:return
        if len(self.window.assets)>98:self.window._show_error('资源已满','请先移除部分资源。');return
        source,target,recipe=example(tool)
        self.window.assets.extend([source,target]);self.window._refresh_asset_widgets();self.window.project_path=None
        self.refresh_assets(source.id,target.id);self.set_recipe(recipe)

    def pick_point(self,side):
        if self.window.busy:return
        asset=self.source_asset() if side=='source' else self.target_asset()
        if asset is None:return
        chooser=QDialog(self);chooser.setWindowTitle('选择测点 · '+asset.name);chooser.resize(650,550)
        layout=QVBoxLayout(chooser);search=QLineEdit();search.setPlaceholderText('输入编号筛选；无编号点按#行号选择');layout.addWidget(search)
        points=asset.world_points();labels=asset.point_ids or [f'#{i+1}' for i in range(len(points))]
        items=QListWidget();layout.addWidget(items,1);note=QLabel();layout.addWidget(note)
        buttons=QDialogButtonBox(QDialogButtonBox.StandardButton.Ok|QDialogButtonBox.StandardButton.Cancel);buttons.accepted.connect(chooser.accept);buttons.rejected.connect(chooser.reject);layout.addWidget(buttons)
        def populate(text=''):
            items.clear();matches=[i for i,label in enumerate(labels) if text.lower() in label.lower()]
            for i in matches[:1000]:
                items.addItem(f'{labels[i]}    ({points[i,0]:.4f}, {points[i,1]:.4f}, {points[i,2]:.4f}) mm');items.item(items.count()-1).setData(Qt.ItemDataRole.UserRole,labels[i])
            note.setText(f'匹配 {len(matches)} 个；当前显示前 {min(1000,len(matches))} 个。输入完整编号可定位其他点。')
        search.textChanged.connect(populate);items.itemDoubleClicked.connect(lambda _:chooser.accept());populate()
        if chooser.exec()==QDialog.DialogCode.Accepted and items.currentItem():
            table=self.gap_table if self.tool_combo.currentData()=='gap_flush' else self.pair_table
            row=max(0,table.currentRow())
            if row>=table.rowCount():table.insertRow(row)
            table.setItem(row,0 if side=='source' else 1,QTableWidgetItem(items.currentItem().data(Qt.ItemDataRole.UserRole)))

    def import_recipe(self):
        if self.window.busy:return
        name,_=QFileDialog.getOpenFileName(self,'载入工程方案','','JSON (*.json)')
        if not name:return
        try:
            if Path(name).stat().st_size>16*1024*1024:raise ValueError('方案超过16 MiB上限。')
            data=json.loads(Path(name).read_text(encoding='utf-8-sig'));self.set_recipe(data.get('recipe',data))
        except (ValueError,OSError,KeyError,TypeError) as e:self.window._show_error('方案无法载入',str(e))

    def save_recipe(self):
        if self.window.busy:return
        try:recipe=self.current_recipe()
        except (ValueError,TypeError) as e:self.window._show_error('方案输入无效',str(e));return
        name,_=QFileDialog.getSaveFileName(self,'保存工程方案','engineering-recipe.json','JSON (*.json)')
        if name:
            try:Path(name).write_text(json.dumps(recipe,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
            except (ValueError,OSError) as e:self.window._show_error('无法保存方案',str(e))

    def import_controls(self):
        if self.window.busy:return
        import csv
        from .emma import is_emma_csv
        from .inspection import load_emma_controls
        name,_=QFileDialog.getOpenFileName(self,'导入坐标控制CSV','','CSV (*.csv)')
        if not name:return
        target=self.target_asset()
        unit=self.control_unit.currentData()
        def load():
            if is_emma_csv(name):
                if target is None:raise ValueError('请先选择名义资源。')
                imported=load_emma_controls(name,target.point_scope,unit=unit)
                return imported['controls'],True,imported
            if Path(name).stat().st_size>16*1024*1024:raise ValueError('控制表过大。')
            with open(name,encoding='utf-8-sig',newline='') as f:
                reader=csv.DictReader(f,strict=True)
                expected={'name','point_id','nx','ny','nz','lower_mm','upper_mm'}
                if not reader.fieldnames or len(reader.fieldnames)!=7 or set(reader.fieldnames)!=expected:raise ValueError('控制CSV表头不符合界面说明。')
                controls=[]
                for row in reader:
                    if None in row or any(v is None for v in row.values()):raise ValueError('控制CSV存在列数错误。')
                    controls.append(dict(name=row['name'],point_id=row['point_id'],axis=[float(row[k]) for k in ('nx','ny','nz')],lower_mm=float(row['lower_mm']) if row['lower_mm'].strip() else None,upper_mm=float(row['upper_mm']) if row['upper_mm'].strip() else None))
                    if len(controls)>50000:raise ValueError('控制表超过50000项。')
                if not controls:raise ValueError('控制表为空。')
                json.dumps(controls,allow_nan=False)
                return controls,False,None
        def done(loaded):
            controls,local,provenance=loaded
            self._setting=True
            try:
                _write_table(self.control_table,[[c['name'],c['point_id'],*c['axis'],c.get('lower_mm'),c.get('upper_mm')] for c in controls])
                self.reference_axes.setChecked(local)
            finally:self._setting=False
            if provenance:
                self.window.history.append(dict(type='controls_import',result={k:v for k,v in provenance.items() if k!='controls'}))
                self.window.project_path=None
            self.invalidate()
        self.window._start_operation('读取坐标控制…',load,done)

    def replay_history(self):
        if self.window.busy:return
        i=self.history_combo.currentData()
        if i is None:return
        receipt=self.window.history[i]['result']
        if not self.window._asset_by_id(receipt['source_id']) or not self.window._asset_by_id(receipt['target_id']):
            self.window._show_error('历史无法重算','原计算引用的几何已移除。');return
        self.refresh_assets(receipt['source_id'],receipt['target_id']);self.set_recipe(receipt['recipe']);self.run_current()

    def closeEvent(self,event):
        if self.window.busy:event.ignore();return
        super().closeEvent(event)

    def reject(self):
        if not self.window.busy:super().reject()
