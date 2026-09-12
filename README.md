# 虚拟装配工作台 · Virtual Assembly Workbench

本地运行的点云与 CAD 装配分析桌面软件。v0.2 使用成熟几何引擎完成 **导入 → 基准建系/配准 → 尺寸测量 → 刚体装调 → 工程重算与报告导出**，提供中文 Qt 界面和可交互三维视图。

[自动构建](https://github.com/QuJindai/virtual-assembly-workbench/actions/workflows/build.yml) · [发行包](https://github.com/QuJindai/virtual-assembly-workbench/releases) · [尺寸工程指南](docs/engineering-guide.md) · [2026-09-12 研发调研](docs/research-2026-09-12.md)

## 开始使用

Windows 10/11 x64：在发行页下载 `AssemblyWorkbench-Windows-x64.zip`，完整解压后双击 `AssemblyWorkbench.exe`。整个目录需要一起保留。便携包包含 Python 及原生依赖，运行时无需联网、账户或服务器。三维视图需要支持 **OpenGL 3.2 或更高版本**的显卡驱动；无显卡的 CI 构建机使用固定版本、校验哈希的 Mesa 软件驱动验证。发行页只发布通过程序自检和窗口启动检查的构建。

源码运行需要 **64 位 Python 3.12**，Windows 或有桌面/OpenGL 环境的 Linux：

```bash
git clone https://github.com/QuJindai/virtual-assembly-workbench.git
cd virtual-assembly-workbench
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux: source .venv/bin/activate
python -m pip install -r requirements.lock
python -m pip install --no-deps -e .
python -m assembly_workbench --demo
```

以后可以运行仓库中的 `start-windows.cmd` / `start-linux.sh`，或命令 `assembly-workbench`。Linux 需要 Qt xcb、OpenGL 和中文字体；CI 的依赖安装命令见工作流。

## 操作流程

1. 点击“演示”载入带孔曲面板的合成扫描件与基准件，或“导入”自己的点云、网格、CAD。导入时明确单位，内部统一为 **mm**。
2. 在右侧选择“移动件”和“参考件”。左侧勾选控制显示；三维窗口支持旋转、缩放和平移。
3. 在“配准”页运行 GICP 或 ICP，检查 RMSE、匹配率、迭代次数和收敛状态，再点击“应用计算变换”。算法是局部配准，初始位姿偏差较大时先用“刚体调整”靠近基准。
4. 在“偏差”页设置距离阈值及采样上限，计算颜色分布、RMS、P95、最大值和阈值内比例。显示采样点数与总点数，未测点不计入统计。
5. 点击“尺寸工程”，使用 3-2-1 / RPS、几何特征拟合、平面/圆候选、间隙面差、截面、闭合 CAD 干涉和逐点坐标公差；可新增合成示例、保存方案和历史重算。
6. “导出报告”生成离线 HTML、完整 JSON、逐点 CSV 和变换后 XYZ；“保存项目”生成包含几何、CAD、位姿、历史的 `.vaw`，可移动到其他电脑恢复。

计算在后台线程执行；改变选择、几何或位姿会清除当前结果，导出还会校验几何指纹，防止复用旧数据。遇到同名报告文件时自动建立新的报告子目录，保护原始输入和旧报告。尺寸工程历史保留完整方案、选点、结果和资产指纹，普通距离分析保留摘要及导出文件。3-2-1 / RPS 与刚体装调先显示六轴调整量，须显式应用后再复测。

## 几何与算法

### eMMA测量CSV

v0.1.1起，“导入”可以自动识别带`IPE.Origin.X/Y/Z`表头的eMMA导出，按检测计划、零件、样本编号与时间拆分名义值/实测值。`MPT`为名义记录；具有样本编号和时间的`A`记录为实测；类别为空但样本编号和时间齐全的记录会按此规则识别为实测，并明确计数。缺失坐标不会补零，冲突测点编号会隔离；缺少表头或行列截断的文件会被拒绝。v0.2 的尺寸工程页可读取原始名义 XYZ 坐标偏差上下限；其他计算特征和完整 GD&T 公差尚未解释。

两组带编号数据的偏差计算使用**同一检测计划与零件内的测点编号对应**，保留无对应点计数，避免最近邻把不同测点错配。CSV报告包含测点编号和三个方向的坐标差。测点是稀疏特征位置（例如孔中心），不是连续扫描表面；一部分测点有结果不代表完整测量覆盖。导入历史保留缺失、推断和冲突计数，悬停可查看完整文字。

带测点编号的工程使用`.vaw`格式版本2，需要v0.1.1或更新程序读取，以防旧程序忽略测点编号。普通点云/CAD工程保持版本1兼容。包含尺寸工程完整历史的项目使用版本3，需要v0.2读取；新程序可读版本1/2。重复导入相同样本前应先移除已有对象，避免在项目中混淆实例。CATPart、CATProduct、CATDrawing、3DXML当前需要在CATIA或合规转换器中导出STEP/IGES后再使用。

批量验证私有数据：`python scripts/validate_emma_folder.py INPUT_FOLDER --output NEW_OUTPUT_FOLDER`。输出包含原始坐标工程、逐点报告和ICP/GICP运行记录；配准只在副本上测试，不修改交付工程中的实测坐标。报告、截图、项目和原始数据可能含有敏感零件信息，请保存在自己的私有目录。

| 环节 | 实现 | 结果含义 |
|---|---|---|
| 桌面与三维交互 | PySide6 / Qt + VTK | 原始几何、位姿和采样偏差的真实渲染 |
| ICP / GICP | small_gicp 原生 C++ 扩展 | 移动件到基准件的刚体变换；支持已有位姿叠加 |
| 点云基准 | SciPy cKDTree | 最近点无符号距离，受点云密度影响 |
| 带编号测点 | 同范围的测点编号对应 | 同一特征位置的三维差值，明确无对应点数量 |
| 网格基准 | VTK 三角面定位器 | 到三角形表面的距离，包含面内部 |
| CAD 基准 | OCCT 经 cadquery-ocp 绑定 | 到原始 CAD 面的几何距离，遵循 CAD 自身容差；显示网格不参与此距离计算 |
| 3-2-1 / RPS / 装调 | 分层基准 + SciPy 有界优化 | 显式法向、自由度、权重、中心、行程和约束秩 |
| 平面/线/圆/球/圆柱 | 正交/几何最小二乘；平面/圆有界稳健候选 | 几何参数、残差、内点和病态性诊断，不是认证 GD&T |
| 间隙/面差与截面 | 配对边缘投影 + OCCT/VTK | 有符号量规；真实几何截面或现有点云薄层 |
| 实体干涉 | OCCT 最小距离 + Common 公共体积 | 有效闭合 CAD 的分离/接触/干涉，不是接触力学 |
| 逐点公差 | 编号对应 + 显式方向限值 | 缺测/未判独立，支持名义局部坐标系 |
| 工程与报告 | NumPy + JSON + ZIP | 无 pickle；完整方案/结果、单位、位姿、方法、参数和指纹 |

| 输入 | 支持范围 |
|---|---|
| XYZ / TXT / CSV | 前三列坐标，或含 x/y/z 列名的 CSV；可附加列 |
| PCD | ASCII，支持 x/y/z 字段重排；不支持二进制/压缩 PCD |
| PLY | VTK 支持的点云或三角网格 |
| STL / OBJ | 三角网格；多边形会三角化 |
| STEP / STP / IGES / IGS | OCCT 导入实体和曲面，按文件单位转换为 mm；需要可显示的面 |

文本、点云与网格导入可选 mm 或 m。单资源最多 200 万点、单文件最多 512 MiB、单工程最多 100 个资源；大模型请预先裁剪或降采样。网格/CAD 配准使用按三角面面积采样的表面点。偏差测量使用移动件的点或网格顶点，并不自动做连续面的全面检测。

## 自检与开发

```bash
python -m assembly_workbench --self-test artifacts/self-test
python -m pytest -q
python -m assembly_workbench --demo --screenshot artifacts/workbench.png
python -m assembly_workbench --engineering-demo feature --screenshot artifacts/engineering.png
```

自检会执行真实 GICP、已知变换校验、OCCT 表面距离和公共体积、九类尺寸工程工具、方案重算、工程往返和报告导出，生成 `self-test.json` 与可检查文件。没有 Linux DISPLAY 时仅跳过 GUI 测试；CI 在真实窗口系统下执行 GUI 流程。测试数据是固定种子的合成几何，精度阈值用于验证实现，不能解释为现场测量精度。

Windows 便携打包：

```bash
python scripts/collect_licenses.py
python -m PyInstaller --noconfirm packaging/workbench.spec
dist/AssemblyWorkbench/AssemblyWorkbench.exe --self-test artifacts/packaged-self-test
```

`src/assembly_workbench/core.py` 是无界面的几何引擎，`io.py` 负责格式和工程，`gui.py` 组织桌面工作流，`viewport.py` 负责 VTK 显示，`__main__.py` 提供启动和自检。全部依赖锁定在 `requirements.lock`；构建由 GitHub Actions 自动验证。

## 当前边界

v0.2 提供尺寸工程与刚体装调预览工作流。当前无全局特征粗配准、非刚性变形、接触力学、FEA、PhysicsNeMo、完整 GD&T 或生产设备接口。普通最近面距离仍为无符号，间隙面差与实体干涉使用独立的明确量规和实体算法。逐点坐标筛查不是工艺放行结论；合成已知答案与真实 CSV 数值验证不构成计量认证。不能据此宣称已全面超越 PolyWorks。

应用代码为 MIT；依赖分别遵循其上游许可证，详见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。合成示例不包含真实零件图纸、扫描数据或专有软件源码。
Local-first Qt/VTK virtual assembly workbench with point-cloud registration, OCCT CAD comparison and reproducible engineering reports.
