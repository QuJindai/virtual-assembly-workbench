中文本地虚拟装配工作台 v0.1.1 预览版。下载 Windows x64 ZIP 并完整解压，运行 `AssemblyWorkbench.exe`；无需另外安装 Python。

- 新增eMMA测量CSV导入，拆分名义值与各次实测，保留缺失和推断计数，拒绝截断数据。
- 带编号测点按编号对应比较，报告保留测点编号、三个方向坐标差及无对应数量。
- 带编号工程采用版本2，防止旧版忽略编号后误用最近邻；普通工程保持兼容。
- 增加私有数据批量验证脚本；生产图纸、样本和测量结果不随公开程序发布。

- 点云、STL/OBJ/PLY 网格和 STEP/IGES CAD 导入，统一 mm。
- small_gicp 原生 ICP/GICP、手动刚体位姿、显式应用配准结果。
- 最近点 / 三角面 / OCCT CAD 表面无符号距离、采样统计、三维着色。
- 自包含 `.vaw` 工程与离线 HTML / JSON / CSV / XYZ 导出。

发布前自动执行 Linux、Windows 数值与 Qt 工作流测试，并对打包后的 EXE 执行合成已知答案自检和真实三维帧缓冲检查。`self-test.json` 是本构建的计算证据，`SHA256SUMS.txt` 可校验 ZIP，`workbench-windows.png` 是包含原生三维画面的程序截图；`workbench-windows-viewport.png` 与 `workbench-windows.render.json` 保留原始三维画面和检查记录。

这是几何分析预览版。现场导出样本用于验证格式兼容性和数值流程，不构成生产测量精度认证。程序不包含 FEA、AI 变形预测、GD&T、碰撞或有符号间隙；稀疏测点的几何距离不能替代连续表面检测。完整依赖许可证随程序提供。
