中文本地虚拟装配工作台 v0.1 预览版。下载 Windows x64 ZIP 并完整解压，运行 `AssemblyWorkbench.exe`；无需另外安装 Python。

- 点云、STL/OBJ/PLY 网格和 STEP/IGES CAD 导入，统一 mm。
- small_gicp 原生 ICP/GICP、手动刚体位姿、显式应用配准结果。
- 最近点 / 三角面 / OCCT CAD 表面无符号距离、采样统计、三维着色。
- 自包含 `.vaw` 工程与离线 HTML / JSON / CSV / XYZ 导出。

发布前自动执行 Linux、Windows 数值与 Qt 工作流测试，并对打包后的 EXE 执行合成已知答案自检和真实窗口截图。`self-test.json` 是本构建的计算证据，`SHA256SUMS.txt` 可校验 ZIP，`workbench-windows.png` 是程序截图。

这是几何分析预览版，尚未用真实工厂数据验证，不包含 FEA、AI 变形预测、GD&T、碰撞或有符号间隙。合成数据上的测试误差不代表生产测量精度。完整依赖许可证随程序提供。
