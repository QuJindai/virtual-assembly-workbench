# Third-party components

The MIT license applies to this application's original code. It does not replace the separate licenses of dependencies. The application uses unmodified upstream wheels and dynamic native libraries; the portable distribution preserves the directory layout and separately replaceable Qt libraries. No restriction on reverse engineering for debugging modifications to LGPL components is added by this application.

| Component | Version | License / source |
|---|---|---|
| small_gicp | 1.0.0 | MIT — https://github.com/koide3/small_gicp |
| VTK | 9.3.1 | BSD 3-Clause — https://gitlab.kitware.com/vtk/vtk/-/tree/v9.3.1 |
| cadquery-ocp bindings | 7.8.1.1.post1 | Apache-2.0 — https://github.com/CadQuery/OCP |
| Open CASCADE Technology | 7.8.1 | LGPL-2.1 with OCCT exception — https://github.com/Open-Cascade-SAS/OCCT/tree/V7_8_1 |
| Qt / PySide6 / Shiboken6 | 6.8.3 | LGPL-3.0 option for the modules used (Qt Core, Gui, Widgets) — https://code.qt.io/cgit/qt/qtbase.git/tree/?h=v6.8.3 and https://code.qt.io/cgit/pyside/pyside-setup.git/tree/?h=v6.8.3 |
| NumPy | 2.3.5 | BSD 3-Clause; bundled numerical-library notices in wheel — https://github.com/numpy/numpy/tree/v2.3.5 |
| SciPy | 1.17.0 | BSD 3-Clause; bundled numerical-library notices in wheel — https://github.com/scipy/scipy/tree/v1.17.0 |
| Python runtime | 3.12.x | Python Software Foundation license — https://www.python.org/downloads/source/ |
| PyInstaller bootloader | 6.16.0 | GPL with bootloader distribution exception — https://github.com/pyinstaller/pyinstaller/tree/v6.16.0 |

The Windows build runs `scripts/collect_licenses.py` to include the installed distributions' license and notice files, the Python license and the additional Qt/OCCT license texts in `_internal/licenses`. Transitive dependencies and exact build versions are recorded in `requirements.lock` and in the generated distribution inventory. Unused Qt Addons are not intentionally collected in the executable. Refer to the actual package notices for bundled OpenBLAS and other native code.

The above source repositories and tagged source trees provide the corresponding upstream library source. The application source and build scripts are available at https://github.com/QuJindai/virtual-assembly-workbench. The application and demo geometry are provided without warranty; dependency warranties are governed by their own license texts.
