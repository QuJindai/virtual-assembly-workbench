# PyInstaller onedir build. Run from the repository root.
from pathlib import Path
from importlib import metadata
from PyInstaller.utils.hooks import collect_submodules, collect_dynamic_libs

root = Path(SPECPATH).parent
binaries = collect_dynamic_libs('OCP') + collect_dynamic_libs('vtkmodules')
# Wheels can keep vendored DLLs beside, rather than inside, the import package.
for distribution_name in ('cadquery-ocp', 'small-gicp'):
    distribution = metadata.distribution(distribution_name)
    for entry in distribution.files or []:
        if '.libs/' in str(entry).replace('\\', '/') and str(entry).endswith(('.dll', '.so', '.dylib')):
            binaries.append((str(distribution.locate_file(entry)), str(Path(entry).parent)))

a = Analysis(
    [str(root / 'scripts' / 'desktop_entry.py')],
    pathex=[str(root / 'src')],
    binaries=binaries,
    datas=[(str(root / 'build' / 'licenses'), 'licenses'),
           (str(root / 'LICENSE'), '.'), (str(root / 'THIRD_PARTY_NOTICES.md'), '.')],
    hiddenimports=collect_submodules('OCP') + ['small_gicp', 'vtkmodules.qt.QVTKRenderWindowInteractor'],
    hookspath=[], hooksconfig={}, runtime_hooks=[],
    excludes=['PyQt5', 'PyQt6', 'PySide2', 'tkinter', 'IPython', 'matplotlib', 'pandas', 'torch', 'tensorflow'],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name='AssemblyWorkbench',
          debug=False, bootloader_ignore_signals=False, strip=False, upx=False,
          console=True, disable_windowed_traceback=False)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name='AssemblyWorkbench')
