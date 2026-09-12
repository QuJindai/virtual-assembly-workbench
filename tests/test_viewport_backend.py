import json
import subprocess
import sys


def test_fresh_desktop_import_registers_real_vtk_implementations():
    # Must be a fresh process: OCCT or another test can register VTK factories
    # and accidentally conceal a blank standalone desktop window.
    code = '''
import json
import assembly_workbench.gui
from vtkmodules.vtkRenderingCore import vtkRenderWindow, vtkRenderer, vtkRenderWindowInteractor
print(json.dumps([vtkRenderWindow().GetClassName(), vtkRenderer().GetClassName(),
                  vtkRenderWindowInteractor().GetInteractorStyle().GetClassName()]))
'''
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    window, renderer, style = json.loads(result.stdout)
    assert "OpenGL" in window
    assert renderer == "vtkOpenGLRenderer"
    assert style == "vtkInteractorStyleSwitch"
