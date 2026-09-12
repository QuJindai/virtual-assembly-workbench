# Virtual Assembly Workbench Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development for the independent GUI task, with local core implementation and final review.

**Goal:** Ship a local desktop workflow from geometry import through alignment, unsigned deviation and reproducible export.

**Architecture:** Python orchestration over native C++ libraries: PySide6/Qt, VTK, OCCT and small_gicp. The numerical core and file operations are callable headlessly. The UI controls asynchronous jobs and explicit transform application.

**Tech Stack:** Python 3.12; dependency versions in pyproject.toml.

**Spec:** docs/design.md

## Global Constraints

- Units inside the application are mm and rigid rotations are degrees at the UI boundary.
- Public code and synthetic samples only; no inherited private attachments or project claims.
- ICP and GICP use small_gicp. Do not label the ICP comparison as PCL.
- Tests run native algorithms; no generated/mock numerical result is displayed as a computed result.
- The spec contains the exact shared API. UI implementation owns only gui.py, viewport.py and tests/test_gui.py.
- Do not publish until the real workflow and relevant geometry tests pass.

### Task 1: Numerical core and recoverable file operations

Files: src/assembly_workbench/core.py, io.py, __init__.py; tests/test_core.py, test_io.py.

- [ ] Write tests for known transforms, nonoverlap, unit conversion, exact plane/box distances, self-contained project restore and escaped/stale exports.
- [ ] Run `python -m pytest tests/test_core.py tests/test_io.py` and confirm new contracts are missing.
- [ ] Implement the API in docs/design.md using small_gicp, SciPy, VTK and OCP; reject invalid geometry/transforms; preserve absolute transform semantics.
- [ ] Rerun the same suite and fix behavior defects.
- [ ] Commit the tested core and IO.

### Task 2: Chinese Qt/VTK desktop workflow

Read docs/design.md first; it is the binding requirements and exact engine API. Implement only src/assembly_workbench/gui.py, viewport.py and tests/test_gui.py. Other files belong to the root implementer.

Interfaces: consume Dataset, make_demo, rigid_transform, register, apply_registration, measure_deviation, RegistrationResult, DeviationResult from core.py; load_dataset, load_project, save_project, export_report from io.py. Produce `MainWindow` and `launch(demo=False, screenshot=None)` in gui.py. MainWindow must expose `load_demo()`, `run_registration()`, `apply_result()`, `run_measurement()`, `assets` (list), `last_registration`, `last_deviation`, `busy` (bool) for consumer-level integration tests. Screenshot argument captures the real shown window and exits, via a QTimer after painting.

The interface has an asset list, source/reference selectors, VTK viewport, registration/deviation/transform tabs and results history. Name the app "虚拟装配工作台" with secondary brand "Assembly Workbench". Make controls clear in Chinese. Use mm consistently. Add toolbar buttons for demo, import, project open/save and report export; point size and fit-view. The 3D view has useful empty-state text, distinct source/target colours, real deviations/scalar legend, and per-asset visibility.

Implement background workers with safe lifetime and GUI-thread signal delivery, disabled state mutations while busy, actionable errors, stale result invalidation, and safe close while busy. Native registration cannot be killed safely: do not offer a misleading cancel. Do not create placeholder modules or mock data when the core is not yet available. Direct environment python: /workspace/scratch/4b70c15738bd/assembly-env/bin/python. Repo working directory: /workspace/scratch/4b70c15738bd/virtual-assembly-workbench. Root is implementing core/IO in parallel; request API clarification if necessary.

- [ ] Write Qt workflow tests before implementing UI. Test user-visible source/target selection, demo, native registration/measurement completion, explicit apply and invalidation.
- [ ] Implement the real Qt/VTK workflow without changing numerical core or project packaging.
- [ ] Run tests under a working X display; if display tooling is missing, report that precisely and still verify syntax/imports.
- [ ] Commit owned files only and write a report in artifacts/gui-report.md containing files, test results and remaining concerns.

### Task 3: CLI, documentation, packaging and public release

Files: __main__.py, tests/test_cli.py, README.md, THIRD_PARTY_NOTICES.md, requirements lock, scripts, .github/workflows/build.yml.

- [ ] Add a headless CLI self-test that runs demo registration, deviation and file export/restore, emitting a JSON receipt and nonzero exit on failure.
- [ ] Document installation, algorithm meanings, available formats, startup, walkthrough and limits.
- [ ] Package Windows with PyInstaller; execute packaged self-test before upload. Include source distribution, dependency lock, license notices, checksums and synthetic examples.
- [ ] Review all source, run numerical, file, CLI and GUI suites, and inspect a real screenshot.
- [ ] Commit, publish to the already authorized public repository, and verify remote files/commit plus CI/build artifacts.
