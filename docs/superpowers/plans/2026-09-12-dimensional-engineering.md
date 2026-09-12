# Dimensional Engineering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development for independent numerical tasks and review; controller owns integration. Do not dispatch multiple implementation subagents concurrently. Steps use checkbox syntax.

**Goal:** Deliver datum-based geometric measurement and constrained rigid assembly workflows in the existing offline desktop application.

**Architecture:** Small pure numerical modules return JSON-safe receipts. A headless engineering facade binds point selections, recipes and geometry fingerprints. A Chinese modal Qt workspace uses the existing worker lifecycle and stores receipts in the existing self-contained project history.

**Tech Stack:** Existing locked Python 3.12, NumPy, SciPy, OCP, VTK, PySide6 and small_gicp.

**Spec:** `docs/superpowers/specs/2026-09-12-dimensional-engineering-design.md`

## Global Constraints

- All lengths are mm and user-facing angles degrees; current-world delta transforms compose on the left.
- No mutation during computation; explicit application checks asset IDs and full geometry fingerprints.
- Missing, degenerate, nonfinite, rank-deficient or unsupported geometry must not produce a successful measurement.
- No factory data or identifiers in public git or CI. Numerical fitting is not certified GD&T, rigid motion is not elastic deformation, and signed edge projections are not solid-interference proof.
- Keep dependencies locked; preserve existing v0.1.1 workflow and project compatibility.
- Each implementer owns only the files listed in its task; no child subagents. Parent integrates and publishes after review and verification.

## Task 1: Datum alignment and geometric features

**Files:** Create `src/assembly_workbench/datums.py`, `features.py`, `tests/test_datums.py`, `tests/test_features.py`.

**Interfaces:** All functions return finite JSON-safe dicts/lists (no ndarray in output).

```python
def align_321(source, target) -> dict: ...  # each shape (6,3), A1,A2,A3,B1,B2,C1
def align_rps(source, target, normals, *, weights=None,
              dofs=(True,True,True,True,True,True), pivot=None,
              translation_limit_mm=100.0, rotation_limit_deg=30.0) -> dict: ...
# Both results: method, transform (4x4 list), residuals_mm, rms_mm, max_mm,
# converged; RPS additionally parameters [tx,ty,tz,rx,ry,rz], pivot,
# before_residuals_mm, rank, condition, at_bounds, dofs.
def fit_feature(points, kind) -> dict: ...  # line, plane, circle, sphere, cylinder
# kind, method, count, center [3], rms_mm, max_mm, form_error_mm;
# direction [3] for line/plane/circle/cylinder, radius_mm for circle/sphere/cylinder;
# diagnostics dict for finite conditioning/coverage data and warnings list.
def detect_features(points, kind='plane', *, threshold_mm=0.1,
                    min_points=20, max_features=5, seed=0) -> list[dict]: ...
# kind plane or circle only; each fitted result plus indices of ORIGINAL points.
```

- [ ] Write behavior tests before code. Independent known-pose fixture:

```python
def test_321_recovers_known_pose():
    from assembly_workbench.datums import align_321
    from assembly_workbench.core import rigid_transform
    import numpy as np
    p=np.array([[0,0,0],[10,0,0],[0,10,0],[0,0,3],[10,0,3],[0,4,5]],float)
    t=rigid_transform(3,-2,1,5,-3,7)
    q=p@t[:3,:3].T+t[:3,3]
    np.testing.assert_allclose(align_321(p,q)['transform'],t,atol=1e-8)
```

- [ ] Run `PYTHONPATH=src /workspace/scratch/4b70c15738bd/assembly-env/bin/python -m pytest tests/test_datums.py tests/test_features.py -q`; record expected missing implementation failure.
- [ ] Implement centered/scaled feature fits using NumPy SVD and SciPy `least_squares`; cylinder axis needs multistart and local rank/coverage checks, not a fixed PCA-axis assumption. Plane/circle proposals use bounded seeded consensus, refit inliers, cap input at 200000 points and return full original indices. Reject fits with insufficient dimensional support; prevent unbounded consensus work.
- [ ] Implement analytic hierarchical 3-2-1 and weighted constrained RPS. Validate finite positive weights, unit normals, boolean DOFs, bounds and full active-parameter Jacobian rank; nonzero fixed axes remain fixed and scalar residual semantics preserved. Use source-centroid pivot for conditioning.
- [ ] Test tilted shapes, noise, outliers, collinear/zero-radius/nonfinite/undersampled inputs, large coordinate offsets, RPS translation-only/rotation/bounds and degeneracy. Record tests and commit only owned files.

## Task 2: Edge gauges, cross-sections and solid interference

**Files:** Create `src/assembly_workbench/assembly.py`, `tests/test_assembly.py`.

**Interfaces:** Consume `Dataset` from core; all outputs finite JSON-safe dicts.

```python
def gap_flush(source_points, reference_points, gap_axis, flush_axis) -> dict: ...
# paired Nx3; method='edge_projection', count, gap_mm list, flush_mm list,
# seam_offset_mm list, source_points, reference_points, gap_axis, flush_axis,
# statistics {gap_min_mm,gap_max_mm,gap_mean_mm,flush_min_mm,flush_max_mm,flush_mean_mm}
def section_geometry(asset, origin, normal, *, slab_mm=0.2,
                     max_points=20000) -> dict: ...
# points (Nx3), method='cad_section'/'mesh_section'/'cloud_slab', source_id,
# origin, normal, slab_mm, sampled flag. CAD uses OCCT section edges;
# mesh uses triangle-plane intersection. Cloud selects existing points in slab.
def cad_contact(source, target, *, tolerance_mm=1e-5) -> dict: ...
# method='occt_solid_common', state='clear'/'contact'/'interference',
# distance_mm, common_volume_mm3, volume_tolerance_mm3, tolerance_mm.
```

- [ ] Write and run failing known-answer tests:

```python
def test_signed_edge_gauge():
    from assembly_workbench.assembly import gap_flush
    r=gap_flush([[4,0,1]],[[0,0,0]],[1,0,0],[0,0,1])
    assert r['gap_mm']==[4.0]
    assert r['flush_mm']==[1.0]
```

- [ ] Implement vectorized signed gauge projections with explicit orthogonal axes, finite checks, equal correspondence lengths and seam offsets.
- [ ] Implement world-space sections with native OCCT/VTK geometry. Closed CAD solid overlap is tested with OCCT Common + volume, minimum distance from DistShapeShape; verify shape validity/closed solids and operation success. Compound solids may be supported only with explicit validation of contents. Never return clear on kernel failure.
- [ ] Test rotated axes, invalid axes, CAD transformed boxes with 2 mm separation, face contact, containment and 8 mm3 intersection; open CAD faces reject contact classification. Test CAD/mesh plane sections and cloud slab boundaries. Commit owned files and provide a spec/quality self-review report.

## Task 3: Reproducible engineering facade and desktop workflow

**Files:** Create `engineering.py`, `engineering_gui.py`, `inspection.py`, `tests/test_engineering.py`, `tests/test_inspection.py`, `tests/test_engineering_gui.py`, `scripts/validate_engineering.py`; modify `gui.py`, `__main__.py` and version metadata.

**Interfaces:**

```python
def run_engineering(source, target, recipe) -> dict: ...
# recipe {'tool': 'datum321'|'rps'|'adjustment'|'feature'|'detect'|'gap_flush'|'section'|'contact'|'inspection', ...}
# receipt includes source/target IDs/signatures, transforms, recipe, result, version.
def apply_engineering(source, target, receipt) -> None: ...
def export_engineering(folder, source, target, receipt) -> list: ...
def evaluate_points(source, target, controls) -> dict: ...
# controls list: {'name','point_id','axis':[3],'lower_mm','upper_mm'},
# signed coordinate delta projected on explicit axis; absent points -> missing,
# absent bounds -> unjudged; exact same scope required. Results include coverage.
```

- [ ] Test missing labels, mismatched scopes, inclusive limits, absent bounds, stale geometry, JSON escaping, project history roundtrip, replay and guarded transform application before implementing the facade.
- [ ] Provide point selection by explicit IDs or `#1` one-based source row references, plus prefix filtering for feature regions; do not silently sample selected datum points. Reject duplicate references.
- [ ] Add a scrollable Chinese engineering dialog with asset selectors and pages for datum/RPS, feature/automatic proposal, edge gap/flush, section/CAD interference, per-point limits and rigid adjustment. Use tables and labeled numeric controls; no JSON editing required for ordinary operations. Import/export recipe JSON remains available for repeatability.
- [ ] Add synthetic engineering examples and populated example recipes (clearly marked examples) for every new user workflow. Inputs, result tables, before/after metrics, explicit apply, export, replay and save into project history are operational.
- [ ] Reuse main `_start_operation`, guard modal closure while work is active, disable parameter edits while busy, preserve main window rendering and controls. Use the engineering facade for both GUI and CLI.
- [ ] Drive workflows through Qt tests. Headless self-test exercises new numerical tools; new screenshot command captures engineering UI and actual VTK scene. Private validation runs original CSVs without altering them and records explicitly synthetic limit/basis demonstrations separately from approved manufacturing controls.

## Task 4: Research record, integration verification and release

**Files:** `docs/research-2026-09-12.md`, `docs/engineering-guide.md`, README, release notes, workflow; private validation outputs remain outside git.

- [ ] Record actual R&D Radar scan/inspection outcomes; distinguish empty keyword scan, unsuitable hits and verified candidates. Verify primary current sources for PolyWorks 2026, SciPy constrained least squares, Open3D proposal workflows, pyRANSAC-3D and OCCT sections/common. No vendor-code performance superiority claims without a baseline.
- [ ] Write usage instructions with datum/axis semantics, geometric vs manufacturing verdict boundaries, CATIA conversion dependency and replay/export examples. Keep research citations by the supported decision.
- [ ] Run full existing/new numerical and Qt suites, inspect real native screenshots, and perform independent code review before publish. Repair blockers and repeat only affected checks before the final gate.
- [ ] Publish public source and tested Windows portable v0.2 build via the authorized GitHub workflow. Verify exact remote commit, release, hashes and packaged self-test; keep raw input and private report evidence private.
