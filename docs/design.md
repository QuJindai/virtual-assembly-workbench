# Virtual Assembly Workbench 0.1

This standalone implementation uses public dependencies and synthetic examples. The first release implements an end-to-end local engineering workflow; it makes no manufacturing accuracy certification or FEA/AI claims.

## Product workflow

1. Start the Chinese desktop UI and load the included deterministic assembly demo, or import point clouds (CSV/XYZ/ASCII PCD/PLY), triangle meshes (STL/OBJ/PLY) and CAD (STEP/IGES).
2. Select a moving/source part and fixed/reference part. Coordinates inside the app are millimetres. Text/point/mesh imports accept mm or m; STEP/IGES units are handled by OCCT.
3. Inspect both parts in an interactive VTK viewport. Edit a rigid transform or run ICP/GICP in a background worker, then explicitly apply the chosen transform.
4. Compare source points against reference cloud points, mesh triangles, or exact OCCT CAD shape. Show unsigned geometric distance, RMS/P95/max, tolerance coverage and sampling count. No signed gap, interference or metrology certification is implied.
5. Export HTML, JSON, per-sample CSV and transformed XYZ; save/load a self-contained .vaw project with geometry and transforms. All work stays local. The app requires no account or server.

## Shared Python API (binding contract for GUI and CLI)

Package: `src/assembly_workbench`. NumPy arrays use float64, lengths in mm. Invalid input raises `ValueError` with an actionable message. The engine never mutates input datasets during registration/measurement.

```python
# core.py
@dataclass
class Dataset:
    name: str
    points: np.ndarray             # Nx3, raw geometry in mm
    kind: str = "cloud"            # cloud, mesh, cad
    triangles: np.ndarray | None = None  # Mx3 vertex indices
    source_path: str | None = None
    cad_shape: object | None = None
    transform: np.ndarray = field(default_factory=lambda: np.eye(4))
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    def world_points(self) -> np.ndarray: ...

@dataclass
class RegistrationResult:
    source_id: str
    target_id: str
    method: str
    transform: np.ndarray          # absolute new source transform
    elapsed_s: float
    rmse_mm: float
    fitness: float                 # inlier fraction, 0..1
    converged: bool
    iterations: int
    parameters: dict
    source_transform: np.ndarray   # transform at computation time
    target_transform: np.ndarray
    def to_dict(self) -> dict: ...

@dataclass
class DeviationResult:
    source_id: str
    target_id: str
    indices: np.ndarray
    distances_mm: np.ndarray
    method: str                    # cloud_nearest_point, mesh_surface, cad_exact
    tolerance_mm: float
    total_count: int
    elapsed_s: float
    source_transform: np.ndarray
    target_transform: np.ndarray
    @property
    def statistics(self) -> dict: ...  # rms_mm, p95_mm, max_mm, mean_mm, within_fraction, sample_count, total_count
    def to_dict(self) -> dict: ...

def make_demo() -> tuple[Dataset, Dataset, np.ndarray]: ... # source,target,known transform
def rigid_transform(tx=0,ty=0,tz=0,rx=0,ry=0,rz=0) -> np.ndarray: ... # degrees, Rz Ry Rx
def register(source, target, method="gicp", voxel_mm=2.0,
             max_distance_mm=20.0, threads=4, max_iterations=60) -> RegistrationResult: ...
def apply_registration(source, result) -> None: ... # check source id and snapshot
def measure_deviation(source,target,tolerance_mm=0.2,max_samples=5000) -> DeviationResult: ...
def fit_landmarks(source_points,target_points) -> np.ndarray: ... # >=3 corresponding noncollinear points

# io.py
def load_dataset(path, unit="mm") -> Dataset: ...
def save_project(path, assets: list[Dataset], history: list[dict] | None = None) -> None: ...
def load_project(path) -> tuple[list[Dataset], list[dict]]: ...
def export_report(folder,source,target,deviation,registrations=None) -> list[Path]: ...
```

## UI design

A desktop engineering layout: dark navy top bar, left asset list, large central 3D viewport, right tabs for registration/deviation/rigid adjustment, bottom result history. Cyan source, amber target; distance colours with an explicit mm legend. All actions use real engine results. Fit-view, point size, visibility, import, remove, demo, save/open and report export are operational. Labels distinguish original geometry from applied transforms. Display may deterministically subsample large clouds, but analysis counts and method are reported.

Every operation that can take noticeable time (import, CAD load, registration, measurement, project and report IO) runs outside the GUI thread. State-changing controls are disabled while a task is active, worker completion updates are delivered on the GUI thread, and closing during a task is deferred safely. Changing geometry/transforms invalidates pending analysis/results for export. Project paths and geometry are self-contained; no pickle loading or archive extraction to user-supplied paths.

## Acceptance

- Known rigid transform recovered by real ICP/GICP on synthetic geometry with correct source/target direction and compositional transforms.
- Empty, nonfinite, degenerate input and nonsensical units/parameters rejected.
- Non-overlap is not called a successful registration.
- Distance to a triangle interior and exact CAD box verified against hand-derived values.
- Metres-to-mm conversion, file roundtrips, project relocation including CAD and malformed archive rejection.
- Export counts, matrices and units match calculations; stale results cannot be exported; HTML escapes user filenames.
- Qt tests drive the demo, execute registration and measurement, save/open, export, and capture a real screenshot. Windows build runs CLI self-test from packaged exe before producing a portable ZIP.

## Release

Public repo `QuJindai/virtual-assembly-workbench`, MIT application code, third-party attribution and source links. Python 3.12, explicit direct dependency versions and generated complete dependency lock. Windows portable PyInstaller build and launcher scripts; Linux source execution. Real factory samples, existing source integration, PhysicsNeMo/FEA, GD&T, certified manufacturing metrics and GPU backends are future integration work and are absent from the GUI until implemented.
