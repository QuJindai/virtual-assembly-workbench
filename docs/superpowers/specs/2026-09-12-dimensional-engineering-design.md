# Dimensional engineering upgrade v0.2

The user approved upgrading the existing product after the gap review, and explicitly requested current research with R&D Radar. This spec turns the approved datum, measurement and rigid assembly workflow into executable tools. It preserves the v0.1.1 geometry/import/report workflow.

## User workflow

1. Open an existing project or import measured and nominal geometry. Open the new Chinese **尺寸工程** workspace from the toolbar.
2. Select source and reference assets. For datum operations explicitly select six corresponding datum targets in primary 3, secondary 2, tertiary 1 order. For RPS and adjustment choose corresponding points, their constraint directions, weights, permitted degrees of freedom and travel limits. No arbitrary production points are automatically designated as datums.
3. Compute a datum transform or bounded adjustment proposal, review residuals and remaining errors, then explicitly apply. Calculations never modify the input; stale proposals cannot be applied.
4. Fit line, plane, circle, sphere or cylinder to an explicit point selection. Automatic plane/circle proposals use seeded robust consensus and retain inliers; they are geometric candidates, not semantic identification of a manufactured hole. Measured feature centers alone do not define hole diameter.
5. Compare same-scope labeled measurements using an explicit table of per-point coordinate or directional limits. Missing measurements remain missing. Compare fitted geometric feature dimensions where selected. No supplied limit means no pass/fail verdict.
6. Evaluate gap and flush using explicitly paired edge points and user-defined orthogonal gap/flush directions. Cross-section extraction on CAD/mesh produces selectable sampled section geometry; arbitrary nearest-point distances are never labeled gap. On valid closed CAD solids also compute minimum separation and common volume to distinguish clear/contact/interference.
7. Save the input recipe and calculation evidence into project history, replay a prior recipe, and export standalone HTML, JSON and CSV with all inputs, axes, units, feature IDs, point selections, fingerprints and before/after results. Open the same project on another machine without the original file paths.

## Numerical and state contracts

- Python 3.12 with the existing locked NumPy, SciPy, VTK, OCP, small_gicp and PySide6 dependencies. No new heavy inference framework.
- All lengths are mm; angles displayed to users are degrees. Numerical results are finite, JSON-safe dictionaries. Rigid transforms are 4x4 lists mapping CURRENT WORLD source points to proposed world points. Application composes `delta @ source.transform`.
- Invalid, underconstrained, degenerate or unsupported inputs raise actionable `ValueError`. No NaN, fake zero measurement, or synthetic fallback is accepted.
- A datum frame from six points uses Z = ordered primary-plane normal from points 1/2/3, X = normalized B1-to-B2 direction projected into that plane, and Y = Z cross X. The origin combines its Z coordinate from A1, Y coordinate from B1, and X coordinate from C1. Primary constraints act along Z, secondary constraints along Y, tertiary along X; the secondary line tangent is X, not Y. Point order controls orientation explicitly. Source and target frames yield `target_frame @ inverse(source_frame)`.
- General RPS/rigid adjustment minimizes weighted scalar normal residuals for explicit corresponding points. Free parameters are TX,TY,TZ,RX,RY,RZ; fixed DOFs remain exactly fixed. Rotations are about an explicit pivot (default source centroid), composed Rz Ry Rx. Bounds constrain each permitted parameter. Report rank, conditioning, active bounds, before/after residuals and convergence. Do not claim success on rank-deficient parameterization. This is a local bounded solve, not guaranteed global optimality.
- Feature fitting uses orthogonal geometric residuals, centered/scaled numerics and explicit degeneracy checks. Cylinder fitting handles unknown axis and radius. Reports retain residual metrics and conditioning/coverage warnings appropriate to partial arcs or short cylinders; a fit is not certified GD&T.
- Gap/flush values are signed projections of `(source_edge - reference_edge)` onto explicit normalized perpendicular directions. Third-direction separation is reported. A negative projected gap is not itself proof of solid interference.
- CAD intersection requires valid closed solids and successful native OCCT operations. A positive common volume beyond a scale-aware tolerance is interference. Near-zero minimum distance with negligible common volume is contact; separated solids are clear. Open faces/clouds cannot receive a solid-interference verdict.
- Engineering results are immutable receipts tied to asset IDs, full geometry fingerprints and transform snapshots. Change of geometry/pose invalidates application/export as a current result. Historical reports remain labeled snapshots. Result rows, recipes and referenced selections survive `.vaw` history roundtrips.
- All long calculations and report operations use the existing Qt worker lifecycle. Modal engineering workspace prevents background geometry edits; safe close and main-window close must wait for workers. UI remains usable at 1100x700 with scrolling, no clipped critical buttons.
- Factory samples, paths, identifiers and private reports never enter public git or CI. Public tests use analytic/synthetic geometry. Actual factory datasets are tested privately and explicitly distinguish numerical validation from Windows packaged GUI validation.

## Deliverable boundary

v0.2 adds real datum/feature/gauge/rigid adjustment tools, not a PTB-certified universal GD&T engine. CATIA native translators, hardware CMM drivers, contact-force/elastic deformation, FEA and learned deformation prediction remain absent. They require proprietary translators or calibrated physical inputs and independent reference data not present in the supplied files. Do not relabel rigid movement as deformation. End-to-end comparison with the original product or PolyWorks remains unverified without a runnable baseline and agreed reference measurements.

## Acceptance

- Analytic tilted/transformed datum fixtures recover expected transforms, preserve datum hierarchy and reject collinear definitions.
- Constrained RPS recovers known translations/rotations, reports bound saturation and rejects unconstrained axes. Locked DOFs and original arrays stay unchanged.
- Tilted noisy circles/planes/lines/spheres/cylinders recover dimensions against independently generated geometry; tiny arcs/degenerate samples are rejected or explicitly qualified. Outlier proposal tests verify inlier recovery and deterministic output.
- Gap and flush give signed known values in a rotated coordinate frame; nonorthogonal axes and bad correspondence fail. CAD boxes test clear/contact/overlap and transformed geometry. Section tests use analytic mesh/CAD intersections.
- Per-point limit tests cover missing IDs, scope mismatch, inclusive bounds, absent tolerances and actual row CSV exports. Recipe replay and `.vaw` relocation preserve results.
- Existing tests continue passing. New real Qt workflows run on Linux/Windows CI; packaged EXE runs expanded self-test plus true native framebuffer capture. Private real-data validation preserves every raw input hash.

Cloud section contract: `slab_mm` is the full slab thickness, so selected measured points satisfy `abs(distance_to_plane) <= slab_mm / 2` within justified projected floating-point roundoff. GUI and recipes use the same definition.

Coordinate screening conservatively labels nonzero bound differences at projected floating-point roundoff scale as unjudged, with a reason and separate numerical_boundary_count. Exact-bound values remain inclusive; no fixed engineering tolerance floor is introduced. The estimate is not measurement uncertainty, and original computed values and limits are preserved.
