"""Desktop entry point and a repeatable, headless native-engine diagnostic."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__


def self_test(directory: Path) -> dict:
    """Exercise the installed native libraries, report export, and project restore."""
    import numpy as np
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox

    from .core import Dataset, apply_registration, make_demo, measure_deviation, register
    from .io import export_report, load_project, save_project

    directory.mkdir(parents=True, exist_ok=True)
    source, target, expected = make_demo()
    registration = register(source, target)
    if not registration.converged:
        raise RuntimeError("Synthetic GICP registration did not converge")
    if not np.allclose(registration.transform, expected, atol=0.05):
        raise RuntimeError("Synthetic registration transform failed the known-answer check")
    apply_registration(source, registration)
    deviation = measure_deviation(source, target, tolerance_mm=0.2)
    if deviation.statistics["p95_mm"] >= 0.08:
        raise RuntimeError("Synthetic registration residual is outside the test bound")
    export_report(directory / "report", source, target, deviation, [registration])
    save_project(directory / "demo.vaw", [source, target], [registration.to_dict()])
    restored, _ = load_project(directory / "demo.vaw")
    restored_ok = all(
        np.array_equal(a.points, b.points) and np.array_equal(a.transform, b.transform)
        for a, b in zip([source, target], restored, strict=True)
    )
    if not restored_ok:
        raise RuntimeError("Project restore changed geometry or placement")
    shape = BRepPrimAPI_MakeBox(10.0, 10.0, 10.0).Shape()
    cad = Dataset("Synthetic CAD box", [[0, 0, 0]], kind="cad", cad_shape=shape)
    probes = Dataset("Synthetic probes", [[5, 5, 5], [12, 5, 5]])
    cad_result = measure_deviation(probes, cad)
    if not np.allclose(cad_result.distances_mm, [5, 2], atol=1e-9):
        raise RuntimeError("CAD surface-distance known-answer check failed")
    from .engineering_examples import self_test as engineering_self_test
    engineering = engineering_self_test(directory / 'engineering')
    receipt = {
        "application": "virtual-assembly-workbench", "version": __version__,
        "passed": True, "data": "deterministic synthetic fixtures; not production validation",
        "registration": {"backend": "small_gicp", "method": registration.method,
                         "rmse_mm": registration.rmse_mm, "fitness": registration.fitness},
        "deviation": deviation.statistics,
        "project_restored": restored_ok,
        "cad_surface_distances_mm": cad_result.distances_mm.tolist(),
        "engineering": engineering,
    }
    (directory / "self-test.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Virtual Assembly Workbench / 虚拟装配工作台")
    parser.add_argument("--version", action="version", version=f"Assembly Workbench {__version__}")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--demo", action="store_true", help="打开内置合成装配演示")
    mode.add_argument("--self-test", type=Path, metavar="DIRECTORY", help="无界面验证原生计算并导出证据")
    mode.add_argument("--engineering-demo", choices=("datum321","rps","adjustment","feature","detect","gap_flush","section","contact","inspection"), help="打开尺寸工程合成示例并计算")
    parser.add_argument("--screenshot", type=Path, metavar="PNG", help="保存真实窗口截图后退出")
    args = parser.parse_args(argv)
    if args.self_test and args.screenshot:
        parser.error("--self-test cannot be combined with --screenshot")
    try:
        if args.self_test:
            receipt = self_test(args.self_test)
            print(json.dumps(receipt, ensure_ascii=True, indent=2))
            return 0
        from .gui import launch

        return launch(demo=args.demo, screenshot=args.screenshot, engineering_demo=args.engineering_demo)
    except Exception as exc:
        print(f"Assembly Workbench: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
