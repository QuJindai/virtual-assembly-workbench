"""Preserve installed wheel license texts and an exact build inventory."""
from __future__ import annotations

import importlib.metadata as metadata
import json
from pathlib import Path
import shutil
import sys


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "build" / "licenses"
    output.mkdir(parents=True, exist_ok=True)
    inventory = []
    for distribution in sorted(metadata.distributions(), key=lambda d: d.metadata["Name"].lower()):
        name = distribution.metadata["Name"]
        inventory.append({"name": name, "version": distribution.version})
        for entry in distribution.files or []:
            if not any(word in entry.name.lower() for word in ("license", "copying", "notice", "copyright")):
                continue
            source = Path(distribution.locate_file(entry))
            if source.is_file() and source.suffix.lower() not in {".py", ".pyc", ".so", ".dll", ".pyd"}:
                destination = output / name / str(entry).replace("..", "parent")
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
    for source in (root / "licenses").glob("*.txt"):
        shutil.copy2(source, output / source.name)
    python_license = Path(sys.base_prefix) / "LICENSE.txt"
    if python_license.is_file():
        shutil.copy2(python_license, output / "Python-LICENSE.txt")
    (output / "inventory.json").write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    print(f"Collected licenses and inventory for {len(inventory)} distributions")


if __name__ == "__main__":
    main()
