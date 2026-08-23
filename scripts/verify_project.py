from __future__ import annotations

import compileall
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ok = compileall.compile_dir(ROOT / "src", quiet=1)
    ok &= compileall.compile_dir(ROOT / "scripts", quiet=1)
    assert ok, "Python compilation failed"
    json.loads((ROOT / "configs" / "paper_boat.json").read_text(encoding="utf-8"))
    required = [
        "README.md", "requirements.txt", "src/environment.py", "src/agents/smc_learning.py",
        "scripts/run_particle_demo.py", "scripts/run_demo.py", "scripts/run_paper_scale.py", "tests/test_core.py",
    ]
    for rel in required:
        assert (ROOT / rel).exists(), rel
    print("Project verification passed.")


if __name__ == "__main__":
    main()
