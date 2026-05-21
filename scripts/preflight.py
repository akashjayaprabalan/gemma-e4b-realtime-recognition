from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "mlx-community/gemma-4-e4b-it-8bit"
MIN_DISK_GIB = 12


def main() -> int:
    checks = [
        check_python(),
        check_machine(),
        check_disk(),
        check_imports(),
        check_hub_model(),
    ]

    print("\nPreflight")
    for ok, message in checks:
        print(f"{'OK' if ok else 'FAIL'}  {message}")

    return 0 if all(ok for ok, _ in checks) else 1


def check_python() -> tuple[bool, str]:
    version = sys.version_info
    ok = version.major == 3 and version.minor == 13
    return ok, f"Python {platform.python_version()} (expected 3.13)"


def check_machine() -> tuple[bool, str]:
    machine = platform.machine()
    ok = machine == "arm64"
    return ok, f"Machine {machine} (expected Apple Silicon arm64)"


def check_disk() -> tuple[bool, str]:
    usage = shutil.disk_usage(ROOT)
    free_gib = usage.free / (1024**3)
    return free_gib >= MIN_DISK_GIB, f"{free_gib:.1f} GiB free disk"


def check_imports() -> tuple[bool, str]:
    missing = []
    for module in ("fastapi", "PIL", "huggingface_hub", "mlx", "mlx_vlm"):
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    if missing:
        return False, f"Missing imports: {', '.join(missing)}"
    return True, "Python dependencies import"


def check_hub_model() -> tuple[bool, str]:
    try:
        output = subprocess.check_output(
            ["curl", "-fsSL", f"https://huggingface.co/api/models/{MODEL_ID}"],
            text=True,
            timeout=20,
        )
        data = json.loads(output)
    except Exception as exc:
        return False, f"Hub lookup failed for {MODEL_ID}: {exc}"

    if data.get("gated"):
        return False, f"{MODEL_ID} is gated"

    modified = data.get("lastModified", "unknown")
    sha = str(data.get("sha", "unknown"))[:12]
    return True, f"{MODEL_ID} available, modified {modified}, sha {sha}"


if __name__ == "__main__":
    raise SystemExit(main())
