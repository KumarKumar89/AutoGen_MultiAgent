#!/usr/bin/env python3
"""
One-command runner for the AutoGen Multi-Agent repository.

Usage:
    python run.py            # start FastAPI web service on http://0.0.0.0:8000
    python run.py --cli      # run the multi-agent pipeline once in the terminal
    python run.py --test     # run the forensic test-suite (pytest)
    python run.py --port 9000

What it does automatically:
  1. Verifies Python version
  2. Extracts bundled data/ and imgs/ assets from the repo zips if missing
  3. Installs required dependencies (fastapi, uvicorn, pydantic) if missing
  4. Launches the requested mode
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PKG = ROOT / "AutoGen_MultiAgent"
DATA_DIR = PKG / "data"
IMGS_DIR = PKG / "imgs"


def step(msg: str) -> None:
    print(f"[runner] {msg}")


def ensure_python() -> None:
    if sys.version_info < (3, 10):
        sys.exit(f"[runner] ERROR: Python >= 3.10 required, found {sys.version.split()[0]}")
    step(f"Python {sys.version.split()[0]} OK")


def ensure_assets() -> None:
    """Extract cv.txt/courses.txt and cover image from the bundled zips."""
    if not DATA_DIR.exists():
        zips = sorted(PKG.glob("data-*.zip"))
        if not zips:
            sys.exit("[runner] ERROR: data/ missing and no data-*.zip found to extract")
        step(f"Extracting {zips[0].name} -> data/")
        zipfile.ZipFile(zips[0]).extractall(PKG)
    for f in ("cv.txt", "courses.txt"):
        if not (DATA_DIR / f).exists():
            sys.exit(f"[runner] ERROR: data/{f} still missing after extraction")
    if not IMGS_DIR.exists():
        zips = sorted(PKG.glob("imgs-*.zip"))
        if zips:
            step(f"Extracting {zips[0].name} -> imgs/")
            zipfile.ZipFile(zips[0]).extractall(PKG)
    step("Assets OK (data/cv.txt, data/courses.txt)")


def ensure_deps() -> None:
    missing = [m for m in ("fastapi", "uvicorn", "pydantic") if importlib.util.find_spec(m) is None]
    if missing:
        step(f"Installing missing dependencies: {', '.join(missing)}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", *missing])
    step("Dependencies OK")


def run_cli() -> None:
    step("Running multi-agent pipeline in CLI mode …")
    subprocess.check_call([sys.executable, str(PKG / "run_live_multi_agent.py")], cwd=PKG)


def run_tests() -> None:
    if importlib.util.find_spec("pytest") is None:
        step("Installing pytest …")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "pytest"])
    step("Running forensic test-suite …")
    # -p no:cacheprovider keeps the repo clean; -q handled by caller via -v
    env = dict(__import__("os").environ, PYTHONDONTWRITEBYTECODE="1")
    subprocess.check_call(
        [sys.executable, "-m", "pytest", str(ROOT / "tests"), "-v",
         "-p", "no:libtmux"],  # libtmux ships a broken plugin in some images
        cwd=ROOT, env=env,
    )


def run_server(host: str, port: int) -> None:
    step(f"Starting FastAPI service  →  http://{host if host != '0.0.0.0' else 'localhost'}:{port}")
    step("  HTML console :  /          API docs :  /docs")
    step("  Health       :  /health    Run      :  POST /api/run")
    sys.path.insert(0, str(PKG))
    import uvicorn
    uvicorn.run("app:app", host=host, port=port, reload=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Runner for the AutoGen Multi-Agent repository")
    ap.add_argument("--cli", action="store_true", help="run pipeline once in terminal instead of serving")
    ap.add_argument("--test", action="store_true", help="run the forensic test suite")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    ensure_python()
    ensure_assets()
    ensure_deps()

    if args.test:
        run_tests()
    elif args.cli:
        run_cli()
    else:
        run_server(args.host, args.port)


if __name__ == "__main__":
    main()
