"""One-time publishing script — run when you have a PyPI token.

Usage:
  export TWINE_USERNAME=__token__
  export TWINE_PASSWORD=pypi-AgE...your-token...
  .venv/bin/python scripts/publish.py

This exists so the publish step is reproducible and doesn't get lost in
shell history. NOT run automatically — only when you decide v0.1.0 is ready.
"""
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DIST = REPO_ROOT / "dist"


def main(target: str = "pypi") -> int:
    """Build then upload. Target: 'pypi' (production) or 'testpypi'."""
    print(f"Building sdist + wheel...")
    r = subprocess.run(
        ["python3", "-m", "build", "--sdist", "--wheel"],
        cwd=REPO_ROOT,
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(r.stderr)
        return 1

    print(f"Uploading to {target}...")
    r = subprocess.run(
        ["python3", "-m", "twine", "upload", "--repository", target, str(DIST / "*")],
        cwd=REPO_ROOT,
    )
    return r.returncode


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "pypi"
    sys.exit(main(target))
