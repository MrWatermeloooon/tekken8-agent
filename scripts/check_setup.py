from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys


def has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False


def command_output(command: list[str]) -> str:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"error: {exc}"
    output = (result.stdout or result.stderr).strip()
    return output.splitlines()[0] if output else f"exit={result.returncode}"


def main() -> int:
    print(f"python: {sys.version.split()[0]}")
    print(f"docker: {command_output(['docker', '--version']) if shutil.which('docker') else 'missing'}")
    print(f"diambra module: {'ok' if has_module('diambra') else 'missing'}")
    print(f"diambra module CLI: {command_output([sys.executable, '-m', 'diambra', '--version'])}")
    print(f"diambra executable: {shutil.which('diambra') or 'not on PATH'}")
    print(f"diambra.arena module: {'ok' if has_module('diambra.arena') else 'missing'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
