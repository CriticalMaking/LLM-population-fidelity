from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

DRIVERS = (
    "run_experiment.sh",
    "run_experiment_add_culture.sh",
    "run_experiment_base_models.sh",
    "run_reports.sh",
)


def _entry_points(repo_root: Path) -> list[Path]:
    scripts = sorted(
        path for path in (repo_root / "scripts").glob("*.sh") if path.name != "_common.sh"
    )
    return [*(repo_root / driver for driver in DRIVERS), *scripts]


def test_every_shell_entry_point_has_substantial_help(repo_root: Path) -> None:
    for script in _entry_points(repo_root):
        result = subprocess.run(
            ["/usr/bin/env", "bash", str(script), "--help"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
            cwd=repo_root,
        )
        assert result.returncode == 0, result.stderr
        assert "Usage:" in result.stdout
        assert len(result.stdout.splitlines()) >= 5


@pytest.mark.parametrize("driver", DRIVERS)
def test_root_driver_rejects_unknown_command(repo_root: Path, driver: str) -> None:
    result = subprocess.run(
        [f"./{driver}", "unknown-command"],
        capture_output=True,
        text=True,
        check=False,
        cwd=repo_root,
    )
    assert result.returncode == 2
    assert "Unknown command" in result.stderr
