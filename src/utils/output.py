"""
Output management with timestamped folders, latest symlinks, and git logging.
"""

import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional


class OutputManager:
    """
    Manages timestamped output directories for runs.

    Creates directories like: data/{task}/YYYY-MM-DD_HH-MM-SS/
    Maintains a `latest` symlink pointing to the most recent successful run.
    Logs git commit hash and diff into each folder.
    """

    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._run_folder: Optional[Path] = None

    def create_run_folder(self) -> Path:
        """Create a new timestamped run folder and log git info."""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
        folder = self.base_dir / timestamp
        folder.mkdir(parents=True, exist_ok=True)
        self._run_folder = folder
        self._log_git_info(folder)
        return folder

    def mark_success(self) -> None:
        """Update the `latest` symlink to point to the current run folder."""
        if self._run_folder is None:
            raise RuntimeError("No run folder created yet. Call create_run_folder() first.")

        latest = self.base_dir / "latest"
        if latest.is_symlink() or latest.exists():
            if latest.is_dir() and not latest.is_symlink():
                import shutil
                shutil.rmtree(latest)
            else:
                latest.unlink()
        latest.symlink_to(self._run_folder.name)

    @property
    def run_folder(self) -> Optional[Path]:
        return self._run_folder

    @staticmethod
    def _log_git_info(folder: Path) -> None:
        git_info_path = folder / "git_info.txt"
        try:
            head = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=10,
            )
            diff = subprocess.run(
                ["git", "diff"],
                capture_output=True, text=True, timeout=30,
            )
            with open(git_info_path, "w") as f:
                f.write(head.stdout)
                if diff.stdout:
                    f.write("\n--- git diff ---\n")
                    f.write(diff.stdout)
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
