import os
from pathlib import Path


def ensure_dir(directory):
    """Create directory if it doesn't exist."""
    Path(directory).mkdir(parents=True, exist_ok=True)
    return directory


def get_project_root():
    """Get the project root directory."""
    return Path(__file__).parent.parent


def get_data_path(subdir=""):
    """Get path to data directory or subdirectory."""
    root = get_project_root()
    path = root / "data" / subdir
    ensure_dir(path)
    return path


def get_results_path(subdir=""):
    """Get path to results directory or subdirectory."""
    root = get_project_root()
    path = root / "results" / subdir
    ensure_dir(path)
    return path