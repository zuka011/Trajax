from pathlib import Path


def project_root(*, marker: str = "pyproject.toml") -> Path:
    """Find project root by searching upward for a marker file."""
    current = Path.cwd()
    for parent in [current, *current.parents]:
        if (parent / marker).exists():
            return parent

    raise FileNotFoundError(f"Could not find {marker} in any parent directory")
