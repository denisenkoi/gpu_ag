"""
Cross-platform path utilities for Windows <-> WSL compatibility.

Automatically converts paths based on current platform:
- On Linux (WSL): E:/Projects/... -> /mnt/e/Projects/...
- On Windows: /mnt/e/Projects/... -> E:/Projects/...

Usage:
    from path_utils import normalize_path, to_wsl_path, to_windows_path

    # Auto-detect and convert
    path = normalize_path("E:/Projects/Rogii/ss")

    # Explicit conversion
    wsl_path = to_wsl_path("E:/Projects/Rogii/ss")
    win_path = to_windows_path("/mnt/e/Projects/Rogii/ss")
"""
import platform
import re
from pathlib import Path
from typing import Union

# Detect platform once at import
IS_WSL = platform.system() == 'Linux'
IS_WINDOWS = platform.system() == 'Windows'


def to_wsl_path(path: Union[str, Path]) -> str:
    """
    Convert Windows path to WSL path.

    E:/Projects/Rogii/ss -> /mnt/e/Projects/Rogii/ss
    E:\\Projects\\Rogii\\ss -> /mnt/e/Projects/Rogii/ss
    """
    path_str = str(path)

    # Already WSL path
    if path_str.startswith('/mnt/'):
        return path_str

    # Windows path with drive letter (E:/ or E:\)
    match = re.match(r'^([A-Za-z]):[/\\](.*)$', path_str)
    if match:
        drive = match.group(1).lower()
        rest = match.group(2).replace('\\', '/')
        return f"/mnt/{drive}/{rest}"

    # Not a Windows path, return as-is
    return path_str


def to_windows_path(path: Union[str, Path]) -> str:
    """
    Convert WSL path to Windows path.

    /mnt/e/Projects/Rogii/ss -> E:/Projects/Rogii/ss
    """
    path_str = str(path)

    # Already Windows path
    if re.match(r'^[A-Za-z]:[/\\]', path_str):
        return path_str.replace('\\', '/')

    # WSL path
    match = re.match(r'^/mnt/([a-z])/(.*)$', path_str)
    if match:
        drive = match.group(1).upper()
        rest = match.group(2)
        return f"{drive}:/{rest}"

    # Not a WSL path, return as-is
    return path_str


def normalize_path(path: Union[str, Path]) -> str:
    """
    Normalize path for current platform.

    On WSL: converts Windows paths to /mnt/x/... format
    On Windows: converts WSL paths to X:/... format

    Args:
        path: Path string or Path object (Windows or WSL format)

    Returns:
        Path string in format appropriate for current platform
    """
    if IS_WSL:
        return to_wsl_path(path)
    elif IS_WINDOWS:
        return to_windows_path(path)
    else:
        # Unknown platform, return as-is
        return str(path)


def normalize_path_obj(path: Union[str, Path]) -> Path:
    """
    Normalize path and return as Path object.
    """
    return Path(normalize_path(path))


def get_env_path(env_var: str, default: str = '') -> str:
    """
    Get path from environment variable and normalize for current platform.

    Args:
        env_var: Environment variable name
        default: Default value if not set

    Returns:
        Normalized path string
    """
    import os
    value = os.getenv(env_var, default)
    if value:
        return normalize_path(value)
    return value


def get_env_path_obj(env_var: str, default: str = '') -> Path:
    """
    Get path from environment variable and return as normalized Path object.
    """
    path_str = get_env_path(env_var, default)
    if path_str:
        return Path(path_str)
    return Path(default) if default else Path()
