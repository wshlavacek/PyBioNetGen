from pathlib import Path
from typing import Final, List, Tuple, Union


VersionPart = Union[int, str]
VersionTuple = Tuple[VersionPart, ...]

_VERSION_FILE: Final = Path(__file__).resolve().parents[1] / "assets" / "VERSION"
_SUFFIXES: Final = {
    "alpha": "a",
    "beta": "b",
    "candidate": "rc",
    "rc": "rc",
    "final": "",
}


def _parse_version_text(text: str) -> VersionTuple:
    parts: List[VersionPart] = []
    for token in text.split():
        try:
            parts.append(int(token))
        except ValueError:
            parts.append(token)
    return tuple(parts)


def _format_version(version: VersionTuple) -> str:
    if len(version) < 3:
        raise ValueError(f"Expected at least major/minor/patch version parts, got: {version!r}")

    major, minor, patch = version[:3]
    base = f"{major}.{minor}.{patch}"
    if len(version) < 5:
        return base

    stage = str(version[3]).lower()
    suffix = _SUFFIXES.get(stage)
    if suffix is None:
        raise ValueError(f"Unsupported release stage {version[3]!r} in VERSION file")
    if not suffix:
        return base
    return f"{base}{suffix}{version[4]}"


def _read_version_file(path: Path = _VERSION_FILE) -> VersionTuple:
    return _parse_version_text(path.read_text(encoding="utf-8"))


VERSION: Final[VersionTuple] = _read_version_file()
__version__ = _format_version(VERSION)


def get_version(version: VersionTuple = VERSION) -> str:
    return _format_version(version)
