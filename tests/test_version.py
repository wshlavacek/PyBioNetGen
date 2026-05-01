import pytest

from bionetgen.core.version import VERSION, __version__, get_version


def test_dunder_version_matches_default_get_version():
    assert __version__ == get_version()
    assert __version__ == get_version(VERSION)


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ((1, 2, 3, "final", 0), "1.2.3"),
        ((1, 2, 3, "alpha", 4), "1.2.3a4"),
        ((1, 2, 3, "beta", 5), "1.2.3b5"),
        ((1, 2, 3, "candidate", 6), "1.2.3rc6"),
        ((1, 2, 3, "rc", 7), "1.2.3rc7"),
    ],
)
def test_get_version_formats_supported_release_stages(version, expected):
    assert get_version(version) == expected
