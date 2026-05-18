"""Feature under test: ``detect_profile`` returns a usable system profile
on every platform the tests run on (we run on macOS in dev; CI may run on
Linux). Specifically: ``os_name`` and ``arch`` are non-empty, ``cpu_count``
is positive, and ``total_ram_gb`` is either a positive float or ``None``.
"""

from __future__ import annotations

from lookback.system import SystemProfile, describe_profile, detect_profile


def test_detect_profile_returns_populated_fields() -> None:
    p = detect_profile()
    assert isinstance(p, SystemProfile)
    assert p.os_name in {"Darwin", "Linux", "Windows"}
    assert p.arch  # not empty
    assert p.cpu_count >= 1
    assert p.total_ram_gb is None or p.total_ram_gb > 0


def test_apple_silicon_flag_consistent_with_os_and_arch() -> None:
    p = detect_profile()
    if p.is_apple_silicon:
        assert p.os_name == "Darwin"
        assert p.arch == "arm64"
    else:
        assert not (p.os_name == "Darwin" and p.arch == "arm64")


def test_describe_profile_is_human_readable() -> None:
    p = detect_profile()
    s = describe_profile(p)
    assert p.os_name in s
    assert p.arch in s
    assert "CPU" in s
