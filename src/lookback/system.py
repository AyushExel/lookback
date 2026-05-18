"""Lightweight system probe + per-model recommendation.

Detects total RAM, OS, architecture, and the Apple-Silicon flag without
adding heavy dependencies. The recommendation logic is intentionally simple
and conservative: when in doubt, recommend the smaller / faster model.
"""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass

from lookback.embed.models import IMAGE_MODELS, TEXT_MODELS


@dataclass(frozen=True, slots=True)
class SystemProfile:
    os_name: str  # "Darwin" | "Linux" | "Windows"
    arch: str  # "arm64" | "x86_64" | ...
    is_apple_silicon: bool
    total_ram_gb: float | None  # None if we couldn't detect
    cpu_count: int


def _detect_ram_gb() -> float | None:
    """Best-effort total physical RAM in GB. Returns None if we can't tell."""
    # POSIX: derive from sysconf where available (macOS, most Linuxen).
    try:
        if (
            hasattr(os, "sysconf")
            and "SC_PAGE_SIZE" in os.sysconf_names
            and "SC_PHYS_PAGES" in os.sysconf_names
        ):
            page_size = os.sysconf("SC_PAGE_SIZE")
            phys_pages = os.sysconf("SC_PHYS_PAGES")
            if page_size > 0 and phys_pages > 0:
                return (page_size * phys_pages) / (1024**3)
    except (ValueError, OSError):
        pass

    # /proc/meminfo fallback for Linux.
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return kb / (1024**2)
    except (FileNotFoundError, ValueError, PermissionError):
        pass

    return None


def detect_profile() -> SystemProfile:
    os_name = platform.system()
    arch = platform.machine()
    return SystemProfile(
        os_name=os_name,
        arch=arch,
        is_apple_silicon=(os_name == "Darwin" and arch == "arm64"),
        total_ram_gb=_detect_ram_gb(),
        cpu_count=os.cpu_count() or 1,
    )


def recommend_text_model(profile: SystemProfile | None = None) -> str:
    """Return the name of the recommended text model for this system.

    Logic: prefer ``nomic-v1.5`` (the safe default) unless the user has at
    least 16 GB RAM AND we're confident enough about the measurement, in
    which case we mention ``nomic-v2-moe`` as an upgrade. We default to
    v1.5 even on big machines because the MoE variant trades ~1.5 GB of RAM
    for a modest retrieval quality bump that most users won't notice.
    """
    _ = profile or detect_profile()
    # v1.5 is always the recommended default in v0.
    if "nomic-v1.5" not in TEXT_MODELS:  # pragma: no cover - defensive
        return next(iter(TEXT_MODELS))
    return "nomic-v1.5"


def recommend_image_model(profile: SystemProfile | None = None) -> str:
    """Return the recommended image model — currently always ``mobileclip-s2``."""
    _ = profile or detect_profile()
    return "mobileclip-s2"


def describe_profile(profile: SystemProfile) -> str:
    """One-line human-readable summary, used by ``lookback init``."""
    parts = [profile.os_name, profile.arch]
    if profile.is_apple_silicon:
        parts.append("Apple Silicon")
    if profile.total_ram_gb is not None:
        parts.append(f"{profile.total_ram_gb:.1f} GB RAM")
    else:
        parts.append("RAM unknown")
    parts.append(f"{profile.cpu_count} CPU")
    return " · ".join(parts)


def model_choices_text() -> list[str]:
    return sorted(TEXT_MODELS.keys())


def model_choices_image() -> list[str]:
    return sorted(IMAGE_MODELS.keys())
