"""Helpers to locate .cti GenTL producer files from various sources
(explicit, env vars, glob patterns, etc.) for GenTL-based camera backends."""

# dlclivegui/cameras/backends/utils/gentl_discovery.py
from __future__ import annotations

import glob
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path


class GenTLDiscoveryPolicy(Enum):
    FIRST = auto()  # default: take first N candidates in order found
    NEWEST = auto()  # take N candidates with most recent modification time (mtime)
    RAISE_IF_MULTIPLE = auto()  # if > N candidates, raise an error to avoid ambiguity (forces explicit config)


@dataclass
class CTIDiscoveryDiagnostics:
    explicit_files: list[str] = field(default_factory=list)
    glob_patterns: list[str] = field(default_factory=list)
    env_vars_used: dict[str, str] = field(default_factory=dict)  # name -> raw value
    env_paths_expanded: list[str] = field(default_factory=list)  # directories/files derived from env vars
    extra_dirs: list[str] = field(default_factory=list)

    candidates: list[str] = field(default_factory=list)
    rejected: list[tuple[str, str]] = field(default_factory=list)  # (path, reason)

    def summarize(self) -> str:
        lines = []
        if self.explicit_files:
            lines.append(f"Explicit CTI file(s): {self.explicit_files}")
        if self.glob_patterns:
            lines.append(f"CTI glob pattern(s): {self.glob_patterns}")
        if self.env_vars_used:
            # Keep raw env var values in summary; you can redact if needed
            lines.append(f"Env vars: {self.env_vars_used}")
        if self.env_paths_expanded:
            lines.append(f"Env-derived path entries: {self.env_paths_expanded}")
        if self.extra_dirs:
            lines.append(f"Extra CTI dirs: {self.extra_dirs}")
        if self.candidates:
            lines.append(f"CTI candidate(s) ({len(self.candidates)}): {self.candidates}")
        if self.rejected:
            lines.append(f"Rejected ({len(self.rejected)}): " + "; ".join([f"{p} ({r})" for p, r in self.rejected]))
        return "\n".join(lines)


def cti_files_as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value if v is not None and str(v).strip()]
    s = str(value).strip()
    return [s] if s else []


def _normalize_path(p: str) -> str:
    """
    Normalize a filesystem path in a cross-platform way:
    - expands ~ and environment variables
    - resolves to absolute where possible (without requiring existence)
    """
    pp = Path(os.path.expandvars(os.path.expanduser(p)))
    try:
        # resolve(False) avoids raising if parts don't exist (py>=3.9)
        return str(pp.resolve(strict=False))
    except Exception:
        return str(pp.absolute())


def _iter_cti_files_in_dir(directory: str, recursive: bool = False) -> Iterable[str]:
    """
    Yield *.cti files in directory. Non-recursive by default (faster, safer).
    """
    d = Path(directory)
    if not d.is_dir():
        return
    if recursive:
        yield from (str(p) for p in d.rglob("*.cti"))
    else:
        yield from (str(p) for p in d.glob("*.cti"))


def _split_env_paths(raw: str) -> list[str]:
    """
    Split environment variable paths using os.pathsep (cross-platform).
    Also trims whitespace and strips surrounding quotes.
    """
    out: list[str] = []
    for item in (raw or "").split(os.pathsep):
        s = item.strip().strip('"').strip("'")
        if s:
            out.append(s)
    return out


def discover_cti_files(
    *,
    cti_file: str | None = None,
    cti_files: Sequence[str] | None = None,
    cti_search_paths: Sequence[str] | None = None,
    include_env: bool = True,
    env_vars: Sequence[str] = ("GENICAM_GENTL64_PATH", "GENICAM_GENTL32_PATH"),
    extra_dirs: Sequence[str] | None = None,
    recursive_env_search: bool = False,
    recursive_extra_search: bool = False,
    must_exist: bool = True,
) -> tuple[list[str], CTIDiscoveryDiagnostics]:
    """
    Discover candidate GenTL producer (.cti) files from multiple sources.

    Returns:
        (candidates, diagnostics)

    Notes:
    - If must_exist=True (recommended), only existing files are returned.
    - Env vars are parsed as path lists; each entry may be a directory OR a .cti file.
    """
    diag = CTIDiscoveryDiagnostics()

    # 1) Explicit CTI file(s)
    explicit = []
    explicit += cti_files_as_list(cti_file)
    explicit += cti_files_as_list(cti_files)
    diag.explicit_files = explicit[:]

    # 2) Glob patterns
    patterns = cti_files_as_list(cti_search_paths)
    diag.glob_patterns = patterns[:]

    # 3) Env var paths
    env_entries: list[str] = []
    if include_env:
        for name in env_vars:
            raw = os.environ.get(name, "")
            if raw:
                diag.env_vars_used[name] = raw
                env_entries.extend(_split_env_paths(raw))
    diag.env_paths_expanded = env_entries[:]

    # 4) Extra directories
    extras = cti_files_as_list(extra_dirs)
    diag.extra_dirs = extras[:]

    candidates: list[str] = []
    rejected: list[tuple[str, str]] = []

    def _add_candidate(path: str, reason_ctx: str) -> None:
        norm = _normalize_path(path)
        if must_exist and not Path(norm).is_file():
            rejected.append((norm, f"not a file ({reason_ctx})"))
            return
        if not norm.lower().endswith(".cti"):
            rejected.append((norm, f"not a .cti ({reason_ctx})"))
            return
        candidates.append(norm)

    # Process explicit files
    for p in explicit:
        _add_candidate(p, "explicit")

    # Process glob patterns
    for pat in patterns:
        # Normalize only for readability; glob needs pattern semantics, so we expanduser/vars but keep globbing
        expanded_pat = os.path.expandvars(os.path.expanduser(pat))
        for hit in glob.glob(expanded_pat):
            _add_candidate(hit, f"glob:{pat}")

    # Process env var entries
    for entry in env_entries:
        norm_entry = _normalize_path(entry)
        p = Path(norm_entry)
        if p.is_file() and p.suffix.lower() == ".cti":
            _add_candidate(norm_entry, "env:file")
        elif p.is_dir():
            for f in _iter_cti_files_in_dir(norm_entry, recursive=recursive_env_search):
                _add_candidate(f, "env:dir")
        else:
            rejected.append((norm_entry, "env entry missing (not file/dir)"))

    # Process extra dirs
    for d in extras:
        norm_d = _normalize_path(d)
        if Path(norm_d).is_dir():
            for f in _iter_cti_files_in_dir(norm_d, recursive=recursive_extra_search):
                _add_candidate(f, "extra:dir")
        elif Path(norm_d).is_file():
            _add_candidate(norm_d, "extra:file")
        else:
            rejected.append((norm_d, "extra entry missing (not file/dir)"))

    # Deduplicate while preserving order
    seen = set()
    unique: list[str] = []
    for c in candidates:
        key = c.lower() if os.name == "nt" else c  # Windows case-insensitive
        if key in seen:
            continue
        seen.add(key)
        unique.append(c)

    diag.candidates = unique[:]
    diag.rejected = rejected[:]
    return unique, diag


def choose_cti_files(
    candidates: Sequence[str],
    *,
    policy: GenTLDiscoveryPolicy = GenTLDiscoveryPolicy.FIRST,
    max_files: int = 1,
) -> list[str]:
    """
    Choose which CTI file(s) to load from candidates.

    policy:
        - FIRST: take the first N candidates (default)
        - NEWEST: take the N most recently modified candidates
        - RAISE_IF_MULTIPLE: if more than N candidates, raise an error (to avoid ambiguity)
    """
    cand = [str(c) for c in candidates if c]
    if not cand:
        return []

    if policy == GenTLDiscoveryPolicy.NEWEST:
        cand_sorted = sorted(cand, key=lambda p: Path(p).stat().st_mtime if Path(p).exists() else 0.0, reverse=True)
        return cand_sorted[:max_files]

    if policy == GenTLDiscoveryPolicy.FIRST:
        return cand[:max_files]

    if policy == GenTLDiscoveryPolicy.RAISE_IF_MULTIPLE:
        if len(cand) > max_files:
            raise RuntimeError(
                f"Multiple GenTL producers (.cti) found ({len(cand)}). "
                f"Please set properties.gentl.cti_file explicitly. Candidates: {cand}"
            )
        return cand[:max_files]

    raise ValueError(f"Unknown policy: {policy!r}")
