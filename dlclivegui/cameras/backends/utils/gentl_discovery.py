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

    def summarize(self, redact_env: bool = True) -> str:
        lines = []
        if self.explicit_files:
            lines.append(f"Explicit CTI file(s): {self.explicit_files}")
        if self.glob_patterns:
            lines.append(f"CTI glob pattern(s): {self.glob_patterns}")
        if self.env_vars_used:
            if redact_env:
                redacted_env = {k: ("<redacted>" if v else "<empty>") for k, v in self.env_vars_used.items()}
                lines.append(f"Env vars used: {redacted_env}")
            else:
                lines.append(f"Env vars used: {self.env_vars_used}")
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


def _expand_user_and_env(value: str) -> str:
    """
    Expand environment variables and '~' in a string path/pattern.
    pathlib does not expand env vars, so we use os.path.expandvars for that part.
    """
    if value is None:
        return ""
    s = str(value).strip()
    if not s:
        return ""
    # Expand env vars first (e.g., %VAR% / $VAR), then user home (~)
    s = os.path.expandvars(s)
    try:
        s = str(Path(s).expanduser())
    except Exception:
        # If expanduser fails for some reason, keep the env-expanded string
        pass
    return s


def _normalize_path(p: str) -> str:
    """
    Normalize a filesystem path in a cross-platform way:
    - expands ~ and environment variables
    - resolves to absolute where possible (without requiring existence)
    """
    expanded = _expand_user_and_env(p)
    pp = Path(expanded)
    try:
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


def _dedup_key(path_str: str) -> str:
    # Windows filesystem is case-insensitive by default -> normalize key case
    return path_str.casefold() if os.name == "nt" else path_str


_GLOB_META_CHARS = set("*?[")


def _pattern_has_glob(s: str) -> bool:
    return any(ch in s for ch in _GLOB_META_CHARS)


def _pattern_static_prefix(pattern: str) -> str:
    """
    Return the substring up to the first glob metacharacter (* ? [).
    This is used as a "base path" to constrain globbing.
    """
    for i, ch in enumerate(pattern):
        if ch in _GLOB_META_CHARS:
            return pattern[:i]
    return pattern


def _is_path_within(child: Path, parent: Path) -> bool:
    """
    Cross-version safe "is_relative_to" implementation.
    """
    try:
        child.relative_to(parent)
        return True
    except Exception:
        return False


def _validate_glob_pattern(
    pattern: str,
    *,
    allowed_roots: Sequence[str] | None = None,
    require_cti_suffix: bool = True,
) -> tuple[bool, str | None]:
    """
    Validate user-supplied glob patterns to reduce filesystem probing risk.

    Rules (conservative but practical):
    - Must expand (~ and env vars) into an absolute-ish location (prefix must exist as a path parent)
    - Must not include '..' path traversal segments
    - Must have a non-trivial static prefix (not empty / not root-only like '/' or 'C:\\')
    - Optionally restrict to allowed roots (directories)
    - Optionally require that the pattern looks like it targets .cti files
    """
    if not pattern or not str(pattern).strip():
        return False, "empty glob pattern"

    expanded = _expand_user_and_env(pattern).strip()

    # Basic traversal guard
    parts = Path(expanded).parts
    if any(p == ".." for p in parts):
        return False, "glob pattern contains '..' traversal"

    if require_cti_suffix:
        # Encourage patterns that clearly target CTIs, e.g. '*.cti' or 'foo*.cti'
        lower = expanded.lower()
        if ".cti" not in lower:
            return False, "glob pattern does not target .cti files"

    # Compute static prefix up to first glob meta-char
    prefix = _pattern_static_prefix(expanded).strip()
    if not prefix:
        return False, "glob pattern has no static base path"

    prefix_path = Path(prefix)

    # If prefix is a file-like thing, use its parent as base; otherwise use itself.
    # Example: "C:\\dir\\*.cti" -> base = "C:\\dir"
    base = prefix_path.parent if prefix_path.suffix else prefix_path

    # Prevent overly broad patterns like "/" or "C:\\"
    try:
        resolved_base = base.resolve(strict=False)
    except Exception:
        resolved_base = base

    # If base is a drive root or filesystem root, reject
    # - POSIX: "/" -> parent == itself
    # - Windows: "C:\\" -> parent often == itself
    try:
        if resolved_base == resolved_base.parent:
            return False, "glob pattern base is filesystem root (too broad)"
    except Exception:
        # If we can't determine, err on conservative side
        return False, "glob pattern base could not be validated"

    # Optional allowlist enforcement
    if allowed_roots:
        ok = False
        for root in allowed_roots:
            try:
                r = Path(_normalize_path(root))
            except Exception:
                r = Path(root)
            try:
                r_resolved = r.resolve(strict=False)
            except Exception:
                r_resolved = r

            try:
                b_resolved = resolved_base.resolve(strict=False)
            except Exception:
                b_resolved = resolved_base

            if _is_path_within(b_resolved, r_resolved):
                ok = True
                break
        if not ok:
            return False, "glob pattern base is outside allowed roots"

    return True, None


def _glob_limited(pattern: str, *, max_hits: int = 200) -> list[str]:
    """
    Iterate matches with an upper bound to prevent expensive scans.
    Uses iglob to avoid materializing huge lists.
    """
    out: list[str] = []
    # Note: recursive globbing via "**" typically requires recursive=True.
    # We intentionally keep recursive off here to reduce scanning.
    for hit in glob.iglob(pattern, recursive=False):
        out.append(hit)
        if len(out) >= max_hits:
            break
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
    allow_globs: bool = True,
    root_globs_allowed: Sequence[str] | None = None,
    max_glob_hits_per_pattern: int = 200,
) -> tuple[list[str], CTIDiscoveryDiagnostics]:
    """
    Discover candidate GenTL producer (.cti) files from multiple sources.

    Returns:
        (candidates, diagnostics)

    Notes:
    - If must_exist=True (recommended), only existing files are returned at duscovery time.
        - Best-effort checks, files may still be missing at load time (e.g. deleted after discovery).
        - Callers should handle load-time errors gracefully regardless.
        - Glob patterns can enumerate filesystem entries is user-controlled.
          Use allow_globs=False to disable globbing and treat patterns as literal paths.
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
        p = Path(norm)
        if must_exist and not p.is_file():
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
        expanded_pat = _expand_user_and_env(pat)

        if not allow_globs:
            rejected.append((_normalize_path(expanded_pat), "glob patterns disabled"))
            continue

        ok, reason = _validate_glob_pattern(
            expanded_pat,
            allowed_roots=root_globs_allowed,
            require_cti_suffix=True,
        )
        if not ok:
            rejected.append((_normalize_path(expanded_pat), f"glob pattern rejected: {reason}"))
            continue

        for hit in _glob_limited(expanded_pat, max_hits=max_glob_hits_per_pattern):
            _add_candidate(hit, f"glob:{pat}")

    # Process env var entries
    for entry in env_entries:
        norm_entry = _normalize_path(entry)
        p = Path(norm_entry)
        if p.is_file():  # let _add_candidate check .cti extension and existence
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
        key = _dedup_key(c)
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

        def _newest_mtime(p: str) -> float:
            try:
                if not Path(p).exists():
                    return 0.0
                return Path(p).stat().st_mtime
            except OSError:
                return 0.0

        cand_sorted = sorted(cand, key=_newest_mtime, reverse=True)
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
