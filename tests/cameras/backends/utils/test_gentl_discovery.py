from __future__ import annotations

import os
from pathlib import Path

import pytest

from dlclivegui.cameras.backends.utils import gentl_discovery as gd

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def clear_shared_harvester_pool():
    gd.SharedHarvesterPool._entries.clear()
    yield
    gd.SharedHarvesterPool._entries.clear()


def test_cti_files_as_list_handles_none_strings_and_sequences():
    assert gd.cti_files_as_list(None) == []
    assert gd.cti_files_as_list("") == []
    assert gd.cti_files_as_list("  ") == []
    assert gd.cti_files_as_list("camera.cti") == ["camera.cti"]
    assert gd.cti_files_as_list(["a.cti", None, "", "  ", Path("b.cti")]) == ["a.cti", "b.cti"]


def test_discover_explicit_cti_file_without_harvester(tmp_path: Path):
    cti = tmp_path / "producer.cti"
    cti.write_text("", encoding="utf-8")

    candidates, diag = gd.discover_cti_files(
        cti_file=str(cti),
        include_env=False,
    )

    assert candidates == [str(cti.resolve())]
    assert diag.explicit_files == [str(cti)]
    assert diag.candidates == [str(cti.resolve())]
    assert diag.rejected == []


def test_discover_rejects_missing_explicit_file(tmp_path: Path):
    missing = tmp_path / "missing.cti"

    candidates, diag = gd.discover_cti_files(
        cti_file=str(missing),
        include_env=False,
    )

    assert candidates == []
    assert diag.rejected == [(str(missing.resolve()), "not a file (explicit)")]


def test_discover_rejects_non_cti_file(tmp_path: Path):
    not_cti = tmp_path / "producer.txt"
    not_cti.write_text("", encoding="utf-8")

    candidates, diag = gd.discover_cti_files(
        cti_file=str(not_cti),
        include_env=False,
    )

    assert candidates == []
    assert diag.rejected == [(str(not_cti.resolve()), "not a .cti (explicit)")]


def test_discover_accepts_missing_cti_when_must_exist_false(tmp_path: Path):
    missing = tmp_path / "missing.cti"

    candidates, diag = gd.discover_cti_files(
        cti_file=str(missing),
        include_env=False,
        must_exist=False,
    )

    assert candidates == [str(missing.resolve())]
    assert diag.rejected == []


def test_discover_extra_dir_collects_cti_files_non_recursive(tmp_path: Path):
    root = tmp_path / "ctis"
    root.mkdir()

    a = root / "a.cti"
    b = root / "b.cti"
    ignored = root / "ignored.txt"
    nested = root / "nested"
    nested.mkdir()
    nested_cti = nested / "nested.cti"

    a.write_text("", encoding="utf-8")
    b.write_text("", encoding="utf-8")
    ignored.write_text("", encoding="utf-8")
    nested_cti.write_text("", encoding="utf-8")

    candidates, diag = gd.discover_cti_files(
        include_env=False,
        extra_dirs=[str(root)],
        recursive_extra_search=False,
    )

    assert candidates == [str(a.resolve()), str(b.resolve())]
    assert diag.extra_dirs == [str(root)]


def test_discover_extra_dir_collects_cti_files_recursive(tmp_path: Path):
    root = tmp_path / "ctis"
    nested = root / "nested"
    nested.mkdir(parents=True)

    top = root / "top.cti"
    child = nested / "child.cti"

    top.write_text("", encoding="utf-8")
    child.write_text("", encoding="utf-8")

    candidates, _diag = gd.discover_cti_files(
        include_env=False,
        extra_dirs=[str(root)],
        recursive_extra_search=True,
    )

    assert candidates == [str(top.resolve()), str(child.resolve())]


def test_discover_deduplicates_candidates_preserving_order(tmp_path: Path):
    cti = tmp_path / "producer.cti"
    cti.write_text("", encoding="utf-8")

    candidates, diag = gd.discover_cti_files(
        cti_file=str(cti),
        cti_files=[str(cti)],
        extra_dirs=[str(tmp_path)],
        include_env=False,
    )

    assert candidates == [str(cti.resolve())]
    assert diag.candidates == [str(cti.resolve())]


def test_discover_env_var_direct_file(monkeypatch, tmp_path: Path):
    cti = tmp_path / "producer.cti"
    cti.write_text("", encoding="utf-8")

    monkeypatch.setenv("MY_GENTL_PATH", str(cti))

    candidates, diag = gd.discover_cti_files(
        include_env=True,
        env_vars=("MY_GENTL_PATH",),
    )

    assert candidates == [str(cti.resolve())]
    assert diag.env_vars_used == {"MY_GENTL_PATH": str(cti)}
    assert diag.env_paths_expanded == [str(cti)]


def test_discover_env_var_directory(monkeypatch, tmp_path: Path):
    cti = tmp_path / "producer.cti"
    cti.write_text("", encoding="utf-8")

    monkeypatch.setenv("MY_GENTL_PATH", str(tmp_path))

    candidates, diag = gd.discover_cti_files(
        include_env=True,
        env_vars=("MY_GENTL_PATH",),
    )

    assert candidates == [str(cti.resolve())]
    assert diag.env_vars_used == {"MY_GENTL_PATH": str(tmp_path)}
    assert diag.env_paths_expanded == [str(tmp_path)]


def test_discover_env_var_multiple_entries(monkeypatch, tmp_path: Path):
    d1 = tmp_path / "one"
    d2 = tmp_path / "two"
    d1.mkdir()
    d2.mkdir()

    cti1 = d1 / "one.cti"
    cti2 = d2 / "two.cti"
    cti1.write_text("", encoding="utf-8")
    cti2.write_text("", encoding="utf-8")

    monkeypatch.setenv("MY_GENTL_PATH", os.pathsep.join([str(d1), str(d2)]))

    candidates, _diag = gd.discover_cti_files(
        include_env=True,
        env_vars=("MY_GENTL_PATH",),
    )

    assert candidates == [str(cti1.resolve()), str(cti2.resolve())]


def test_validate_glob_pattern_rejects_empty_pattern():
    ok, reason = gd._validate_glob_pattern("")

    assert ok is False
    assert reason == "empty glob pattern"


def test_validate_glob_pattern_rejects_traversal(tmp_path: Path):
    pattern = str(tmp_path / ".." / "*.cti")

    ok, reason = gd._validate_glob_pattern(pattern)

    assert ok is False
    assert reason == "glob pattern contains '..' traversal"


def test_validate_glob_pattern_rejects_non_cti_pattern(tmp_path: Path):
    pattern = str(tmp_path / "*.txt")

    ok, reason = gd._validate_glob_pattern(pattern)

    assert ok is False
    assert reason == "glob pattern does not target .cti files"


def test_validate_glob_pattern_rejects_outside_allowed_roots(tmp_path: Path):
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()

    pattern = str(outside / "*.cti")

    ok, reason = gd._validate_glob_pattern(
        pattern,
        allowed_roots=[str(allowed)],
    )

    assert ok is False
    assert reason == "glob pattern base is outside allowed roots"


def test_discover_glob_pattern_with_allowed_root(tmp_path: Path):
    cti_dir = tmp_path / "ctis"
    cti_dir.mkdir()

    cti = cti_dir / "producer.cti"
    cti.write_text("", encoding="utf-8")

    candidates, diag = gd.discover_cti_files(
        cti_search_paths=[str(cti_dir / "*.cti")],
        include_env=False,
        root_globs_allowed=[str(tmp_path)],
    )

    assert candidates == [str(cti.resolve())]
    assert diag.rejected == []


def test_choose_cti_files_first_policy():
    assert gd.choose_cti_files(
        ["a.cti", "b.cti", "c.cti"],
        policy=gd.GenTLDiscoveryPolicy.FIRST,
        max_files=2,
    ) == ["a.cti", "b.cti"]


def test_choose_cti_files_raise_if_multiple_policy_raises():
    with pytest.raises(RuntimeError, match="Multiple GenTL producers"):
        gd.choose_cti_files(
            ["a.cti", "b.cti"],
            policy=gd.GenTLDiscoveryPolicy.RAISE_IF_MULTIPLE,
            max_files=1,
        )


def test_choose_cti_files_raise_if_multiple_policy_allows_within_limit():
    assert gd.choose_cti_files(
        ["a.cti"],
        policy=gd.GenTLDiscoveryPolicy.RAISE_IF_MULTIPLE,
        max_files=1,
    ) == ["a.cti"]


def test_choose_cti_files_newest_policy(tmp_path: Path):
    old = tmp_path / "old.cti"
    new = tmp_path / "new.cti"

    old.write_text("", encoding="utf-8")
    new.write_text("", encoding="utf-8")

    os.utime(old, (1000, 1000))
    os.utime(new, (2000, 2000))

    assert gd.choose_cti_files(
        [str(old), str(new)],
        policy=gd.GenTLDiscoveryPolicy.NEWEST,
        max_files=1,
    ) == [str(new)]


def test_choose_cti_files_empty_candidates():
    assert gd.choose_cti_files([]) == []


def test_choose_cti_files_unknown_policy_raises():
    with pytest.raises(ValueError, match="Unknown policy"):
        gd.choose_cti_files(
            ["a.cti"],
            policy=object(),  # type: ignore[arg-type]
        )


def test_shared_harvester_pool_reuses_entry_and_refcounts(monkeypatch, tmp_path: Path):
    calls: list[tuple[str, str | None]] = []

    class FakeHarvester:
        def __init__(self):
            calls.append(("init", None))

        def add_file(self, path: str) -> None:
            calls.append(("add_file", path))

        def update(self) -> None:
            calls.append(("update", None))

        def reset(self) -> None:
            calls.append(("reset", None))

    cti = tmp_path / "producer.cti"
    cti.write_text("", encoding="utf-8")

    monkeypatch.setattr(gd, "Harvester", FakeHarvester)
    gd.SharedHarvesterPool._entries.clear()

    entry1 = gd.SharedHarvesterPool.acquire([str(cti)])
    entry2 = gd.SharedHarvesterPool.acquire([str(cti)])

    assert entry1 is entry2
    assert gd.SharedHarvesterPool.get_refcount(entry1) == 2

    gd.SharedHarvesterPool.release(entry1)
    assert gd.SharedHarvesterPool.get_refcount(entry2) == 1

    gd.SharedHarvesterPool.release(entry2)
    assert gd.SharedHarvesterPool.get_refcount(entry2) == 0
    assert ("reset", None) in calls


def test_shared_harvester_entry_reports_failed_files(monkeypatch, tmp_path: Path):
    class FakeHarvester:
        def add_file(self, path: str) -> None:
            raise RuntimeError("load failed")

        def update(self) -> None:
            raise AssertionError("update should not be called")

        def reset(self) -> None:
            pass

    cti = tmp_path / "bad.cti"
    cti.write_text("", encoding="utf-8")

    monkeypatch.setattr(gd, "Harvester", FakeHarvester)

    with pytest.raises(RuntimeError, match="No GenTL producer"):
        gd.SharedHarvesterEntry([str(cti)])
