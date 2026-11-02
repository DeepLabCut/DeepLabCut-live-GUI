from dlclivegui.services.inference.processor import PoseProcessor


def test_set_backend_switches_to_poet(
    window,
) -> None:
    window._set_backend("poet")

    assert window._backend_name == "poet"
    assert window.action_backend_poet.isChecked()
    assert not window.action_backend_dlc.isChecked()
    assert window.inference_group.title() == "POET"
    assert "poet_weights" in (window.model_path_edit.placeholderText())


def test_poet_processor_is_created_lazily(
    window,
) -> None:
    assert "poet" not in window._pose_processors

    window._set_backend("poet")

    assert "poet" in window._pose_processors
    assert isinstance(
        window._pose_processors["poet"],
        PoseProcessor,
    )


def test_backend_switch_is_rejected_during_inference(
    window,
    monkeypatch,
) -> None:
    warnings: list[str] = []

    window._dlc_active = True
    monkeypatch.setattr(
        window,
        "_show_warning",
        warnings.append,
    )

    window._set_backend("poet")

    assert window._backend_name == "dlc"
    assert warnings == ["Stop pose inference before switching backends."]


def test_switching_backends_restores_separate_paths(
    window,
    tmp_path,
    monkeypatch,
) -> None:
    dlc_path = tmp_path / "dlc_model.pth"
    poet_path = tmp_path / "poet_weights.pth"

    dlc_path.touch()
    poet_path.touch()

    monkeypatch.setattr(
        window._model_path_store,
        "resolve",
        lambda _configured: str(dlc_path),
    )

    window.model_path_edit.setText(str(dlc_path))
    window._set_last_poet_weights_path(str(poet_path))

    window._set_backend("poet")
    assert window.model_path_edit.text() == str(poet_path)

    window._set_backend("dlc")
    assert window.model_path_edit.text() == str(dlc_path)


def test_restore_pose_backend_restores_poet(
    window,
) -> None:
    window.settings.setValue(
        "app/backend",
        "poet",
    )

    window._restore_pose_backend()

    assert window._backend_name == "poet"
    assert window.action_backend_poet.isChecked()


def test_configure_poet_registers_backend_factory(
    window,
    tmp_path,
) -> None:
    checkpoint = tmp_path / "weights.pth"
    checkpoint.touch()

    window._set_backend("poet")
    window.model_path_edit.setText(str(checkpoint))

    result = window._configure_poet()

    processor = window._active_pose_processor

    assert result is True
    assert isinstance(processor, PoseProcessor)
    assert processor.is_configured()
    assert processor._backend is None
