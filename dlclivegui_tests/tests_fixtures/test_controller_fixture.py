@pytest.mark.integration
def test_controller_emits_started_and_frames(qtbot, multi_camera_controller, app_config_two_cams):
    ctrl = multi_camera_controller
    cams = app_config_two_cams.multi_camera.get_active_cameras()

    # Start and wait until all_started is emitted
    with qtbot.waitSignal(ctrl.all_started, timeout=3000):
        ctrl.start(cams)

    assert ctrl.is_running()
    assert ctrl.get_active_count() == 2

    # Wait for a frame_ready emission from worker threads
    with qtbot.waitSignal(ctrl.frame_ready, timeout=3000) as blocker:
        pass

    frame_data = blocker.args[0]
    assert frame_data is not None
    assert len(frame_data.frames) >= 1
    assert frame_data.source_camera_id in frame_data.frames

    # Since we started two cameras, we usually get both quickly.
    # But allow some jitter and wait briefly for the second camera frame to appear.
    expected_ids = {f"{c.backend}:{c.index}" for c in cams}

    qtbot.waitUntil(
        lambda: expected_ids.issubset(set(ctrl.get_all_frames().keys())),
        timeout=3000,
    )

    ctrl.stop(wait=True)
    assert not ctrl.is_running()
