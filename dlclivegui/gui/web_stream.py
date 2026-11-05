"""View-menu actions and lifecycle for the web stream.

On/off and network exposure are local QSettings preferences, written only by the
user's menu toggles. Port, fps, size and quality come from the config JSON.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable

import numpy as np
from PySide6.QtCore import QSignalBlocker
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QMenu, QMessageBox, QWidget

from ..config import WebStreamSettings
from ..services.web_streamer import LOOPBACK_HOST, NETWORK_HOST, SHUTDOWN_TIMEOUT, WebStreamer
from ..utils.settings_store import DLCLiveGUISettingsStore

logger = logging.getLogger(__name__)

_NETWORK_WARNING = (
    "Anyone on your network who can reach this computer will be able to watch the camera "
    "stream, without a password.\n\n"
    "The stream is view-only: it cannot control the app or read files.\n\n"
    "Allow network access?"
)


def _set_checked_silently(action: QAction, checked: bool) -> None:
    with QSignalBlocker(action):
        action.setChecked(checked)


class WebStreamController:
    """Owns the web stream's menu actions, preferences and server lifecycle."""

    def __init__(
        self,
        parent: QWidget,
        settings_store: DLCLiveGUISettingsStore,
        frame_source: Callable[[], np.ndarray | None],
        settings: WebStreamSettings,
        show_status: Callable[[str, int], None],
    ):
        self._parent = parent
        self._store = settings_store
        self._frame_source = frame_source
        self._settings = settings.model_copy()
        self._show_status = show_status
        self._streamer: WebStreamer | None = None
        self._pending_shutdown: threading.Thread | None = None

        self.action_enable = QAction(parent, checkable=True)
        self.action_enable.toggled.connect(self._on_enable_toggled)
        self.action_network = QAction("Allow network access", parent, checkable=True)
        self.action_network.setChecked(self._store.get_web_stream_allow_network())  # before connect: no prompt
        self.action_network.toggled.connect(self._on_network_toggled)
        self._update_label()

    # ------------------------------------------------------------------
    # Public API (main window)
    def add_to_menu(self, menu: QMenu) -> QMenu:
        """Add a "Web stream" submenu holding the stream's actions to ``menu``."""
        submenu = menu.addMenu("Web stream")
        submenu.addActions([self.action_enable, self.action_network])
        return submenu

    @property
    def settings(self) -> WebStreamSettings:
        return self._settings.model_copy()

    @property
    def streamer(self) -> WebStreamer | None:
        return self._streamer

    def apply_settings(self, settings: WebStreamSettings, *, restore_enabled: bool = False) -> None:
        """Adopt config parameters; restart a running stream only if they changed.

        ``restore_enabled`` starts the stream if the persisted preference says so (startup only).
        """
        changed = settings != self._settings
        self._settings = settings.model_copy()
        self._update_label()
        if self._streamer is not None and changed:
            self._restart()
        elif restore_enabled and self._streamer is None and self._store.get_web_stream_enabled():
            self._start()

    def shutdown(self) -> None:
        """Non-blocking stop for app exit; server threads are daemons and the OS releases the port."""
        self._stop()

    # ------------------------------------------------------------------
    # Menu toggles: the only writers of the persisted preferences
    def _on_enable_toggled(self, enabled: bool) -> None:
        if enabled:
            self._store.set_web_stream_enabled(self._start())
        else:
            self._store.set_web_stream_enabled(False)
            if self._streamer is not None:
                self._stop()
                self._show_status("Web stream stopped", 3000)

    def _on_network_toggled(self, allow: bool) -> None:
        if allow and not self._confirm_network():
            _set_checked_silently(self.action_network, False)
            return
        self._store.set_web_stream_allow_network(allow)
        self._update_label()
        if self._streamer is not None:
            self._restart()

    def _confirm_network(self) -> bool:
        reply = QMessageBox.question(
            self._parent,
            "Allow network access?",
            _NETWORK_WARNING,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return reply == QMessageBox.StandardButton.Yes

    # ------------------------------------------------------------------
    # Lifecycle
    def _host(self) -> str:
        return NETWORK_HOST if self.action_network.isChecked() else LOOPBACK_HOST

    def _update_label(self) -> None:
        scope = "network" if self._host() == NETWORK_HOST else "localhost only"
        self.action_enable.setText(f"Enable (port {self._settings.port}, {scope})")

    def _start(self) -> bool:
        """Start the stream and sync the menu check state. Returns whether it is running."""
        # A quick off/on toggle can race the previous server's background shutdown; wait for the port.
        pending, self._pending_shutdown = self._pending_shutdown, None
        if pending is not None:
            pending.join(timeout=SHUTDOWN_TIMEOUT)

        host = self._host()
        streamer = WebStreamer.from_settings(self._frame_source, self._settings, host=host)
        try:
            streamer.start()
        except OSError as exc:
            logger.warning("Failed to start web stream on %s:%s: %s", host, self._settings.port, exc)
            self._show_status(f"Web stream failed to start on port {self._settings.port}: {exc}", 5000)
            _set_checked_silently(self.action_enable, False)
            return False

        self._streamer = streamer
        _set_checked_silently(self.action_enable, True)
        message = f"Web stream at {streamer.url}"
        if host == NETWORK_HOST:
            network_url = streamer.network_url
            message += f" (network: {network_url})" if network_url else " (network: no LAN address found)"
            logger.warning("Web stream is reachable from the network without authentication")
        self._show_status(message, 8000)
        return True

    def _stop(self, wait: bool = False) -> None:
        streamer, self._streamer = self._streamer, None
        if streamer is not None:
            self._pending_shutdown = streamer.stop(wait=wait)
        _set_checked_silently(self.action_enable, False)

    def _restart(self) -> None:
        # Blocking stop so the new server can rebind the port; user/config actions only, not a hot path.
        self._stop(wait=True)
        self._start()
