"""The menu-bar icon (macOS): status, Start/Stop, Pause, Show Cortex, Quit.

It shares the app's run loop with the window. ``install`` and the timer run on
the main thread.
"""
from __future__ import annotations

import logging
import threading
from typing import Callable

import AppKit
import objc
from Foundation import NSObject, NSTimer

from .controller import AppController

log = logging.getLogger(__name__)
ICONS = {"idle": "◉", "starting": "⏳", "running": "▶︎", "paused": "⏸", "error": "⚠️", "recording": "⏺", "learning": "🧠"}


class _Target(NSObject):
    """Objective-C target for the menu items and the refresh timer."""

    def initWithOwner_(self, owner):
        self = objc.super(_Target, self).init()
        if self is not None:
            self.owner = owner
        return self

    def tick_(self, _timer):
        self.owner.refresh()

    def startStop_(self, _sender):
        self.owner.start_stop()

    def pause_(self, _sender):
        self.owner.ctl.toggle_pause()
        self.owner.refresh()

    def show_(self, _sender):
        self.owner.show_window()

    def quit_(self, _sender):
        self.owner.quit_app()


class MenuBar:
    def __init__(self, ctl: AppController, show_window: Callable[[], None], quit_app: Callable[[], None]):
        self.ctl = ctl
        self.show_window = show_window
        self.quit_app = quit_app
        self.item = None

    def _menu_item(self, title: str, action: str | None, key: str = ""):
        mi = AppKit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(title, action, key)
        if action:
            mi.setTarget_(self._target)
        return mi

    def install(self) -> None:
        self._target = _Target.alloc().initWithOwner_(self)
        self.item = AppKit.NSStatusBar.systemStatusBar().statusItemWithLength_(AppKit.NSVariableStatusItemLength)
        menu = AppKit.NSMenu.alloc().init()
        menu.setAutoenablesItems_(False)
        self.status_mi = self._menu_item("Ready", None)
        self.status_mi.setEnabled_(False)
        self.start_mi = self._menu_item("Start", "startStop:")
        self.pause_mi = self._menu_item("Pause", "pause:")
        menu.addItem_(self.status_mi)
        menu.addItem_(AppKit.NSMenuItem.separatorItem())
        menu.addItem_(self.start_mi)
        menu.addItem_(self.pause_mi)
        menu.addItem_(AppKit.NSMenuItem.separatorItem())
        menu.addItem_(self._menu_item("Show Cortex", "show:"))
        menu.addItem_(self._menu_item("Quit Cortex", "quit:", "q"))
        self.item.setMenu_(menu)
        self.timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            0.5, self._target, "tick:", None, True
        )
        self.refresh()
        log.info("menu-bar icon installed")

    def start_stop(self) -> None:
        if self.ctl.busy:
            threading.Thread(target=self.ctl.stop, daemon=True).start()
            if self.ctl.recording:
                threading.Thread(target=self.ctl.stop_recording, daemon=True).start()
            return
        try:
            self.ctl.start()
        except Exception as e:
            self.ctl.status = f"Error: {e}"
            self.show_window()

    def refresh(self) -> None:
        if self.item is None:
            return
        c = self.ctl
        self.item.button().setTitle_(ICONS[c.mode])
        game = c.game_title()
        status = c.status if c.busy else (f"{game}: ready" if game else "Pick a game in the Cortex window")
        elapsed = c.elapsed
        self.status_mi.setTitle_((status[:60] + "…" if len(status) > 60 else status) + (f"  ({elapsed})" if elapsed else ""))
        self.start_mi.setTitle_("Stop  (F12)" if c.busy else "Start")
        self.start_mi.setEnabled_(c.busy or bool(c.settings.game))
        self.pause_mi.setTitle_("Resume  (F11)" if c.paused.is_set() else "Pause  (F11)")
        self.pause_mi.setEnabled_(c.running)
