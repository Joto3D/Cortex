"""A small stand-in for `rumps` so the menu-bar app can be built and driven in tests.

It copies the real library's quirks that matter to Cortex. In particular, a MenuItem
has no submenu (`_menu is None`) until its first child is added, and `clear()` on
such an item raises, like rumps 0.4's `self._menu.removeAllItems()` does.
"""
from __future__ import annotations

separator = object()


class _NSMenu:
    def __init__(self):
        self.items = []

    def removeAllItems(self):
        self.items.clear()


class MenuItem:
    def __init__(self, title, callback=None, key=None, icon=None, dimensions=None, template=None):
        if key is not None and not isinstance(key, str):
            raise TypeError("key must be a string or None")
        self.title = str(title)
        self.callback = callback
        self.key = key
        self.state = 0
        self._menu = None
        self._children: dict = {}

    def set_callback(self, callback, key=None):
        self.callback = callback

    def add(self, item):
        if self._menu is None:
            self._menu = _NSMenu()
        key = "---sep%d" % len(self._children) if item is None else item.title
        if key not in self._children:  # real rumps keys children by title and ignores duplicates
            self._children[key] = item
            self._menu.items.append(item)

    def clear(self):
        self._menu.removeAllItems()  # AttributeError when no child was ever added, as in rumps
        self._children.clear()

    @property
    def items(self):
        return [] if self._menu is None else list(self._menu.items)


class App:
    def __init__(self, name, title=None, icon=None, quit_button="Quit"):
        self.name = name
        self.title = title
        self._menu_items = []

    @property
    def menu(self):
        return self._menu_items

    @menu.setter
    def menu(self, items):
        self._menu_items = list(items)

    def run(self, **options):
        pass


class Timer:
    def __init__(self, callback, interval):
        self.callback = callback

    def start(self):
        pass

    def stop(self):
        pass


window_responses: list = []  # tests push (clicked, text) tuples here


class Window:
    def __init__(self, *a, **k):
        pass

    def run(self):
        clicked, text = window_responses.pop(0) if window_responses else (0, "")
        return type("Response", (), {"clicked": clicked, "text": text})()


alert_answers: list = []  # tests push 1/0 here
alerts: list = []
notifications: list = []


def alert(title=None, message="", ok=None, cancel=None, other=None, icon_path=None):
    alerts.append((title, message))
    return alert_answers.pop(0) if alert_answers else 0


def notification(title, subtitle, message, **kw):
    notifications.append((title, message))


def quit_application(sender=None):
    pass
