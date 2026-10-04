"""macOS permissions and secrets for the app.

Every function degrades gracefully off macOS or when pyobjc/keyring are missing,
so tests and the CLI still work.
"""
from __future__ import annotations

import logging
import os
import subprocess

log = logging.getLogger(__name__)

KEYCHAIN_SERVICE = "Cortex"
KEYCHAIN_ACCOUNT = "anthropic_api_key"

SETTINGS_PANES = {
    "screen": "x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture",
    "accessibility": "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility",
    "input": "x-apple.systempreferences:com.apple.preference.security?Privacy_ListenEvent",
}


def has_screen_recording() -> bool | None:
    try:
        import Quartz

        return bool(Quartz.CGPreflightScreenCaptureAccess())
    except Exception:
        return None


def request_screen_recording() -> None:
    try:
        import Quartz

        Quartz.CGRequestScreenCaptureAccess()
    except Exception as e:
        log.debug("screen recording request unavailable: %s", e)


def has_accessibility(prompt: bool = False) -> bool | None:
    try:
        from ApplicationServices import AXIsProcessTrustedWithOptions, kAXTrustedCheckOptionPrompt

        return bool(AXIsProcessTrustedWithOptions({kAXTrustedCheckOptionPrompt: prompt}))
    except Exception:
        return None


def open_settings(pane: str) -> None:
    subprocess.run(["open", SETTINGS_PANES[pane]], check=False)


def get_api_key() -> str | None:
    """API key from the environment, else from the macOS Keychain."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return os.environ["ANTHROPIC_API_KEY"]
    try:
        import keyring

        return keyring.get_password(KEYCHAIN_SERVICE, KEYCHAIN_ACCOUNT)
    except Exception as e:
        log.debug("keychain unavailable: %s", e)
        return None


def save_api_key(key: str) -> None:
    import keyring

    keyring.set_password(KEYCHAIN_SERVICE, KEYCHAIN_ACCOUNT, key.strip())
    os.environ["ANTHROPIC_API_KEY"] = key.strip()


def load_api_key_into_env() -> bool:
    """Make the Keychain key visible to the Anthropic SDK. Returns True if a key is available."""
    key = get_api_key()
    if key:
        os.environ["ANTHROPIC_API_KEY"] = key
    return bool(key)
