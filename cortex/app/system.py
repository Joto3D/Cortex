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
# provider -> (Keychain account, environment variable)
PROVIDERS = {
    "gemini": ("gemini_api_key", "GEMINI_API_KEY"),
    "anthropic": ("anthropic_api_key", "ANTHROPIC_API_KEY"),
}
KEYCHAIN_ACCOUNT = PROVIDERS["anthropic"][0]

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


def get_key(provider: str) -> str | None:
    """API key from the environment, else from the macOS Keychain."""
    account, env = PROVIDERS[provider]
    if os.environ.get(env):
        return os.environ[env]
    if provider == "gemini" and os.environ.get("GOOGLE_API_KEY"):
        return os.environ["GOOGLE_API_KEY"]
    try:
        import keyring

        return keyring.get_password(KEYCHAIN_SERVICE, account)
    except Exception as e:
        log.debug("keychain unavailable: %s", e)
        return None


def save_key(provider: str, key: str) -> None:
    """Store a key in the Keychain (an empty key removes it)."""
    import keyring

    account, env = PROVIDERS[provider]
    key = key.strip()
    if key:
        keyring.set_password(KEYCHAIN_SERVICE, account, key)
        os.environ[env] = key
    else:
        try:
            keyring.delete_password(KEYCHAIN_SERVICE, account)
        except Exception:
            pass
        os.environ.pop(env, None)


def load_key_into_env(provider: str) -> bool:
    key = get_key(provider)
    if key:
        os.environ[PROVIDERS[provider][1]] = key
    return bool(key)


# Claude (kept for the optional "thinking mode" and older callers)
def get_api_key() -> str | None:
    return get_key("anthropic")


def save_api_key(key: str) -> None:
    save_key("anthropic", key)


def load_api_key_into_env() -> bool:
    """Make the Keychain key visible to the Anthropic SDK. Returns True if a key is available."""
    return load_key_into_env("anthropic")
