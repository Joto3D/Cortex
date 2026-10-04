# PyInstaller spec for Cortex.app (Apple Silicon).
#   pip install -e ".[build]"
#   python packaging/make_icon.py && pyinstaller packaging/cortex.spec --noconfirm
# Output: dist/Cortex.app
import os

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

import cortex

HERE = SPECPATH  # noqa: F821 (provided by PyInstaller)

datas = collect_data_files("cortex") + collect_data_files("open_clip")
for dist in ("torch", "open_clip_torch", "anthropic", "huggingface_hub", "safetensors", "tqdm", "regex", "timm", "keyring"):
    try:
        datas += copy_metadata(dist)
    except Exception:
        pass

hiddenimports = (
    collect_submodules("cortex")
    + collect_submodules("keyring.backends")
    + ["pynput.keyboard._darwin", "pynput.mouse._darwin", "rumps", "ApplicationServices"]
)

a = Analysis(
    [os.path.join(HERE, "cortex_app.py")],
    pathex=[os.path.dirname(HERE)],
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["tkinter", "matplotlib", "IPython", "pytest"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Cortex",
    console=False,
    target_arch="arm64",
    codesign_identity=os.environ.get("CODESIGN_IDENTITY") or None,
    entitlements_file=os.path.join(HERE, "entitlements.plist"),
)
coll = COLLECT(exe, a.binaries, a.datas, name="Cortex")
app = BUNDLE(
    coll,
    name="Cortex.app",
    icon=os.path.join(HERE, "build", "Cortex.icns") if os.path.exists(os.path.join(HERE, "build", "Cortex.icns")) else None,
    bundle_identifier="io.github.joto3d.cortex",
    version=cortex.__version__,
    info_plist={
        "CFBundleDisplayName": "Cortex",
        "CFBundleShortVersionString": cortex.__version__,
        "LSUIElement": True,  # menu-bar app: no Dock icon
        "LSMinimumSystemVersion": "13.0",
        "NSHighResolutionCapable": True,
        "NSHumanReadableCopyright": "MIT License",
    },
)
