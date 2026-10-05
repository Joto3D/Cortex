# PyInstaller spec for Cortex.app (Apple Silicon).
#   pip install -e ".[build]"
#   python packaging/make_icon.py && pyinstaller packaging/cortex.spec --noconfirm
# Output: dist/Cortex.app
import os

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules, copy_metadata

import cortex

HERE = SPECPATH  # noqa: F821 (provided by PyInstaller)

datas = collect_data_files("cortex") + collect_data_files("open_clip") + collect_data_files("webview") + collect_data_files("certifi")
for dist in ("torch", "open_clip_torch", "anthropic", "huggingface_hub", "safetensors", "tqdm", "regex", "timm", "keyring", "pywebview"):
    try:
        datas += copy_metadata(dist)
    except Exception:
        pass

# torchvision loads its compiled ops (_C*.so) with torch.ops.load_library, which PyInstaller
# can't see; without them `import open_clip` fails with "operator torchvision::nms does not exist".
binaries = collect_dynamic_libs("torchvision", search_patterns=["*.so", "*.dylib"])

hiddenimports = (
    collect_submodules("cortex")
    + collect_submodules("keyring.backends")
    + ["pynput.keyboard._darwin", "pynput.mouse._darwin", "ApplicationServices", "certifi",
       "webview", "webview.platforms.cocoa", "WebKit", "PyObjCTools.AppHelper"]
)

a = Analysis(
    [os.path.join(HERE, "cortex_app.py")],
    pathex=[os.path.dirname(HERE)],
    binaries=binaries,
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
        "LSMinimumSystemVersion": "13.0",
        "NSHighResolutionCapable": True,
        "NSHumanReadableCopyright": "MIT License",
    },
)
