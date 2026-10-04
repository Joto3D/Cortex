"""Export the profile's tasks for the GitHub Pages demo (docs/tasks.json).

    python scripts/export_site_data.py

tests/test_site.py fails if docs/tasks.json is out of date with the profile.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cortex.config import load_profile  # noqa: E402

OUT = ROOT / "docs" / "tasks.json"


def site_data(profile_name: str = "stardew") -> dict:
    p = load_profile(profile_name)
    return {
        "profile": p.name,
        "tasks": [
            {"name": t.name, "description": t.description, "keywords": list(t.keywords), "tool": t.tool, "action": t.action}
            for t in p.tasks
        ],
    }


def render(profile_name: str = "stardew") -> str:
    return json.dumps(site_data(profile_name), indent=2) + "\n"


if __name__ == "__main__":
    OUT.write_text(render())
    print(f"wrote {OUT.relative_to(ROOT)}")
