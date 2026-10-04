"""Keep the GitHub Pages demo in sync with the Python code."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from cortex.assignment import KeywordInterpreter
from cortex.config import load_profile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from export_site_data import OUT, render  # noqa: E402

SENTENCES = [
    "Harvest everything, then water the crops and break 5 rocks",
    "Water my plants first, then pick the ripe ones",
    "Do all the chores",
    "Chop twelve sticks then mow the weeds",
    "Clear the rocks and feed the chickens",
    "use the pickaxe; gather 3 ripe parsnips. finally irrigate",
    "",
]


def test_tasks_json_is_up_to_date():
    assert OUT.read_text() == render(), "run: python scripts/export_site_data.py"


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_js_parser_matches_python():
    script = (
        "import { interpret } from './docs/assign.mjs';"
        "import fs from 'fs';"
        "const tasks = JSON.parse(fs.readFileSync('docs/tasks.json')).tasks;"
        f"const sentences = {json.dumps(SENTENCES)};"
        "console.log(JSON.stringify(sentences.map((s) => interpret(s, tasks))));"
    )
    out = subprocess.run(
        ["node", "--input-type=module", "-e", script], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout
    js = json.loads(out)
    kw = KeywordInterpreter(load_profile("stardew"))
    for sentence, got in zip(SENTENCES, js):
        m = kw.interpret(sentence)
        expected = {
            "steps": [{"task": s.task, "limit": s.limit} for s in m.steps],
            "unsupported": list(m.unsupported),
        }
        assert got == expected, sentence
