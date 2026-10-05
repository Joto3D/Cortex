# Cortex

A Mac app that plays single-player games for you, 2D or 3D. Open your game, click its window in
Cortex, type what to do ("explore and collect wood, then come back"), and press **Start**. It
watches the game window and plays with the keyboard and mouse.

- **Free AI brain:** Google Gemini, with a free API key from [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
  (about a minute, no card).
- **Fast hands:** keys and mouse never wait for the network. Moves run on your Mac, and taught skills react in about 50 ms.
- **Teach by showing:** record yourself playing and Cortex repeats it. This works fully offline, with no key.

> **Use it only in single-player or offline games.** Botting in online games usually
> breaks their Terms of Service and can get your account banned.

**Website:** https://joto3d.github.io/Cortex/

## Install (Mac app)

1. Download **Cortex.dmg** from the [latest release](https://github.com/Joto3D/Cortex/releases/latest),
   open it and drag **Cortex** into **Applications**.
2. Open Cortex. If macOS says it can't verify the app, go to **System Settings → Privacy & Security** and
   click **Open Anyway**. (Builds are unsigned until the project has an Apple Developer ID.)
3. The welcome screen walks you through it: allow Screen Recording, Accessibility and Input Monitoring,
   then paste your free Gemini key. You can skip the key.
4. **Play:** open your game, click its window in Cortex, type what to do, press **Start** and switch to the
   game. **F12** stops Cortex at any time and **F11** pauses it.

Cortex has a normal window (Play, Teach, Settings) and a menu-bar icon for Start/Stop while you're in the game.
The live view shows what Cortex sees, what it plans, and how fast it is reacting.

Requires an Apple Silicon Mac with macOS 13 or later. Skills need a small vision model, which is downloaded
the first time (about 300 MB).

## How it stays fast: two speeds

A cloud AI needs about a second to answer, which is too slow to steer every keypress. So Cortex splits the work:

| Lane | Where | Speed | What it does |
|---|---|---|---|
| **Hands** | your Mac | ~50 ms | Runs the current plan back-to-back (hold keys, turn the camera, click), plays taught skills, and fires CLIP reflexes |
| **Brain** | Gemini | ~0.5–1.5 s | Looks at one screenshot and plans the next few seconds as a list of actions |

The next screenshot goes to Gemini *before* the current plan runs out (`cortex/agent/gemini.py`, `GeminiPilot`),
so a new plan is ready when it's needed and the keys never stand still waiting for the network. Gemini can also
call `use_skill` to run a skill you taught, which reacts locally ten times a second. Requests stay under the free
tier's limit (Settings → Requests per minute, default 12). If Google says to slow down, Cortex backs off and
keeps playing.

What leaves your Mac: with a Gemini key, your instruction and a small JPEG of the game window (only that window)
every few seconds. Without a key, nothing. Keys are stored in the macOS Keychain.

## Teach it by playing (no key needed)

1. Open the **Teach** page, name a skill (for example "collect wood") and press **Record**.
2. Switch to the game and play normally for 2–5 minutes, then press **F12**.
3. Type the skill's name as the task (or click it) and press **Start**.

While you play, Cortex saves 10 screen "fingerprints" per second (MobileCLIP embeddings) together with the keys
you held, mouse movement and clicks. When it plays, it fingerprints the live screen, finds the most similar moment
in your recordings, and does what you did next for half a second. Then it looks again.

- It only records while the game window is in front, and never records ⌘-shortcuts or F11/F12.
- It can only repeat what it has seen, so record a few varied examples. Recording the same name again adds more.
- With a Gemini key, Gemini decides when to use which skill and fills the gaps with its own moves.
- From the terminal: `python -m cortex.teach record --game my_game "collect wood"`, then
  `python -m cortex.teach play --game my_game`.

## Which brain plays

Every game you pick gets a profile in `~/Library/Application Support/Cortex/games/` with `engine: auto`:

1. With a **Gemini key**, Gemini plays, using your taught skills when they fit.
2. Otherwise, with a **Claude key** (optional, paid, Settings → Claude), Claude plays (`cortex/agent/agent.py`).
3. Otherwise, it plays your **taught skills**.

Add `agent.keys` (action → key) and `agent.tips` to a profile to tell the AI your game's controls.
`cortex/profiles/generic_3d.yaml` shows every option, including CLIP **reflexes**: instant local reactions such as
"health bar nearly empty → back off".

From the terminal: `cortex-games add "My Game"`, then `GEMINI_API_KEY=... cortex --game my_game -a "collect wood"`.

## Built-in example: fast 2D mode

For top-down tile games, the `grid` engine needs no AI while playing. The bundled Stardew Valley profile is an
example. MobileCLIP labels every tile in milliseconds, and a rule-based planner walks, waters and harvests.
Assignments are turned into steps once, before the game starts.

## Give it an assignment

Tell Cortex what to do in plain English:

```bash
python -m cortex.loop -a "harvest everything, then water the crops and break 5 rocks"
```

```
Plan: Harvest the ripe crops, water the dry ones, then break 5 rocks.
  1. harvest (until done)
  2. water (until done)
  3. clear_stone x5
Start? Switch to the game window after pressing Enter. [Y/n]
```

- Run it with no `-a` and it asks you what to do. Press Enter with nothing typed to do every chore.
- **Claude** (`claude-opus-5-5`) reads the assignment *once, before the game starts*, and returns
  a plan restricted to the tasks your profile defines. Anything it can't do (like "feed the chickens")
  is listed instead of guessed. Set `ANTHROPIC_API_KEY` (or run `ant auth login`) to enable it.
- With no credentials or no network, it falls back automatically to an **offline keyword parser**
  (each task's `keywords:` in the profile). Use `--offline` to force it.
- Steps run in order. A step ends when its count is reached, or when its targets have been gone
  from the screen for a few frames. Preview a plan without the game: `python -m cortex.assignment "..."`.

## How it works

```
 capture ──► perceive ──────────────────────► plan ─────────► act
 (window)    tile grid + CLIP zero-shot        BFS + task       held keys,
  ~8 ms      + content-hash cache  ~5-20 ms    priorities <1ms  mouse clicks
```

1. **Capture.** Quartz finds the game window and mss grabs its pixels.
2. **Perceive.** The frame is cut into game tiles. The grid offset is estimated every frame
   from where colour edges line up. Each tile is classified **zero-shot** by comparing its CLIP
   image embedding with text embeddings of prompts like *"a small green sprout on light dry soil"*.
   No training data is needed: to add a new object you add a prompt in the profile YAML.
3. **Speed.** Text prompts are embedded once at start-up. Tiles are cached by a hash of
   their pixels. Pixel-art tilesets repeat a lot, so after the first second almost every
   tile is a cache hit and CLIP only runs on new-looking tiles (one batched call).
4. **Plan.** A rule-based planner, with no LLM in the loop, picks the highest-priority task
   that has a target on screen (harvest → water → clear debris). It finds a path with BFS,
   walks with held direction keys, stops next to the target, selects the tool and clicks the tile.
5. **Safety.** F12 is the kill switch and F11 pauses. The bot also pauses automatically when the
   game window loses focus, and it stops when energy is low or no work is visible.

## Developer setup (from source)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"
python -m cortex.app          # run the app from source
```

When running from a terminal, grant your terminal app these permissions in **System Settings → Privacy & Security**:
- **Screen Recording**, to capture the game window
- **Accessibility**, to send key and mouse events
- **Input Monitoring**, for the F11/F12 global hotkeys

The first run downloads the CLIP weights (MobileCLIP-S1 by default, with ViT-B-32 as a fallback).

## Usage

1. In Stardew, set **zoom to 100%** and check `tile_px` in `cortex/profiles/stardew.yaml`
   (128 on Retina, 64 otherwise). Set the `tools:` hotbar keys to match your toolbar.
2. Stand on your farm, then run:

```bash
python -m cortex.loop --debug --dry-run   # watch the overlay; no input is sent
python -m cortex.loop --debug             # let it play
python -m cortex.loop --record recordings # save frames for prompt tuning
python -m cortex.bench --image recordings/frame_000000.png   # latency + label counts
```

In the debug overlay, each label is drawn as a coloured tint. The cyan box is the player,
white boxes show the path and the red box is the current target.

## Tuning perception

Perception quality depends on the prompts. Workflow:
1. Run `--record` while playing normally.
2. Run `cortex.bench --image ...` on the saved frames and look at the label counts and the `--debug` overlay.
3. Edit the prompts in the profile (several prompts per label are averaged together) and repeat.

**Grid alignment** is detected automatically from where colour edges line up, and that works
well for normal tilesets. If the overlay's tile boxes look offset from the real tiles
(very noisy art, unusual zoom), set `game.grid_phase: [x, y]` in the profile to pin it.

## Project layout

| Path | What it does |
|---|---|
| `cortex/capture/screen.py` | Window lookup and capture (Quartz and mss) |
| `cortex/perception/clip_model.py` | CLIP encoder (open_clip on MPS/CUDA/CPU), GPU-side preprocessing |
| `cortex/perception/zeroshot.py` | Prompt ensembling and zero-shot classification |
| `cortex/perception/tilegrid.py` | Grid phase estimation, tile slicing, hash cache → `SemanticMap` |
| `cortex/perception/hud.py` | Energy bar reader (pixel colours, no network) |
| `cortex/perception/perceiver.py` | Frame → `WorldState` |
| `cortex/assignment.py` | Plain-English assignment → Mission (Claude, with a keyword fallback) |
| `cortex/planner/` | Pathing (BFS), actions, farm task planner (runs missions step by step) |
| `cortex/control/` | Quartz CGEvent input and the action → input controller |
| `cortex/loop.py` | Runners: `run_gemini`, `run_agent` (Claude), `run_skill`, `run` (grid); hotkeys, dry-run |
| `cortex/agent/gemini.py` | Gemini REST client and the pipelined two-speed `GeminiPilot` |
| `cortex/agent/` | Game tools and action executor (shared by Gemini and Claude), Claude agent, CLIP reflexes |
| `cortex/telemetry.py` | Live numbers for the window: latest frame, reaction and AI time, action log |
| `cortex/teach/` | Teach by showing: recorder, skills (embeddings + action ticks), nearest-moment player |
| `cortex/games.py` | Game profiles for open windows (`game_for_window`); `cortex-games list/add/show/windows` |
| `cortex/app/` | The app: window UI (`ui/index.html` in pywebview), JS↔Python `bridge.py`, `menubar.py`, UI-free `AppController`, permissions and Keychain |
| `packaging/` | PyInstaller spec, icon and entitlements for `Cortex.app` |
| `cortex/profiles/generic_3d.yaml` | Example profile showing controls, tips and reflexes for a 3D game |
| `cortex/profiles/stardew.yaml` | Labels/prompts, tools, tasks (with descriptions and keywords), controls, HUD for Stardew |
| `docs/` | GitHub Pages website (`index.html`, the JS parser port `assign.mjs`, generated `tasks.json`) |

## Website

`docs/` is deployed to GitHub Pages by `.github/workflows/pages.yml` on every push to `main`
that touches `docs/`. Before the first deploy, enable it once in the repo's
**Settings → Pages → Source: GitHub Actions**. If you change tasks in the profile, run
`python scripts/export_site_data.py`; a test fails if `docs/tasks.json` is stale.

## Releasing the Mac app

`.github/workflows/mac-app.yml` builds `Cortex.app` and `Cortex.dmg` on a macOS runner. Every pull
request that touches the app is built and self-tested, and every `v*` tag publishes a GitHub Release:

```bash
git tag v0.3.0 && git push origin v0.3.0
```

Or, without git: **Actions → Mac app → Run workflow**, on `main`, with a `release_tag` such as `v0.3.0`.
The workflow creates the tag and the release.

To ship **signed and notarized** builds, so there is no "Open Anyway" step, add these repository
secrets from an Apple Developer account: `APPLE_CERT_P12` (a base64 Developer ID Application
certificate), `APPLE_CERT_PASSWORD`, `APPLE_ID`, `APPLE_TEAM_ID` and `APPLE_APP_PASSWORD`
(an app-specific password). The workflow signs and notarizes automatically when they are present.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

The tests use a fake colour-based encoder in place of CLIP, so they run anywhere (no
torch, no macOS) and cover grid detection, caching, pathing, planning and input.

## Roadmap

- **Core ML export** of the MobileCLIP image tower so it runs on the Neural Engine.
- **Distillation:** auto-label thousands of tiles with CLIP and train a tiny CNN (<2 ms/frame),
  keeping CLIP as the teacher for new labels.
- **ScreenCaptureKit** streaming capture backend.
- Exploring beyond the visible screen, going to bed at night, refilling the watering can.
- More bundled game profiles; a shared gallery of community profiles.
- Stream Gemini's function calls and start the first action before the whole plan arrives.
