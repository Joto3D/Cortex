# Cortex

A bot that plays single-player farming sims for you. It watches the game window,
works out what's on screen with a contrastive vision-language model (CLIP), and
drives keyboard and mouse fast enough to keep up with a human player.

Stardew Valley on macOS (Apple Silicon) is the reference target.

> **Use it only in single-player or offline games.** Botting in online games usually
> breaks their Terms of Service and can get your account banned.

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

## Setup (macOS)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"
```

Grant your terminal app these permissions in **System Settings → Privacy & Security**:
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

## Project layout

| Path | What it does |
|---|---|
| `cortex/capture/screen.py` | Window lookup and capture (Quartz and mss) |
| `cortex/perception/clip_model.py` | CLIP encoder (open_clip on MPS/CUDA/CPU), GPU-side preprocessing |
| `cortex/perception/zeroshot.py` | Prompt ensembling and zero-shot classification |
| `cortex/perception/tilegrid.py` | Grid phase estimation, tile slicing, hash cache → `SemanticMap` |
| `cortex/perception/hud.py` | Energy bar reader (pixel colours, no network) |
| `cortex/perception/perceiver.py` | Frame → `WorldState` |
| `cortex/planner/` | Pathing (BFS), actions, farm task planner |
| `cortex/control/` | Quartz CGEvent input and the action → input controller |
| `cortex/loop.py` | Main loop, hotkeys, dry-run and recording |
| `cortex/profiles/stardew.yaml` | Labels/prompts, tools, tasks, controls, HUD for Stardew |

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
- An optional Claude call *once per in-game day* (outside the hot loop) to set high-level goals.
- More game profiles.
