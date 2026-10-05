"""Skills learned from your own play, and the policy that replays them.

A skill is one or more recordings ("segments") of you playing. Each recorded tick
(10 per second) has a screen fingerprint (a MobileCLIP embedding) and the
*action state* during that tick: which keys/buttons were held, how far the mouse
moved, and any clicks.

Playing a skill means: fingerprint the live screen, find the most similar
recorded moment, and replay what you did for the next half second. Then look
again. There is no training step and no network. It's all local.
"""
from __future__ import annotations

import json
import re
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from cortex.config import cortex_home

FPS = 10  # recording / playback ticks per second


@dataclass
class Tick:
    """What you were doing during one 1/FPS-second tick."""

    held: list[str] = field(default_factory=list)  # keys / mouse buttons held down ("w", "shift", "mouse_left")
    dx: float = 0.0                                 # relative mouse movement during the tick (points)
    dy: float = 0.0
    clicks: list[dict] = field(default_factory=list)  # {"button", "x", "y"} quick clicks (window fractions)

    @property
    def idle(self) -> bool:
        return not (self.held or self.dx or self.dy or self.clicks)


def shrink(frame: np.ndarray, width: int = 256) -> np.ndarray:
    """Downscale a frame to at most `width` px wide, the same way for recording and playback."""
    step = max(1, -(-frame.shape[1] // width))  # ceil division
    return np.ascontiguousarray(frame[::step, ::step, :3])


def skills_dir(game: str) -> Path:
    return cortex_home() / "skills" / game


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "skill"


@dataclass
class Skill:
    name: str
    game: str
    embeddings: np.ndarray            # (N, D) float32, L2-normalised
    ticks: list[Tick]                 # N action states
    segments: np.ndarray              # (N,) int: which recording each tick came from

    @property
    def path(self) -> Path:
        return skills_dir(self.game) / slug(self.name)

    @property
    def seconds(self) -> float:
        return len(self.ticks) / FPS

    # -- persistence ------------------------------------------------------------
    def save(self) -> Path:
        d = self.path
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "embeddings.npy", self.embeddings.astype(np.float16))
        np.save(d / "segments.npy", self.segments.astype(np.int32))
        with open(d / "ticks.jsonl", "w") as f:
            for t in self.ticks:
                f.write(json.dumps(asdict(t)) + "\n")
        (d / "meta.json").write_text(json.dumps({"name": self.name, "game": self.game, "fps": FPS}, indent=2))
        return d

    @classmethod
    def load(cls, path: Path) -> "Skill":
        meta = json.loads((path / "meta.json").read_text())
        ticks = [Tick(**json.loads(line)) for line in (path / "ticks.jsonl").read_text().splitlines() if line]
        emb = np.load(path / "embeddings.npy").astype(np.float32)
        seg = np.load(path / "segments.npy")
        return cls(meta["name"], meta["game"], emb, ticks, seg)

    def add_recording(self, embeddings: np.ndarray, ticks: list[Tick]) -> None:
        """Append another recording of the same skill (more examples = better play)."""
        seg_id = int(self.segments.max()) + 1 if len(self.segments) else 0
        self.embeddings = np.concatenate([self.embeddings, embeddings]) if len(self.embeddings) else embeddings
        self.ticks = self.ticks + ticks
        self.segments = np.concatenate([self.segments, np.full(len(ticks), seg_id, np.int32)])

    def delete(self) -> None:
        shutil.rmtree(self.path, ignore_errors=True)


def list_skills(game: str) -> list[Skill]:
    d = skills_dir(game)
    if not d.is_dir():
        return []
    out = []
    for p in sorted(d.iterdir()):
        if (p / "meta.json").exists():
            try:
                out.append(Skill.load(p))
            except (OSError, ValueError, KeyError):
                continue
    return out


class NearestMomentPolicy:
    """Pick the recorded moment most like the current screen; return what you did next.

    * Cosine similarity over all recorded ticks that still have ``chunk`` ticks after them
      in the same recording.
    * A continuity bonus favours the moment right after the one we replayed last, so
      playback follows one recording smoothly instead of flickering between near-ties.
    """

    def __init__(self, skill: Skill, chunk: int = FPS // 2, continuity_bonus: float = 0.03):
        self.skill = skill
        self.continuity_bonus = continuity_bonus
        seg = skill.segments
        # Very short recordings: shrink the chunk so at least some moments are playable.
        longest = int(np.bincount(seg).max()) if len(seg) else 0
        self.chunk = max(1, min(chunk, longest - 1))
        self._usable = self._usable_mask(seg, self.chunk)
        self.last: int | None = None

    @staticmethod
    def _usable_mask(seg: np.ndarray, chunk: int) -> np.ndarray:
        """A tick is usable if the next `chunk` ticks belong to the same recording."""
        n = len(seg)
        ok = np.zeros(n, bool)
        if n > chunk:
            ok[: n - chunk] = seg[: n - chunk] == seg[chunk:]
        return ok

    def reset(self) -> None:
        self.last = None

    def choose(self, embedding: np.ndarray) -> tuple[int, float]:
        """Return (index of the chosen recorded tick, its raw similarity)."""
        if not self._usable.any():
            raise ValueError("this skill's recordings are too short to play")
        sims = self.skill.embeddings @ embedding.astype(np.float32)
        scores = np.where(self._usable, sims, -np.inf)
        if self.last is not None:
            nxt = self.last + self.chunk
            if nxt < len(scores) and self._usable[nxt]:
                scores = scores.copy()
                scores[nxt] += self.continuity_bonus
        i = int(np.argmax(scores))
        self.last = i
        return i, float(sims[i])

    def actions(self, i: int) -> list[Tick]:
        return self.skill.ticks[i : i + self.chunk]
