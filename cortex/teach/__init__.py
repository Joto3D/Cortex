"""Teach by showing: record yourself playing any game, and Cortex repeats it. Fully offline, no API key."""
from .choose import choose_skill
from .player import PlayResult, TickPlayer, play_skill
from .recorder import InputLog, Recorder
from .skill import FPS, NearestMomentPolicy, Skill, Tick, list_skills

__all__ = [
    "FPS", "InputLog", "NearestMomentPolicy", "PlayResult", "Recorder", "Skill", "Tick", "TickPlayer",
    "choose_skill", "list_skills", "play_skill",
]
