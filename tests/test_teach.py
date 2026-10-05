import numpy as np
import pytest

from cortex.teach import FPS, InputLog, NearestMomentPolicy, Recorder, Skill, Tick, TickPlayer, choose_skill, list_skills, play_skill

from .conftest import COLORS, FakeEncoder


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_HOME", str(tmp_path))
    return tmp_path


def screen(color: str, size=(40, 60)) -> np.ndarray:
    return np.full((*size, 3), COLORS[color], np.uint8)


class RecordingIO:
    def __init__(self):
        self.events = []

    def key(self, name, down):
        self.events.append(("key", name, down))

    def mouse_move(self, x, y):
        self.events.append(("move", round(x), round(y)))

    def mouse_delta(self, dx, dy):
        self.events.append(("delta", dx, dy))

    def mouse_button(self, x, y, button, down):
        self.events.append(("button", button, down))


# --- recording -------------------------------------------------------------------

def test_input_log_ticks():
    log = InputLog(ignore={"f12"})
    log.key("w", True)
    log.mouse_move(3, -1)
    log.mouse_move(2, 0)
    t = log.sample()
    assert t.held == ["w"] and (t.dx, t.dy) == (5, -1)
    # held keys stay held in later ticks; mouse motion resets
    assert log.sample().held == ["w"] and log.sample().dx == 0
    log.key("w", False)
    assert log.sample().held == []

    # a key tapped within one tick still shows up for that tick
    log.key("e", True)
    log.key("e", False)
    assert log.sample().held == ["e"]

    # a quick click becomes a positioned click; a long press becomes a hold
    log.mouse_button("left", True, 0.25, 0.5)
    log.mouse_button("left", False, 0.25, 0.5)
    t = log.sample()
    assert t.held == [] and t.clicks == [{"button": "left", "x": 0.25, "y": 0.5}]
    log.mouse_button("right", True, 0.5, 0.5)
    assert log.sample().held == ["mouse_right"]
    log.mouse_button("right", False, 0.5, 0.5)
    assert log.sample() == Tick()

    # ignored keys and ⌘-shortcuts are never recorded
    log.key("f12", True)
    log.key("cmd", True)
    log.key("r", True)
    assert log.sample().held == []


def test_recorder_only_records_while_focused():
    focused = iter([False, True, True, False, True])
    log = InputLog()
    rec = Recorder(grab=lambda: screen("grass", (80, 600)), is_focused=lambda: next(focused), log_=log)

    log.key("w", True)          # typed in another app: ignored
    assert rec.step() is False
    log.key("w", True)          # now in the game
    assert rec.step() is True
    assert rec.step() is True
    log.key("d", True)          # pressed while unfocused again
    assert rec.step() is False  # ...and all keys are released
    assert rec.step() is True
    assert [t.held for t in rec.ticks] == [["w"], ["w"], []]
    assert rec.frames[0].shape[1] <= 256  # frames are downscaled


def test_recording_becomes_a_skill_and_round_trips(home):
    log = InputLog()
    seq = ["grass", "grass", "stone", "stone", "weed"]
    frames = iter(seq)
    rec = Recorder(grab=lambda: screen(next(frames)), is_focused=lambda: True, log_=log)
    rec.step()                       # idle tick at the start: trimmed
    log.key("d", True); rec.step()
    log.key("d", False); log.key("w", True); rec.step()
    rec.step()
    log.key("w", False); rec.step()  # idle tick at the end: trimmed

    skill = rec.to_skill("walk around", "stardew", FakeEncoder())
    assert [t.held for t in skill.ticks] == [["d"], ["w"], ["w"]]
    assert skill.embeddings.shape[0] == 3
    skill.save()

    loaded = list_skills("stardew")
    assert [s.name for s in loaded] == ["walk around"]
    assert loaded[0].ticks == skill.ticks
    assert np.allclose(loaded[0].embeddings, skill.embeddings, atol=1e-3)


def test_empty_recording_is_rejected():
    rec = Recorder(grab=lambda: screen("grass"), is_focused=lambda: True, log_=InputLog())
    rec.step()
    with pytest.raises(ValueError, match="nothing was recorded"):
        rec.to_skill("x", "g", FakeEncoder())


# --- policy ----------------------------------------------------------------------

def make_skill(colors_and_keys, chunk=2):
    enc = FakeEncoder()
    imgs = np.stack([screen(c) for c, _ in colors_and_keys])
    ticks = [Tick(held=[k] if k else []) for _, k in colors_and_keys]
    return Skill("demo", "game", enc.encode_images(imgs), ticks, np.zeros(len(ticks), np.int32))


def test_policy_picks_most_similar_moment_and_returns_next_actions():
    skill = make_skill([("grass", "w"), ("grass", "w"), ("stone", "d"), ("stone", "d"), ("weed", "a"), ("weed", "a"), ("grass", None)])
    pol = NearestMomentPolicy(skill, chunk=2)
    emb = FakeEncoder().encode_images(screen("stone")[None])[0]
    i, sim = pol.choose(emb)
    assert skill.ticks[i].held == ["d"] and sim > 0.99
    assert [t.held for t in pol.actions(i)] == [["d"], ["d"]]


def test_policy_never_picks_a_moment_without_enough_future():
    # "weed" only appears in the final tick, so it can't be chosen
    skill = make_skill([("grass", "w"), ("grass", "w"), ("grass", "w"), ("weed", "a")])
    pol = NearestMomentPolicy(skill, chunk=2)
    i, _ = pol.choose(FakeEncoder().encode_images(screen("weed")[None])[0])
    assert i <= 1


def test_continuity_bonus_follows_one_recording():
    # Two identical-looking stretches: without the bonus argmax would always pick the first.
    skill = make_skill([("grass", "w")] * 4 + [("grass", "s")] * 4, chunk=2)
    pol = NearestMomentPolicy(skill, chunk=2, continuity_bonus=0.05)
    emb = FakeEncoder().encode_images(screen("grass")[None])[0]
    seq = [pol.choose(emb)[0] for _ in range(3)]
    assert seq == [0, 2, 4]


def test_short_recordings_shrink_the_chunk():
    skill = make_skill([("grass", "w"), ("stone", "d"), ("weed", "a")])
    pol = NearestMomentPolicy(skill, chunk=5)
    assert pol.chunk == 2 and pol._usable.tolist() == [True, False, False]


def test_segments_do_not_bleed_into_each_other():
    skill = make_skill([("grass", "w"), ("grass", "w"), ("stone", "d")])
    skill.add_recording(skill.embeddings[:1], [Tick(held=["x"])])
    pol = NearestMomentPolicy(skill, chunk=2)
    assert not pol._usable[2] and not pol._usable[3]


# --- playback --------------------------------------------------------------------

def test_tick_player_syncs_held_keys_and_clicks():
    io = RecordingIO()
    p = TickPlayer(io, to_screen=lambda fx, fy: (fx * 100, fy * 100))
    p.apply(Tick(held=["w", "shift"]))
    p.apply(Tick(held=["w"], dx=5, dy=0))
    p.apply(Tick(held=["w", "unknown-key"], clicks=[{"button": "left", "x": 0.5, "y": 0.2}]))
    p.release_all()
    assert io.events == [
        ("key", "shift", True), ("key", "w", True),
        ("key", "shift", False), ("delta", 5, 0),
        ("move", 50, 20), ("button", "left", True), ("button", "left", False),
        ("key", "w", False),
    ]


def test_play_skill_replays_and_stops():
    skill = make_skill([("grass", "w"), ("grass", "w"), ("stone", "d"), ("stone", "d"), ("weed", None)], chunk=2)
    io = RecordingIO()
    screens = iter(["stone", "grass", "grass", "grass"])
    calls = {"n": 0}

    def stop():
        calls["n"] += 1
        return calls["n"] > 7

    res = play_skill(skill, lambda: screen(next(screens)), FakeEncoder(), io, lambda x, y: (x, y),
                     should_stop=stop, sleep=lambda s: None, chunk=2)
    assert res.reason == "stopped"
    keys = [e for e in io.events if e[0] == "key"]
    assert keys[0] == ("key", "d", True)         # it saw stone, so it does what you did on stone
    assert ("key", "w", True) in keys            # then grass -> w
    assert keys[-1][2] is False                  # everything released at the end


def test_play_skill_stops_when_lost():
    skill = make_skill([("grass", "w")] * 6, chunk=2)
    t = {"now": 0.0}

    def clock():
        t["now"] += 1.0
        return t["now"]

    res = play_skill(skill, lambda: screen("crop_ripe"), FakeEncoder(), RecordingIO(), lambda x, y: (x, y),
                     min_similarity=0.99, lost_after_s=3, sleep=lambda s: None, clock=clock)
    assert res.reason == "lost"


# --- choosing a skill -------------------------------------------------------------

class SynonymEncoder:
    """Text encoder where 'wood/tree/chop/cut' and 'fish/rod/water' land close together."""

    GROUPS = [{"wood", "tree", "trees", "chop", "cut", "log"}, {"fish", "fishing", "rod", "lake"}]

    def encode_text(self, texts):
        out = []
        for t in texts:
            words = set(t.lower().split())
            v = np.array([len(words & g) for g in self.GROUPS] + [0.1], np.float32)
            out.append(v / np.linalg.norm(v))
        return np.stack(out)


def test_choose_skill():
    skills = [Skill(n, "g", np.zeros((0, 3)), [], np.zeros(0, np.int32)) for n in ["chop trees", "go fishing"]]
    assert choose_skill("", skills).name == "chop trees"
    assert choose_skill("go fishing", skills).name == "go fishing"
    assert choose_skill("fishing at the lake please", skills).name == "go fishing"  # word overlap, no encoder
    assert choose_skill("cut some wood", skills, SynonymEncoder()).name == "chop trees"
    assert choose_skill("anything", []) is None
