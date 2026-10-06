"""Tempo -> meter -> bar grid on synthetic tracker output (no model weights needed)."""

import unittest
from collections import Counter

try:
    import numpy as np
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False


def tracker(bpm, beats_per_bar, n_bars, start=1.0, drop_every=0, bar_lengths=None, extra_downbeats=(),
            missing_downbeats=()):
    """Synthetic beat-tracker output with optional missed beats / downbeat errors."""
    P = 60.0 / bpm
    beats, downbeats, t = [], [], start
    lengths = bar_lengths or [beats_per_bar] * n_bars
    for i, n in enumerate(lengths):
        if i not in missing_downbeats:
            downbeats.append(t)
        for k in range(n):
            beats.append(t + k * P)
        t += n * P
    beats = [b for j, b in enumerate(beats) if not (drop_every and j % drop_every == drop_every - 1 and b not in downbeats)]
    downbeats = sorted(downbeats + list(extra_downbeats))
    return np.array(beats), np.array(downbeats), t  # audio ends at the last bar's end


@unittest.skipUnless(HAVE_NUMPY, "numpy not installed")
class TestMeterGrid(unittest.TestCase):
    def test_bpm_and_meter_with_missed_beats(self):
        from harmonia.frontend.meter import build_grid
        b, d, dur = tracker(120, 4, 40, drop_every=3)  # a third of the beats missing
        g = build_grid(b, d, dur)
        self.assertAlmostEqual(g.bpm, 120, delta=0.5)
        self.assertEqual(g.beats_per_bar, 4)
        self.assertTrue(g.time_signature.startswith("4 beats/bar"))
        per_bar = Counter(Counter(i for i in g.bar_index if i > 0).values())
        self.assertEqual(per_bar[4], len(g.bar_starts))

    def test_compound_six_eight(self):
        # 春日影-like: 6/8, eighth = 194 BPM, bar ≈ 1.856 s
        from harmonia.frontend.meter import build_grid
        b, d, dur = tracker(194, 6, 30, drop_every=4)
        g = build_grid(b, d, dur)
        self.assertEqual(g.beats_per_bar, 6)
        self.assertEqual(g.time_signature, "6/8")
        self.assertAlmostEqual(g.bar_period, 6 * 60 / 194, delta=0.01)

    def test_missing_and_spurious_downbeats(self):
        from harmonia.frontend.meter import build_grid
        b, d, dur = tracker(100, 4, 20, missing_downbeats=(5, 6), extra_downbeats=(1.0 + 0.6 * 10 + 0.3,))
        g = build_grid(b, d, dur)
        self.assertEqual(len(g.bar_starts), 20)
        self.assertEqual(g.irregular_bars, [])

    def test_genuine_short_bar_is_kept_and_reported(self):
        # one 2/4 bar inside 4/4 (common in J-pop) must not be "repaired" away
        from harmonia.frontend.meter import build_grid
        lengths = [4] * 8 + [2] + [4] * 8
        b, d, dur = tracker(120, 4, len(lengths), bar_lengths=lengths)
        g = build_grid(b, d, dur)
        self.assertEqual(g.beats_per_bar, 4)
        self.assertEqual(g.irregular_bars, [(9, 2)])
        self.assertTrue(any("irregular" in w for w in g.warnings))

    def test_double_triggers_do_not_halve_the_tempo(self):
        # tracker occasionally fires twice per beat (音一会 regression: was read as 400 BPM)
        from harmonia.frontend.meter import build_grid
        b, d, dur = tracker(200, 4, 40)
        rng = np.random.default_rng(0)
        extra_b = b[rng.choice(len(b), 40, replace=False)] + 0.14
        extra_d = d[rng.choice(len(d), 8, replace=False)] + 0.14
        g = build_grid(np.sort(np.concatenate([b, extra_b])), np.sort(np.concatenate([d, extra_d])), dur)
        self.assertAlmostEqual(g.bpm, 200, delta=1.0)
        self.assertEqual(g.beats_per_bar, 4)

    def test_pickup_beats(self):
        from harmonia.frontend.meter import build_grid
        b, d, dur = tracker(120, 4, 10, start=1.7)  # pickups at 1.2, 0.7, 0.2 s
        g = build_grid(b, d, dur)
        self.assertEqual(g.bar_index[:3], [0, 0, 0])
        self.assertEqual(g.beat_in_bar[:4], [2, 3, 4, 1])


if __name__ == "__main__":
    unittest.main()
