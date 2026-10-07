"""Recognition-layer logic that does not need model weights: beat grid and HMM decoding on
synthetic frame evidence. Skipped when numpy is unavailable."""

import unittest

try:
    import numpy as np
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False


@unittest.skipUnless(HAVE_NUMPY, "numpy not installed")
class TestBeatGrid(unittest.TestCase):
    def test_intervals_bars_and_pickup(self):
        from harmonia.frontend.beats import BeatGrid, beat_intervals
        beats = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
        grid = BeatGrid(beats, np.array([1.0, 3.0]), "test", [])
        edges, bars, pos = beat_intervals(grid, 4.5)
        self.assertEqual(list(edges), [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5])
        # pre-roll (None), pickup beat in bar 0, then bars 1 and 2 of 4 beats
        self.assertEqual(bars, [None, 0, 1, 1, 1, 1, 2, 2, 2])
        self.assertEqual(pos, [None, 1, 1, 2, 3, 4, 1, 2, 3])


@unittest.skipUnless(HAVE_NUMPY, "numpy not installed")
class TestDecode(unittest.TestCase):
    def _evidence(self, seq, noise_at=None):
        from harmonia.frontend.chords import FrameEvidence
        labels = ["N", "C:maj", "F:maj", "G:maj", "A:min", "G:7"]
        fps = 10
        T = len(seq) * fps
        ll = np.full((T, len(labels)), -6.0)
        for i, lab in enumerate(seq):
            ll[i * fps:(i + 1) * fps, labels.index(lab)] = -0.1
        if noise_at is not None:  # one beat where the evidence is split
            i, a, b = noise_at
            ll[i * fps:(i + 1) * fps] = -6.0
            ll[i * fps:(i + 1) * fps, labels.index(a)] = -0.6
            ll[i * fps:(i + 1) * fps, labels.index(b)] = -0.8
        times = (np.arange(T) + 0.5) / fps
        bass = np.zeros((T, 13))
        bass[:, 1] = 1.0
        return FrameEvidence(times, labels, ll, bass, T / fps, "synthetic"), np.arange(len(seq) + 1, dtype=float)

    def test_clear_evidence_is_decoded(self):
        from harmonia.config import load_config
        from harmonia.frontend.decode import decode
        seq = ["C:maj"] * 4 + ["F:maj"] * 2 + ["G:7"] * 2 + ["C:maj"] * 4
        ev, edges = self._evidence(seq)
        d = decode(ev, edges, load_config()["frontend"]["decode"])
        self.assertEqual([d.labels[i] for i in d.path], seq)
        top = [d.labels[int(np.argmax(p))] for p in d.posteriors]
        self.assertEqual(top, seq)
        self.assertTrue(np.allclose(d.posteriors.sum(axis=1), 1.0))

    def test_ambiguous_beat_keeps_both_candidates(self):
        from harmonia.config import load_config
        from harmonia.frontend.decode import decode
        seq = ["C:maj"] * 3 + ["A:min"] + ["F:maj"] * 3
        ev, edges = self._evidence(seq, noise_at=(3, "A:min", "C:maj"))
        d = decode(ev, edges, load_config()["frontend"]["decode"])
        p = d.posteriors[3]
        top2 = {d.labels[i] for i in np.argsort(-p)[:2]}
        self.assertEqual(top2, {"A:min", "C:maj"})
        # The runner-up must survive as a candidate (soft evidence for the analysis layer).
        # How soft the posterior is depends on obs_weight / change_penalty, which are
        # uncalibrated until tuned on data/eval/dev — so only the survival is asserted here.
        second = np.sort(p)[-2]
        self.assertGreaterEqual(second, load_config()["frontend"]["min_candidate_prob"])

    def test_seventh_bias_only_lifts_seventh_chords(self):
        from harmonia.config import load_config
        from harmonia.frontend.decode import beat_observations
        ev, edges = self._evidence(["G:maj"] * 2)
        dc = dict(load_config()["frontend"]["decode"], seventh_bias=0.0)
        o0, _ = beat_observations(ev, edges, dc)
        o1, _ = beat_observations(ev, edges, dict(dc, seventh_bias=1.0))
        diff = (o1 - o0)[0]
        self.assertEqual([ev.labels[i] for i in np.flatnonzero(diff)], ["G:7"])
        self.assertAlmostEqual(float(diff.max()), 1.0)


if __name__ == "__main__":
    unittest.main()
