"""Song-structure analysis (frontend/structure.py) on synthetic bar features with a known form.
No audio needed; skipped without numpy."""

import unittest

try:
    import numpy as np
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False


def _bars(roots, m=4, shift=0):
    """One bar per root: chord identity one-hot (major chord on root), chroma left empty."""
    out = np.zeros((len(roots), m, 3, 12))
    for i, r in enumerate(roots):
        out[i, :, 0, (r + shift) % 12] = 1.0
    return out


@unittest.skipUnless(HAVE_NUMPY, "numpy not installed")
class TestStructure(unittest.TestCase):
    def setUp(self):
        from harmonia.config import load_config
        self.cfg = load_config()
        verse = [0, 0, 5, 5, 9, 9, 7, 7]          # C C F F A A G G
        pre = [2, 2, 4, 4, 5, 5, 7, 7]            # D D E E F F G G
        chorus = [5, 7, 4, 9, 5, 7, 0, 0]         # F G E A F G C C
        bridge = [8, 8, 10, 10, 1, 1, 3, 3]       # Ab Ab Bb Bb Db Db Eb Eb (no transposed verse inside)
        intro, outro = [0, 7], [0, 5, 0, 0]
        plan = [("intro", intro, 0), ("verse", verse, 0), ("pre-chorus", pre, 0), ("chorus", chorus, 0),
                ("verse", verse, 0), ("pre-chorus", pre, 0), ("chorus", chorus, 0), ("bridge", bridge, 0),
                ("chorus", chorus, 0), ("chorus", chorus, 2), ("outro", outro, 0)]
        self.truth, self.plan = [], []
        harm, loud = [], []
        for label, roots, shift in plan:
            self.plan.append((label, len(self.truth), len(self.truth) + len(roots)))
            harm.append(_bars(roots, shift=shift))
            loud += [(-10.0 if label == "chorus" else -18.0)] * len(roots)
            self.truth += [label] * len(roots)
        self.harm = np.concatenate(harm)
        self.loud = np.array(loud)
        rng = np.random.default_rng(0)
        self.timbre = np.concatenate([rng.normal(0, 0.1, (len(loud), 12)),
                                      ((self.loud - self.loud.mean()) / self.loud.std())[:, None]], axis=1)
        self.bar_times = [2.0 * i for i in range(len(loud))]

    def analyse(self):
        from harmonia.frontend.structure import analyse
        return analyse(self.harm, self.timbre, self.loud, self.bar_times, 2.0 * len(self.loud), self.cfg)

    def test_labels_follow_the_form(self):
        st = self.analyse()
        pred = []
        for s in st.sections:
            pred += [s.label] * (s.end_bar - s.start_bar)
        agree = sum(p == t for p, t in zip(pred, self.truth)) / len(self.truth)
        self.assertGreaterEqual(agree, 0.9, [(s.start_bar, s.end_bar, s.label) for s in st.sections])

    def test_transposed_last_chorus_is_a_chorus_with_its_shift(self):
        st = self.analyse()
        last = [s for s in st.sections if s.label == "chorus"][-1]
        self.assertEqual(last.shift, 2)

    def test_external_labels_with_our_grouping(self):
        # SongFormer-style input: its boundaries (slightly off the bar lines) and labels are kept,
        # a leading "silence" is absorbed; repetition groups and the modulation come from us.
        from harmonia.frontend.structure import analyse
        ext = [(0.0, 0.5, "silence")]
        for k, (lab, a, b) in enumerate(self.plan):
            jitter = 0.4 if k % 2 else -0.4
            start = 0.5 if k == 0 else ext[-1][1]
            end = 2.0 * b + (jitter if k < len(self.plan) - 1 else 0.0)
            ext.append((start, end, lab))
        st = analyse(self.harm, self.timbre, self.loud, self.bar_times, 2.0 * len(self.loud), self.cfg, external=ext)
        pred = []
        for sec in st.sections:
            pred += [sec.label] * (sec.end_bar - sec.start_bar)
        self.assertEqual(pred, self.truth)
        choruses = [sec for sec in st.sections if sec.label == "chorus"]
        self.assertEqual(len({sec.cluster for sec in choruses}), 1)
        self.assertEqual([sec.shift for sec in choruses], [0, 0, 0, 2])
        verse = [sec for sec in st.sections if sec.label == "verse"]
        self.assertNotEqual(verse[0].cluster, choruses[0].cluster)

    def test_too_short_input_is_reported_not_guessed(self):
        from harmonia.frontend.structure import analyse
        st = analyse(self.harm[:4], self.timbre[:4], self.loud[:4], self.bar_times[:4], 8.0, self.cfg)
        self.assertEqual(st.sections, [])
        self.assertTrue(st.warnings)


if __name__ == "__main__":
    unittest.main()
