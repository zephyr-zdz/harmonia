"""Phase 2 evaluation framework. Needs mir_eval (``uv sync --extra eval``); skipped otherwise.
Fixtures in tests/fixtures/eval are synthetic, not real songs."""

import tempfile
import unittest
from pathlib import Path

try:
    import mir_eval  # noqa: F401
    HAVE_MIR_EVAL = True
except ImportError:
    HAVE_MIR_EVAL = False

from harmonia.eval.align import align, est_to_ref_map
from harmonia.theory.chord import parse_chord

FIX = Path(__file__).parent / "fixtures" / "eval"


class TestAlign(unittest.TestCase):
    def test_alignment_with_oversegmentation_and_substitution(self):
        ref = [parse_chord(x) for x in "C F G C".split()]
        est = [parse_chord(x) for x in "C F F7 G Am C".split()]
        cost, path = align(ref, est)
        self.assertEqual(est_to_ref_map(path, len(est)), [0, 1, 1, 2, 2, 3])
        self.assertAlmostEqual(cost, 2.0)


@unittest.skipUnless(HAVE_MIR_EVAL, "mir_eval not installed")
class TestEventMatching(unittest.TestCase):
    def setUp(self):
        from harmonia.eval.events import Span
        self.S = Span

    def test_type_and_overlap(self):
        from harmonia.eval.events import match, prf
        g = [self.S("ii_V_I", 4, 10), self.S("borrowed_chord", 16, 18)]
        p = [self.S("ii_V_I", 6, 10, confidence=0.9), self.S("secondary_dominant", 16, 18),
             self.S("borrowed_chord", 17.5, 20)]
        m = match(g, p)
        self.assertEqual(m.pairs, [(0, 0)])
        self.assertEqual(sorted(m.false_pos), [1, 2])
        self.assertEqual(m.missed, [1])
        r = prf(g, p)["overall"]
        self.assertAlmostEqual(r["precision"], 1 / 3)
        self.assertAlmostEqual(r["recall"], 1 / 2)

    def test_point_events(self):
        from harmonia.eval.events import match
        g = [self.S("modulation", 60, 60)]
        self.assertEqual(len(match(g, [self.S("modulation", 61.5, 61.5)]).pairs), 1)
        self.assertEqual(len(match(g, [self.S("modulation", 63, 63)]).pairs), 0)

    def test_attribution_categories(self):
        from harmonia.eval.events import attribute
        g = [self.S("ii_V_I", 0, 6), self.S("ii_V_I", 10, 16), self.S("borrowed_chord", 20, 22)]
        oracle = [self.S("ii_V_I", 0, 6), self.S("ii_V_I", 10, 16), self.S("tritone_sub", 30, 32)]
        system = [self.S("ii_V_I", 0, 6), self.S("tritone_sub", 30, 32), self.S("secondary_dominant", 40, 42)]
        c = attribute(g, oracle, system)["counts"]
        self.assertEqual(c, {"ok": 1, "recognition_miss": 1, "analysis_miss": 1,
                             "analysis_fp": 1, "recognition_fp": 1})


@unittest.skipUnless(HAVE_MIR_EVAL, "mir_eval not installed")
class TestRunner(unittest.TestCase):
    def test_reference_as_estimate_is_perfect(self):
        from harmonia.eval.dataset import discover
        from harmonia.eval.runner import evaluate_song
        from harmonia.config import load_config
        for ref in discover(FIX, "dev"):
            with self.subTest(song=ref.song_id):
                r = evaluate_song(ref, ref.recognition, load_config())
                self.assertEqual(r["chord"]["cer_majmin"], 0.0)
                self.assertEqual(r["events_agreement"]["overall"]["f1"], 1.0)
                self.assertEqual(r["events_oracle"]["overall"]["recall"], 1.0)
                self.assertEqual(r["attribution"]["counts"], {"ok": len(ref.events)})
                self.assertEqual(r["numeral_agreement"], 1.0)

    def test_chart_reference_vs_estimate_in_seconds(self):
        # The same chords performed at 120 BPM (seconds timeline) must align to the chart.
        from harmonia.eval.dataset import load_reference
        from harmonia.eval.runner import evaluate_song
        from harmonia.config import load_config
        from harmonia.schema import ChordCandidate, Frame, RecognitionResult
        ref = load_reference(FIX / "dev" / "toy_chart", "dev")
        est = RecognitionResult(frames=[Frame(f.time * 0.5, f.duration * 0.5, [ChordCandidate(f.candidates[0].label, 0.9)])
                                        for f in ref.recognition.frames], time_unit="second")
        r = evaluate_song(ref, est, load_config())
        self.assertEqual(r["chord"]["aligned_majmin"], 1.0)
        self.assertEqual(r["events_system"]["overall"]["f1"], 1.0)

    def test_simulation_is_seeded_and_errors_lower_scores(self):
        from harmonia.eval.dataset import load_reference
        from harmonia.eval.simulate import SimConfig, simulate
        ref = load_reference(FIX / "dev" / "toy_lab", "dev")
        iv, lab = ref.intervals_labels()
        a = simulate(iv, lab, SimConfig(0.3, 0.3, 0.1, seed=7))
        b = simulate(iv, lab, SimConfig(0.3, 0.3, 0.1, seed=7))
        c = simulate(iv, lab, SimConfig(0.3, 0.3, 0.1, seed=8))
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        from harmonia.eval.runner import run
        clean = run(FIX, "dev", sim=SimConfig(0, 0, 0, 0))["aggregate"]
        noisy = run(FIX, "dev", sim=SimConfig(0.3, 0.4, 0.1, 0))["aggregate"]
        self.assertAlmostEqual(clean["chord_mir_eval"]["majmin"], 1.0)
        self.assertLess(noisy["chord_mir_eval"]["majmin"], 1.0)
        self.assertLessEqual(noisy["events_agreement"]["overall"]["f1"], clean["events_agreement"]["overall"]["f1"])

    def test_cli_writes_report(self):
        from harmonia.cli import main
        with tempfile.TemporaryDirectory() as d:
            import contextlib
            import io
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                main(["eval", "--data", str(FIX), "--simulate", "--out", d])
            files = sorted(p.suffix for p in Path(d).iterdir())
            self.assertEqual(files, [".json", ".md"])


if __name__ == "__main__":
    unittest.main()
