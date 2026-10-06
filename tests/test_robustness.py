"""Design principle 2 (tolerate recognition errors) and 'never swallow failures'."""

import json
import unittest

from harmonia import analyze
from harmonia.io import parse_progression
from harmonia.schema import ChordCandidate, Frame, RecognitionResult
from harmonia.theory.chord import ChordParseError

from .helpers import events


def soft(*frames: list[tuple[str, float]], key=None):
    fr = [Frame(time=4.0 * i, duration=4.0, beats=4.0, candidates=[ChordCandidate(l, p) for l, p in c])
          for i, c in enumerate(frames)]
    return analyze(RecognitionResult(frames=fr, time_unit="beat"), key=key)


class TestSoftEvidence(unittest.TestCase):
    def test_triads_without_sevenths_still_ii_V_I(self):
        res = soft([("Dm", 1.0)], [("G", 1.0)], [("C", 1.0)], key="C")
        self.assertEqual(len(events(res, "ii_V_I")), 1)
        self.assertGreaterEqual(events(res, "ii_V_I")[0].confidence, 0.9)

    def test_power_chords_give_low_confidence_ii_V_I(self):
        # quality uninformative → root motion alone carries the event, flagged low-confidence
        res = soft([("D5", 1.0)], [("G5", 1.0)], [("C5", 1.0)], key="C")
        ev = events(res, "ii_V_I")
        self.assertEqual(len(ev), 1)
        self.assertTrue(ev[0].low_confidence)

    def test_misrecognised_ii_from_top_k(self):
        # Dm7 often comes out as F (shares F-A-C); the ii–V–I survives via the 2nd candidate.
        res = soft([("C", 1.0)], [("F", 0.55), ("Dm7", 0.45)], [("G7", 0.9), ("Em", 0.1)], [("C", 1.0)], key="C")
        ev = events(res, "ii_V_I")
        self.assertEqual(len(ev), 1)
        self.assertTrue(0.35 <= ev[0].confidence < 0.6)
        self.assertTrue(ev[0].low_confidence)

    def test_quality_confusion_lowers_but_keeps_secondary_dominant(self):
        res = soft([("C", 1.0)], [("E7", 0.6), ("Em7", 0.4)], [("Am", 1.0)], key="C")
        sd = events(res, "secondary_dominant")
        self.assertEqual(len(sd), 1)
        self.assertAlmostEqual(sd[0].confidence, 0.6, places=2)

    def test_low_confidence_segments_flagged(self):
        res = soft([("C", 0.4), ("Am", 0.3)], [("G", 1.0)], key="C")
        self.assertTrue(res.segments[0].low_confidence)
        self.assertTrue(any("chord confidence" in w for w in res.segments[0].warnings))
        self.assertFalse(res.segments[1].low_confidence)


class TestFailuresAreVisible(unittest.TestCase):
    def test_text_parser_raises_by_default(self):
        with self.assertRaises(ChordParseError):
            parse_progression("| C | Hm | G |")

    def test_flag_mode_keeps_going_with_warning(self):
        rec = parse_progression("| C | Cfoo | G | C |", on_error="flag")
        res = analyze(rec, key="C")
        self.assertTrue(any("Cfoo" in w for w in res.warnings))
        bad = res.segments[1]
        self.assertEqual(bad.chord, "X")
        self.assertTrue(bad.low_confidence)

    def test_unparseable_candidate_dropped_with_warning(self):
        res = soft([("C", 0.7), ("C??", 0.3)], [("G", 1.0)], key="C")
        self.assertTrue(any("C??" in w for w in res.warnings))
        self.assertEqual(res.segments[0].chord, "C")


class TestSchema(unittest.TestCase):
    def test_recognition_roundtrip_and_json(self):
        rec = parse_progression("[Verse] | C | Am7 . . G | [Chorus] | F | % |")
        self.assertEqual([f.section for f in rec.frames], ["Verse", "Verse", "Verse", "Chorus", "Chorus"])
        self.assertEqual([f.beats for f in rec.frames], [4.0, 3.0, 1.0, 4.0, 4.0])
        rec2 = RecognitionResult.from_dict(json.loads(json.dumps(rec.to_dict())))
        self.assertEqual(rec2, rec)
        res = analyze(rec2)
        d = json.loads(res.to_json())
        self.assertEqual(d["schema_version"], "0.1.0")
        for k in ("global_key", "key_regions", "segments", "events", "warnings"):
            self.assertIn(k, d)
        # repeated F bars merge into one segment
        self.assertEqual([s["chord"] for s in d["segments"]], ["C", "Am7", "G", "F"])


if __name__ == "__main__":
    unittest.main()
