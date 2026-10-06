"""Each rule tested in isolation on a context with a fixed key."""

import unittest

from harmonia.analysis.pipeline import build_context
from harmonia.analysis.rules import borrowed, ii_v_i, patterns, secondary, tritone
from harmonia.io import parse_progression


def ctx(text: str, key: str = "C major"):
    c, _ = build_context(parse_progression(text), key=key)
    return c


class TestRulesInIsolation(unittest.TestCase):
    def test_ii_v_i(self):
        evs = ii_v_i.detect(ctx("Dm7 G7 Cmaj7"))
        self.assertEqual([e.label for e in evs], ["ii–V–I"])
        self.assertEqual(ii_v_i.detect(ctx("D7 G7 C")), [])  # major "ii" contradicts

    def test_secondary_dominant_resolution_states(self):
        self.assertEqual(secondary.detect_dominants(ctx("E7 Am"))[0].attributes["resolution"], "resolved")
        dec = secondary.detect_dominants(ctx("E7 F"))[0]
        self.assertEqual(dec.attributes["resolution"], "deceptive")
        self.assertEqual(dec.label, "V7/vi (deceptive)")
        un = secondary.detect_dominants(ctx("E7 G"))[0]
        self.assertEqual(un.attributes["resolution"], "unresolved")
        self.assertLess(un.confidence, dec.confidence)

    def test_leading_tone(self):
        self.assertEqual([e.label for e in secondary.detect_leading_tone(ctx("F#dim7 G"))], ["vii°7/V"])
        self.assertEqual(secondary.detect_leading_tone(ctx("Bdim C")), [])  # diatonic vii° → I

    def test_tritone(self):
        self.assertEqual([e.label for e in tritone.detect(ctx("Ab7 G7 C"))], ["subV7/V → V"])

    def test_borrowed_table(self):
        got = {e.attributes["entry"] for e in borrowed.detect(ctx("Fm Ab Bb Eb Gm Cm Dm7b5 Db"))}
        self.assertEqual(got, {"iv", "♭VI", "♭VII", "♭III", "v", "i", "iiø7", "♭II"})
        self.assertEqual(borrowed.detect(ctx("C Dm Em F G Am Bdim")), [])

    def test_borrowed_minor_key(self):
        got = {e.attributes["entry"] for e in borrowed.detect(ctx("A D Bm", key="A minor"))}
        self.assertEqual(got, {"I", "IV", "ii"})

    def test_aeolian(self):
        self.assertEqual(len(patterns.detect_aeolian(ctx("Ab Bb C"))), 1)
        self.assertEqual(patterns.detect_aeolian(ctx("F G Am", key="A minor")), [])

    def test_describe_modulation(self):
        from harmonia.theory.key import parse_key as k
        self.assertEqual(patterns.describe_modulation(k("C"), k("Db")), "half step up")
        self.assertEqual(patterns.describe_modulation(k("C"), k("Am")), "relative key")
        self.assertEqual(patterns.describe_modulation(k("C"), k("Cm")), "parallel key")


if __name__ == "__main__":
    unittest.main()
