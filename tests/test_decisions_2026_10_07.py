"""Behaviour decided by the user on 2026-10-07 (see CLAUDE.md "Decisions log")."""

import unittest

from harmonia.analysis.pipeline import build_context
from harmonia.analysis.rules import ii_v_i
from harmonia.io import parse_progression

from .helpers import events, run


class TestBoundaryTonicBonus(unittest.TestCase):
    """Bonus switched on: a song that begins and ends on a chord is probably in that key."""

    def test_mixolydian_loop_stays_on_its_tonic(self):
        # I–♭VII–IV–I: without the bonus the ♭VII makes this diatonic in the key of IV.
        self.assertEqual(run("G F C G F C G").global_key.key.label, "G major")
        self.assertEqual(run("C Bb F C Bb F C").global_key.key.label, "C major")

    def test_named_progression_boundaries_are_exempt(self):
        # 王道 opens on IV, 丸サ loops back to IVmaj7: the idiom decides, not "first chord = I"
        # (the 2026-10-06 tests for 丸サ / 王道 cover the key; this checks a restart chord).
        res = run("Fmaj7 E7 Am7 Gm7 C7 Fmaj7")
        self.assertEqual(res.global_key.key.label, "C major")


class TestDeceptiveCadenceGrading(unittest.TestCase):
    """Too many reports on J-pop: nearly all were IV–V–vi, stock phrase motion."""

    def test_iv_v_vi_triads_is_weak(self):
        ev = events(run("C F G Am F G C", key="C"), "deceptive_cadence")
        self.assertTrue(all(e.low_confidence for e in ev))

    def test_iv_v7_vi_and_ii_v_vi_stay(self):
        ev = events(run("C F G7 Am F G C", key="C"), "deceptive_cadence")
        self.assertEqual(len(ev), 1)
        self.assertFalse(ev[0].low_confidence)
        ev = events(run("C Dm G Am F G C", key="C"), "deceptive_cadence")
        self.assertEqual([e.confidence for e in ev], [1.0])


class TestFewerFalseAppliedChords(unittest.TestCase):
    """Secondary-dominant false positives (ChoCo dev applied precision 0.23)."""

    def test_unresolved_ii_V_needs_chromatic_V(self):
        # Gm–C in C: a "ii–V/IV" whose V is the plain tonic triad and never reaches IV is not
        # an applied ii–V (design decision 2: applied dominants carry an out-of-key tone).
        c, _ = build_context(parse_progression("C Gm C Am F G C"), key="C major")
        self.assertEqual([e for e in ii_v_i.detect(c) if e.type == "ii_V"], [])
        # with the chromatic V7 (b7 = B♭) it is still reported
        c, _ = build_context(parse_progression("C Gm7 C7 Am F G C"), key="C major")
        self.assertEqual([e.label for e in ii_v_i.detect(c) if e.type == "ii_V"], ["ii–V/IV (unresolved)"])

    def test_unresolved_dominant_of_nondiatonic_target_dropped(self):
        # blues IV7 in A (D7 → A): not "V7/♭VII (unresolved)"
        self.assertEqual(events(run("A D7 A E A", key="A"), "secondary_dominant"), [])
        # resolved V7/♭VII is still reported
        self.assertEqual([e.label for e in events(run("A D7 G A", key="A"), "secondary_dominant")], ["V7/♭VII"])


if __name__ == "__main__":
    unittest.main()
