"""Behaviour decided by the user on 2026-10-06 (see CLAUDE.md "Decisions log")."""

import unittest

from .helpers import at, events, numerals, run


class TestNamedProgressions(unittest.TestCase):
    def test_marusa_auto_key_uses_conventional_reference(self):
        # Decision Q1: without a given key, 丸サ進行 is analysed in its naming reference.
        res = run("Fmaj7 E7 Am7 Gm7 C7 Fmaj7")
        self.assertEqual(res.global_key.key.label, "C major")
        self.assertEqual(numerals(res), ["IVmaj7", "III7", "vim7", "vm7", "I7", "IVmaj7"])
        self.assertEqual([p.name for p in res.progressions], ["丸サ進行"])
        self.assertTrue(res.progressions[0].key_prior_applied)

    def test_marusa_original_key(self):
        # Just the Two of Us in its original D♭ voicing → reference A♭ major
        res = run("Dbmaj7 C7 Fm7 Ebm7 Ab7 Dbmaj7")
        self.assertEqual(res.global_key.key.label, "Ab major")
        self.assertEqual(res.progressions[0].numerals, ["IVmaj7", "III7", "vim7", "vm7", "I7"])

    def test_royal_road_is_named_but_keeps_ambiguity(self):
        # Decision Q2: no key prior for 王道進行; the ambiguity flag stays.
        res = run("F G Em Am F G Em Am")
        self.assertEqual([p.name for p in res.progressions], ["王道進行", "王道進行"])
        self.assertFalse(res.progressions[0].key_prior_applied)
        self.assertTrue(res.global_key.ambiguous)

    def test_other_idioms(self):
        self.assertEqual([p.name for p in run("C G Am Em F C F G C").progressions], ["カノン進行"])
        self.assertEqual([p.name for p in run("Am F G C Am F G C", key="C").progressions],
                         ["小室進行", "小室進行"])
        self.assertEqual(run("C F G C", key="C").progressions, [])


class TestUnresolvedIiV(unittest.TestCase):
    def test_tonic_ii_V_unresolved(self):
        res = run("C Dm7 G7 Em7 Am", key="C")
        ev = events(res, "ii_V")
        self.assertEqual([e.label for e in ev], ["ii–V (unresolved)"])
        self.assertEqual(ev[0].attributes["resolution"], "unresolved")
        self.assertEqual(events(res, "ii_V_I"), [])

    def test_ii_V_at_end_of_input(self):
        ev = events(run("C Am Dm7 G7", key="C"), "ii_V")
        self.assertEqual([e.attributes["resolution"] for e in ev], ["end"])

    def test_secondary_ii_V_deceptive(self):
        res = run("C Bm7b5 E7 F G C", key="C")
        ev = events(res, "ii_V")
        self.assertEqual([e.label for e in ev], ["ii–V/vi (deceptive)"])
        self.assertEqual(ev[0].functions[:2], ["iiø7/vi", "V7/vi"])

    def test_secondary_ii_V_unresolved_target_quality(self):
        self.assertEqual([e.label for e in events(run("C Em7 A7 F G C", key="C"), "ii_V")],
                         ["ii–V/ii (unresolved)"])

    def test_resolved_ii_V_I_is_not_also_unresolved(self):
        self.assertEqual(events(run("C Dm7 G7 C", key="C"), "ii_V"), [])

    def test_backdoor_is_not_a_failed_ii_V(self):
        # iv–♭VII7–I: Fm7–B♭7 is not a "ii–V/♭III that failed"; it is the backdoor progression.
        res = run("C F Fm7 Bb7 C", key="C")
        self.assertEqual(events(res, "ii_V"), [])
        self.assertTrue(at(res, "borrowed_chord", 3)[0].attributes.get("backdoor"))

    def test_diatonic_fifth_chains_do_not_trigger(self):
        self.assertEqual(events(run("C Em Am Dm G C", key="C"), "ii_V"), [])
        self.assertEqual(events(run("F G Em Am", key="C"), "ii_V"), [])


class TestDeceptiveCadence(unittest.TestCase):
    def test_prepared_V_to_vi(self):
        res = run("C Dm7 G7 Am F G C", key="C")
        ev = events(res, "deceptive_cadence")
        self.assertEqual([e.label for e in ev], ["V–vi (deceptive cadence)"])
        self.assertEqual(ev[0].chords, ["G7", "Am"])
        # reported once: not also as an unresolved ii–V
        self.assertEqual(events(res, "ii_V"), [])

    def test_V7_without_predominant(self):
        ev = events(run("C G7 Am", key="C"), "deceptive_cadence")
        self.assertEqual(len(ev), 1)
        self.assertLess(ev[0].confidence, 1.0)

    def test_minor_key_V_to_bVI(self):
        ev = events(run("Am Dm E7 F G Am", key="Am"), "deceptive_cadence")
        self.assertEqual([e.label for e in ev], ["V–♭VI (deceptive cadence)"])

    def test_passing_V_vi_in_loops_is_not_a_cadence(self):
        # カノン進行 I–V–vi and I–V–vi–IV: unprepared triad V → vi is passing motion
        self.assertEqual(events(run("C G Am Em F C F G C"), "deceptive_cadence"), [])
        self.assertEqual(events(run("C G Am F", key="C"), "deceptive_cadence"), [])


if __name__ == "__main__":
    unittest.main()
