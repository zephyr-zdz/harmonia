"""End-to-end tests on the progressions required for Phase 1 plus negative (no false
positive) cases. Expected values come from standard pop / jazz harmony analysis."""

import unittest

from .helpers import NOTABLE, at, events, labels, numerals, run


class TestRequiredProgressions(unittest.TestCase):
    def test_royal_road_iv_v_iii_vi(self):
        # 王道進行: IV–V–iii–vi, fully diatonic → numerals only, no notable events.
        res = run("F G Em Am F G Em Am", key="C major")
        self.assertEqual(numerals(res)[:4], ["IV", "V", "iii", "vi"])
        self.assertEqual(events(res), [])

    def test_royal_road_with_III7(self):
        # Common J-pop variant IV–V–III7–vi: III7 is V7/vi.
        res = run("F G E7 Am", key="C major")
        self.assertEqual(numerals(res), ["IV", "V", "III7", "vi"])
        sd = at(res, "secondary_dominant", 2)
        self.assertEqual([e.label for e in sd], ["V7/vi"])
        self.assertEqual(sd[0].attributes["resolution"], "resolved")

    def test_just_the_two_of_us(self):
        # 丸サ進行 IVmaj7–III7–vim7–vm7–I7 (→ IVmaj7), labelled in C as the user specified.
        res = run("Fmaj7 E7 Am7 Gm7 C7 Fmaj7", key="C major")
        self.assertEqual(numerals(res), ["IVmaj7", "III7", "vim7", "vm7", "I7", "IVmaj7"])
        # III7 → vim7 : V7/vi
        self.assertEqual([e.label for e in at(res, "secondary_dominant", 1)], ["V7/vi"])
        # vm7 – I7 – IVmaj7 : ii–V into IV
        iv = events(res, "ii_V_I")
        self.assertEqual(len(iv), 1)
        self.assertEqual(iv[0].label, "ii–V–I → IV")
        self.assertEqual(iv[0].segment_indices, [3, 4, 5])
        self.assertEqual(iv[0].functions[:2], ["iim7/IV", "V7/IV"])
        # Gm7 is explained by the ii–V, so it is NOT reported as a borrowed v (kept as alternative).
        self.assertEqual(at(res, "borrowed_chord", 3), [])
        self.assertTrue(any(a["type"] == "borrowed_chord" for a in iv[0].alternatives))
        self.assertEqual(res.segments[3].functions, ["iim7/IV"])
        self.assertEqual(res.segments[4].functions, ["V7/IV"])

    def test_canon(self):
        res = run("C G Am Em F C F G C")
        self.assertEqual(res.global_key.key.label, "C major")
        self.assertEqual(numerals(res), ["I", "V", "vi", "iii", "IV", "I", "IV", "V", "I"])
        self.assertEqual(events(res), [])

    def test_secondary_dominants_V_of_V_and_V_of_vi(self):
        res = run("C D7 G7 C E7 Am F G7 C")
        self.assertEqual(res.global_key.key.label, "C major")
        self.assertEqual([e.label for e in at(res, "secondary_dominant", 1)], ["V7/V"])
        self.assertEqual([e.label for e in at(res, "secondary_dominant", 4)], ["V7/vi"])
        # II7–V7–I is V7/V–V7–I, not a ii–V–I (ii must not be major).
        self.assertEqual(events(res, "ii_V_I"), [])
        # the primary V7 is not a secondary dominant
        self.assertEqual(at(res, "secondary_dominant", 7), [])

    def test_triad_secondary_dominant_and_V_of_ii(self):
        res = run("C A7 Dm7 G7 C D G C", key="C major")
        self.assertEqual([e.label for e in at(res, "secondary_dominant", 1)], ["V7/ii"])
        self.assertEqual([e.label for e in at(res, "secondary_dominant", 5)], ["V/V"])
        self.assertEqual(labels(res, "ii_V_I"), ["ii–V–I"])

    def test_bVI_bVII_I(self):
        res = run("C F G C Ab Bb C")
        self.assertEqual(res.global_key.key.label, "C major")
        self.assertEqual(numerals(res)[4:], ["♭VI", "♭VII", "I"])
        self.assertEqual(labels(res, "aeolian_cadence"), ["♭VI–♭VII–I"])
        self.assertEqual(len(at(res, "borrowed_chord", 4)), 1)
        self.assertEqual(len(at(res, "borrowed_chord", 5)), 1)

    def test_minor_ii_V_i(self):
        res = run("Am Dm Bm7b5 E7 Am")
        self.assertEqual(res.global_key.key.label, "A minor")
        iv = events(res, "ii_V_I")
        self.assertEqual(len(iv), 1)
        self.assertEqual(iv[0].label, "ii–V–i")
        self.assertEqual(iv[0].numerals, ["iiø7", "V7", "i"])
        self.assertEqual(iv[0].attributes["mode"], "minor")
        # E7 is the primary dominant of A minor, not a secondary dominant
        self.assertEqual(events(res, "secondary_dominant"), [])

    def test_minor_ii_V_into_vi_in_major(self):
        res = run("Cmaj7 Dm7 G7 Cmaj7 Bm7b5 E7 Am7 F G7 Cmaj7")
        self.assertEqual(res.global_key.key.label, "C major")
        lbl = labels(res, "ii_V_I")
        self.assertIn("ii–V–I", lbl)
        self.assertIn("ii–V–i → vi", lbl)
        sec = [e for e in events(res, "ii_V_I") if e.attributes["tonicization"]][0]
        self.assertEqual(sec.functions[:2], ["iiø7/vi", "V7/vi"])

    def test_tritone_substitution(self):
        res = run("Cmaj7 Am7 Dm7 Db7 Cmaj7")
        self.assertEqual(res.global_key.key.label, "C major")
        tt = events(res, "tritone_sub")
        self.assertEqual(len(tt), 1)
        self.assertEqual(tt[0].label, "subV7 → I")
        self.assertEqual(tt[0].numerals[0], "♭II7")
        self.assertTrue(tt[0].attributes["preceded_by_ii"])
        self.assertEqual(at(res, "borrowed_chord", 3), [])

    def test_borrowed_iv(self):
        res = run("C F Fm C", key="C major")
        self.assertEqual(at(res, "borrowed_chord", 1), [])
        self.assertEqual(len(at(res, "borrowed_chord", 2)), 1)
        self.assertEqual(at(res, "borrowed_chord", 2)[0].attributes["entry"], "iv")

    def test_backdoor_bVII7(self):
        res = run("C F Fm7 Bb7 C", key="C major")
        bd = at(res, "borrowed_chord", 3)
        self.assertEqual(len(bd), 1)
        self.assertTrue(bd[0].attributes.get("backdoor"))

    def test_secondary_leading_tone(self):
        res = run("C C#dim7 Dm7 G7 C")
        self.assertEqual([e.label for e in events(res, "secondary_leading_tone")], ["vii°7/ii"])

    def test_modulation_whole_step_up(self):
        # J-pop "ラスサビ転調": final chorus a whole step up.
        res = run("C F G C Am Dm G C Em Am Dm G C F G C "
                  "D G A D Bm Em A D G A Bm G Em A D D")
        mods = events(res, "modulation")
        self.assertEqual(len(mods), 1)
        self.assertEqual(mods[0].attributes["from"], "C major")
        self.assertEqual(mods[0].attributes["to"], "D major")
        self.assertEqual(mods[0].attributes["interval"], "whole step up")
        self.assertIn(mods[0].segment_indices[1], (15, 16, 17))


class TestNoFalsePositives(unittest.TestCase):
    def assertNoNotable(self, res):
        self.assertEqual([(e.type, e.label) for e in res.events if e.type in NOTABLE and e.type != "ii_V_I"], [])

    def test_diatonic_I_IV_V_I(self):
        self.assertEqual(events(run("C F G C")), [])

    def test_komuro_and_fifties(self):
        self.assertEqual(events(run("C Am F G C", key="C")), [])
        self.assertEqual(events(run("Am F G C", key="C")), [])

    def test_I_to_IV_is_not_V_of_IV(self):
        self.assertEqual(events(run("C F C F", key="C"), "secondary_dominant"), [])

    def test_primary_V7_is_not_secondary(self):
        self.assertEqual(events(run("F G7 C", key="C"), "secondary_dominant"), [])

    def test_diatonic_iii_vi_is_not_V_of_vi(self):
        self.assertEqual(events(run("C Em Am F", key="C"), "secondary_dominant"), [])

    def test_diatonic_fifth_chain_has_one_ii_V_I_only(self):
        # Em–Am–Dm–G–C: only Dm–G–C is a ii–V–I (vi–ii–V and iii–vi–ii have a minor "V").
        res = run("C Em Am Dm G C", key="C")
        iv = events(res, "ii_V_I")
        self.assertEqual(len(iv), 1)
        self.assertEqual(iv[0].chords, ["Dm", "G", "C"])
        self.assertNoNotable(res)

    def test_minor_key_diatonic_bVI_bVII_i_not_borrowed(self):
        res = run("Am F G Am Dm E7 Am", key="A minor")
        self.assertEqual(events(res, "borrowed_chord"), [])
        self.assertEqual(events(res, "aeolian_cadence"), [])
        self.assertEqual(events(res, "secondary_dominant"), [])

    def test_bII_triad_is_not_tritone_sub(self):
        res = run("C Db C", key="C")
        self.assertEqual(events(res, "tritone_sub"), [])
        self.assertEqual(len(events(res, "borrowed_chord")), 1)


if __name__ == "__main__":
    unittest.main()
