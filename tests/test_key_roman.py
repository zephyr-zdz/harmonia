import unittest

from harmonia.theory.chord import parse_chord
from harmonia.theory.key import Key, parse_key
from harmonia.theory.roman import roman

from .helpers import run


def rn(sym: str, key: str) -> str:
    return roman(parse_chord(sym), parse_key(key)).display


class TestKeyParsing(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(parse_key("C major"), Key(0, "major"))
        self.assertEqual(parse_key("Am"), Key(9, "minor"))
        self.assertEqual(parse_key("A:min"), Key(9, "minor"))
        self.assertEqual(parse_key("F# minor"), Key(6, "minor"))
        self.assertEqual(parse_key("Eb"), Key(3, "major"))
        self.assertEqual(parse_key("a"), Key(9, "minor"))

    def test_spelling_follows_major_scale_degrees(self):
        c = parse_key("C")
        self.assertEqual([c.spell(p) for p in (1, 3, 6, 8, 10)], ["Db", "Eb", "F#", "Ab", "Bb"])
        a = parse_key("Am")
        self.assertEqual([a.spell(p) for p in (0, 5, 7, 8)], ["C", "F", "G", "G#"])


class TestRomanNumerals(unittest.TestCase):
    def test_major(self):
        cases = {"Cmaj7": "Imaj7", "Dm7": "iim7", "Em": "iii", "F": "IV", "G7": "V7", "Am7": "vim7",
                 "Bm7b5": "viiø7", "E7": "III7", "Fm": "iv", "Ab": "♭VI", "Bb": "♭VII", "Eb": "♭III",
                 "Db7": "♭II7", "C#dim7": "♯i°7", "F#m7b5": "♯ivø7", "Gm7": "vm7", "C/E": "I/3", "G7/F": "V7/4"}
        for sym, exp in cases.items():
            with self.subTest(sym=sym):
                self.assertEqual(rn(sym, "C"), exp)

    def test_minor_uses_major_scale_reference(self):
        cases = {"Am": "i", "Bm7b5": "iiø7", "C": "♭III", "Dm": "iv", "Em": "v", "E7": "V7",
                 "F": "♭VI", "G": "♭VII", "G#dim7": "vii°7", "D": "IV"}
        for sym, exp in cases.items():
            with self.subTest(sym=sym):
                self.assertEqual(rn(sym, "Am"), exp)

    def test_diatonic_flag(self):
        k = parse_key("C")
        self.assertTrue(roman(parse_chord("Dm7"), k).diatonic)
        self.assertFalse(roman(parse_chord("D7"), k).diatonic)
        # harmonic-minor V7 counts as diatonic in minor
        self.assertTrue(roman(parse_chord("E7"), parse_key("Am")).diatonic)


class TestKeyEstimation(unittest.TestCase):
    def test_canon_is_c_major(self):
        res = run("C G Am Em F C F G C")
        self.assertEqual(res.global_key.key.label, "C major")
        self.assertFalse(res.global_key.ambiguous)

    def test_minor_with_harmonic_dominant(self):
        res = run("Am Dm E7 Am F Dm E7 Am")
        self.assertEqual(res.global_key.key.label, "A minor")

    def test_royal_road_in_song_context_is_major(self):
        # 王道進行 inside a passage that also cadences on I.
        res = run("C G Am Em F G C C F G Em Am F G C C")
        self.assertEqual(res.global_key.key.label, "C major")

    def test_isolated_royal_road_loop_is_flagged_ambiguous(self):
        # A bare IV–V–iii–vi loop never states the tonic: C major vs A minor is a genuine
        # ambiguity and must be reported, not hidden.
        res = run("F G Em Am F G Em Am")
        self.assertTrue(res.global_key.ambiguous)
        top2 = {res.global_key.key.label, res.global_key.alternatives[0].label}
        self.assertEqual(top2, {"C major", "A minor"})
        self.assertTrue(all(s.low_confidence for s in res.segments))

    def test_tonicization_is_not_modulation(self):
        res = run("C E7 Am Dm G7 C A7 Dm G7 C")
        self.assertEqual(len(res.key_regions), 1)
        self.assertEqual(res.key_regions[0].key.label, "C major")

    def test_given_key_overrides(self):
        res = run("F G Em Am", key="C major")
        self.assertEqual(res.global_key.source, "given")
        self.assertTrue(all(s.key.label == "C major" for s in res.segments))


if __name__ == "__main__":
    unittest.main()
