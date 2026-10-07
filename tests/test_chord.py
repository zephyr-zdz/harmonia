import unittest

from harmonia.theory.chord import ChordParseError, parse_chord


class TestChordParsing(unittest.TestCase):
    CASES = {
        # pop / jazz
        "C": "C:maj", "Cm": "C:min", "C7": "C:7", "Cmaj7": "C:maj7", "CM7": "C:maj7", "CΔ": "C:maj7",
        "Cm7": "C:min7", "C-7": "C:min7", "Cmin7": "C:min7", "Cmi7": "C:min7",
        "Cm7b5": "C:hdim7", "Cø": "C:hdim7", "Cø7": "C:hdim7",
        "Cdim": "C:dim", "C°": "C:dim", "Cdim7": "C:dim7", "C°7": "C:dim7",
        "Caug": "C:aug", "C+": "C:aug", "C+7": "C:aug(b7)", "C7#5": "C:aug(b7)",
        "Csus4": "C:sus4", "Csus": "C:sus4", "Csus2": "C:sus2", "C7sus4": "C:sus4(b7)",
        "C6": "C:maj6", "Cm6": "C:min6", "C69": "C:maj6(9)", "C6/9": "C:maj6(9)",
        "C9": "C:7(9)", "Cmaj9": "C:maj7(9)", "Cm9": "C:min7(9)", "C13": "C:7(9,13)",
        "C7(b9)": "C:7(b9)", "C7(b9,13)": "C:7(b9,13)", "C7#9": "C:7(#9)",
        "Cadd9": "C:maj(9)", "CmM7": "C:minmaj7", "Cm(maj7)": "C:minmaj7", "C5": "C:5",
        "Bbm7b5": "Bb:hdim7", "F#7": "F#:7", "B♭maj7": "Bb:maj7", "E♭m": "Eb:min",
        # Japanese chord-sheet conventions
        "Cm7-5": "C:hdim7", "C7-9": "C:7(b9)", "C7+9": "C:7(#9)", "C7+5": "C:aug(b7)", "C(9)": "C:maj(9)",
        # slash / on-chord bass
        "C/E": "C:maj/3", "C/Bb": "C:maj/b7", "C(onE)": "C:maj/3", "ConE": "C:maj/3",
        "Am7onD": "A:min7/4", "D/F#": "D:maj/3", "C/C": "C:maj",
        # Harte
        "C:maj/3": "C:maj/3", "A:min7": "A:min7", "B:hdim7": "B:hdim7", "C:sus4(b7)": "C:sus4(b7)",
        "C:(1,b3,5)": "C:min", "C:7(b9)": "C:7(b9)", "C:9": "C:7(9)", "G:maj(*3)": "G:5",
        # Harte-style extensions used by IdolSongsJp; redundant chord tones are tolerated
        "Db:7sus4": "Db:sus4(b7)", "Eb:aug7": "Eb:aug(b7)", "Bb:maj9(7)/2": "Bb:maj7(9)/2",
        # no-chord
        "N": "N", "N.C.": "N", "X": "X",
    }

    def test_cases(self):
        for sym, harte in self.CASES.items():
            with self.subTest(sym=sym):
                self.assertEqual(parse_chord(sym).harte(), harte)

    def test_roundtrip_harte(self):
        for sym in self.CASES:
            h = parse_chord(sym).harte()
            with self.subTest(sym=sym):
                self.assertEqual(parse_chord(h).harte(), h)

    def test_quality_classes(self):
        self.assertEqual(parse_chord("G7").qclass, "dom")
        self.assertEqual(parse_chord("Fmaj7").qclass, "maj")
        self.assertEqual(parse_chord("Dm7").qclass, "min")
        self.assertEqual(parse_chord("Bm7b5").qclass, "hdim")
        self.assertEqual(parse_chord("G7sus4").qclass, "sus")

    def test_pitch_classes(self):
        self.assertEqual(parse_chord("G7").core_pcs(), {7, 11, 2, 5})
        self.assertEqual(parse_chord("C7(b5)").core_pcs(), {0, 4, 6, 10})
        self.assertEqual(parse_chord("Cadd9").pitch_classes(), {0, 2, 4, 7})

    def test_errors_are_raised_not_swallowed(self):
        for bad in ["", "H7", "Cfoo", "Cm7sus4", "C/Q", "C:weird"]:
            with self.subTest(bad=bad), self.assertRaises(ChordParseError):
                parse_chord(bad)


if __name__ == "__main__":
    unittest.main()
