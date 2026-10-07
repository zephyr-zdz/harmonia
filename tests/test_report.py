"""Section report (harmonia/report.py) on a symbolic chart with [Section] markers."""

import unittest

from harmonia.analysis import analyze
from harmonia.io import parse_progression
from harmonia.report import summarise, to_markdown


class TestSectionReport(unittest.TestCase):
    def test_loops_keys_and_progressions_per_section(self):
        rec = parse_progression("[Verse] | C | F | C | G | C | F | C | G | "
                                "[Chorus] | F | G | Em | Am | F | G | Em | Am |")
        res = analyze(rec)
        secs = summarise(rec, res)
        self.assertEqual([s.label for s in secs], ["Verse", "Chorus"])
        self.assertEqual(secs[0].loop, ["I", "IV", "I", "V"])
        self.assertEqual(secs[1].loop, ["IV", "V", "iii", "vi"])
        self.assertEqual(secs[1].progressions, ["王道進行"])
        self.assertEqual(secs[1].keys[0][0], "C major")
        md = to_markdown("demo", rec, res, secs)
        self.assertIn("‖ IV │ V │ iii │ vi ‖", md)   # no raw '|' inside table cells

    def test_no_loop_when_bars_do_not_repeat(self):
        rec = parse_progression("[A] | C | Dm | Em | F | G | Am | Bdim | C |")
        secs = summarise(rec, analyze(rec))
        self.assertEqual(secs[0].loop, [])
        self.assertEqual(len(secs[0].bars), 8)


if __name__ == "__main__":
    unittest.main()
