"""Keys (tonic + mode) and their diatonic pitch-class sets."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .pitch import LETTER_PC, note_name

MAJOR_SCALE = (0, 2, 4, 5, 7, 9, 11)
NATURAL_MINOR = (0, 2, 3, 5, 7, 8, 10)
MODES = ("major", "minor")

_LETTERS = "CDEFGAB"
# degree (semitones above tonic) -> (letter steps above the tonic letter, accidental hint)
_DEGREE_SPELLING = {0: (0, 0), 1: (1, -1), 2: (1, 0), 3: (2, -1), 4: (2, 0), 5: (3, 0), 6: (3, 1),
                    7: (4, 0), 8: (5, -1), 9: (5, 0), 10: (6, -1), 11: (6, 0)}

# Conventional tonic spellings (pop / J-pop practice).
_MAJOR_TONIC_NAMES = ("C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")
_MINOR_TONIC_NAMES = ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "G#", "A", "Bb", "B")


@dataclass(frozen=True)
class Key:
    tonic: int
    mode: str  # "major" | "minor"

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        if not 0 <= self.tonic < 12:
            raise ValueError(f"tonic must be a pitch class 0..11, got {self.tonic}")

    @property
    def is_major(self) -> bool:
        return self.mode == "major"

    @property
    def tonic_name(self) -> str:
        return (_MAJOR_TONIC_NAMES if self.is_major else _MINOR_TONIC_NAMES)[self.tonic]

    @property
    def label(self) -> str:
        return f"{self.tonic_name} {self.mode}"

    @property
    def harte(self) -> str:
        return f"{self.tonic_name}:{'maj' if self.is_major else 'min'}"

    @property
    def prefers_flats(self) -> bool:
        name = self.tonic_name
        if "b" in name:
            return True
        return (self.is_major and self.tonic == 5) or (not self.is_major and self.tonic in (0, 2, 5, 7))

    @property
    def scale_pcs(self) -> tuple[int, ...]:
        steps = MAJOR_SCALE if self.is_major else NATURAL_MINOR
        return tuple((self.tonic + s) % 12 for s in steps)

    @property
    def diatonic_pcs(self) -> frozenset[int]:
        """Pitch classes treated as non-chromatic in this key.

        Minor = natural minor + raised 7th (harmonic-minor leading tone), because V / V7 /
        vii°7 are the normal dominant chords of a minor key and must not count as
        "chromatic". The raised 6th is NOT included (treated as chromatic / Dorian colour).
        """
        pcs = set(self.scale_pcs)
        if not self.is_major:
            pcs.add((self.tonic + 11) % 12)
        return frozenset(pcs)

    def degree(self, pc: int) -> int:
        """Semitones above the tonic."""
        return (pc - self.tonic) % 12

    def spell(self, pc: int) -> str:
        """Spell ``pc`` as a scale degree of the tonic's MAJOR scale, matching the numeral
        convention: chromatic degrees are ♭2 ♭3 ♯4 ♭6 ♭7 (A♭ not G♯ in C; C♯ / F♯ / G♯ in A minor)."""
        letter_idx = _LETTERS.index(self.tonic_name[0])
        step, _ = _DEGREE_SPELLING[self.degree(pc)]
        letter = _LETTERS[(letter_idx + step) % 7]
        diff = (pc - LETTER_PC[letter] + 6) % 12 - 6
        if abs(diff) > 2:
            return note_name(pc, self.prefers_flats)
        return letter + ("#" * diff if diff > 0 else "b" * -diff)

    @property
    def relative(self) -> "Key":
        if self.is_major:
            return Key((self.tonic + 9) % 12, "minor")
        return Key((self.tonic + 3) % 12, "major")

    def __str__(self) -> str:
        return self.label


ALL_KEYS: tuple[Key, ...] = tuple(Key(t, m) for m in MODES for t in range(12))
KEY_INDEX = {k: i for i, k in enumerate(ALL_KEYS)}

_KEY_RE = re.compile(
    r"^(?P<letter>[A-Ga-g])(?P<acc>[#b]*)\s*:?\s*(?P<mode>major|minor|maj|min|M|m)?$"
)


def parse_key(text: str) -> Key:
    """Parse "C major", "C", "Am", "A minor", "A:min", "F# minor", "Eb", "a" (lowercase = minor)."""
    s = text.strip().replace("♭", "b").replace("♯", "#")
    m = _KEY_RE.match(s)
    if not m:
        raise ValueError(f"cannot parse key {text!r}")
    letter = m.group("letter")
    acc = m.group("acc")
    tonic = (LETTER_PC[letter.upper()] + acc.count("#") - acc.count("b")) % 12
    mode_s = m.group("mode")
    if mode_s is None:
        mode = "minor" if letter.islower() else "major"
    else:
        mode = "major" if mode_s in ("major", "maj", "M") else "minor"
    return Key(tonic, mode)
