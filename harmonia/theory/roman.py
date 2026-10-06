"""Roman-numeral labelling of a chord relative to a key.

Convention (see CLAUDE.md): numerals are relative to the MAJOR scale of the tonic in both
modes, with explicit accidentals (Berklee / Japanese 度数 style), e.g. in A minor:
i, iiø7, ♭III, iv, v, V7, ♭VI, ♭VII. Case shows chord quality (lower = minor/diminished).
Suffixes follow pop usage: ``m7`` for minor sevenths (``vim7``), ``maj7``, ``7``, ``ø7``,
``°7``. Inversions are written with an arabic scale degree of the bass: ``I/3``, ``V7/4``.
"""

from __future__ import annotations

from dataclasses import dataclass

from .chord import Chord
from .key import Key

_UPPER = {0: "I", 1: "♭II", 2: "II", 3: "♭III", 4: "III", 5: "IV",
          7: "V", 8: "♭VI", 9: "VI", 10: "♭VII", 11: "VII"}
SCALE_DEGREE_NAMES = {0: "1", 1: "♭2", 2: "2", 3: "♭3", 4: "3", 5: "4", 6: "♯4",
                      7: "5", 8: "♭6", 9: "6", 10: "♭7", 11: "7"}
_LOWER_QCLASSES = {"min", "dim", "hdim"}
_SHARP_DIM = {1: "♯i", 3: "♯ii", 8: "♯v", 10: "♯vi"}

NUMERAL_SUFFIX = {
    "maj": "", "min": "", "dim": "°", "aug": "+", "sus2": "sus2", "sus4": "sus4", "5": "5",
    "7": "7", "maj7": "maj7", "min7": "m7", "hdim7": "ø7", "dim7": "°7", "minmaj7": "mM7",
    "aug7": "+7", "augmaj7": "+maj7", "7sus4": "7sus4", "maj6": "6", "min6": "m6",
}

# Diatonic triad class on each scale degree (semitones above tonic). Minor uses the natural
# minor scale (v is minor); the harmonic-minor V is handled as diatonic by Key.diatonic_pcs.
DIATONIC_TRIAD_CLASS = {
    "major": {0: "maj", 2: "min", 4: "min", 5: "maj", 7: "maj", 9: "min", 11: "dim"},
    "minor": {0: "min", 2: "dim", 3: "maj", 5: "min", 7: "min", 8: "maj", 10: "maj"},
}


def base_numeral(degree: int, qclass: str | None) -> str:
    """Numeral without suffix, e.g. (9, "min") -> "vi", (8, "maj") -> "♭VI"."""
    lower = qclass in _LOWER_QCLASSES
    if degree == 6:  # tritone above tonic: ♯iv° / ♯ivø7 when diminished-ish, else ♭V
        return "♯iv" if lower else "♭V"
    if qclass in ("dim", "hdim") and degree in _SHARP_DIM:
        # chromatic diminished chords are leading-tone chords: spell as raised degree
        return _SHARP_DIM[degree]
    up = _UPPER[degree]
    return up.lower() if lower else up


@dataclass(frozen=True)
class RomanNumeral:
    degree: int            # root, semitones above the tonic
    base: str              # e.g. "♭VI", "vi"
    suffix: str            # e.g. "maj7", "m7", "ø7"
    bass_degree: str | None  # scale degree of a non-root bass, e.g. "3", "♭7"
    diatonic: bool         # all core chord tones in Key.diatonic_pcs

    @property
    def numeral(self) -> str:
        return self.base + self.suffix

    @property
    def display(self) -> str:
        return self.numeral + (f"/{self.bass_degree}" if self.bass_degree else "")


def roman(chord: Chord, key: Key) -> RomanNumeral | None:
    if chord.root is None:
        return None
    deg = key.degree(chord.root)
    bass_deg = None
    if chord.bass is not None and chord.bass != chord.root:
        bass_deg = SCALE_DEGREE_NAMES[key.degree(chord.bass)]
    return RomanNumeral(
        degree=deg,
        base=base_numeral(deg, chord.qclass),
        suffix=NUMERAL_SUFFIX[chord.quality],
        bass_degree=bass_deg,
        diatonic=chord.core_pcs() <= key.diatonic_pcs,
    )


def diatonic_triad_class(key: Key, degree: int) -> str | None:
    return DIATONIC_TRIAD_CLASS[key.mode].get(degree)


def target_numeral(key: Key, target_root: int, target_qclass: str | None = None) -> str:
    """Numeral naming a tonicised target (the part after the slash in "V7/vi").

    Uses the actual target chord's quality when known, otherwise the diatonic triad on
    that degree (major if the degree is not diatonic).
    """
    deg = key.degree(target_root)
    qc = target_qclass or diatonic_triad_class(key, deg) or "maj"
    if qc not in ("maj", "min", "dim", "hdim"):
        qc = "maj"
    return base_numeral(deg, qc)


def applied_label(chord: Chord, target_root: int, target_mode: str, target_num: str) -> str:
    """Applied-chord label such as "V7/vi" or "iim7/IV": the chord's numeral relative to the
    temporary key of its target, slash, the target's numeral in the home key."""
    rn = roman(chord, Key(target_root, target_mode))
    assert rn is not None
    return f"{rn.base}{rn.suffix}/{target_num}"
