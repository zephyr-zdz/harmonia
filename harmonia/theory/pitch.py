"""Pitch classes and note-name spelling."""

from __future__ import annotations

LETTER_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
ACCIDENTALS = {"#": 1, "♯": 1, "b": -1, "♭": -1}
SHARP_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
FLAT_NAMES = ("C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B")


class NoteParseError(ValueError):
    pass


def parse_note_prefix(text: str) -> tuple[int, str, int]:
    """Parse a note name (uppercase letter + accidentals) at the start of ``text``.

    Returns ``(pitch_class, ascii_name, n_chars_consumed)``. Accidentals are consumed
    greedily, so ``"Bbm7"`` -> B-flat.
    """
    if not text or text[0] not in LETTER_PC:
        raise NoteParseError(f"expected a note name at the start of {text!r}")
    shift = 0
    name = text[0]
    i = 1
    while i < len(text) and text[i] in ACCIDENTALS:
        step = ACCIDENTALS[text[i]]
        shift += step
        name += "#" if step > 0 else "b"
        i += 1
    return (LETTER_PC[text[0]] + shift) % 12, name, i


def parse_note(text: str) -> tuple[int, str]:
    text = text.strip()
    pc, name, n = parse_note_prefix(text)
    if n != len(text):
        raise NoteParseError(f"trailing characters in note name {text!r}")
    return pc, name


def note_name(pc: int, prefer_flats: bool = False) -> str:
    return (FLAT_NAMES if prefer_flats else SHARP_NAMES)[pc % 12]
