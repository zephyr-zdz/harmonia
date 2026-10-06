"""Chord symbols: parsing (pop / Japanese chord-sheet style and Harte syntax) and a
small immutable chord model.

Conventions (see CLAUDE.md):
- The canonical machine label is Harte syntax (``C:min7``, ``Bb:7(b9)``, ``C:maj/3``) so
  the evaluation layer can hand labels to mir_eval unchanged.
- Pop input accepts m/min/mi/-, M/maj/ma/Δ, 7, maj7/M7, m7b5/m7-5/ø, dim/°/o, aug/+,
  sus2/sus4, 6/69, add9, tensions with or without parentheses (``7(b9,13)``, ``7-9``),
  slash bass ``C/E`` and the Japanese on-chord form ``C(onE)`` / ``ConE``.
- In Japanese chord sheets ``-5`` means ♭5 (``Cm7-5``) and ``-9`` means ♭9 (``C7-9``).
  A ``-`` directly after the root means minor (``C-7``) unless it is followed by ``5``.
- Parsing never guesses: unrecognised text raises ChordParseError.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .pitch import NoteParseError, note_name, parse_note_prefix


class ChordParseError(ValueError):
    """A chord symbol could not be parsed. Callers must surface this, never swallow it."""


# Core chord tones, in semitones above the root.
QUALITY_INTERVALS: dict[str, tuple[int, ...]] = {
    "maj": (0, 4, 7),
    "min": (0, 3, 7),
    "dim": (0, 3, 6),
    "aug": (0, 4, 8),
    "sus2": (0, 2, 7),
    "sus4": (0, 5, 7),
    "5": (0, 7),
    "7": (0, 4, 7, 10),
    "maj7": (0, 4, 7, 11),
    "min7": (0, 3, 7, 10),
    "hdim7": (0, 3, 6, 10),
    "dim7": (0, 3, 6, 9),
    "minmaj7": (0, 3, 7, 11),
    "aug7": (0, 4, 8, 10),
    "augmaj7": (0, 4, 8, 11),
    "7sus4": (0, 5, 7, 10),
    "maj6": (0, 4, 7, 9),
    "min6": (0, 3, 7, 9),
}

# Coarse quality classes used by the key model and the rules.
#   maj: major third, no minor seventh     min: minor third, perfect fifth
#   dom: major third + minor seventh       dim / hdim: diminished fifth
#   aug: augmented fifth, no ♭7            sus: no third                pow: root+fifth only
QUALITY_CLASS: dict[str, str] = {
    "maj": "maj", "maj7": "maj", "maj6": "maj",
    "min": "min", "min7": "min", "min6": "min", "minmaj7": "min",
    "7": "dom", "aug7": "dom",
    "dim": "dim", "dim7": "dim",
    "hdim7": "hdim",
    "aug": "aug", "augmaj7": "aug",
    "sus2": "sus", "sus4": "sus", "7sus4": "sus",
    "5": "pow",
}

SEVENTH_QUALITIES = frozenset({"7", "maj7", "min7", "hdim7", "dim7", "minmaj7", "aug7", "augmaj7", "7sus4"})

# The added tone (7th / 6th) that the coarse class "maj"/"min" does not encode.
ADDED_TONE: dict[str, int] = {"maj7": 11, "maj6": 9, "min7": 10, "min6": 9, "minmaj7": 11}

EXTENSION_INTERVALS: dict[str, int] = {
    "b5": 6, "#5": 8, "b9": 1, "9": 2, "#9": 3, "11": 5, "#11": 6, "b13": 8, "13": 9,
}
_EXT_ORDER = list(EXTENSION_INTERVALS)

POP_SUFFIX: dict[str, str] = {
    "maj": "", "min": "m", "dim": "dim", "aug": "aug", "sus2": "sus2", "sus4": "sus4",
    "5": "5", "7": "7", "maj7": "maj7", "min7": "m7", "hdim7": "m7b5", "dim7": "dim7",
    "minmaj7": "mM7", "aug7": "aug7", "augmaj7": "augmaj7", "7sus4": "7sus4",
    "maj6": "6", "min6": "m6",
}

HARTE_SHORTHAND: dict[str, str] = {
    "maj": "maj", "min": "min", "dim": "dim", "aug": "aug", "sus2": "sus2", "sus4": "sus4",
    "5": "5", "7": "7", "maj7": "maj7", "min7": "min7", "hdim7": "hdim7", "dim7": "dim7",
    "minmaj7": "minmaj7", "aug7": "aug(b7)", "augmaj7": "aug(7)", "7sus4": "sus4(b7)",
    "maj6": "maj6", "min6": "min6",
}

_SEMITONE_TO_DEGREE = {0: "1", 1: "b2", 2: "2", 3: "b3", 4: "3", 5: "4", 6: "b5",
                       7: "5", 8: "b6", 9: "6", 10: "b7", 11: "7"}
_DEGREE_BASE = {1: 0, 2: 2, 3: 4, 4: 5, 5: 7, 6: 9, 7: 11, 9: 2, 11: 5, 13: 9}


@dataclass(frozen=True)
class Chord:
    """A parsed chord. ``root``/``bass`` are pitch classes; ``quality`` is a key of
    QUALITY_INTERVALS, or "N" (no chord) / "X" (unknown)."""

    root: int | None
    quality: str
    extensions: tuple[str, ...] = ()
    bass: int | None = None
    root_name: str | None = None
    bass_name: str | None = None

    @property
    def is_no_chord(self) -> bool:
        return self.quality == "N"

    @property
    def is_unknown(self) -> bool:
        return self.quality == "X"

    @property
    def qclass(self) -> str | None:
        return QUALITY_CLASS.get(self.quality)

    @property
    def has_seventh(self) -> bool:
        return self.quality in SEVENTH_QUALITIES

    @property
    def intervals(self) -> tuple[int, ...]:
        ivs = list(QUALITY_INTERVALS.get(self.quality, ()))
        if 7 in ivs and "b5" in self.extensions:
            ivs[ivs.index(7)] = 6
        elif 7 in ivs and "#5" in self.extensions:
            ivs[ivs.index(7)] = 8
        return tuple(ivs)

    def core_pcs(self) -> frozenset[int]:
        if self.root is None:
            return frozenset()
        return frozenset((self.root + i) % 12 for i in self.intervals)

    def pitch_classes(self) -> frozenset[int]:
        if self.root is None:
            return frozenset()
        ext = {(self.root + EXTENSION_INTERVALS[e]) % 12 for e in self.extensions
               if e not in ("b5", "#5")}
        return self.core_pcs() | ext

    @property
    def bass_pc(self) -> int | None:
        return self.bass if self.bass is not None else self.root

    def _root_str(self) -> str:
        return self.root_name or note_name(self.root)

    def label(self) -> str:
        """Pop-style label, e.g. ``Bbm7b5``, ``C7(b9,13)/E``."""
        if self.root is None:
            return self.quality
        s = self._root_str() + POP_SUFFIX[self.quality]
        if self.extensions:
            s += "(" + ",".join(self.extensions) + ")"
        if self.bass is not None and self.bass != self.root:
            s += "/" + (self.bass_name or note_name(self.bass))
        return s

    def harte(self) -> str:
        """Harte-syntax label, e.g. ``Bb:hdim7``, ``C:7(b9)/3``."""
        if self.root is None:
            return self.quality
        short = HARTE_SHORTHAND[self.quality]
        items: list[str] = []
        if "(" in short:
            short, inner = short[:-1].split("(")
            items.extend(inner.split(","))
        items.extend(self.extensions)
        s = f"{self._root_str()}:{short}"
        if items:
            s += "(" + ",".join(items) + ")"
        if self.bass is not None and self.bass != self.root:
            s += "/" + _SEMITONE_TO_DEGREE[(self.bass - self.root) % 12]
        return s

    def __str__(self) -> str:
        return self.label()


NO_CHORD = Chord(None, "N")
UNKNOWN_CHORD = Chord(None, "X")

_NO_CHORD_LABELS = {"N", "N.C.", "N.C", "NC", "n.c."}

# --------------------------------------------------------------------------- parsing

_UNICODE_REPLACEMENTS = [
    ("♭", "b"), ("♯", "#"), ("−", "-"), ("–", "-"),
    ("Δ7", "maj7"), ("Δ9", "maj9"), ("Δ", "maj7"),
    ("ø7", "m7b5"), ("ø", "m7b5"), ("°7", "dim7"), ("°", "dim"),
    ("6/9", "69"),
]
_ON_BASS_RE = re.compile(r"^(?P<body>.+?)\(?on(?P<bass>[A-G][#b]*)\)?$")
_NOTE_RE = re.compile(r"[A-G][#b]*")
_DEGREE_RE = re.compile(r"[#b]*\d+")
_NUMBER_RE = re.compile(r"69|13|11|9|7|6|5")
_REST_TOKEN_RE = re.compile(r"sus2|sus4|sus|add[#b]?\d+|alt|[#b+\-]?\d+|[(),]")
_HARTE_RE = re.compile(
    r"^(?P<root>[A-G][#b]*):(?P<short>[a-z0-9]*)(?:\((?P<extra>[^)]*)\))?(?:/(?P<bass>[#b]*\d+))?$"
)

_MINMAJ_PREFIXES = ("mMaj7", "mmaj7", "mM7", "m(maj7)", "m(Maj7)", "m(M7)", "minmaj7", "-maj7", "-M7")

# Base quality from (family, leading number) in pop symbols.
_FAMILY_NUMBER: dict[str, dict[str | None, tuple[str, tuple[str, ...]]]] = {
    "plain": {None: ("maj", ()), "7": ("7", ()), "9": ("7", ("9",)), "11": ("7", ("9", "11")),
              "13": ("7", ("9", "13")), "6": ("maj6", ()), "69": ("maj6", ("9",)), "5": ("5", ())},
    "min": {None: ("min", ()), "7": ("min7", ()), "9": ("min7", ("9",)), "11": ("min7", ("9", "11")),
            "13": ("min7", ("9", "13")), "6": ("min6", ()), "69": ("min6", ("9",))},
    "majmark": {None: ("maj", ()), "7": ("maj7", ()), "9": ("maj7", ("9",)),
                "11": ("maj7", ("9", "11")), "13": ("maj7", ("9", "13")), "6": ("maj6", ())},
    "aug": {None: ("aug", ()), "5": ("aug", ()), "7": ("aug7", ()), "9": ("aug7", ("9",))},
}

_HARTE_SHORT: dict[str, tuple[str, tuple[str, ...]]] = {
    "maj": ("maj", ()), "min": ("min", ()), "dim": ("dim", ()), "aug": ("aug", ()),
    "maj7": ("maj7", ()), "min7": ("min7", ()), "7": ("7", ()), "dim7": ("dim7", ()),
    "hdim7": ("hdim7", ()), "minmaj7": ("minmaj7", ()), "maj6": ("maj6", ()), "min6": ("min6", ()),
    "9": ("7", ("9",)), "maj9": ("maj7", ("9",)), "min9": ("min7", ("9",)),
    "11": ("7", ("9", "11")), "min11": ("min7", ("9", "11")),
    "13": ("7", ("9", "13")), "maj13": ("maj7", ("9", "13")), "min13": ("min7", ("9", "13")),
    "sus2": ("sus2", ()), "sus4": ("sus4", ()), "5": ("5", ()), "1": ("5", ()),
}


def _degree_to_semitones(deg: str, label: str) -> int:
    m = re.fullmatch(r"([#b]*)(\d+)", deg)
    if not m or int(m.group(2)) not in _DEGREE_BASE:
        raise ChordParseError(f"bad interval degree {deg!r} in {label!r}")
    shift = m.group(1).count("#") - m.group(1).count("b")
    return (_DEGREE_BASE[int(m.group(2))] + shift) % 12


def _apply_token(quality: str, ext: list[str], tok: str, label: str) -> str:
    """Apply one added-degree / alteration token (``b5``, ``#9``, ``6``, ``b7`` ...)."""
    if tok == "b5":
        if quality == "min7":
            return "hdim7"
        if quality == "min":
            return "dim"
        if quality in ("maj", "7"):
            ext.append("b5")
            return quality
    elif tok == "#5":
        upgrade = {"7": "aug7", "maj": "aug", "maj7": "augmaj7", "aug": "aug", "aug7": "aug7"}
        if quality in upgrade:
            return upgrade[quality]
    elif tok in ("9", "b9", "#9", "11", "#11", "13", "b13"):
        if quality != "5":
            ext.append(tok)
            return quality
    elif tok == "2":
        if quality in ("maj", "min", "7", "maj7", "min7", "maj6", "min6"):
            ext.append("9")
            return quality
    elif tok == "4":
        if quality in ("maj", "min", "7", "min7"):
            ext.append("11")
            return quality
    elif tok == "6":
        upgrade = {"maj": "maj6", "min": "min6"}
        if quality in upgrade:
            return upgrade[quality]
        if quality in ("7", "min7", "maj7"):
            ext.append("13")
            return quality
    elif tok == "b7":
        upgrade = {"maj": "7", "min": "min7", "sus4": "7sus4", "aug": "aug7", "dim": "hdim7"}
        if quality in upgrade:
            return upgrade[quality]
    elif tok == "7":
        upgrade = {"maj": "maj7", "min": "minmaj7", "aug": "augmaj7", "sus4": "7sus4"}
        if quality in upgrade:
            return upgrade[quality]
    elif tok == "bb7":
        if quality == "dim":
            return "dim7"
    elif tok in ("1", "3", "5", "b3"):
        semis = {"1": 0, "3": 4, "5": 7, "b3": 3}[tok]
        if semis in QUALITY_INTERVALS[quality]:
            return quality  # redundant restatement of a chord tone
    raise ChordParseError(f"cannot apply {tok!r} to quality {quality!r} in {label!r}")


def _normalize_ext(ext: list[str]) -> tuple[str, ...]:
    return tuple(e for e in _EXT_ORDER if e in set(ext))


def _parse_pop_quality(q: str, label: str) -> tuple[str, tuple[str, ...]]:
    pos = 0
    base: str | None = None
    ext: list[str] = []
    family = "plain"

    def take(options: tuple[str, ...]) -> str | None:
        for o in options:
            if q.startswith(o, pos):
                return o
        return None

    if tok := take(("dim7", "o7")):
        base = "dim7"
    elif tok := take(("dim", "o")):
        base = "dim"
    elif tok := take(("aug", "+")):
        family = "aug"
    elif tok := take(_MINMAJ_PREFIXES):
        base = "minmaj7"
    elif tok := take(("maj", "Maj", "ma", "M")):
        family = "majmark"
    elif tok := take(("min", "mi", "m")):
        family = "min"
    elif q.startswith("-") and not q.startswith("-5"):
        tok = "-"
        family = "min"
    pos += len(tok or "")

    if base is None:
        m = _NUMBER_RE.match(q, pos)
        num = m.group(0) if m else None
        table = _FAMILY_NUMBER[family]
        if num not in table:
            raise ChordParseError(f"unsupported chord quality {q!r} in {label!r}")
        base, init_ext = table[num]
        ext.extend(init_ext)
        pos += len(num or "")

    rest = q[pos:]
    tokens = _REST_TOKEN_RE.findall(rest)
    if "".join(tokens) != rest:
        raise ChordParseError(f"unrecognised text {rest!r} in chord {label!r}")
    for t in tokens:
        if t in "(),":
            continue
        if t in ("sus4", "sus"):
            if base == "maj":
                base = "sus4"
            elif base == "7":
                base = "7sus4"
            else:
                raise ChordParseError(f"sus4 on {base!r} unsupported in {label!r}")
        elif t == "sus2":
            if base != "maj":
                raise ChordParseError(f"sus2 on {base!r} unsupported in {label!r}")
            base = "sus2"
        elif t.startswith("add"):
            deg = t[3:]
            deg = {"2": "9", "4": "11"}.get(deg, deg)
            base = _apply_token(base, ext, deg, label)
        elif t == "alt":
            if base != "7":
                raise ChordParseError(f"'alt' requires a dominant seventh in {label!r}")
            ext.extend(["b9", "#9", "#11", "b13"])
        else:
            sign, num = (t[0], t[1:]) if t[0] in "#b+-" else ("", t)
            acc = {"#": "#", "+": "#", "b": "b", "-": "b", "": ""}[sign]
            base = _apply_token(base, ext, acc + num, label)
    return base, _normalize_ext(ext)


def _parse_pop(label: str) -> Chord:
    s = label
    for a, b in _UNICODE_REPLACEMENTS:
        s = s.replace(a, b)
    s = s.replace(" ", "")

    bass_pc: int | None = None
    bass_name: str | None = None
    body = s
    m = _ON_BASS_RE.match(s)
    if m and m.group("body") and _NOTE_RE.match(m.group("body")):
        body = m.group("body")
        bass_pc, bass_name, _ = parse_note_prefix(m.group("bass"))
    elif "/" in s:
        body, rhs = s.rsplit("/", 1)
        if _NOTE_RE.fullmatch(rhs):
            bass_pc, bass_name, _ = parse_note_prefix(rhs)
        elif not _DEGREE_RE.fullmatch(rhs):
            raise ChordParseError(f"bad bass {rhs!r} in {label!r}")
        else:
            bass_name = rhs  # resolved to a pitch class after the root is known

    try:
        root, root_name, n = parse_note_prefix(body)
    except NoteParseError as e:
        raise ChordParseError(f"cannot parse root of {label!r}: {e}") from None
    quality, ext = _parse_pop_quality(body[n:], label)
    if bass_name is not None and bass_pc is None:
        bass_pc = (root + _degree_to_semitones(bass_name, label)) % 12
        bass_name = None
    if bass_pc == root:
        bass_pc, bass_name = None, None
    return Chord(root, quality, ext, bass_pc, root_name, bass_name)


def _parse_harte(label: str) -> Chord:
    m = _HARTE_RE.match(label)
    if not m:
        raise ChordParseError(f"malformed Harte chord {label!r}")
    root, root_name, _ = parse_note_prefix(m.group("root"))
    short, extra = m.group("short"), m.group("extra")
    ext: list[str] = []
    if short:
        if short not in _HARTE_SHORT:
            raise ChordParseError(f"unknown Harte shorthand {short!r} in {label!r}")
        quality, init = _HARTE_SHORT[short]
        ext.extend(init)
    else:
        if extra is None:
            raise ChordParseError(f"Harte chord without quality {label!r}")
        degrees = [d.strip() for d in extra.split(",") if d.strip() and not d.strip().startswith("*")]
        semis = {0} | {_degree_to_semitones(d, label) for d in degrees}
        matches = [q for q, ivs in QUALITY_INTERVALS.items() if set(ivs) == semis]
        if not matches:
            raise ChordParseError(f"unsupported Harte interval set {extra!r} in {label!r}")
        quality, extra = matches[0], None
    if extra:
        for item in (x.strip() for x in extra.split(",")):
            if not item:
                continue
            if item.startswith("*"):
                if item == "*3" and quality in ("maj", "min"):
                    quality = "5"
                continue  # other omissions do not change the coarse quality
            quality = _apply_token(quality, ext, item, label)
    bass_pc = None
    if m.group("bass"):
        bass_pc = (root + _degree_to_semitones(m.group("bass"), label)) % 12
        if bass_pc == root:
            bass_pc = None
    return Chord(root, quality, _normalize_ext(ext), bass_pc, root_name, None)


def parse_chord(label: str) -> Chord:
    """Parse a chord symbol (pop or Harte). Raises ChordParseError on failure."""
    s = label.strip()
    if not s:
        raise ChordParseError("empty chord label")
    if s in _NO_CHORD_LABELS:
        return NO_CHORD
    if s == "X":
        return UNKNOWN_CHORD
    if ":" in s:
        return _parse_harte(s)
    return _parse_pop(s)
