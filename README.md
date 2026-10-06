# Harmonia

Local harmonic analysis for pop music (J-pop / City Pop): chord progression → local keys,
roman numerals, and annotated harmonic events (ii–V–I, secondary dominants, borrowed chords,
tritone substitutions, modulations), each with a confidence and the evidence behind it.

Status: Phase 1 (symbolic analysis layer) done; audio front end not yet built.
See [CLAUDE.md](CLAUDE.md) for architecture and conventions, [docs/schema.md](docs/schema.md)
for the JSON contract between the recognition and analysis layers.

```bash
python3 -m harmonia analyze "| Fmaj7 | E7 | Am7 | Gm7 C7 | Fmaj7 |" --key "C major"
python3 -m unittest discover -s tests -t .
```

Requires Python ≥ 3.11; the analysis layer uses only the standard library.
