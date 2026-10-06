# Evaluation set (Phase 2)

```
data/eval/
  dev/    <song_id>/audio.(mp3|m4a|flac)   — used for development and any calibration
          <song_id>/chords.lab              — time-aligned Harte labels (start end label), or
          <song_id>/chords.txt              — bar-wise chord chart (aligned by edit distance)
          <song_id>/events.json             — optional hand-labelled harmonic events
          <song_id>/meta.toml               — title, artist, source, key, license / provenance
  test/   same layout — NEVER used for tuning; only for final reported numbers
```

Rules: raw audio and annotations here are immutable inputs; derived outputs go to
`outputs/` (not here). Do not tune anything on `test/`.
