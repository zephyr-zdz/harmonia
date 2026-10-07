# Song structure (verse / chorus / bridge): options

User request 2026-10-07: identify sections. Decision: implement **A** (own, no new
dependencies) now; feasibility study for **B** and **C** below. Facts checked 2026-10-07
(PyPI JSON, GitHub, HuggingFace pages listed at the end).

## A — own repetition-based analysis (implemented: `harmonia/frontend/structure.py`)

Bar grid from `meter.py` → chord-identity repetition (root × maj/min posteriors, one
transposition per section pair) + timbre/loudness novelty → DP segmentation with a 4/8/16-bar
phrase prior → clustering (±2 bar slack) → J-pop rules (Aメロ / Bメロ / サビ / Cメロ).

Measured (IdolSongsJp dev, no structure ground truth exists there): bars paired as "the same
section" carry the same reference chord (root, maj/min) 85 % of the time; 66 % of bars fall in
repeated groups. Labels are heuristic and **unscored** until annotated sections exist.

## B — all-in-one (`allin1`, Kim & Nam, WASPAA 2023)

| | |
|---|---|
| Output | beats, downbeats, segments: intro / verse / chorus / bridge / inst / solo / break / outro (no pre-chorus) |
| Training | Harmonix Set (912 Western pop songs), 8-fold ensemble |
| License | MIT (code + weights) |
| Package | `allin1` 1.1.0, **last release 2023-10-10**, pure-Python wheel |
| Dependencies | torch, demucs (have), **natten**, **madmom** (git master), hydra-core, omegaconf |
| natten | macOS arm64: **source only** (no wheels); current 0.21.x targets NVIDIA kernels; allin1 is written against the old API (open issue #30: incompatible with NATTEN 0.17.5; #11: fails to install on macOS; #17: asks for an MPS alternative) |
| madmom | last PyPI release 2017, sdist only (Cython build); needs git master; Python 3.14 / numpy 2 support unverified |
| Maintenance | open issues from 2024–2026 without maintainer answers |
| Reported accuracy | SongFormBench-CN (Chinese pop) 0.834 (from the SongFormer paper) |

**Feasibility on this Mac (arm64, Python 3.14, torch 2.14): not installable as is.** Paths:
1. Separate venv with an old stack (Python ≤ 3.11, torch ≈ 2.1, natten ≤ 0.15 compiled from
   source for CPU) — high risk of build failures on arm64, frozen stack.
2. Re-implement the 1-D/2-D dilated neighbourhood attention in plain PyTorch (inference only,
   ~100–200 lines) and load the MIT weights; drop madmom (use our Beat This! grid). Medium
   effort (≈ 1–2 days), moderate risk (numerical parity must be verified against reference
   outputs, which we cannot produce without a working natten — would need a Linux/CUDA box,
   e.g. the user's RTX 6000).
Domain: Western-pop training; no pre-chorus label (J-pop Bメロ would become verse or chorus).

## C — SongFormer (ASLP-lab, 2025, arXiv 2510.02797)

| | |
|---|---|
| Output | intro / verse / **pre-chorus** / chorus / bridge / inst / outro / silence |
| Training | SongFormDB (large multilingual, incl. Chinese pop) + heterogeneous supervision |
| Backbones | MuQ (~0.3 B params) + MusicFM, each at 30 s and 420 s windows |
| Weights | `SongFormer.safetensors` 104 MB + `model.safetensors` **2.76 GB** (≈ 3 GB total, HuggingFace ASLP-lab/SongFormer) |
| Licenses | repository CC-BY-4.0; **MuQ weights CC-BY-NC 4.0** (non-commercial; fine for this local research tool, must be recorded); SongFormer weight license not stated on the model page → [ASK] before redistribution of anything derived |
| Environment | tested Python 3.10 on Ubuntu / NVIDIA GPU; README says version pins may need loosening |
| Speed | 2–4 s per song on an NVIDIA L40; CPU / MPS speed unknown (two ~0.3 B backbones over a full song: expect tens of seconds to minutes on CPU) |
| Reported accuracy | SongFormBench-CN 0.891 vs all-in-one 0.834 |

**Feasibility: plausible in an isolated venv (Python 3.10–3.12, its own torch), CPU or MPS.**
Costs: ~3 GB download (needs approval), new dependencies in a separate environment (not in
our lockfile), non-commercial weight license. Best label set for J-pop (has pre-chorus) and the
best reported accuracy on Chinese pop. Effort ≈ 1 day to wrap it as an optional structure
backend writing the same `Structure` sections; risk medium (pins, MPS ops).

## Evaluation data (needed for A, B and C alike)

* **MyGO official scores** (user is transcribing): rehearsal marks → `[Verse]` / `[Chorus]`
  markers in the chord text. Best domain match (J-pop band).
* **SongFormBench** (CC BY 4.0): 300 expert-annotated songs, 100 of them Chinese pop — but
  distributed as **mel spectrograms**, not audio (reconstruction via a BigVGAN vocoder is
  suggested). Usable for scoring structure if vocoded audio is acceptable to our pipeline;
  extra download (vocoder) and an unknown accuracy penalty from vocoding.
* mir_eval already provides the standard metrics (`mir_eval.segment`: boundary hit rate at
  0.5 s / 3 s, pairwise frame clustering F, normalized conditional entropies) — to be wired
  into `harmonia eval` once references exist.

## Recommendation

Keep A as the default (no dependencies, explainable, uses our bar grid). If a learned labeller
is wanted, try **C** first in an isolated environment as an optional backend and compare on
the same references; **B** only via a PyTorch re-implementation of natten, verified on a Linux
GPU machine. Nothing installed or downloaded for B / C yet.

## Sources

- allin1 on PyPI: https://pypi.org/project/allin1/ · GitHub: https://github.com/mir-aidj/all-in-one (issues #11, #17, #30)
- natten on PyPI: https://pypi.org/project/natten/ · madmom on PyPI: https://pypi.org/project/madmom/
- SongFormer: https://github.com/aslp-lab/songformer · https://huggingface.co/ASLP-lab/SongFormer · https://arxiv.org/abs/2510.02797
- MuQ weights: https://huggingface.co/OpenMuQ/MuQ-large-msd-iter
- SongFormBench: https://huggingface.co/datasets/ASLP-lab/SongFormBench

## C — trial on this Mac (2026-10-07, user: "试试 C")

Setup (all under gitignored `outputs/songformer/`): code + weights from HuggingFace
ASLP-lab/SongFormer @ a75880ed (code reviewed before running: no network / exec calls; weights
are safetensors), 2.76 GB `model.safetensors`, plus the 2 KB config of
facebook/wav2vec2-conformer-rope-large-960h-ft. Isolated venv: uv-managed CPython 3.11
(Homebrew's framework Python ran throttled; Python 3.10 + scipy 1.15 wheels fail to load on this
macOS), torch 2.4.0, transformers 4.51.1, muq 0.1.0, msaf 0.1.80, scipy 1.17.1, numpy 1.26.4,
setuptools < 80 (msaf needs pkg_resources). Runner: `tools/songformer_infer.py`.

CPU adaptations, each verified:
1. MusicFM attention: transformers' plain conformer materialises T×T matrices (T ≈ 10.5 k for
   the 420 s window). Swapped for MusicFM's own fused-attention twin with the same weights
   (2-layer check: hidden states equal to 1e-6; weight-norm key names mapped, strict load).
2. Forced CUDA flash kernel → PyTorch's own kernel choice; `transformers.deepspeed` shim.
3. Attention dropout (p = 0.1) was applied even in eval by upstream → set to 0 (deterministic).
4. `--low-mem` loading: peak memory 6.87 → 5.79 GB on a 60 s clip, sections identical.

Results: 60 s clip 27 s, 120 s clip 53 s (CPU). A full 6-min song exhausted RAM + swap (the
Mac already had 8–10 GB swapped by other apps) and was stopped by a watchdog; bf16 autocast was
> 5× slower on this CPU and abandoned. → **C works technically, but full songs need roughly
≥ 10 GB of free RAM**, which this 17 GB Mac does not have with the usual apps open.
Ways forward: (a) the user's Linux / RTX 6000 machine (SongFormer's native platform, 2–4 s per
song); (b) close memory-heavy apps here and retry (≈ 1.5–3 min per song); (c) run in ≤ 120 s
chunks — fits easily but departs from the model's 420 s context (quality unknown, would need
checking against full-context output from (a) or (b)).

## C — GPU run (user decision 2026-10-07: run on the RTX 6000 host)

`tools/songformer_gpu/` (README there): `make_bundle.sh` (Mac: scripts + list + audio links,
m4a decoded) → `setup_env.sh` (host: torch 2.7.1 + cu128 for Blackwell, rest as upstream) →
`submit.sh` (Slurm / PBS / LSF / direct) or the host's Claude `gsched` skill with `run_job.sh`
→ results back to `outputs/songformer/runs/gpu_v1/` → `tools/compare_structure.py` (A vs C:
boundary F@3s, pairwise F, label agreement — agreement, not accuracy).
Note: torch 2.7.1 on GPU vs 2.4.0 in the CPU trial; outputs of the 60 s clip can be compared
to `kaeri_60s` from the CPU run as a cross-check.
