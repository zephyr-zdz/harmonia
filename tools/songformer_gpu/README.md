# SongFormer on the GPU host (structure option C)

Runs SongFormer (ASLP-lab, HuggingFace `ASLP-lab/SongFormer` @ `a75880ed`) on all songs and
writes one `<song>.songformer.json` per song (`sections: [{start, end, label}]`). Why a GPU
host: on the 17 GB Mac a full song needs ≈ 10 GB free RAM (see docs/structure_options.md).

Licences: SongFormer repo CC-BY-4.0; MuQ weights inside the checkpoint CC-BY-NC 4.0
(non-commercial). Audio: IdolSongsJp (evaluation only, no training) + your own MyGO files —
keep them on your machines.

## 0. On the Mac

```bash
bash tools/songformer_gpu/make_bundle.sh
rsync -avL outputs/songformer/gpu_bundle/ GPUHOST:songformer/      # ≈ 2.6 GB (audio)
```

## 1. On the GPU host — environment (once, login node is fine)

```bash
cd ~/songformer
bash setup_env.sh "$PWD/sf_env" "$PWD/sf_model"
```
Downloads PyTorch 2.7.1 + CUDA 12.8 wheels (Blackwell / sm_120 needs ≥ 2.7; SongFormer's own pin
2.4.0 cannot run on an RTX 6000 Pro) and the 2.76 GB checkpoint. No internet on the cluster?
Copy `outputs/songformer/model/` from the Mac to `sf_model/` instead (the script then skips the
download) and pre-fetch the 2 KB config into `~/.cache/huggingface` the same way.
Check the last lines: `available True` and `capability (12, 0)` on a GPU node.

## 2. Submit

```bash
cd ~/songformer
bash submit.sh sf_env sf_model list.txt results
# site options if needed:
SF_PARTITION=gpu SF_ACCOUNT=xxx SF_GRES=gpu:1 bash submit.sh sf_env sf_model list.txt results
```
`submit.sh` detects Slurm (`sbatch`), PBS (`qsub`) or LSF (`bsub`); without a scheduler it runs
directly. Requests 1 GPU, 8 cores, 32 GB, 1 h. The job is resumable (finished songs are skipped).

### 2b. Or let Claude on the GPU host submit it with its `gsched` skill

Start Claude Code in `~/songformer` on the host and paste:

> Use the gsched skill to run one GPU job in this directory. Environment and weights are set
> up (`sf_env/`, `sf_model/`; if `sf_env` is missing, run `bash setup_env.sh "$PWD/sf_env"
> "$PWD/sf_model"` first and show me its last lines). The job command is
> `bash run_job.sh "$PWD/sf_env" "$PWD/sf_model" "$PWD/list.txt" "$PWD/results"` — needs 1 GPU
> (any NVIDIA card with ≥ 24 GB; Blackwell is fine), 8 CPU cores, 32 GB RAM, ≤ 1 h; it is
> resumable. Do not change the scripts or the model. When it finishes, show me `results/run.log`
> (tail), `results/PROVENANCE.txt` and the number of `*.songformer.json` files (expected: as many
> lines as `list.txt`).

## 3. Bring the results back

```bash
rsync -av GPUHOST:songformer/results/ outputs/songformer/runs/gpu_v1/      # on the Mac
.venv/bin/python tools/compare_structure.py outputs/songformer/runs/gpu_v1
```

Differences from upstream inference (all verified on CPU, see docs/structure_options.md):
MusicFM uses its fused-attention twin (outputs equal to 1e-6), attention dropout is 0 at
inference (upstream applied p = 0.1 even in eval), kernel choice left to PyTorch,
`transformers.deepspeed` shim.
