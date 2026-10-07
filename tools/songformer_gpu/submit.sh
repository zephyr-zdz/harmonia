#!/usr/bin/env bash
# Submit the SongFormer job to whatever scheduler this host has (Slurm, PBS/Torque, LSF),
# or run it directly if there is none.
#
#   bash submit.sh ENV_DIR MODEL_DIR LIST OUT_DIR
#
# Resources: 1 GPU, 8 CPU cores, 32 GB RAM, 1 h (29 songs take a few minutes on one GPU).
# Site-specific options go in environment variables, e.g.
#   SF_PARTITION=gpu SF_ACCOUNT=myproj SF_QOS=normal bash submit.sh ...
#   SF_GRES=gpu:rtx6000:1   (Slurm GRES string if the site names GPU types)
set -euo pipefail
ENV_DIR="$(realpath "$1")"; MODEL_DIR="$(realpath "$2")"; LIST="$(realpath "$3")"; OUT_DIR="$(realpath -m "$4")"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB="$HERE/run_job.sh"
mkdir -p "$OUT_DIR"
NAME=songformer
TIME="${SF_TIME:-01:00:00}"

if command -v sbatch >/dev/null 2>&1; then
  echo "scheduler: Slurm"
  opts=(--job-name="$NAME" --gres="${SF_GRES:-gpu:1}" --cpus-per-task=8 --mem=32G --time="$TIME"
        --output="$OUT_DIR/slurm-%j.out")
  [ -n "${SF_PARTITION:-}" ] && opts+=(--partition="$SF_PARTITION")
  [ -n "${SF_ACCOUNT:-}" ] && opts+=(--account="$SF_ACCOUNT")
  [ -n "${SF_QOS:-}" ] && opts+=(--qos="$SF_QOS")
  sbatch "${opts[@]}" "$JOB" "$ENV_DIR" "$MODEL_DIR" "$LIST" "$OUT_DIR"
  echo "watch: squeue -u \$USER ; log: $OUT_DIR/slurm-<jobid>.out"
elif command -v qsub >/dev/null 2>&1; then
  echo "scheduler: PBS/Torque (qsub)"
  q=(-N "$NAME" -l "select=1:ncpus=8:ngpus=1:mem=32gb" -l "walltime=$TIME" -o "$OUT_DIR/pbs.out" -j oe)
  [ -n "${SF_PARTITION:-}" ] && q+=(-q "$SF_PARTITION")
  [ -n "${SF_ACCOUNT:-}" ] && q+=(-A "$SF_ACCOUNT")
  echo "bash '$JOB' '$ENV_DIR' '$MODEL_DIR' '$LIST' '$OUT_DIR'" | qsub "${q[@]}"
  echo "watch: qstat -u \$USER   (if the site uses -l nodes=1:gpus=1 instead of select=, edit this line)"
elif command -v bsub >/dev/null 2>&1; then
  echo "scheduler: LSF"
  b=(-J "$NAME" -n 8 -gpu "num=1" -R "rusage[mem=32G]" -W "${TIME%:*}" -o "$OUT_DIR/lsf-%J.out")
  [ -n "${SF_PARTITION:-}" ] && b+=(-q "$SF_PARTITION")
  [ -n "${SF_ACCOUNT:-}" ] && b+=(-P "$SF_ACCOUNT")
  bsub "${b[@]}" bash "$JOB" "$ENV_DIR" "$MODEL_DIR" "$LIST" "$OUT_DIR"
  echo "watch: bjobs"
else
  echo "no scheduler found: running directly on this machine (needs a free GPU)"
  bash "$JOB" "$ENV_DIR" "$MODEL_DIR" "$LIST" "$OUT_DIR"
fi
