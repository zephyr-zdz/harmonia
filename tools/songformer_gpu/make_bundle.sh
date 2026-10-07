#!/usr/bin/env bash
# (Mac side) Assemble outputs/songformer/gpu_bundle/ for the GPU host: scripts + audio list +
# symlinks to the audio (no extra disk here). Copy it with symlinks dereferenced:
#
#   bash tools/songformer_gpu/make_bundle.sh
#   rsync -avL outputs/songformer/gpu_bundle/ GPUHOST:songformer/
#
# Audio: IdolSongsJp plain mixes (eval-only licence) + the user's MyGO files — copyrighted /
# licensed material: keep it on your own machines, never in the public repo.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
B="$ROOT/outputs/songformer/gpu_bundle"
rm -rf "$B"   # only this generated directory
mkdir -p "$B/audio"
cp "$ROOT/tools/songformer_gpu/"{setup_env.sh,run_job.sh,submit.sh,README.md} "$ROOT/tools/songformer_infer.py" "$B/"

: > "$B/list.txt"
add() {  # add <file> <unique name>
  ln -s "$1" "$B/audio/$2"
  echo "audio/$2" >> "$B/list.txt"
}
for f in "$ROOT"/data/external/idolsongsjp/master_48k32b_-9LUFS/*.wav; do
  add "$f" "$(basename "$f")"
done
while IFS= read -r -d '' f; do
  case "$f" in
    *.flac|*.wav) add "$f" "$(basename "$f")" ;;
    *)  # m4a / mp3: decode here so the GPU host needs no ffmpeg (real file, not a link)
        w="$B/audio/$(basename "${f%.*}").wav"
        ffmpeg -nostdin -loglevel error -y -i "$f" -ac 2 -ar 44100 "$w"
        echo "audio/$(basename "$w")" >> "$B/list.txt" ;;
  esac
done < <(find "$ROOT/data/external/user_music" -type f \( -name '*.flac' -o -name '*.m4a' -o -name '*.mp3' -o -name '*.wav' \) -print0 | sort -z)

# paths are relative to the bundle: run everything from inside it on the GPU host
n=$(wc -l < "$B/list.txt")
size=$(du -shL "$B/audio" | cut -f1)
echo "bundle: $B  ($n songs, audio $size)"
echo "next:   rsync -avL \"$B/\" GPUHOST:songformer/"
