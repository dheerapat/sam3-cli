#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# segment.sh — run SAM3 segmentation over an entire video in 50-frame chunks
#
# Usage:
#   ./segment.sh <input> <prompt>
#
# Examples:
#   ./segment.sh video.mp4 "penguin"
#   ./segment.sh /path/to/clip.mp4 "ear"
#
# Outputs:
#   <stem>-1.sam.mp4, <stem>-2.sam.mp4, ... in the current directory
# ---------------------------------------------------------------------------

CHUNK=50

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <input> <prompt>" >&2
    exit 1
fi

INPUT="$1"
PROMPT="$2"

# Derive output stem from the input filename (strip directory and extension)
STEM="$(basename "${INPUT%.*}")"

# Step 1: get total frame count
echo "Counting frames in '${INPUT}'..."
TOTAL=$(uv run sam video -i "$INPUT" --frames)

# Guard: make sure we got a valid positive integer
if ! [[ "$TOTAL" =~ ^[0-9]+$ ]] || [[ "$TOTAL" -eq 0 ]]; then
    echo "error: could not determine frame count (got: '${TOTAL}')" >&2
    exit 1
fi

echo "Total frames: ${TOTAL}"
CHUNKS=$(( (TOTAL + CHUNK - 1) / CHUNK ))
echo "Processing ${CHUNKS} chunk(s) of ${CHUNK} frames each"
echo ""

# Step 2: loop over chunks
START=0
CHUNK_NUM=1

while [[ "$START" -lt "$TOTAL" ]]; do
    OUTPUT="${STEM}-${CHUNK_NUM}.sam.mp4"
    echo "[${CHUNK_NUM}/${CHUNKS}] frames ${START}..$((START + CHUNK - 1)) -> ${OUTPUT}"

    uv run sam video \
        -i "$INPUT" \
        -p "$PROMPT" \
        --start-frame "$START" \
        --max-frames "$CHUNK" \
        -o "$OUTPUT"

    START=$(( START + CHUNK ))
    CHUNK_NUM=$(( CHUNK_NUM + 1 ))
    echo ""
done

echo "Done. ${CHUNKS} segment(s) written."

# Step 3: stitch all chunks into a single output file
FINAL="${STEM}.sam.mp4"
echo "Stitching chunks into '${FINAL}'..."

FILELIST="$(mktemp /tmp/sam_chunks_XXXXXX.txt)"
trap 'rm -f "$FILELIST"' EXIT

for (( i=1; i<CHUNK_NUM; i++ )); do
    echo "file '$(realpath "${STEM}-${i}.sam.mp4")'" >> "$FILELIST"
done

ffmpeg -f concat -safe 0 -i "$FILELIST" -c copy "$FINAL" -y

echo "Saved stitched video to '${FINAL}'"
