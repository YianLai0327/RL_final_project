#!/bin/bash

# --- Check parameters ---
if [ $# -ne 2 ]; then
    echo "Usage: $0 <input_mp4_directory> <output_mp3_directory>"
    exit 1
fi

INPUT_DIR="$1"
OUTPUT_DIR="$2"

# --- Validate input directory ---
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory does not exist: $INPUT_DIR"
    exit 1
fi

# --- Create output directory if missing ---
mkdir -p "$OUTPUT_DIR"

# --- Concurrency limit (modify as needed) ---
MAX_JOBS=20
# --- Function to limit parallel jobs ---
function wait_for_slot() {
    while [ "$(jobs -r | wc -l)" -ge "$MAX_JOBS" ]; do
        sleep 1
    done
}

echo "Starting async MP4 → MP3 conversions..."
echo "Max parallel jobs: $MAX_JOBS"
echo

for f in "$INPUT_DIR"/*.mp4; do
    [ -e "$f" ] || { echo "No .mp4 files found in $INPUT_DIR"; exit 0; }

    wait_for_slot   # wait if too many jobs running

    base_name=$(basename "$f" .mp4)
    output_file="$OUTPUT_DIR/$base_name.mp3"

    echo "Launching async conversion: $f → $output_file"

    # Run the conversion asynchronously
    ffmpeg -i "$f" -q:a 0 -map a "$output_file" > /dev/null 2>&1 &
done

echo
echo "Waiting for all jobs to complete..."
wait

echo "✅ All async conversions finished."
