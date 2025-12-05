#!/bin/bash

SRC_DIR="mp3"
OUT1="mp3_1"
OUT2="mp3_2"
OUT3="mp3_3"
OUT4="mp3_4"

# Create output dirs if they do not exist
mkdir -p "$OUT1" "$OUT2" "$OUT3" "$OUT4"

# Shuffle files and assign them in round-robin order
i=0
find "$SRC_DIR" -type f | shuf | while read -r f; do
    case $(( i % 4 )) in
        0) dest="$OUT1" ;;
        1) dest="$OUT2" ;;
        2) dest="$OUT3" ;;
        3) dest="$OUT4" ;;
    esac
    echo "Moving $f → $dest"
    mv "$f" "$dest"/
    i=$((i+1))
done
