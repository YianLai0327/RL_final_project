import os
import csv
from music2emo import Music2emo

# === 1. SET THIS PATH ===
BGM_DIR = r"PATH/TO/YOUR/BGM_LIBRARY"   # <- change this
OUTPUT_CSV = "bgm_va_tags.csv"          # output file name

# === 2. INIT MODEL ===
music2emo = Music2emo()

AUDIO_EXTS = (".mp3", ".wav", ".flac", ".m4a", ".ogg")

rows = []

for root, _, files in os.walk(BGM_DIR):
    for fname in files:
        if not fname.lower().endswith(AUDIO_EXTS):
            continue

        fpath = os.path.join(root, fname)
        print(f"Processing: {fname}")

        try:
            out = music2emo.predict(fpath)

            valence = out.get("valence")
            arousal = out.get("arousal")
            moods = out.get("predicted_moods") or []
            mood_tag = "|".join(moods)

            rows.append([fname, valence, arousal, mood_tag])

        except Exception as e:
            print(f"!!! Error on {fname}: {e}")

# === 3. WRITE CSV ===
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "valence", "arousal", "mood_tags"])
    writer.writerows(rows)

print(f"Done. Wrote {len(rows)} tracks to {OUTPUT_CSV}")
