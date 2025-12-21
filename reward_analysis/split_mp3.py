import json
import subprocess
import os

JSON_FILE = "split.json"
ROOT_DIR = "separated"
OUTPUT_DIR = "splited"
with open(JSON_FILE, "r") as f:
    data = json.load(f)

for mp3_file, timestamps in data.items():
    mp3_path = os.path.join(ROOT_DIR, mp3_file)
    base = os.path.splitext(mp3_file)[0]

    prev_time = 0
    for i, t in enumerate(timestamps):
        duration = t - prev_time
        output = os.path.join(OUTPUT_DIR, f"{base}_{i}.mp3") 

        cmd = [
            "ffmpeg",
            "-y",
            "-i", mp3_path,
            "-ss", str(prev_time),
            "-t", str(duration),
            "-c", "copy",
            output
        ]

        subprocess.run(cmd, check=True)
        prev_time = t
