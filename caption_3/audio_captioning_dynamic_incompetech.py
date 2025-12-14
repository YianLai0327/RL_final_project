import os
import time
import json
import numpy as np
import librosa
from google import genai
from google.genai import types

# 1. CONFIGURATION
# ---------------------------------------------------------
# ⚠️ SECURITY: Use environment variables or paste your key here!
API_KEY = "AIzaSyArd-u_e2VhEw6huNpN7x4Mu8KsQ4oSuNo" 
INPUT_DIR = "./bgm_library_incompetech_chorus_start"
OUTPUT_FILE = "./data/audio_captions_incompetech_chorus_start.json"

client = genai.Client(api_key=API_KEY)

# UPDATED: Aligned with your 'download_free_incompetech.py' Base Mood mappings
MOOD_LIST = [
    "Happy",        # Maps from 'upbeat'
    "Epic",         # Maps from 'epic'
    "Dark",         # Maps from 'horror'
    "Romantic",     # Maps from 'romantic'
    "Playful",      # Maps from 'comedy'
    "World",        # Maps from 'world'
    "Cinematic",    # Maps from 'scoring'
    "Electronic",   # Maps from 'electronic'
    "Classy",       # Maps from 'jazz'
    "Neutral"       # Fallback
]

# 2. ANALYSIS FUNCTION (Signal Processing Only)
# ---------------------------------------------------------
def analyze_signal(file_path):
    """
    Extracts objective metrics (BPM, RMS) to ground the LLM's hallucination,
    but skips the subjective 'Energy Level' classification.
    """
    try:
        y, sr = librosa.load(file_path, duration=30)
        
        # 1. BPM (Standard Librosa)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        bpm = float(tempo) if np.isscalar(tempo) else float(tempo[0])

        # 2. RMS (Loudness)
        rms = float(np.mean(librosa.feature.rms(y=y)))
        
        return bpm, rms

    except Exception as e:
        print(f"⚠️ Error classifying {file_path}: {e}")
        return 0.0, 0.0

# 3. MAIN PIPELINE
# ---------------------------------------------------------
def process_audio_library():
    if not os.path.exists(INPUT_DIR):
        print(f"❌ Input directory '{INPUT_DIR}' not found.")
        return

    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(('.mp3', '.wav'))]
    print(f"Found {len(files)} tracks.")

    results = []
    
    for i, filename in enumerate(files):
        file_path = os.path.join(INPUT_DIR, filename)
        print(f"[{i+1}/{len(files)}] {filename}...", end="", flush=True)

        try:
            # 1. Analyze (Get BPM/RMS only)
            bpm, rms = analyze_signal(file_path)
            print(f" [BPM:{int(bpm)}]", end="", flush=True)

            # 2. Upload
            audio_file = client.files.upload(file=file_path)
            while audio_file.state.name == "PROCESSING":
                time.sleep(1)
                audio_file = client.files.get(name=audio_file.name)

            # 3. Generate Caption
            # We provide BPM/RMS as context, but let the LLM decide the description.
            prompt = f"""
            You are a Music Supervisor for a video editing AI.
            CONTEXT: BPM={int(bpm)}.
            
            INSTRUCTIONS:
            1. Output strictly valid JSON.
            2. "rich_caption": A concise description of the track's musical style and atmosphere.
            3. "mood_tags": Select EXACTLY 1 or 2 tags from this list ONLY: {json.dumps(MOOD_LIST)}. 
               - Do not invent new tags. 
               - If unsure, use "Neutral".
            4. "has_vocals": true/false.
            5. "instrumentation": List key instruments (e.g. "Piano, Synth, Drums").
            """

            response = client.models.generate_content(
                model="gemini-2.5-flash", 
                contents=[audio_file, prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.1
                )
            )

            data = json.loads(response.text)
            
            # 4. Enforce Data & Validation
            data["filename"] = filename
            data["bpm"] = float(f"{bpm:.1f}")
            data["rms"] = float(f"{rms:.4f}")
            # Note: "energy_level" is removed per your request
            
            # Strict Mood Validation (Case-Insensitive Match)
            valid_set = {m.lower() for m in MOOD_LIST}
            clean_tags = []
            for t in data.get('mood_tags', []):
                if t.lower() in valid_set:
                    # Restore canonical capitalization from MOOD_LIST
                    canonical = next(m for m in MOOD_LIST if m.lower() == t.lower())
                    clean_tags.append(canonical)
            
            data['mood_tags'] = clean_tags if clean_tags else ["Neutral"]

            results.append(data)
            client.files.delete(name=audio_file.name)
            print(" ✅ Done")
            time.sleep(1) # Rate limit safety

        except Exception as e:
            print(f" ❌ {e}")

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Done! Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    process_audio_library()