import os
import time
import csv
import urllib.parse
import warnings
import requests
import librosa
import numpy as np

# Suppress librosa warnings
warnings.filterwarnings('ignore')

# === Config ===
# FINAL CLEAN MAPPING:
# 1. Feel-Based: Matches the 'feel' metadata field (e.g., "Bright, Grooving").
# 2. Keyword-Based: Searches the song's 'title' and 'description' for the keyword 
#    (e.g., finds "Jazz" in the title "Space Jazz").

CATEGORY_RULES = {
    # --- FEEL-BASED (Moods) ---
    "upbeat":     {"filter_type": "feel", "values": ["Bright", "Uplifting", "Bouncy"]},
    "epic":       {"filter_type": "feel", "values": ["Epic", "Action"]},
    "romantic":   {"filter_type": "feel", "values": ["Calming", "Relaxed"]}, 
    "comedy":     {"filter_type": "feel", "values": ["Humorous"]},
    
    # --- KEYWORD-BASED (Styles) ---
    "world":      {"filter_type": "keyword", "values": ["World", "Ethnic", "Tribal"]}, 
    "scoring":    {"filter_type": "keyword", "values": ["Score", "Soundtrack", "Cinematic"]}, 
    "electronic": {"filter_type": "keyword", "values": ["Electronic", "Synth", "Techno"]},  
    "jazz":       {"filter_type": "keyword", "values": ["Jazz", "Vibraphone", "Saxophone"]}, 
    "horror":     {"filter_type": "keyword", "values": ["Horror", "Eerie", "Suspenseful", "Dark", "SCP"]}, 
}

CATALOG_URL = "https://incompetech.com/music/royalty-free/pieces.json"
BASE_MP3_URL = "https://incompetech.com/music/royalty-free/mp3-royaltyfree/"

DEST_DIR = "bgm_library_incompetech_raw"
OUT_CSV = "audio_features_incompetech.csv"
REQUEST_DELAY = 0.5
MAX_PER_CATEGORY = 10
MAX_DURATION_SECONDS = 300  # 5 Minutes

BASE_MOOD_MAP = {
    "upbeat":     "happy",
    "epic":       "epic",
    "horror":     "dark",
    "romantic":   "romantic",
    "comedy":     "playful",
    "world":      "world",
    "scoring":    "cinematic",
    "electronic": "electronic",
    "jazz":       "classy",
}

# === Helpers (Same as before) ===

def get_catalog():
    # ... (unchanged) ...
    print(f"[INIT] Fetching catalog from {CATALOG_URL}...")
    try:
        resp = requests.get(CATALOG_URL, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"  [ERROR] Could not fetch catalog: {e}")
        return []

def parse_duration(length_str):
    # ... (unchanged) ...
    if not length_str: return 9999
    try:
        parts = list(map(int, length_str.split(':')))
        if len(parts) == 3: return parts[0] * 3600 + parts[1] * 60 + parts[2]
        elif len(parts) == 2: return parts[0] * 60 + parts[1]
        else: return 9999
    except ValueError:
        return 9999

def download_file(filename, dest_dir):
    # ... (unchanged) ...
    if not filename: return None
    safe_filename = urllib.parse.quote(filename)
    url = f"{BASE_MP3_URL}{safe_filename}"
    
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, filename)

    if os.path.exists(dest_path):
        return dest_path

    print(f"  [DOWNLOAD] {filename}")
    try:
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            with open(dest_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        time.sleep(REQUEST_DELAY)
        return dest_path
    except Exception as e:
        print(f"    [ERROR] Download failed: {e}")
        return None

def extract_features(path):
    # ... (unchanged) ...
    try:
        y, sr = librosa.load(path, sr=None, mono=True, duration=30.0)
    except Exception: return 0, None, 0, 0
    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        if np.ndim(tempo) > 0: tempo = tempo.item()
    except: tempo = 0.0
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1)
    rms = librosa.feature.rms(y=y).mean()
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    return float(tempo), mfcc, float(rms), float(centroid)

def classify_mood(category_tag, bpm, rms):
    # ... (unchanged) ...
    base_mood = BASE_MOOD_MAP.get(category_tag, "neutral")
    if bpm >= 130 or rms >= 0.12: energy = "high"
    elif bpm >= 100 or rms >= 0.06: energy = "medium"
    else: energy = "low"
    return f"{base_mood}_{energy}", energy

def init_csv(filename):
    # ... (unchanged) ...
    if not os.path.exists(filename):
        headers = ["filename", "base_mood", "mood_tag", "energy", "bpm", "rms", "spectral_centroid"] + [f"mfcc_{i}" for i in range(1, 14)]
        with open(filename, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=headers).writeheader()

def append_to_csv(filename, row):
    # ... (unchanged) ...
    with open(filename, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=row.keys()).writerow(row)

# === Main ===
def main():
    catalog = get_catalog()
    if not catalog: return

    init_csv(OUT_CSV)
    processed_files = set() 

    for category, rule in CATEGORY_RULES.items():
        print(f"\n=== CATEGORY: {category} ===")
        
        candidates = []
        filter_type = rule["filter_type"] 
        target_values = [val.lower() for val in rule["values"]] # Convert values to lower for case-insensitive search

        for track in catalog:
            if track['filename'] in processed_files:
                continue

            # 1. Duration Check
            duration_sec = parse_duration(track.get("length", ""))
            if duration_sec >= MAX_DURATION_SECONDS:
                continue

            # 2. Match Check
            match = False
            
            if filter_type == "feel":
                # Match against the track's 'feel' field
                track_field = track.get("feel", "").lower()
                if any(val in track_field for val in target_values):
                    match = True
            
            elif filter_type == "keyword":
                # Match against Title and Description
                track_title = track.get("title", "").lower()
                track_desc = track.get("description", "").lower()
                
                if any(val in track_title or val in track_desc for val in target_values):
                    match = True

            if match:
                candidates.append(track)

        print(f"  Found {len(candidates)} candidates (< 5 mins).")

        # --- Downloading/Processing ---
        success_count = 0
        for track in candidates:
            if success_count >= MAX_PER_CATEGORY: 
                break
            
            filename = track.get("filename")
            local_path = download_file(filename, DEST_DIR)
            
            if not local_path: continue
            
            # Feature Extraction (uses first 30s)
            try:
                bpm, mfcc, rms, centroid = extract_features(local_path)
            except Exception as e:
                print(f"    [ERROR] Extract failed: {e}")
                continue

            if bpm < 10: continue

            mood_tag, energy = classify_mood(category, bpm, rms)
            
            row = {
                "filename": filename,
                "base_mood": BASE_MOOD_MAP.get(category, "neutral"),
                "mood_tag": mood_tag,
                "energy": energy,
                "bpm": bpm,
                "rms": rms,
                "spectral_centroid": centroid,
            }
            if mfcc is not None:
                for i, v in enumerate(mfcc): row[f"mfcc_{i+1}"] = float(v)
            
            append_to_csv(OUT_CSV, row)
            processed_files.add(filename)
            success_count += 1
            print(f"    [SUCCESS] {success_count}/{MAX_PER_CATEGORY}: {mood_tag}")

if __name__ == "__main__":
    main()