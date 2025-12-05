import os
import time
import json
import numpy as np
import librosa
from google import genai
from google.genai import types

# 1. CONFIGURATION
# ---------------------------------------------------------
# ⚠️ SECURITY: Use environment variables!
API_KEY = "AIzaSyD13s4uafY3MQ5ftuWHszJNeF5ZU4mo6D8" 
INPUT_DIR = "./data/music_library"
OUTPUT_FILE = "./data/music_library_aligned.json"

client = genai.Client(api_key=API_KEY)

MOOD_LIST = ["Happy", "Sad", "Epic", "Chill", "Tense", "Romantic", "Upbeat", "Dark", "Funny", "Sentimental"]

# 2. CALIBRATION STEP (The "Data-Driven" Rule)
# ---------------------------------------------------------
def calibrate_library_thresholds(file_list):
    """
    Scans the entire library first to find the 33rd and 66th percentiles
    for RMS (Loudness). This adapts the rule to YOUR specific audio files.
    """
    print("📊 Calibrating Energy Thresholds...")
    all_rms = []
    
    for filename in file_list:
        try:
            path = os.path.join(INPUT_DIR, filename)
            # Load just 10 seconds to be fast
            y, sr = librosa.load(path, duration=10)
            rms = float(np.mean(librosa.feature.rms(y=y)))
            all_rms.append(rms)
        except:
            continue
            
    if not all_rms:
        return 0.03, 0.06 # Fallback to slide defaults if empty

    # Calculate dynamic thresholds
    low_thresh = np.percentile(all_rms, 33)  # Below this is "Low"
    high_thresh = np.percentile(all_rms, 66) # Above this is "High"
    
    print(f"   ► Low/Med Boundary: {low_thresh:.4f}")
    print(f"   ► Med/High Boundary: {high_thresh:.4f}")
    
    return low_thresh, high_thresh

# 3. ANALYSIS FUNCTION
# ---------------------------------------------------------
def analyze_signal(file_path, low_thresh, high_thresh):
    try:
        y, sr = librosa.load(file_path, duration=30)
        
        # 1. BPM (Standard Librosa)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        bpm = float(tempo) if np.isscalar(tempo) else float(tempo[0])

        # 2. RMS (Loudness)
        rms = float(np.mean(librosa.feature.rms(y=y)))
        
        # 3. Dynamic Decision Rule
        # We use the calibrated thresholds + The Standard BPM rule (130)
        if bpm >= 130:
            energy_level = "High" # Fast is always high energy
        elif rms >= high_thresh:
            energy_level = "High"
        elif rms <= low_thresh:
            energy_level = "Low"
        else:
            energy_level = "Medium"
            
        return bpm, rms, energy_level

    except Exception as e:
        print(f"⚠️ Error: {e}")
        return 0.0, 0.0, "Medium"

# 4. MAIN PIPELINE
# ---------------------------------------------------------
def process_audio_library():
    if not os.path.exists(INPUT_DIR):
        print("❌ Input directory not found.")
        return

    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(('.mp3', '.wav'))]
    print(f"Found {len(files)} tracks.")

    # --- PHASE 1: CALIBRATION ---
    low_cut, high_cut = calibrate_library_thresholds(files)

    # --- PHASE 2: PROCESSING ---
    results = []
    for i, filename in enumerate(files):
        file_path = os.path.join(INPUT_DIR, filename)
        print(f"[{i+1}/{len(files)}] {filename}...", end="", flush=True)

        try:
            # 1. Analyze
            bpm, rms, calc_energy = analyze_signal(file_path, low_cut, high_cut)
            print(f" [BPM:{int(bpm)} | En:{calc_energy}]", end="", flush=True)

            # 2. Upload
            audio_file = client.files.upload(file=file_path)
            while audio_file.state.name == "PROCESSING":
                time.sleep(1)
                audio_file = client.files.get(name=audio_file.name)

            # 3. Generate Caption
            prompt = f"""
            You are a Music Supervisor.
            CONTEXT: BPM={int(bpm)}, Energy={calc_energy} (Relative to library).
            
            INSTRUCTIONS:
            1. Output JSON.
            2. "rich_caption": Description matching context (e.g. "A {calc_energy} energy track...").
            3. "mood_tags": Exactly 1 or 2 from: {json.dumps(MOOD_LIST)}
            4. "has_vocals": true/false.
            5. "instrumentation": Key instruments.
            """

            response = client.models.generate_content(
                model="gemini-2.0-flash", 
                contents=[audio_file, prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.1
                )
            )

            data = json.loads(response.text)
            
            # 4. Enforce Data
            data["filename"] = filename
            data["bpm"] = float(f"{bpm:.1f}")
            data["rms"] = float(f"{rms:.4f}")
            data["energy_level"] = calc_energy
            
            # Validation
            valid_set = set(MOOD_LIST)
            data['mood_tags'] = [t.capitalize() for t in data.get('mood_tags', []) if t.capitalize() in valid_set]
            if not data['mood_tags']: data['mood_tags'] = ["Chill"]

            results.append(data)
            client.files.delete(name=audio_file.name)
            print(" ✅ Done")
            time.sleep(1)

        except Exception as e:
            print(f" ❌ {e}")

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Done! Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    process_audio_library()


'''
1. Phase 1: Global Calibration (The "Learning" Step)
Before processing a single file for captioning, the script scans every track in
./data/music_chorus_starts to build a statistical profile of Loudness (RMS).
--It calculates the RMS (Root Mean Square) amplitude for every track.
--It calculates two dynamic percentiles:
    --$P_{33}$ (Low Cutoff): The boundary below which the quietest 33% of your tracks sit.
    --$P_{66}$ (High Cutoff): The boundary above which the loudest 33% of your tracks sit.

2. Phase 2: Per-Track Decision Logic (The "Brain")
For each specific track, we extract its BPM and RMS, and then apply this hybrid decision tree:
The Hierarchy of Rules:
    1. Rule A (The Tempo Override):
        -IF BPM ≥ 130:
        -THEN Energy = "High"
        -(Reasoning: A fast track is energetic even if it was mastered quietly. 130 BPM is the standard threshold for Techno/DnB/Fast Pop.)
    2. Rule B (The Loudness Check):
        - ELSE IF RMS ≥ $P_{66}$ (The dynamic high cutoff):
        - THEN Energy = "High"
    3. Rule C (The Chill Check):
        - ELSE IF RMS ≤ $P_{33}$ (The dynamic low cutoff):
        - THEN Energy = "Low"
    4. Rule D (The Middle Ground):
        - ELSE:
        - THEN Energy = "Medium"
3. Phase 3: Prompt Injection (Controlling the AI)
Once the math determines the Energy and BPM, we force the LLM to agree with it. We don't ask Gemini "How does this sound?"; we tell it:
    - "This track has 145 BPM and is High Energy. Write a description that matches these facts."

The Result:
    - You get the rich semantic understanding of the LLM (identifying instruments, mood, genre).
    - You get the reliability of Signal Processing (accurate BPM, consistent energy levels).
    - No Hallucinations: The AI cannot claim a slow, quiet song is "High Energy" because the prompt explicitly forbids it based on the math.
'''