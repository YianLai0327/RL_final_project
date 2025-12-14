import os
import time
import json
import numpy as np
import librosa
from google import genai
from google.genai import types

# 1. CONFIGURATION
# ---------------------------------------------------------
# ⚠️ PASTE YOUR NEW KEY HERE (The old one is expired!)
API_KEY = "" # <--- PASTE YOUR KEY HERE
INPUT_DIR = "./bgm_library"
OUTPUT_FILE = "audio_caption_dataset.json"

client = genai.Client(api_key=API_KEY)

MOOD_LIST = [
    "Happy", "Epic", "Dark", "Romantic", "Playful", 
    "World", "Cinematic", "Electronic", "Classy", "Neutral"
]

# 2. ANALYSIS FUNCTION
# ---------------------------------------------------------
def analyze_signal(file_path):
    try:
        y, sr = librosa.load(file_path, duration=30)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        bpm = float(tempo) if np.isscalar(tempo) else float(tempo[0])
        rms = float(np.mean(librosa.feature.rms(y=y)))
        return bpm, rms
    except Exception as e:
        # Pass the error up to the main loop to handle retry
        raise e 

# 3. HELPER: LOAD/SAVE JSON
# ---------------------------------------------------------
def load_existing_data():
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []
    return []

def save_data(data):
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# 4. MAIN PIPELINE
# ---------------------------------------------------------
def process_audio_library():
    if not os.path.exists(INPUT_DIR):
        print(f"❌ Input directory '{INPUT_DIR}' not found.")
        return

    # Load existing progress to skip done files
    results = load_existing_data()
    processed_files = {item['filename'] for item in results}
    
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(('.mp3', '.wav'))]
    print(f"Found {len(files)} tracks. Already processed: {len(processed_files)}")

    for i, filename in enumerate(files):
        # Skip if already done
        if filename in processed_files:
            continue

        file_path = os.path.join(INPUT_DIR, filename)
        print(f"[{i+1}/{len(files)}] {filename}...", end="", flush=True)

        # --- INFINITE RETRY LOOP ---
        # This will NEVER exit until success is achieved.
        while True:
            audio_file = None
            try:
                # 1. Analyze
                bpm, rms = analyze_signal(file_path)
                print(f" [BPM:{int(bpm)}]", end="", flush=True)

                # 2. Upload
                audio_file = client.files.upload(file=file_path)
                while audio_file.state.name == "PROCESSING":
                    time.sleep(1)
                    audio_file = client.files.get(name=audio_file.name)

                # 3. Generate Caption
                prompt = f"""
                You are a Music Supervisor for a video editing AI.
                CONTEXT: BPM={int(bpm)}.
                
                INSTRUCTIONS:
                1. Output strictly valid JSON.
                2. "rich_caption": A concise description of the track's musical style and atmosphere.
                3. "mood_tags": Select EXACTLY 1 or 2 tags from this list ONLY: {json.dumps(MOOD_LIST)}. 
                4. "has_vocals": true/false.
                5. "instrumentation": List key instruments.
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
                
                # 4. Enforce Data & Validation
                data["filename"] = filename
                data["bpm"] = float(f"{bpm:.1f}")
                data["rms"] = float(f"{rms:.4f}")

                valid_set = {m.lower() for m in MOOD_LIST}
                clean_tags = []
                for t in data.get('mood_tags', []):
                    if t.lower() in valid_set:
                        canonical = next(m for m in MOOD_LIST if m.lower() == t.lower())
                        clean_tags.append(canonical)
                data['mood_tags'] = clean_tags if clean_tags else ["Neutral"]

                # 5. SAVE IMMEDIATELY
                results.append(data)
                save_data(results)
                
                # Cleanup
                client.files.delete(name=audio_file.name)
                print(" ✅ Saved")
                time.sleep(1) 
                
                # SUCCESS: Break the while loop to move to the next file
                break 

            except Exception as e:
                # --- ERROR HANDLING ---
                print(f"\n   ❌ Error: {e}")
                print("   🔄 Retrying in 5 seconds... (Press Ctrl+C to stop)")
                
                # Clean up failed upload if it exists
                if audio_file:
                    try:
                        client.files.delete(name=audio_file.name)
                    except:
                        pass
                
                time.sleep(5)
                # Loop will now restart from the top for the SAME file

    print(f"Done! Final data in {OUTPUT_FILE}")

if __name__ == "__main__":
    process_audio_library()