import os
import time
import json
from google import genai
from google.genai import types

# CONFIGURATION
API_KEY = ""
# Point this to your NEW folder (the smart cuts)
INPUT_DIR = "./data/music_library" 
OUTPUT_FILE = "./data/music_captions.json"

client = genai.Client(api_key=API_KEY)

# THE REFINED PROMPT
# ---------------------------------------------------------
# Changes made:
# 1. "Loop/Theme" context: Tells Gemini this is just the highlight.
# 2. "Voice Interference": Explicit check for vocals (Chorus usually has vocals!).
# 3. "Loopability": Asks if the clip feels repetitive (good for BGM) or dynamic.

PROMPT_TEXT = """
You are a Music Supervisor Curating a Background Music Library.
You are listening to a **30-second highlight clip** (Chorus/Hook) of a track.

Analyze this clip and output a JSON object with exactly these fields:

1.  "filename": (The input filename)
2.  "mood_tags": A list of exactly 2 adjectives from this allowed list: 
    ["Happy", "Sad", "Epic", "Chill", "Tense", "Romantic", "Upbeat", "Dark", "Funny", "Sentimental"].
3.  "energy": (Low, Medium, High).
4.  "instrumentation": Key instruments heard (e.g., "Synth, Bass, Drums").
5.  "has_vocals": (true/false) -> CRITICAL: Set to true if ANY human voice (singing or rapping) is present.
6.  "loop_suitability": (1-10) -> How repetitive/steady is this? (10 = Perfect steady loop, 1 = Too many changes/stops).
7.  "suggested_genre": The specific sub-genre (e.g. "Lo-fi Hip Hop", "Epic Orchestral", "Tech House").
8.  "video_pairing": A one-sentence description of the *perfect* video scene for this track (e.g. "Fast-paced travel montage" or "Sad emotional dialogue").

Output ONLY the JSON object.
"""

def process_audio_library():
    results = []
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(('.mp3', '.wav'))]
    print(f"Found {len(files)} highlights to process.")

    for i, filename in enumerate(files):
        file_path = os.path.join(INPUT_DIR, filename)
        print(f"[{i+1}/{len(files)}] Listening to {filename}...", end="")

        try:
            # A. Upload the Highlight
            audio_file = client.files.upload(file=file_path)
            
            # Wait for processing
            while audio_file.state.name == "PROCESSING":
                time.sleep(1)
                audio_file = client.files.get(name=audio_file.name)
            
            if audio_file.state.name == "FAILED":
                print(" ❌ Failed.")
                continue

            # B. Generate Caption
            response = client.models.generate_content(
                model="gemini-2.0-flash", 
                contents=[audio_file, PROMPT_TEXT],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                )
            )
            
            # C. Parse & Store
            data = json.loads(response.text)
            data["filename"] = filename
            results.append(data)
            
            # Clean up
            client.files.delete(name=audio_file.name)
            print(f" ✅ {data['mood_tags']} | Vocals: {data['has_vocals']}")
            
            # Rate limit safety
            time.sleep(4)

        except Exception as e:
            print(f" ❌ Error: {e}")

    # Save
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Done! Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    process_audio_library()