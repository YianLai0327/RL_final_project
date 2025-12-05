import time
import json
import os
import glob
import numpy as np
import librosa
from moviepy.editor import VideoFileClip
from google import genai
from google.genai import types

# ==========================================
# 1. CONFIGURATION
# ==========================================
API_KEY = "AIzaSyDAXnyYyRgScjcTlYdN94CMOPxaAV0jnIs"  # <--- PASTE YOUR KEY HERE
VIDEO_DIR = "./video_library_all"
OUTPUT_DIR = "./data/captions_raw"
MASTER_FILE = "./data/video_caption_dataset.json"

API_COOLDOWN_SECONDS = 60
MAX_DURATION_SEC = 300  # <--- NEW: Limit processing to 8 minutes (8 * 60)
MOOD_LIST = ["Happy", "Sad", "Epic", "Chill", "Tense", "Romantic", "Upbeat", "Dark", "Funny", "Sentimental"]

client = genai.Client(api_key=API_KEY)

# ==========================================
# 2. AUDIO ANALYSIS ENGINE (Librosa)
# ==========================================
def analyze_video_audio(video_path):
    """
    Extracts audio from video (capped at 8 mins) and identifies 
    timestamps of physically 'LOUD' vs 'QUIET' sections.
    """
    print(f"   🔊 Analyzing audio (0-{MAX_DURATION_SEC}s) for: {os.path.basename(video_path)}...")
    temp_audio = f"temp_{int(time.time())}.wav"
    
    video = None
    try:
        # Load Video
        video = VideoFileClip(video_path)
        
        # TRIM VIDEO IF NEEDED
        duration = video.duration
        if duration > MAX_DURATION_SEC:
            # We only extract audio for the first 8 mins to match our new constraint
            video = video.subclip(0, MAX_DURATION_SEC)
            
        video.audio.write_audiofile(temp_audio, logger=None)
        
        # Load into Librosa
        y, sr = librosa.load(temp_audio)
        rms = librosa.feature.rms(y=y)[0]
        times = librosa.times_like(rms, sr=sr)
        
        # --- Three-Tier Logic ---
        noise_floor = float(np.percentile(rms, 10))
        quiet_threshold = min(0.05, noise_floor * 2.0)
        
        active_signal = rms[rms > quiet_threshold]
        if len(active_signal) > 0:
            loud_threshold = float(np.percentile(active_signal, 70))
        else:
            loud_threshold = 0.15 
        loud_threshold = max(0.08, loud_threshold)

        loud_segments = []
        quiet_segments = []
        
        current_state = None 
        start_t = 0
        
        for i, amp in enumerate(rms):
            t = times[i]
            if amp >= loud_threshold: state = "loud"
            elif amp <= quiet_threshold: state = "quiet"
            else: state = "med"
            
            if state != current_state:
                if current_state == "loud" and (t - start_t) > 0.5:
                    loud_segments.append(f"{int(start_t)}s-{int(t)}s")
                elif current_state == "quiet" and (t - start_t) > 1.0: 
                    quiet_segments.append(f"{int(start_t)}s-{int(t)}s")
                current_state = state
                start_t = t
        
        return ", ".join(loud_segments), ", ".join(quiet_segments)

    except Exception as e:
        print(f"   ⚠️ Audio Analysis Warning: {e}")
        return "None", "None"
    finally:
        # Cleanup
        if video: video.close()
        if os.path.exists(temp_audio): 
            try: os.remove(temp_audio)
            except: pass

# ==========================================
# 3. AI CAPTIONING ENGINE (Gemini)
# ==========================================
def repair_json_string(raw_text):
    """
    Simple fallback: If JSON is cut off, try to close the list/object.
    """
    raw_text = raw_text.strip()
    # If it ends with a comma, remove it
    if raw_text.endswith(","):
        raw_text = raw_text[:-1]
    # If it doesn't end with list closer, add it
    if not raw_text.endswith("]"):
        # Check if we are inside an object
        if raw_text.endswith("}"):
            raw_text += "]"
        else:
            # We are probably inside a string or value. 
            # This is hard to fix perfectly, but we can try closing the object then the list.
            raw_text += '", "energy_level": "Medium", "cut_pace": "Moderate", "has_speech": false, "mood_tags": ["Chill"]}]'
    return raw_text

def generate_captions(video_path, loud_str, quiet_str):
    print(f"   🚀 Uploading to Gemini...")
    video_file = client.files.upload(file=video_path)
    
    # Wait for processing
    while video_file.state.name == "PROCESSING":
        time.sleep(2)
        video_file = client.files.get(name=video_file.name)
    
    if video_file.state.name == "FAILED":
        raise ValueError("Video processing failed on Google servers.")

    # The Prompt with 8-MINUTE LIMIT
    PROMPT_TEXT = f"""
    You are a professional Music Supervisor.
    
    ### DATA INPUTS:
    1. **Visuals:** (Analyze the video stream directly)
    2. **Audio Volume (Data):**
       - PHYSICALLY LOUD Segments: [{loud_str}]
       - PHYSICALLY SILENT Segments: [{quiet_str}]
    
    ### TASK:
    Segment the video and determine `energy_level` by **INTEGRATING** Visuals and Audio.
    
    ⛔️ **CRITICAL CONSTRAINT: ONLY ANALYZE FROM 00:00 TO 08:00.**
    - Stop generating output exactly at the 8-minute mark (480 seconds).
    - Do not output any JSON objects for timestamps after 08:00.
    
    ### INTEGRATION MATRIX:
    | Visual Action | Audio Volume (Data) | **Final Energy Level** |
    | :--- | :--- | :--- |
    | **Fast / Chaotic** | Loud / Medium | **High** (Action drives energy) |
    | **Fast / Chaotic** | Quiet | **High** (Visuals override silence) |
    | Static / Slow | **Loud (Shouting/Excitement)** | **High** (Audio drives energy) |
    | Static / Slow | **Loud (Wind/Noise)** | **Medium** (Context matters!) |
    | Static / Slow | Quiet | **Low** |
    | Moderate | Moderate | **Medium** |

    ### RESTRICTED MOOD LIST:
    {json.dumps(MOOD_LIST)}

    ### OUTPUT FORMAT (JSON):
    [
      {{
        "start": "00:00", 
        "end": "00:10",
        "visual_summary": "...",
        "ideal_music_description": "...",
        "energy_level": "Low",
        "cut_pace": "Slow",
        "has_speech": true,
        "mood_tags": ["Chill"]
      }}
    ]
    """

    print("   🤖 Generating intelligence...")
    
    max_retries = 3
    base_delay = 10
    
    response = None
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash", 
                contents=[video_file, PROMPT_TEXT],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.1,
                    max_output_tokens=65536
                )
            )
            break
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait = base_delay * (2 ** attempt)
                print(f"   ⚠️ Quota hit. Waiting {wait}s...")
                time.sleep(wait)
            else:
                client.files.delete(name=video_file.name)
                raise e
    
    client.files.delete(name=video_file.name)
    
    if not response:
        raise RuntimeError("Failed to get response after retries.")

    # Clean Markdown
    raw_text = response.text
    if raw_text.startswith("```json"): raw_text = raw_text[7:]
    if raw_text.startswith("```"): raw_text = raw_text[3:]
    if raw_text.endswith("```"): raw_text = raw_text[:-3]
    
    try:
        return json.loads(raw_text.strip())
    except json.JSONDecodeError:
        print("   ⚠️ JSON incomplete. Attempting repair...")
        # Try to fix truncated JSON
        fixed_text = repair_json_string(raw_text)
        try:
            return json.loads(fixed_text)
        except:
            # Save raw debug if repair fails
            with open("debug_failed_chunk.txt", "w", encoding="utf-8") as f:
                f.write(raw_text)
            raise ValueError("JSON output was truncated and could not be repaired.")

# ==========================================
# 4. MAIN BATCH PIPELINE
# ==========================================
def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    video_files = glob.glob(os.path.join(VIDEO_DIR, "*.mp4"))
    video_files.sort()
    
    print(f"📂 Found {len(video_files)} videos in {VIDEO_DIR}")
    print(f"⏱️  Processing Limit: First {MAX_DURATION_SEC/60} minutes per video.")
    print("-" * 50)

    for i, video_path in enumerate(video_files):
        video_name = os.path.basename(video_path)
        json_name = video_name.replace(".mp4", ".json")
        output_path = os.path.join(OUTPUT_DIR, json_name)
        
        print(f"\n[{i+1}/{len(video_files)}] Checking: {video_name}")

        if os.path.exists(output_path):
            print("   ✅ JSON already exists. Skipping.")
            continue

        try:
            # 1. Analyze Audio (Clipped)
            loud_str, quiet_str = analyze_video_audio(video_path)
            
            # 2. Get AI Captions (Clipped via Prompt)
            data = generate_captions(video_path, loud_str, quiet_str)
            
            # 3. Validation
            for seg in data:
                if not seg.get('mood_tags'): seg['mood_tags'] = ["Chill"]
                seg['mood_tags'] = [t.capitalize() for t in seg['mood_tags'] if t.capitalize() in MOOD_LIST]

            # 4. Save
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"   💾 Saved to: {output_path}")

            if i < len(video_files) - 1:
                print(f"   💤 Cooling down for {API_COOLDOWN_SECONDS}s...")
                time.sleep(API_COOLDOWN_SECONDS)

        except Exception as e:
            print(f"   ❌ FAILED: {str(e)}")
            continue

    # Merge
    print("\n" + "="*50)
    print("🔄 Merging all JSONs...")
    master_data = {}
    json_files = glob.glob(os.path.join(OUTPUT_DIR, "*.json"))
    for jf in json_files:
        key = os.path.basename(jf).replace(".json", ".mp4")
        with open(jf, 'r', encoding='utf-8') as f:
            master_data[key] = json.load(f)
            
    with open(MASTER_FILE, 'w', encoding='utf-8') as f:
        json.dump(master_data, f, indent=2, ensure_ascii=False)
        
    print(f"🎉 DONE! Master dataset saved to: {MASTER_FILE}")

if __name__ == "__main__":
    main()