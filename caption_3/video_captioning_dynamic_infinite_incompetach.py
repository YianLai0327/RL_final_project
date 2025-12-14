import time
import json
import os
import glob
import datetime
import numpy as np
import librosa
from moviepy.editor import VideoFileClip
from google import genai
from google.genai import types

# ==========================================
# 1. CONFIGURATION
# ==========================================
API_KEY = "AIzaSyArd-u_e2VhEw6huNpN7x4Mu8KsQ4oSuNo"  # <--- PASTE YOUR KEY HERE
VIDEO_DIR = "./video_library_large"
OUTPUT_DIR = "./data/captions_raw"
MASTER_FILE = "./data/video_caption_dataset.json"

API_COOLDOWN_SECONDS = 5
MAX_DURATION_SEC = 300  # Strict 5:00 Limit

# --- UPDATED MOOD LIST TO MATCH RL BGM LIBRARY TAGS ---
MOOD_LIST = [
    "Happy", "Epic", "Dark", "Romantic", "Playful", 
    "World", "Cinematic", "Electronic", "Classy", "Neutral",
    "Chill", "Tense", "Sentimental" # Added a few descriptive moods for visual analysis
]
# -------------------------------------------------------

client = genai.Client(api_key=API_KEY)

# ==========================================
# 2. AUDIO ANALYSIS ENGINE
# ==========================================
def analyze_video_audio(video_path):
    print(f"   🔊 Analyzing audio (0-{MAX_DURATION_SEC}s) for: {os.path.basename(video_path)}...")
    temp_audio = f"temp_{int(time.time())}.wav"
    video = None
    try:
        video = VideoFileClip(video_path)
        if video.duration > MAX_DURATION_SEC:
            video = video.subclip(0, MAX_DURATION_SEC)
        video.audio.write_audiofile(temp_audio, logger=None)
        
        y, sr = librosa.load(temp_audio)
        rms = librosa.feature.rms(y=y)[0]
        times = librosa.times_like(rms, sr=sr)
        
        noise_floor = float(np.percentile(rms, 10))
        quiet_threshold = min(0.05, noise_floor * 2.0)
        active_signal = rms[rms > quiet_threshold]
        loud_threshold = float(np.percentile(active_signal, 70)) if len(active_signal) > 0 else 0.15
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
        if video: video.close()
        if os.path.exists(temp_audio): 
            try: os.remove(temp_audio)
            except: pass

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def parse_time_str(t_str):
    try:
        parts = t_str.split(':')
        minutes = int(parts[0])
        seconds = int(parts[1])
        return minutes * 60 + seconds
    except:
        return 0

def enforce_time_limit(data_list, limit_sec=300):
    cleaned_data = []
    for item in data_list:
        start_sec = parse_time_str(item.get("start", "00:00"))
        end_sec = parse_time_str(item.get("end", "00:00"))
        
        if start_sec >= limit_sec: continue
        if end_sec > limit_sec: item["end"] = "05:00"
        
        cleaned_data.append(item)
    return cleaned_data

def repair_json_string(raw_text):
    raw_text = raw_text.strip()
    if raw_text.endswith(","): raw_text = raw_text[:-1]
    if not raw_text.endswith("]"):
        # Use a safe default that includes moods from the NEW list
        if raw_text.endswith("}"): raw_text += "]"
        else: raw_text += '", "energy_level": "Medium", "cut_pace": "Moderate", "has_speech": false, "mood_tags": ["Chill"]}]'
    return raw_text

# ==========================================
# 4. ROBUST GENERATION ENGINE (NUCLEAR RETRY)
# ==========================================
def generate_captions(video_path, loud_str, quiet_str):
    PROMPT_TEXT = f"""
    You are a professional Music Supervisor.
    ### DATA INPUTS:
    1. **Visuals:** (Analyze the video stream directly)
    2. **Audio Volume (Data):**
       - PHYSICALLY LOUD Segments: [{loud_str}]
       - PHYSICALLY SILENT Segments: [{quiet_str}]
    
    ### TASK:
    Segment the video and determine `energy_level`.
    
    ⛔️ **STRICT CONSTRAINT: 00:00 to 05:00 ONLY**
    - You MUST STOP analyzing at exactly 5 minutes (300 seconds).
    - DO NOT generate any JSON objects for timestamps like 05:01, 05:10, etc.
    
    ### INTEGRATION MATRIX:
    | Visual Action | Audio Volume | **Final Energy Level** |
    | :--- | :--- | :--- |
    | Fast / Chaotic | Loud / Medium | **High** |
    | Fast / Chaotic | Quiet | **High** |
    | Static / Slow | Loud (Shouting) | **High** |
    | Static / Slow | Loud (Wind/Noise) | **Medium** |
    | Static / Slow | Quiet | **Low** |
    | Moderate | Moderate | **Medium** |

    ### RESTRICTED MOOD LIST (MUST use these tags ONLY):
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

    attempt = 0
    base_delay = 20   # Start with 20s wait
    max_delay = 300   # Cap at 5 minutes
    
    # --- OUTER LOOP: HANDLES FILE UPLOADS ---
    while True:
        print(f"   🚀 Uploading file to Gemini (Attempt context refresh)...")
        video_file = client.files.upload(file=video_path)
        
        # Wait for file processing
        while video_file.state.name == "PROCESSING":
            time.sleep(2)
            video_file = client.files.get(name=video_file.name)
        
        if video_file.state.name == "FAILED":
            print("   ❌ File processing failed on Google side. Re-uploading...")
            client.files.delete(name=video_file.name)
            time.sleep(5)
            continue # Go back to outer loop start (re-upload)

        # --- INNER LOOP: HANDLES GENERATION RETRIES ---
        # We try generating 5 times with this specific file upload. 
        # If it fails 5 times, we break inner loop -> delete file -> re-upload (Nuclear Refresh)
        generation_attempts_for_this_upload = 0
        
        while generation_attempts_for_this_upload < 2:
            attempt += 1
            generation_attempts_for_this_upload += 1
            
            try:
                print(f"   🤖 Generating intelligence (Global Attempt {attempt})...")
                response = client.models.generate_content(
                    model="gemini-2.5-flash", 
                    contents=[video_file, PROMPT_TEXT],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.1,
                        max_output_tokens=65536
                    )
                )
                
                # If we get here, SUCCESS!
                client.files.delete(name=video_file.name) # Clean up
                
                # Parse JSON
                raw_text = response.text
                if raw_text.startswith("```json"): raw_text = raw_text[7:]
                if raw_text.startswith("```"): raw_text = raw_text[3:]
                if raw_text.endswith("```"): raw_text = raw_text[:-3]
                
                try:
                    return json.loads(raw_text.strip())
                except json.JSONDecodeError:
                    print("   ⚠️ JSON error. Repairing...")
                    return json.loads(repair_json_string(raw_text))

            except Exception as e:
                error_str = str(e)
                if any(code in error_str for code in ["503", "429", "RESOURCE_EXHAUSTED", "UNAVAILABLE", "Overloaded"]):
                    # Exponential Backoff
                    wait_time = min(max_delay, base_delay * (1.5 ** (attempt - 1)))
                    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                    print(f"   ⚠️ [{timestamp}] Model overloaded. Waiting {int(wait_time)}s...")
                    time.sleep(wait_time)
                else:
                    # Fatal error
                    print(f"   ❌ Fatal Error: {error_str}")
                    client.files.delete(name=video_file.name)
                    raise e
        
        # If we hit this point, we failed 5 times in a row with the SAME uploaded file.
        # It's time to delete it and re-upload to clear any "stuck" state.
        print(f"   ♻️  Failed 5 times. Deleting file and RE-UPLOADING to refresh context...")
        try:
            client.files.delete(name=video_file.name)
        except:
            pass
        time.sleep(10) # Small breather before re-uploading

# ==========================================
# 5. MAIN BATCH PIPELINE
# ==========================================
def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    video_files = glob.glob(os.path.join(VIDEO_DIR, "*.mp4"))
    video_files.sort()
    
    print(f"📂 Found {len(video_files)} videos.")
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
            loud_str, quiet_str = analyze_video_audio(video_path)
            
            # This logic now loops infinitely until success
            data = generate_captions(video_path, loud_str, quiet_str)
            
            original_len = len(data)
            data = enforce_time_limit(data, limit_sec=MAX_DURATION_SEC)
            
            if len(data) < original_len:
                print(f"   ✂️  Trimmed {original_len - len(data)} segments exceeding 05:00.")

            # ENFORCEMENT: Filter and capitalize mood tags based on the updated MOOD_LIST
            for seg in data:
                if not seg.get('mood_tags'): seg['mood_tags'] = ["Chill"]
                # This ensures any mood generated by the model is one of the approved tags
                seg['mood_tags'] = [t.capitalize() for t in seg['mood_tags'] if t.capitalize() in MOOD_LIST]

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"   💾 Saved to: {output_path}")

            if i < len(video_files) - 1:
                print(f"   💤 Cooling down for {API_COOLDOWN_SECONDS}s...")
                time.sleep(API_COOLDOWN_SECONDS)

        except Exception as e:
            # Only FATAL errors (like Auth failures) will break the infinite loop above
            print(f"   ❌ UNRECOVERABLE FAILURE: {str(e)}")
            continue

    # Merge
    print("\n" + "="*50)
    print("🔄 Merging...")
    master_data = {}
    json_files = glob.glob(os.path.join(OUTPUT_DIR, "*.json"))
    for jf in json_files:
        key = os.path.basename(jf).replace(".json", ".mp4")
        with open(jf, 'r', encoding='utf-8') as f:
            master_data[key] = json.load(f)
            
    with open(MASTER_FILE, 'w', encoding='utf-8') as f:
        json.dump(master_data, f, indent=2, ensure_ascii=False)
        
    print(f"🎉 DONE! Saved to: {MASTER_FILE}")

if __name__ == "__main__":
    main()