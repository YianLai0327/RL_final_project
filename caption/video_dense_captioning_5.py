import time
import json
import os
from google import genai
from google.genai import types

# 1. CONFIGURATION
# ---------------------------------------------------------
API_KEY = "" 
VIDEO_PATH = "./data/test.mp4"
OUTPUT_PATH = "./data/vlog_captions.json"

client = genai.Client(api_key=API_KEY)

# 2. THE RESTRICTED VOCABULARY
# ---------------------------------------------------------
# We define this here so we can easily inject it into the prompt
MOOD_LIST = ["Happy", "Sad", "Epic", "Chill", "Tense", "Romantic", "Upbeat", "Dark", "Funny", "Sentimental"]

PROMPT_TEXT = f"""
You are a professional Music Supervisor logging footage for an automated DJ.
Your goal is to segment the video into "Micro-Events" (5-20 seconds) and assign a specific mood tag.

### THE RESTRICTED MOOD LIST:
You are STRICTLY limited to choosing from this list. Do not invent new words.
{json.dumps(MOOD_LIST)}

### INSTRUCTIONS:
Analyze the video for visual and energetic shifts. 
For each distinct segment, output a JSON object with these exact keys:

1.  `start`: "MM:SS"
2.  `end`: "MM:SS"
3.  `visual_caption`: Brief description of action (e.g. "Running through terminal").
4.  `scene_category`: "Dialogue" | "Montage" | "Static" | "Transit".
5.  `energy`: "Low" | "Medium" | "High".
6.  `suggested_genre`: A specific genre (e.g. "Lo-Fi", "Synthwave").
7.  `mood_tags`: **Select exactly 1 or 2 tags** from the MOOD LIST above that best fit the scene.

### ALIGNMENT RULES:
- **Dialogue Scenes:** Usually map to "Chill" or "Sentimental".
- **Action/Montage:** Usually map to "Upbeat", "Epic", or "Tense".
- **Jokes/Vlogging:** Usually map to "Funny" or "Happy".

### SEGMENTATION TRIGGERS:
- Create a new segment whenever the mood shifts from one category to another (e.g., from "Happy" to "Tense").
"""

# 3. PROCESSING FUNCTION
# ---------------------------------------------------------
def process_video_timeline(file_path):
    print(f"Uploading {file_path} to Gemini...")
    
    video_file = client.files.upload(file=file_path)
    
    while video_file.state.name == "PROCESSING":
        print(".", end="", flush=True)
        time.sleep(2)
        video_file = client.files.get(name=video_file.name)
    
    if video_file.state.name == "FAILED":
        raise ValueError("Video processing failed.")
    print("\nVideo processed. Generating Fixed-Mood Metadata...")

    response = client.models.generate_content(
        model="gemini-2.0-flash", 
        contents=[video_file, PROMPT_TEXT],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            max_output_tokens=66536,
            temperature=0.1 # Low temperature forces adherence to the list
        )
    )

    client.files.delete(name=video_file.name)
    return json.loads(response.text)

# 4. EXECUTION
# ---------------------------------------------------------
if __name__ == "__main__":
    try:
        if not os.path.exists("./data"):
            os.makedirs("./data")

        timeline_data = process_video_timeline(VIDEO_PATH)
        
        # VALIDATION STEP: Check if AI followed the rules
        print("\n--- Validating Mood Tags ---")
        valid_moods = set(MOOD_LIST)
        
        for i, seg in enumerate(timeline_data):
            # Force tags to match our list just in case of capitalization errors
            cleaned_tags = [t.capitalize() for t in seg.get('mood_tags', [])]
            
            # Filter out invalid tags
            valid_tags = [t for t in cleaned_tags if t in valid_moods]
            
            # If AI failed, default to 'Chill' (Safe fallback)
            if not valid_tags:
                valid_tags = ["Chill"]
                print(f"⚠️ Warning: Segment {i} had invalid tags. Defaulting to 'Chill'.")
            
            seg['mood_tags'] = valid_tags

        # Save to file
        with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
            json.dump(timeline_data, f, indent=2, ensure_ascii=False)
            
        print(f"✅ Generated {len(timeline_data)} segments with strict mood validation.")
        print(f"💾 Saved to: {OUTPUT_PATH}")

    except Exception as e:
        print(f"❌ Error: {e}")