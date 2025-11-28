import json
import os
from pathlib import Path
import google.generativeai as genai
from pydub import AudioSegment
import tempfile
from typing import List, Dict
import time

# 設定你的 Gemini API key
GEMINI_API_KEY = ""
genai.configure(api_key=GEMINI_API_KEY)

# PROMPT_TEXT = """
# You are a Music Supervisor analyzing a background music segment from a vlog.
# You are listening to a **BGM segment** that plays during a specific part of the video.

# Analyze this audio clip and output a JSON object with exactly these fields:

# 1. "mood_tags": A list of exactly 2 adjectives from this allowed list: 
#    ["Happy", "Sad", "Epic", "Chill", "Tense", "Romantic", "Upbeat", "Dark", "Funny", "Sentimental"]
# 2. "energy": ("Low", "Medium", "High")
# 3. "instrumentation": Key instruments heard (e.g., "Acoustic Guitar, Piano", "Synth, Bass, Drums")
# 4. "has_vocals": (true/false) -> Is there ANY human voice (singing/rapping/humming)?
# 5. "loop_suitability": (1-10) -> How repetitive/steady? (10 = Perfect steady loop, 1 = Too many changes/stops)
# 6. "suggested_genre": The specific sub-genre (e.g., "Lo-fi Hip Hop", "Epic Orchestral", "Tech House")
# 7. "video_pairing": A one-sentence description of the *perfect* video scene for this track (e.g., "Fast-paced travel montage" or "Sad emotional dialogue")

# Output ONLY the JSON object. No markdown formatting, no code blocks, just the raw JSON.
# """

PROMPT_TEXT = """
You are a Music Supervisor creating BGM descriptions for a video segment.
You are analyzing a **background music segment** that should match a specific scene.

Given the scene information:
- Visual description: {visual_caption}
- Scene category: {scene_category}
- Duration: {duration} seconds
- Current energy level: {energy}
- Mood tags: {mood_tags}

Output a JSON object with exactly these fields:

1. "start_time": (The segment start time in seconds)
2. "end_time": (The segment end time in seconds)
3. "scene_type": (The scene category: Dialogue, Montage, Transit, Static, or Action)
4. "mood_tags": Keep the original mood tags: {mood_tags}
5. "energy": Keep the original energy level: {energy}
6. "music_description": A concise description of the ideal BGM (e.g., "Light acoustic guitar with soft percussion", "Upbeat electronic beat with synth melody")
7. "instrumentation": Suggested key instruments (e.g., "Acoustic Guitar, Light Percussion", "Synth, Bass, Drums")
8. "has_vocals": (true/false) -> Should this segment have vocals? (Generally false for most BGM, true for intro/outro or special moments)
9. "tempo": (Slow, Medium, Fast) -> Suggested tempo based on scene pacing
10. "transition_type": ("Fade", "Cut", "Crossfade", "Continue") -> How should this transition from previous segment?
11. "suggested_genre": The specific genre from: {suggested_genre}
12. "prominence": ("Background", "Foreground", "Ambient") -> How prominent should music be?
    - "Background": Dialogue scenes, music stays subtle
    - "Foreground": Montages, action, music drives the scene
    - "Ambient": Scenic shots, music creates atmosphere
13. "reference_style": A brief style reference (e.g., "Similar to travel vlog BGM", "Like lo-fi study music", "Cinematic documentary style")

Output ONLY the JSON object.
"""

def time_to_seconds(time_str: str) -> float:
    """將時間字串 (MM:SS) 轉換為秒數"""
    parts = time_str.split(':')
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    elif len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    else:
        return float(time_str)

def extract_audio_segment(audio_path: str, start_time: float, end_time: float, output_path: str):
    """從音訊檔案中提取指定時間段"""
    audio = AudioSegment.from_file(audio_path)
    
    # 轉換為毫秒
    start_ms = int(start_time * 1000)
    end_ms = int(end_time * 1000)
    
    # 提取片段
    segment = audio[start_ms:end_ms]
    
    # 匯出
    segment.export(output_path, format="mp3")
    print(f"  ✓ Extracted segment: {start_time}s - {end_time}s")

def caption_audio_with_gemini(audio_path: str, model_name: str = "gemini-2.0-flash-exp") -> Dict:
    """使用 Gemini 為音訊片段生成 caption"""
    try:
        # 上傳音訊檔案
        print(f"  ⟳ Uploading audio to Gemini...")
        audio_file = genai.upload_file(audio_path)
        
        # 等待檔案處理完成
        while audio_file.state.name == "PROCESSING":
            time.sleep(1)
            audio_file = genai.get_file(audio_file.name)
        
        if audio_file.state.name == "FAILED":
            raise ValueError("Audio file processing failed")
        
        print(f"  ✓ Audio uploaded successfully")
        
        # 建立模型並生成 caption
        model = genai.GenerativeModel(model_name)
        response = model.generate_content([audio_file, PROMPT_TEXT])
        
        # 解析 JSON 回應
        response_text = response.text.strip()
        
        # 移除可能的 markdown 格式
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        
        caption_data = json.loads(response_text.strip())
        
        # 清理上傳的檔案
        genai.delete_file(audio_file.name)
        
        return caption_data
        
    except Exception as e:
        print(f"  ✗ Error captioning audio: {e}")
        return None

def process_vlog_bgm(video_caption_json_path: str, bgm_audio_path: str, output_json_path: str):
    """
    處理整個 vlog BGM，為每個片段生成 audio caption
    
    Args:
        video_caption_json_path: 包含場景資訊的 JSON 檔案路徑
        bgm_audio_path: 背景音樂音訊檔案路徑 (已做 source separation)
        output_json_path: 輸出的 audio caption JSON 檔案路徑
    """
    # 讀取場景資訊
    print(f"📖 Loading video captions from: {video_caption_json_path}")
    with open(video_caption_json_path, 'r', encoding='utf-8') as f:
        scenes = json.load(f)
    
    print(f"📊 Found {len(scenes)} scenes to process")
    print(f"🎵 BGM audio file: {bgm_audio_path}")
    print()
    
    # 建立臨時目錄存放音訊片段
    temp_dir = tempfile.mkdtemp()
    print(f"📁 Temporary directory: {temp_dir}")
    print()
    
    results = []
    
    for idx, scene in enumerate(scenes, 1):
        print(f"[{idx}/{len(scenes)}] Processing scene: {scene['start']} - {scene['end']}")
        print(f"  Scene: {scene['visual_caption'][:60]}...")
        
        # 轉換時間
        start_sec = time_to_seconds(scene['start'])
        end_sec = time_to_seconds(scene['end'])
        
        # 提取音訊片段
        segment_path = os.path.join(temp_dir, f"segment_{idx:03d}.mp3")
        extract_audio_segment(bgm_audio_path, start_sec, end_sec, segment_path)
        
        # 使用 Gemini 生成 caption
        print(f"  🤖 Generating audio caption with Gemini...")
        caption = caption_audio_with_gemini(segment_path)
        
        if caption:
            # 合併原始場景資訊和音訊 caption
            result = {
                "start": scene['start'],
                "end": scene['end'],
                "visual_caption": scene['visual_caption'],
                "scene_category": scene['scene_category'],
                "audio_caption": caption
            }
            results.append(result)
            print(f"  ✓ Caption generated successfully")
            print(f"    Mood: {caption.get('mood_tags', [])}")
            print(f"    Genre: {caption.get('suggested_genre', 'N/A')}")
        else:
            print(f"  ✗ Failed to generate caption")
        
        print()
        
        # 刪除臨時音訊片段
        if os.path.exists(segment_path):
            os.remove(segment_path)
        
        # 避免 API rate limit
        time.sleep(1)
    
    # 清理臨時目錄
    os.rmdir(temp_dir)
    
    # 儲存結果
    print(f"💾 Saving results to: {output_json_path}")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Processing complete! Generated {len(results)} audio captions")

if __name__ == "__main__":
    # 設定檔案路徑
    VIDEO_CAPTION_JSON = "Vlog Captions 2.json"  # 你的場景 JSON 檔案
    BGM_AUDIO_FILE = "../test_audio.mp3"         # 分離出來的背景音樂
    OUTPUT_JSON = "audio_captions_2.json"          # 輸出的 audio caption
    
    # 檢查檔案是否存在
    if not os.path.exists(VIDEO_CAPTION_JSON):
        print(f"❌ Error: Video caption JSON not found: {VIDEO_CAPTION_JSON}")
        exit(1)
    
    if not os.path.exists(BGM_AUDIO_FILE):
        print(f"❌ Error: BGM audio file not found: {BGM_AUDIO_FILE}")
        exit(1)
    
    # 處理
    process_vlog_bgm(VIDEO_CAPTION_JSON, BGM_AUDIO_FILE, OUTPUT_JSON)