import json
import os
import subprocess
import threading
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pytubefix import YouTube
from tqdm import tqdm
from deep_translator import GoogleTranslator

# Settings
DOWNLOAD_DIR = "video_library_large"
CLIP_START = "00:00:00"
CLIP_DURATION = "00:05:00"  # 5 minutes
MAX_WORKERS = 4
TOTAL_VIDEO_LIMIT = 50

# Globals for progress tracking
progress_bars = {}
lock = threading.Lock()

def sanitize_filename(name):
    """Clean the video title to be a valid filename."""
    # Only allow safe characters: alphanumerics, space, dot, dash, underscore
    return "".join([c for c in name if c.isalpha() or c.isdigit() or c in ' .-_']).strip()

def contains_chinese(text):
    """Check if the text contains Chinese characters."""
    # Matches common CJK Unified Ideographs
    return bool(re.search(r'[\u4e00-\u9fff]', text))

def translate_title(title):
    """Translate Chinese titles to English."""
    try:
        # IF Chinese is detected, FORCE the source to be Traditional Chinese (zh-TW)
        # This prevents the 'auto' detector from mistakenly thinking it's English.
        if contains_chinese(title):
            translated = GoogleTranslator(source='zh-TW', target='en').translate(title)
            return translated
        return title
    except Exception as e:
        print(f"Translation failed for {title}: {e}")
        return title

def download_and_rename(video_data, idx):
    video_id = video_data['videoId']
    original_title = video_data.get('title', 'Unknown')
    
    with lock:
        progress_bars[idx] = tqdm(total=100, desc=f"Video {idx}", position=idx, leave=True)

    try:
        # --- STEP 1: PREPARE URL AND TEMP PATH ---
        if "youtube.com" not in video_id and "youtu.be" not in video_id:
            full_url = f"https://www.youtube.com/watch?v={video_id}"
        else:
            full_url = video_id
            if "v=" in full_url:
                video_id = full_url.split("v=")[1].split("&")[0]

        # Use Video ID as the safe temp filename
        temp_filename = f"{video_id}.mp4"
        temp_path = os.path.join(DOWNLOAD_DIR, temp_filename)

        # --- STEP 2: DOWNLOAD ---
        yt = YouTube(full_url)
        stream = yt.streams.get_highest_resolution()
        
        if not stream:
            raise ValueError("No suitable stream found")

        cmd = [
            "ffmpeg", 
            "-y",
            "-ss", CLIP_START, 
            "-i", stream.url, 
            "-t", CLIP_DURATION, 
            "-c", "copy",
            "-loglevel", "error", 
            temp_path
        ]

        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        if result.returncode != 0:
            raise Exception(f"FFmpeg Error: {result.stderr.decode('utf-8')}")

        # --- STEP 3: TRANSLATE AND RENAME ---
        with lock:
            progress_bars[idx].set_description("Translating...")

        # Determine the title to translate
        title_to_process = original_title if original_title != 'Unknown' else yt.title
        
        # Translate (Forcing zh-TW -> en)
        final_title_str = translate_title(title_to_process)
        
        # Sanitize filename
        safe_filename = sanitize_filename(final_title_str)
        if not safe_filename:
            safe_filename = f"video_{idx}"
            
        final_path = os.path.join(DOWNLOAD_DIR, f"{safe_filename}.mp4")
        
        # Rename the file
        if os.path.exists(final_path):
            os.remove(final_path)
        os.rename(temp_path, final_path)

        with lock:
            progress_bars[idx].set_description(f"Done: {safe_filename[:10]}...")
            progress_bars[idx].n = 100
            progress_bars[idx].refresh()
            progress_bars[idx].close()

    except Exception as e:
        with lock:
            progress_bars[idx].set_description(f"Err {idx}: {str(e)[:10]}...")
            progress_bars[idx].close()
        print(f"\n❌ Failed ID {idx} ({video_id}): {e}")

def select_balanced_videos(data, limit):
    """
    Selects videos ensuring variety across uploaders.
    """
    videos_by_uploader = defaultdict(list)
    for item in data:
        uploader = item.get("uploader", "Unknown")
        videos_by_uploader[uploader].append(item)
    
    uploaders = list(videos_by_uploader.keys())
    selected_videos = []
    
    print(f"Found {len(uploaders)} unique uploaders.")
    
    while len(selected_videos) < limit:
        added_in_this_round = 0
        for uploader in uploaders:
            if len(selected_videos) >= limit:
                break
            
            if videos_by_uploader[uploader]:
                video = videos_by_uploader[uploader].pop(0)
                selected_videos.append(video)
                added_in_this_round += 1
        
        if added_in_this_round == 0:
            break
            
    return selected_videos

def main():
    if not os.path.exists(DOWNLOAD_DIR):
        os.makedirs(DOWNLOAD_DIR)

    json_path = "crawl_v3_all.json"
    if not os.path.exists(json_path):
        json_path = "crawl_v3.json" 
        if not os.path.exists(json_path):
            print("❌ JSON file not found.")
            return

    print(f"Reading {json_path}...")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    target_videos = select_balanced_videos(data, TOTAL_VIDEO_LIMIT)
    
    print(f"Selected {len(target_videos)} videos from {len(data)} total entries.")
    print("Starting download -> Translate (Source: zh-TW) -> Rename...")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for idx, video_data in enumerate(target_videos):
            executor.submit(download_and_rename, video_data, idx)

if __name__ == "__main__":
    main()