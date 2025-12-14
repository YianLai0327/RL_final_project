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

# ==========================================
# 1. SETTINGS
# ==========================================
DOWNLOAD_DIR = "video_library_large"
CLIP_START = "00:00:00"
CLIP_DURATION = "00:05:00"  # 5 minutes
MAX_WORKERS = 4
TOTAL_VIDEO_LIMIT = 50
TRANSLATION_LOG_FILE = "translation_log.json"

# ⚠️ NEW: Maximum length for the filename (prevent Windows path errors)
MAX_FILENAME_LEN = 80  

# Globals
progress_bars = {}
translation_log = {}
lock = threading.Lock()

def sanitize_filename(name):
    """
    Clean the video title to be a valid filename AND truncate it.
    """
    # 1. Remove unsafe characters
    clean_name = "".join([c for c in name if c.isalpha() or c.isdigit() or c in ' .-_']).strip()
    
    # 2. Collapse multiple spaces
    clean_name = re.sub(r'\s+', ' ', clean_name)
    
    # 3. Truncate to safe limit (preserve extension space later)
    if len(clean_name) > MAX_FILENAME_LEN:
        clean_name = clean_name[:MAX_FILENAME_LEN].strip()
        
    return clean_name

def contains_chinese(text):
    """Check if the text contains Chinese characters."""
    return bool(re.search(r'[\u4e00-\u9fff]', text))

def translate_title(title):
    """Translate Chinese titles to English."""
    try:
        # Force Traditional Chinese (zh-TW) detection
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
        # --- STEP 1: PREPARE URL ---
        if "youtube.com" not in video_id and "youtu.be" not in video_id:
            full_url = f"https://www.youtube.com/watch?v={video_id}"
        else:
            full_url = video_id
            if "v=" in full_url:
                video_id = full_url.split("v=")[1].split("&")[0]

        # Use Video ID as the safe temp filename first
        temp_filename = f"{video_id}.mp4"
        temp_path = os.path.join(DOWNLOAD_DIR, temp_filename)

        # --- STEP 2: DOWNLOAD CLIP ---
        yt = YouTube(full_url)
        stream = yt.streams.get_highest_resolution()
        
        if not stream:
            raise ValueError("No suitable stream found")

        # Download clip (5 mins) using FFmpeg
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

        # --- STEP 3: TRANSLATE & RENAME ---
        with lock:
            progress_bars[idx].set_description("Translating...")

        title_to_process = original_title if original_title != 'Unknown' else yt.title
        final_title_str = translate_title(title_to_process)
        
        # Save full mapping to log (Full English Title)
        with lock:
            translation_log[title_to_process] = final_title_str

        # Generate SHORT safe filename
        safe_filename = sanitize_filename(final_title_str)
        if not safe_filename:
            safe_filename = f"video_{idx}"
            
        final_path = os.path.join(DOWNLOAD_DIR, f"{safe_filename}.mp4")
        
        # Rename
        if os.path.exists(final_path):
            os.remove(final_path)
        os.rename(temp_path, final_path)

        with lock:
            # Show truncated name in progress bar
            display_name = (safe_filename[:15] + '..') if len(safe_filename) > 15 else safe_filename
            progress_bars[idx].set_description(f"Done: {display_name}")
            progress_bars[idx].n = 100
            progress_bars[idx].refresh()
            progress_bars[idx].close()

    except Exception as e:
        with lock:
            progress_bars[idx].set_description(f"Err {idx}: {str(e)[:10]}...")
            progress_bars[idx].close()
        print(f"\n❌ Failed ID {idx} ({video_id}): {e}")

def select_balanced_videos(data, limit):
    """Selects videos ensuring variety across uploaders."""
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
            if len(selected_videos) >= limit: break
            
            if videos_by_uploader[uploader]:
                video = videos_by_uploader[uploader].pop(0)
                selected_videos.append(video)
                added_in_this_round += 1
        
        if added_in_this_round == 0: break
            
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
    
    print(f"Selected {len(target_videos)} videos.")
    print(f"⚠️ Filenames will be truncated to {MAX_FILENAME_LEN} chars.")
    print("Starting download...")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for idx, video_data in enumerate(target_videos):
            executor.submit(download_and_rename, video_data, idx)

    print(f"Saving translation log to {TRANSLATION_LOG_FILE}...")
    with open(TRANSLATION_LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(translation_log, f, ensure_ascii=False, indent=4)
    print("All tasks completed.")

if __name__ == "__main__":
    main()