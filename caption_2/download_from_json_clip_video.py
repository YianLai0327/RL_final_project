import json
import os
import subprocess
import threading
import re
from concurrent.futures import ThreadPoolExecutor
from pytubefix import YouTube
from tqdm import tqdm

# Settings
DOWNLOAD_DIR = "video_library_all"
CLIP_START = "00:00:00"
CLIP_DURATION = "00:05:00"  # 5 minutes
MAX_WORKERS = 4

# Globals for progress tracking
progress_bars = {}
lock = threading.Lock()

def sanitize_filename(name):
    """Clean the video title to be a valid filename."""
    return "".join([c for c in name if c.isalpha() or c.isdigit() or c in ' .-_']).strip()

def download_video_clip(url, idx):
    # Setup simple progress bar (0 -> 100 on completion)
    # Note: We can't easily track granular progress with ffmpeg via subprocess 
    # without complex parsing, so we use a spinner-style bar.
    with lock:
        progress_bars[idx] = tqdm(total=100, desc=f"Video {idx}", position=idx, leave=True)

    try:
        # 1. Handle URL format (fix if it's just an ID)
        if "youtube.com" not in url and "youtu.be" not in url:
            full_url = f"https://www.youtube.com/watch?v={url}"
        else:
            full_url = url

        # 2. Get the stream URL using pytubefix
        yt = YouTube(full_url)
        # 'get_highest_resolution' gets the progressive stream (video+audio, usually 720p)
        stream = yt.streams.get_highest_resolution()
        
        if not stream:
            raise ValueError("No suitable stream found")

        # 3. Prepare Filename
        title = sanitize_filename(yt.title)
        output_path = os.path.join(DOWNLOAD_DIR, f"{title}.mp4")

        # 4. Use FFMPEG to download just the clip
        # -ss: Start time
        # -t: Duration
        # -i: Input URL
        # -c copy: Copy stream (fast, no re-encoding)
        cmd = [
            "ffmpeg", 
            "-y",               # Overwrite if exists
            "-ss", CLIP_START, 
            "-i", stream.url, 
            "-t", CLIP_DURATION, 
            "-c", "copy",       # Fast copy (may cut at nearest keyframe)
            "-loglevel", "error", # Quiet mode
            output_path
        ]

        # Execute
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        if result.returncode != 0:
            raise Exception(f"FFmpeg Error: {result.stderr.decode('utf-8')}")

        # 5. Mark as done
        with lock:
            progress_bars[idx].n = 100
            progress_bars[idx].refresh()
            progress_bars[idx].close()

    except Exception as e:
        with lock:
            # Show error in the progress bar description if possible
            progress_bars[idx].set_description(f"Err {idx}: {str(e)[:15]}...")
            progress_bars[idx].close()
        print(f"\n❌ Failed ID {idx}: {url}\nReason: {e}")

def main():
    # Ensure output directory exists
    if not os.path.exists(DOWNLOAD_DIR):
        os.makedirs(DOWNLOAD_DIR)

    # Load JSON
    json_path = "crawl_v3.json"
    if not os.path.exists(json_path):
        print(f"File not found: {json_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Extract IDs/URLs
    urls = [item["videoId"] for item in data if "videoId" in item]
    print(f"Found {len(urls)} videos to process.")

    # Multi-threaded Download
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for idx, url in enumerate(urls):
            executor.submit(download_video_clip, url, idx)

if __name__ == "__main__":
    main()