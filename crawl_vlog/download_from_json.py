import json
from pytubefix import YouTube
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import threading

# 用來存放各影片的進度條
progress_bars = {}
lock = threading.Lock()

def download_video(url, idx):
    # 建立進度條（偽，等知道影片大小後更新）
    progress_bars[idx] = tqdm(total=100, desc=f"影片 {idx}", position=idx, leave=True)

    def on_progress(stream, chunk, bytes_remaining):
        total_size = stream.filesize
        bytes_downloaded = total_size - bytes_remaining
        progress = int(bytes_downloaded / total_size * 100)

        with lock:
            progress_bars[idx].n = progress
            progress_bars[idx].refresh()

    try:
        yt = YouTube(url, on_progress_callback=on_progress)
        stream = yt.streams.get_highest_resolution()
        stream.download(output_path="downloads_v1")

        with lock:
            progress_bars[idx].n = 100
            progress_bars[idx].refresh()
            progress_bars[idx].close()

    except Exception as e:
        print(f"下載失敗: {url}, 原因: {e}")


def main():
    # 讀取 JSON
    with open("crawl_v1.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    urls = [item["videoId"] for item in data if "videoId" in item]

    for idx in range(len(urls)):
        urls[idx] = "https://www.youtube.com/watch?v=" + urls[idx]

    # 使用多執行緒下載
    max_workers = 4  # 可自行調整（別設太高 YouTube 會限速）
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for idx, url in enumerate(urls):
            executor.submit(download_video, url, idx)


if __name__ == "__main__":
    main()
