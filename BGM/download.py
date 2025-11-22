import os
import time
import csv
import urllib.parse
import warnings

import requests
from bs4 import BeautifulSoup
import librosa
import numpy as np

# Suppress librosa warnings to keep output clean
warnings.filterwarnings('ignore')

# === Config ===
CATEGORY_PAGES = [
    ("upbeat",     "https://freepd.com/upbeat.php"),
    ("epic",       "https://freepd.com/epic.php"),
    ("horror",     "https://freepd.com/horror.php"),
    ("romantic",   "https://freepd.com/romantic.php"),
    ("comedy",     "https://freepd.com/comedy.php"),
    ("world",      "https://freepd.com/world.php"),
    ("scoring",    "https://freepd.com/scoring.php"),
    ("electronic", "https://freepd.com/electronic.php"),
    ("misc",       "https://freepd.com/misc.php"),
]

DEST_DIR = "bgm_library"         # folder for mp3s
OUT_CSV = "audio_features.csv"   # metadata + features
REQUEST_DELAY = 0.5              # seconds between HTTP requests
MAX_PER_CATEGORY = 10             # 🔹 Target: Get 6 SUCCESSFUL tracks per category

BASE_MOOD_MAP = {
    "upbeat":     "happy",
    "epic":       "epic",
    "horror":     "dark",
    "romantic":   "romantic",
    "comedy":     "playful",
    "world":      "world",
    "scoring":    "cinematic",
    "electronic": "electronic",
    "misc":       "neutral",
}


# === HTTP helpers ===
def download_file(url, dest_dir):
    """
    Download a single file from url into dest_dir.
    Returns the local path if successful, None otherwise.
    """
    if not url:
        return None
        
    os.makedirs(dest_dir, exist_ok=True)

    path = urllib.parse.urlparse(url).path
    filename = os.path.basename(path)
    if not filename:
        filename = f"audio_{int(time.time() * 1000)}.mp3"

    dest_path = os.path.join(dest_dir, filename)

    if os.path.exists(dest_path):
        # Even if it exists, we return the path so we can check if it's a valid audio file
        print(f"[SKIP] {filename} already exists")
        return dest_path

    print(f"[DOWNLOAD] {url} -> {dest_path}")
    try:
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            # Check content length if available
            if 'Content-Length' in r.headers and int(r.headers['Content-Length']) < 1000:
                 print(f"  [SKIP] File too small (likely invalid): {url}")
                 return None
                 
            with open(dest_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
    except Exception as e:
        print(f"  [ERROR] Failed to download {url}: {e}")
        return None

    time.sleep(REQUEST_DELAY)
    return dest_path


def get_mp3_links(page_url):
    """
    Parse a FreePD category page and return ALL available mp3 URLs.
    """
    print(f"[CRAWL] {page_url}")
    try:
        resp = requests.get(
            page_url,
            timeout=20,
            headers={"User-Agent": "freepd-music-crawler/1.0"},
        )
        resp.raise_for_status()
    except Exception as e:
        print(f"  [ERROR] Could not fetch page {page_url}: {e}")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    links = []

    for a in soup.find_all("a", href=True):
        href = a["href"]
        if ".mp3" in href.lower():
            abs_url = urllib.parse.urljoin(page_url, href)
            links.append(abs_url)

    print(f"  Found {len(links)} total mp3 links on page.")
    return links


# === Audio feature extraction & mood tagging ===
def extract_features(path):
    """
    Extract tempo (BPM), MFCC mean vector, RMS (energy), and spectral centroid.
    Analyzes first 30 seconds.
    """
    try:
        # Load audio
        y, sr = librosa.load(path, sr=None, mono=True, duration=30.0)
    except Exception as e:
        print(f"  [ERROR] Librosa load failed for {path}: {e}")
        return 0, None, 0, 0

    # 1. Tempo
    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        if np.ndim(tempo) > 0:
            tempo = tempo.item()
    except Exception:
        tempo = 0.0

    # 2. MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = mfcc.mean(axis=1)

    # 3. RMS Energy
    rms = librosa.feature.rms(y=y).mean()

    # 4. Spectral Centroid
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()

    return float(tempo), mfcc_mean, float(rms), float(centroid)


def classify_mood(category_tag, bpm, rms):
    """
    Returns mood_tag and energy based on optimized thresholds.
    """
    base_mood = BASE_MOOD_MAP.get(category_tag, "neutral")

    # Optimized Thresholds from data analysis
    if bpm >= 128 or rms >= 0.14:
        energy = "high"
    elif bpm >= 110 or rms >= 0.075:
        energy = "medium"
    else:
        energy = "low"

    mood_tag = f"{base_mood}_{energy}"
    return mood_tag, energy


def init_csv(filename):
    """Initializes CSV with headers if it doesn't exist."""
    if not os.path.exists(filename):
        headers = [
            "filename", "base_mood", "mood_tag", "energy", 
            "bpm", "rms", "spectral_centroid"
        ]
        # Add MFCC headers
        for i in range(1, 14):
            headers.append(f"mfcc_{i}")
            
        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()


def append_to_csv(filename, row):
    """Appends a single row to the CSV."""
    with open(filename, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        writer.writerow(row)


def main():
    # Initialize CSV file with headers
    init_csv(OUT_CSV)
    
    for tag, url in CATEGORY_PAGES:
        print(f"\n=== CATEGORY: {tag} ===")
        
        # Get ALL links for this category
        all_links = get_mp3_links(url)
        
        success_count = 0
        
        # Loop through ALL links until we get enough successful ones
        for audio_url in all_links:
            if success_count >= MAX_PER_CATEGORY:
                print(f"  [INFO] Reached target of {MAX_PER_CATEGORY} tracks for {tag}. Moving to next category.")
                break

            local_path = download_file(audio_url, DEST_DIR)
            if local_path is None:
                continue

            try:
                bpm, mfcc_mean, rms, centroid = extract_features(local_path)
            except Exception as e:
                print(f"  [ERROR] Feature extraction failed for {local_path}: {e}")
                continue
            
            # Skip invalid BPM
            if bpm <= 0.1:
                print(f"  [WARN] Skipping {os.path.basename(local_path)} (Invalid BPM: {bpm})")
                continue

            mood_tag, energy = classify_mood(tag, bpm, rms)

            row = {
                "filename": os.path.basename(local_path),
                "base_mood": BASE_MOOD_MAP.get(tag, "neutral"),
                "mood_tag": mood_tag,
                "energy": energy,
                "bpm": bpm,
                "rms": rms,
                "spectral_centroid": centroid,
            }
            
            for i, v in enumerate(mfcc_mean):
                row[f"mfcc_{i+1}"] = float(v)

            # SAVE IMMEDIATELY
            append_to_csv(OUT_CSV, row)
            success_count += 1
            print(f"  [SUCCESS] Processed {success_count}/{MAX_PER_CATEGORY}: {mood_tag}")

        if success_count < MAX_PER_CATEGORY:
            print(f"  [WARN] Could only find {success_count} valid tracks for {tag} (wanted {MAX_PER_CATEGORY}).")

    print(f"\n[DONE] Finished processing. Data saved to {OUT_CSV}")


if __name__ == "__main__":
    main()