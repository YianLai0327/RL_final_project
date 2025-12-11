# Video & Audio Captioning

##  Requirements & Installation

First, ensure you have Python 3.8+ installed. You also need **FFmpeg** installed on your system for audio/video processing.

**Install Python Dependencies:**
```bash
pip install numpy librosa soundfile pychorus moviepy pytubefix tqdm google-genai deep-translator
```

##  Execution Instructions
Run the scripts in the following order to generate the datasets and train the RL agent.

Phase 1: Music Data Pipeline
1. Download & Analyze Music Crawls royalty-free BGM and calculates BPM/Energy features.
```bash
python download_raw_BGM.py
```
2. Select Best Tracks Filters the library to keep the top ~100 high-quality tracks (balanced by mood).
```bash
python select_best_tracks.py
```
3. Chorus Extraction Trims tracks to start exactly at the "hook" (chorus) for immediate impact.
```bash
python extract_chorus_start_only.py
```
  &emsp;&emsp;(Optional: If you prefer looping tracks, run python extract_chorus_looping.py instead.)

4. Audio Captioning (Gemini) Generates semantic descriptions (e.g., "High energy rock...") for the RL agent.
```bash
python audio_captioning_dynamic.py
```

Phase 2: Video Data Pipeline
1. Download Videos Downloads clips from crawl_v3.json, renames them by Video ID, and translates titles to English.
```bash
python download_from_json_clip_2.py
```
2. Dense Video Captioning (Gemini) Extracts visual mood and audio atmosphere from the clips to create the "Ground Truth" for the RL environment.
```bash
python video_captioning_dynamic_final.py
```

