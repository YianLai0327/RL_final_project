# Video & Audio Captioning

## Requirements & Installation

First, ensure you have Python 3.8+ installed. You also need **FFmpeg** installed on your system for audio/video processing.

**Install Python Dependencies:**
```bash
pip install numpy librosa soundfile pychorus moviepy pytubefix tqdm google-genai deep-translator
```

*Note: You will need a Google Gemini API Key to run the captioning scripts (audio_captioning.py and video_captioning.py).*

##  Execution Instructions
Run the scripts in the following order to generate the datasets. 

*Note: The scripts below will output their final JSON/Audio files into the data/ directory.*

### Phase 1: Music Data Pipeline
1. Download & Analyze Music Crawls royalty-free BGM and calculates BPM/Energy features.
```bash
python download_BGM.py
```
2. Chorus Extraction Trims tracks to start exactly at the "hook" (chorus) for immediate impact.
```bash
python extract_chorus_start.py
```
3. Audio Captioning (Gemini) Generates semantic descriptions (e.g., "High energy rock...") for the RL agent.
```bash
python audio_captioning.py
```

### Phase 2: Video Data Pipeline
1. Download Videos Downloads clips from crawl_v3_all.json, renames them by Video ID, and translates titles to English.
```bash
python download_video.py
```
2. Dense Video Captioning (Gemini) Extracts visual mood and audio atmosphere from the clips to create the "Ground Truth" for the RL environment.
```bash
python video_captioning.py
```
