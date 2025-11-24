--------------------------------------------------------------------------------------------------------------------------
BGM Emotion Extractor (Valence–Arousal + Mood Tags)
--------------------------------------------------------------------------------------------------------------------------
This repository provides a simple pipeline to batch-process a background-music (BGM) library using the Music2Emo model.
For every audio file, the script outputs:=
Valence (1–9)
Arousal (1–9)
Predicted Mood Tags
Filename only (no file paths)

The final results are saved in a CSV file such as:
filename,valence,arousal,mood_tags
song1.mp3,6.4,7.1,energetic|happy
song2.mp3,3.2,4.8,calm


--------------------------------------------------------------------------------------------------------------------------
1. Environment Setup
--------------------------------------------------------------------------------------------------------------------------
Recommended Python Version
Use Python 3.10.


git clone https://github.com/AMAAI-Lab/Music2Emotion
cd Music2Emotion

pip install -r requirements.txt


--------------------------------------------
If you want to keep a clean environment:

python3.10 -m venv venv
venv/Scripts/activate  # Windows
# or
source venv/bin/activate  # Mac / Linux


--------------------------------------------------------------------------------------------------------------------------
2. Run the VA Extraction Script
--------------------------------------------------------------------------------------------------------------------------
- python output.py
only output a single song's emotion prediction

- python output2.py
batch process a folder of songs and save the results in a CSV file

