
--------------------------------------------------------------------------------------
Folder Structure
--------------------------------------------------------------------------------------
├── download.py              # Main script for crawling and extraction
├── audio_features.csv       # Output metadata and feature dataset
└── bgm_library/             # Directory containing the actual .mp3 files
    ├── audio_173235.mp3
    ├── audio_173236.mp3
    └── ...

--------------------------------------------------------------------------------------
Quick Start
--------------------------------------------------------------------------------------
Configuration: You can modify the following variables in download.py to change the dataset size:

MAX_PER_CATEGORY: Number of tracks to download per mood category (Default: 6 for dev, 12 for production).

DEST_DIR: Target folder for MP3s.


--------------------------------------------------------------------------------------
Data Documentation: audio_features.csv
--------------------------------------------------------------------------------------
This CSV file serves as the Music Library ($M$) for the RL environment2. Each row represents a track with features used for calculating rewards.
Column	            Type	      Description	                                    Usage in RL
filename	        String	      Filename in bgm_library/	                        Loading audio
base_mood	        String	      Original category (e.g., "upbeat", "horror")	    Ground truth base
energy	            String	      Calculated energy (low, medium, high)	            State representation
mood_tag	        String	      Composite tag (e.g., happy_high, dark_low)	    MoodMatch reward
bpm	                Float	      Beats Per Minute (Tempo)	                        TempoMatch reward
rms	                Float	      Root Mean Square (Loudness)	                    Energy calculation
spectral_centroid	Float	      Center of mass of the spectrum (Brightness)	    Transition Cost (Smoothness)
mfcc_1...13	        Float	      Mel-Frequency Cepstral Coefficients	            Transition Cost (Timbre similarity)


--------------------------------------------------------------------------------------
Tagging Logic (Mood & Energy)
--------------------------------------------------------------------------------------
To simplify the state space for the RL agent, continuous features are discretized into tags based on the following thresholds (derived from dataset distribution):
Energy Levels:

High: BPM ≥ 128 OR RMS ≥ 0.14

Medium: 110 ≤ BPM < 128 OR 0.075 ≤ RMS < 0.14

Low: BPM < 110 AND RMS < 0.075

Mood Tag Construction: The mood_tag is a combination of the base_mood (mapped from Source) and the energy.
Example: An "Epic" track with 140 BPM -> epic_high.
Example: A "Horror" track with low RMS -> dark_low.


--------------------------------------------------------------------------------------
Feature Extraction Details
--------------------------------------------------------------------------------------
Features are extracted using librosa analyzing the first 30 seconds of each track.

Tempo (BPM): Extracted using beat_track. Used to align video motion speed with music speed.

MFCCs (1-13): The "sound fingerprint" used to calculate the distance between tracks. Large distances in MFCC space result in a high Transition Cost penalty.

Spectral Centroid: Measures how "bright" or "dark" the audio sounds. Used to prevent jarring transitions (e.g., switching from dull to sharp audio instantly).