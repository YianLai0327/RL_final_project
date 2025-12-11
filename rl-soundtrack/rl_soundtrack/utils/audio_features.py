from typing import Any, Dict

import librosa
import numpy as np


def compute_audio_features(file_path: str) -> Dict[str, Any]:
    """
    Computes MFCC, Spectral Centroid, RMS (Energy), and BPM for a given audio file.

    Args:
        file_path: Absolute path to the audio file.

    Returns:
        Dict containing:
            - mfcc: Mean MFCC vector (list of floats)
            - spectral_centroid: Mean Spectral Centroid (float)
            - energy: Mean RMS energy (float)
            - bpm: Estimated Tempo (float)
            - duration: Duration in seconds (float)
    """
    try:
        y, sr = librosa.load(file_path, sr=22050)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        # Return zeros/defaults if load fails
        return {
            "mfcc": [0.0] * 13,
            "spectral_centroid": 0.0,
            "energy": 0.0,
            "bpm": 0.0,
            "duration": 0.0,
        }

    # 1. MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1).tolist()

    # 2. Spectral Centroid
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    cent_mean = float(np.mean(cent))

    # 3. Energy (RMS)
    rms = librosa.feature.rms(y=y)
    energy_mean = float(np.mean(rms))

    # 4. BPM
    # onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    # tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
    # Use newer librosa API or standard approach
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(tempo)

    # 5. Duration
    duration = float(librosa.get_duration(y=y, sr=sr))

    return {
        "mfcc": mfcc_mean,
        "spectral_centroid": cent_mean,
        "energy": energy_mean,
        "bpm": bpm,
        "duration": duration,
    }
