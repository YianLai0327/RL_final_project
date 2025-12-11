import os
import pychorus
import librosa
import soundfile as sf
import numpy as np

# CONFIGURATION
INPUT_DIR = "./bgm_selected_raw"
OUTPUT_DIR = "./data/music_chorus_starts"
sr_target = 22050

os.makedirs(OUTPUT_DIR, exist_ok=True)

def find_nearest_zero_crossing(audio, idx, search_range=100):
    """Refines a cut point to the nearest zero-crossing to avoid clicks."""
    if idx >= len(audio): return len(audio) - 1
    start = max(0, idx - search_range)
    end = min(len(audio), idx + search_range)
    zero_crossings = np.where(np.diff(np.signbit(audio[start:end])))[0]
    if len(zero_crossings) == 0:
        return idx
    return start + zero_crossings[np.argmin(np.abs(zero_crossings - (idx - start)))]

def extract_chorus_start(file_path, output_path):
    print(f"Processing: {os.path.basename(file_path)}...")
    
    try:
        # 1. LOAD AUDIO
        y, sr = librosa.load(file_path, sr=sr_target, mono=True)
        
        # 2. FIND CHORUS TIMESTAMP
        # We look for the best 15s clip to identify the "high relevance" section
        chorus_start_sec = pychorus.find_and_output_chorus(
            file_path, None, clip_length=15
        )
        
        # Fallback: If pychorus fails, use the loudest section (RMS energy)
        if chorus_start_sec is None:
            print("  ⚠️ No chroma match found. Using max energy point.")
            rms = librosa.feature.rms(y=y)[0]
            # Skip the first 10% (intro) to avoid false positives
            valid_range_start = int(len(rms) * 0.10)
            peak_frame = valid_range_start + np.argmax(rms[valid_range_start:])
            chorus_start_sec = librosa.frames_to_time(peak_frame, sr=sr)

        # 3. BEAT SNAP (Crucial for Rhythm)
        # We don't want to start 50ms after a beat; it sounds amateur.
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        
        if len(beat_frames) > 0:
            # Find the beat closest to the detected chorus start time
            beat_times = librosa.frames_to_time(beat_frames, sr=sr)
            closest_beat_idx = np.argmin(np.abs(beat_times - chorus_start_sec))
            final_start_frame = beat_frames[closest_beat_idx]
            
            # Refine: Ensure we didn't snap "too far" (e.g., if the beat grid is sparse)
            # If the snap moves us more than 0.5s, stick to original (fallback)
            if abs(librosa.frames_to_time(final_start_frame, sr=sr) - chorus_start_sec) > 0.5:
                 final_start_frame = librosa.time_to_frames(chorus_start_sec, sr=sr)
        else:
            final_start_frame = librosa.time_to_frames(chorus_start_sec, sr=sr)

        # 4. EXPORT
        start_sample = librosa.frames_to_samples(final_start_frame)
        
        # Clean cut at zero crossing
        start_sample = find_nearest_zero_crossing(y, start_sample)
        
        # Slice from start to the end of the file
        y_out = y[start_sample:]
        
        sf.write(output_path, y_out, sr)
        print(f"  ✅ Saved: Starts at {librosa.samples_to_time(start_sample, sr=sr):.2f}s")

    except Exception as e:
        print(f"  ❌ Error: {e}")

# Run Batch
if __name__ == "__main__":
    for f in os.listdir(INPUT_DIR):
        if f.endswith(('.mp3', '.wav')):
            extract_chorus_start(
                os.path.join(INPUT_DIR, f), 
                os.path.join(OUTPUT_DIR, f)
            )