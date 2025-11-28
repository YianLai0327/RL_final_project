import os
import pychorus
import librosa
import soundfile as sf
import numpy as np
import scipy.signal

# CONFIGURATION
INPUT_DIR = "./data/bgm_library_small"
OUTPUT_DIR = "./data/music_choruses"
MIN_LOOP_DURATION = 10  # Don't loop anything shorter than 10s

os.makedirs(OUTPUT_DIR, exist_ok=True)

def find_seamless_loop(file_path, output_path):
    print(f"Analyzing: {os.path.basename(file_path)}...")
    
    try:
        # 1. LOAD AUDIO
        y, sr = librosa.load(file_path, sr=22050, mono=True)
        
        # 2. FIND CHORUS START (The "Anchor")
        # We start by finding the most important part of the song
        chorus_start_sec = pychorus.find_and_output_chorus(
            file_path, None, clip_length=15
        )
        
        if chorus_start_sec is None:
            print("  ⚠️ No chorus found. Using 15% mark.")
            chorus_start_sec = librosa.get_duration(y=y, sr=sr) * 0.15

        # Convert start time to frame index
        start_frame = librosa.time_to_frames(chorus_start_sec, sr=sr)
        
        # 3. COMPUTE FEATURES (Chroma = Musical Notes)
        # We use Chroma because it matches "Melody/Harmony" regardless of loudness
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        
        # 4. FIND THE "REPEAT POINT"
        # We take a small "fingerprint" of the chorus start (e.g., 4 seconds)
        fingerprint_len = librosa.time_to_frames(4, sr=sr)
        if start_frame + fingerprint_len >= chroma.shape[1]:
            print("  ❌ Song too short.")
            return

        target_pattern = chroma[:, start_frame : start_frame + fingerprint_len]
        
        # We scan the REST of the song looking for this pattern
        # We only look AFTER the start point + MIN_LOOP_DURATION
        search_start_frame = start_frame + librosa.time_to_frames(MIN_LOOP_DURATION, sr=sr)
        
        if search_start_frame >= chroma.shape[1]:
            print("  ❌ Song too short for loop.")
            return

        # Calculate similarity (Cross-Correlation)
        # We slide the target_pattern across the rest of the song
        correlation = []
        search_window = chroma[:, search_start_frame:]
        
        # Manual sliding window correlation for multi-dimensional chroma
        # (Simplified: we compare frame-by-frame similarity)
        # A more robust way is using scipy's correlation on flattened arrays or librosa recurrence
        
        # FAST METHOD: Recurrence Matrix
        # We look for the diagonal line in the recurrence matrix starting at start_frame
        # But here is a simpler "Match Hunter" logic:
        
        best_score = -1
        best_end_frame = -1
        
        # Scan every potential end point
        # Optimization: We only check every 10th frame to speed up, then refine
        test_frames = range(0, search_window.shape[1] - fingerprint_len, 10)
        
        for t in test_frames:
            # Compare the 4s window at Search Point vs Anchor Point
            candidate = search_window[:, t : t + fingerprint_len]
            
            # Cosine similarity between the two 12xN chroma matrices
            score = np.mean(np.multiply(target_pattern, candidate))
            
            if score > best_score:
                best_score = score
                # The actual time in the song is search_start + t
                best_end_frame = search_start_frame + t

        # 5. REFINE TO NEAREST BEAT (Crucial for Rhythm)
        # Now we have the "Harmonic Match", let's snap it to the "Rhythmic Match"
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        
        # Find beat closest to our best harmonic match
        closest_beat_idx = np.argmin(np.abs(beat_frames - best_end_frame))
        final_end_frame = beat_frames[closest_beat_idx]
        
        # Also snap start to nearest beat for perfect alignment
        closest_start_beat_idx = np.argmin(np.abs(beat_frames - start_frame))
        final_start_frame = beat_frames[closest_start_beat_idx]
        
        # 6. EXPORT
        start_sample = librosa.frames_to_samples(final_start_frame)
        end_sample = librosa.frames_to_samples(final_end_frame)
        
        y_loop = y[start_sample:end_sample]
        
        # CROSSFADE (The "Magic Glue")
        # Even with perfect beat matching, phase differences cause clicks.
        # We overlap the last 100ms with the first 100ms to hide the seam.
        xfade_len = int(0.1 * sr) 
        if len(y_loop) > 2 * xfade_len:
            # We don't change the file length, we just smooth the edges
            # Fade In Start
            y_loop[:xfade_len] *= np.linspace(0, 1, xfade_len)
            # Fade Out End
            y_loop[-xfade_len:] *= np.linspace(1, 0, xfade_len)

        sf.write(output_path, y_loop, sr)
        
        duration = (end_sample - start_sample) / sr
        print(f"  ✅ Loop Found: {duration:.1f}s")
        print(f"     Start: {librosa.frames_to_time(final_start_frame):.1f}s")
        print(f"     End:   {librosa.frames_to_time(final_end_frame):.1f}s (Matches Start)")

    except Exception as e:
        print(f"  ❌ Error: {e}")

# Run Batch
for f in os.listdir(INPUT_DIR):
    if f.endswith(('.mp3', '.wav')):
        find_seamless_loop(
            os.path.join(INPUT_DIR, f), 
            os.path.join(OUTPUT_DIR, f)
        )