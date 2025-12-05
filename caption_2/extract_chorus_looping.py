import os
import pychorus
import librosa
import soundfile as sf
import numpy as np

# CONFIGURATION
INPUT_DIR = "./bgm_library_small"
OUTPUT_DIR = "./data/music_chorus_loops"
MIN_LOOP_DURATION = 10  # Seconds
sr_target = 22050

os.makedirs(OUTPUT_DIR, exist_ok=True)

def find_nearest_zero_crossing(audio, idx, search_range=100):
    """Refines a cut point to the nearest zero-crossing to avoid clicks."""
    start = max(0, idx - search_range)
    end = min(len(audio), idx + search_range)
    # Find points where sign flips
    zero_crossings = np.where(np.diff(np.signbit(audio[start:end])))[0]
    if len(zero_crossings) == 0:
        return idx
    # Return global index of closest crossing
    return start + zero_crossings[np.argmin(np.abs(zero_crossings - (idx - start)))]

def find_seamless_loop(file_path, output_path):
    print(f"Analyzing: {os.path.basename(file_path)}...")
    
    try:
        # 1. LOAD AUDIO & FEATURES
        y, sr = librosa.load(file_path, sr=sr_target, mono=True)
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        
        # 2. BEAT TRACKING (The "Grid")
        # We need the grid first to snap our chorus search to valid musical starts
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        
        if len(beat_frames) < 16:
            print("  ❌ Song too short/No beats.")
            return

        # 3. FIND CHORUS START (The "Anchor")
        chorus_start_sec = pychorus.find_and_output_chorus(
            file_path, None, clip_length=15
        )
        
        if chorus_start_sec is None:
            # Fallback: Use the loudest section (RMS) roughly 20% in
            print("  ⚠️ No chorus found. Using max energy point.")
            rms = librosa.feature.rms(y=y)[0]
            # Ignore intro (first 10%)
            valid_range_start = int(len(rms) * 0.10)
            peak_frame = valid_range_start + np.argmax(rms[valid_range_start:])
            chorus_start_sec = librosa.frames_to_time(peak_frame, sr=sr)

        # Snap start to the nearest DOWNBEAT or at least a Beat
        # We find the beat frame closest to our timestamp
        start_beat_idx = np.argmin(np.abs(librosa.frames_to_time(beat_frames) - chorus_start_sec))
        start_frame = beat_frames[start_beat_idx]
        
        # 4. FIND THE "REPEAT POINT" (Vectorized)
        # Fingerprint is 4 seconds (or approx 8 beats)
        fingerprint_len = librosa.time_to_frames(4, sr=sr)
        
        # Define search area: Must be after start + MIN_DURATION
        min_gap_frames = librosa.time_to_frames(MIN_LOOP_DURATION, sr=sr)
        search_start_frame = start_frame + min_gap_frames
        
        if search_start_frame + fingerprint_len >= chroma.shape[1]:
            print("  ❌ Song too short for loop.")
            return

        # Target: The chroma pattern at the start of the chorus
        target_pattern = chroma[:, start_frame : start_frame + fingerprint_len]
        
        # Search Area: The rest of the song
        search_area = chroma[:, search_start_frame:]
        
        # SCORING: Compute similarity (Dot Product / Cross-Correlation)
        # This replaces the slow for-loop with matrix multiplication
        # We convolve the target pattern over the search area
        # (Simplified: Normalized Cross-Correlation)
        from scipy.signal import correlate2d
        
        # Efficient check: We only compare at specific intervals to save time, then refine?
        # Actually, let's just check the "Beat Locations" in the search area.
        # This is much faster and musically correct.
        
        best_score = -1
        best_end_beat_idx = -1
        
        # We only check potential end points that fall on beats
        # Look for beats that are after our search_start_frame
        candidate_beat_indices = [i for i, b in enumerate(beat_frames) if b > search_start_frame]
        
        for idx in candidate_beat_indices:
            test_frame = beat_frames[idx]
            if test_frame + fingerprint_len >= chroma.shape[1]: break
            
            candidate_patch = chroma[:, test_frame : test_frame + fingerprint_len]
            
            # Simple metric: Mean Cosine Similarity
            # Flatten to 1D for quick correlation
            score = np.mean(target_pattern * candidate_patch)
            
            # 5. MUSICAL CONSTRAINT (Crucial for BGM)
            # Check if length is a multiple of 4 beats (1 Bar)
            beats_duration = idx - start_beat_idx
            is_bar_aligned = (beats_duration % 4 == 0)
            
            # Boost score if bar-aligned
            if is_bar_aligned:
                score *= 1.2 
            
            if score > best_score:
                best_score = score
                best_end_beat_idx = idx

        if best_end_beat_idx == -1:
            print("  ❌ No good loop point found.")
            return

        final_end_frame = beat_frames[best_end_beat_idx]
        
        # 6. EXPORT WITH ZERO-CROSSING (No Fade)
        start_sample = librosa.frames_to_samples(start_frame)
        end_sample = librosa.frames_to_samples(final_end_frame)
        
        # Refine cuts to nearest zero crossing to avoid "pop" noise
        start_sample = find_nearest_zero_crossing(y, start_sample)
        end_sample = find_nearest_zero_crossing(y, end_sample)
        
        y_loop = y[start_sample:end_sample]
        
        # Double check duration
        duration = len(y_loop) / sr
        beats_len = best_end_beat_idx - start_beat_idx
        
        print(f"  ✅ Loop Found: {duration:.2f}s ({beats_len} beats)")
        if beats_len % 4 == 0:
            print("     Musicality: PERFECT (Bar aligned)")
        else:
            print("     Musicality: OFF-GRID (Might sound jerky)")
            
        sf.write(output_path, y_loop, sr)

    except Exception as e:
        print(f"  ❌ Error processing {os.path.basename(file_path)}: {e}")
        import traceback
        traceback.print_exc()

# Run Batch
if __name__ == "__main__":
    for f in os.listdir(INPUT_DIR):
        if f.endswith(('.mp3', '.wav')):
            find_seamless_loop(
                os.path.join(INPUT_DIR, f), 
                os.path.join(OUTPUT_DIR, f)
            )