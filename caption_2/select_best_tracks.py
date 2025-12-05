import os
import csv
import shutil
from collections import defaultdict

# === CONFIGURATION ===
INPUT_DIR = "bgm_library_large"          # Source: All 100+ downloaded files
CSV_PATH = "audio_features_large.csv"    # Metadata file
OUTPUT_DIR = "bgm_selected_raw"          # Destination: The 50 chosen originals
TARGET_TOTAL = 50

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_library_metadata(csv_path):
    """Reads the CSV and groups tracks by base_mood."""
    if not os.path.exists(csv_path):
        print(f"[ERROR] CSV not found at {csv_path}.")
        return None

    library = defaultdict(list)
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Only consider files that actually exist
            if os.path.exists(os.path.join(INPUT_DIR, row['filename'])):
                library[row['base_mood']].append(row)
    return library

def select_balanced_tracks(library, target_total):
    """
    Selects ~6 tracks per category.
    Ranking Logic: 
    1. Must have valid BPM (bpm > 0)
    2. Highest Energy (RMS) within that category
    """
    selected_files = []
    categories = list(library.keys())
    
    if not categories:
        return []

    # Calculate target per category
    count_per_cat = target_total // len(categories)
    remainder = target_total % len(categories)
    
    print(f"--- Selection Plan: ~{count_per_cat} tracks from each of {len(categories)} categories ---")

    for category in categories:
        tracks = library[category]
        
        # Convert numeric strings to floats for sorting
        for t in tracks:
            t['bpm'] = float(t['bpm']) if t['bpm'] else 0.0
            t['rms'] = float(t['rms']) if t['rms'] else 0.0

        # SORTING LOGIC:
        # Key 1: bpm > 0 (True comes first). Filter out broken analysis.
        # Key 2: rms (Descending). Pick the clearest/loudest in this genre.
        tracks.sort(key=lambda x: (x['bpm'] > 0, x['rms']), reverse=True)
        
        # Determine quota
        quota = count_per_cat + (1 if remainder > 0 else 0)
        remainder -= 1
        
        # Select Top N
        picks = tracks[:quota]
        selected_files.extend([t['filename'] for t in picks])
        
        print(f"  [{category.upper()}] Selected {len(picks)} tracks.")

    return selected_files

def main():
    print("--- 1. Loading Metadata ---")
    library = load_library_metadata(CSV_PATH)
    if not library:
        return

    print("\n--- 2. Selecting Best 50 (Balanced) ---")
    final_list = select_balanced_tracks(library, TARGET_TOTAL)
    
    print(f"\n--- 3. Copying {len(final_list)} Files to '{OUTPUT_DIR}' ---")
    
    # Clean output dir first to avoid stale files
    for f in os.listdir(OUTPUT_DIR):
        os.remove(os.path.join(OUTPUT_DIR, f))

    for filename in final_list:
        src = os.path.join(INPUT_DIR, filename)
        dst = os.path.join(OUTPUT_DIR, filename)
        shutil.copy2(src, dst)
        
    print(f"[DONE] Raw selected files are in '{OUTPUT_DIR}'.")
    print("Now run 'extract_chorus_start_only.py' pointing to this folder.")

if __name__ == "__main__":
    main()



'''
1. Horizontal Logic: Category Balancing (Diversity)
Instead of picking the "Top 50" globally (which would fill your library with only aggressive "Epic/Electronic" tracks), the script forces equal representation across all moods.

Logic: Target Total (50) / Number of Categories (e.g., 9) = ~5-6 tracks per category.

Why: This prevents Mode Collapse. If you only select high-energy music, your RL agent will never learn how to handle sad or quiet scenes because it literally won't have any "sad" actions available to choose.

2. Vertical Logic: Quality Ranking (Validity & Clarity)
Once inside a specific category (e.g., "Romantic"), the script needs to pick the "best" 6 tracks. It sorts them using a two-tier priority key:

Priority 1: Rhythm Validity (bpm > 0)

Logic: The script checks if librosa successfully detected a beat.

Why: If the BPM is 0 or undetectable, the track is likely ambient noise, has no rhythm, or is a bad recording. Since your RL reward function relies on TempoMatch, a track with no detectable tempo is useless to the agent. These are pushed to the bottom.

Priority 2: Signal Clarity (RMS Descending)

Logic: Among valid tracks, it picks the ones with the highest RMS (Energy).

Why: In free music libraries, "louder" tracks tend to be professionally mastered with clear instrumentation. Very quiet tracks are often "dead air" or poor recordings.

Crucial Detail: This comparison happens only within the category. A "Romantic" track only needs to be louder than other "Romantic" tracks to get picked; it does not compete with "Epic" tracks.
'''