import argparse
import difflib
import json
import os
import re
import unicodedata
from pathlib import Path


def slug_filename(filename: str) -> str:
    forbidden = r'[\\/:*?"<>|]'
    new_filename = "".join(ch for ch in filename if ch.isprintable())
    new_filename = re.sub(forbidden, "_", new_filename)
    return new_filename


def slugify_filename(title: str, max_length: int = 100) -> str:
    # Normalize unicode: remove diacritics
    normalized = "".join(title.split(".")[:-1])
    extension = title.split(".")[-1]
    normalized = unicodedata.normalize("NFKD", normalized)
    normalized = normalized.encode("ascii", "ignore").decode("ascii")

    # Remove Windows forbidden characters
    normalized = re.sub(r'[<>:"/\\|?*]', "", normalized)

    # Only keep letters, digits, whitespace, hyphens, and underscores
    normalized = re.sub(r"[^A-Za-z0-9 _.-]+", "", normalized)

    # Replace whitespace with -
    normalized = normalized.strip()
    normalized = re.sub(r"\s+", "-", normalized)

    # Avoid filename too long
    normalized = normalized[:max_length]

    # Remove trailing . or -
    normalized = normalized.rstrip(" .-")

    # If empty string, provide default name
    if not normalized:
        normalized = "file"

    # Avoid Windows reserved names
    reserved = {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }
    if normalized.upper() in reserved:
        normalized = f"_{normalized}"

    return normalized.lower() + "." + extension


def find_file_fuzzy(target_name, file_list, cutoff=0.5):
    """
    Finds the closest matching filename from file_list using fuzzy search.
    Returns the match if found, else None.
    """
    lower_target_name = target_name.lower()
    lower_file_list = [f.lower() for f in file_list]
    lower_file_map = {f.lower(): f for f in file_list}
    matches = difflib.get_close_matches(
        lower_target_name, lower_file_list, n=1, cutoff=cutoff
    )
    if matches:
        return lower_file_map[matches[0]]
    return None


def process_video_captions(data_dir, dry_run=False):
    """
    Process video captions:
    - Reads video_captions_old.json
    - Fuzzy matches filenames in video_library/
    - Writes to video_captions.json
    """
    # Handle naming inconsistency
    old_file_path = os.path.join(data_dir, "video_captions_old.json")

    if not os.path.exists(old_file_path):
        print(f"Warning: No video caption old file found in {data_dir}. Skipping.")
        return

    print(f"Processing {old_file_path}...")

    with open(old_file_path, "r") as f:
        data = json.load(f)

    # Dictionary where keys are filenames
    if not isinstance(data, dict):
        print(
            f"Error: Expected dict format for video captions in {old_file_path}, got {type(data)}. Skipping."
        )
        return

    video_library_dir = os.path.join(data_dir, "video_library")
    if not os.path.exists(video_library_dir):
        print(f"Warning: video_library directory not found in {data_dir}. Skipping.")
        return

    existing_files = os.listdir(video_library_dir)
    # Filter for likely video files if needed, but fuzzy matching might handle it.

    new_data = []
    filename_map = {}

    for filename, segments in data.items():
        match = find_file_fuzzy(filename, existing_files)

        if match:
            safe_filename = slugify_filename(filename)
            filename_map[filename] = safe_filename
            if match != filename:
                print(
                    f"  [Fuzzy Match] {filename[:20]:<20} -> {match[:20]:<20} -> {safe_filename[:20]:<20}"
                )

            # new_data.append({"filename": safe_filename, "segments": segments})
            new_data.append({"filename": match, "segments": segments})
            # os.rename(
            #     os.path.join(video_library_dir, match),
            #     os.path.join(video_library_dir, safe_filename),
            # )
        else:
            filename_map[filename] = filename
            print(
                f"  [Warning] File not found: '{filename}' in {video_library_dir}. Skipping."
            )

    new_data.sort(key=lambda x: x["filename"])
    output_path = os.path.join(data_dir, "video_captions.json")
    filename_map_path = os.path.join(data_dir, "video_filename_map.json")
    if not dry_run:
        with open(output_path, "w") as f:
            json.dump(new_data, f, indent=2)
        with open(filename_map_path, "w") as f:
            json.dump(filename_map, f, indent=2)
        print(f"Wrote {len(new_data)} items to {output_path}")
    else:
        print(f"[Dry Run] Would write {len(new_data)} items to {output_path}")


def process_music_captions(data_dir, dry_run=False):
    """
    Process music captions:
    - Reads music_captions_old.json
    - Fuzzy matches filenames in music_library/
    - Writes to music_captions.json
    """
    old_file_path = os.path.join(data_dir, "music_captions_old.json")

    if not os.path.exists(old_file_path):
        print(f"Warning: No music caption old file found in {data_dir}. Skipping.")
        return

    print(f"Processing {old_file_path}...")

    with open(old_file_path, "r") as f:
        data = json.load(f)

    if not isinstance(data, list):
        print(
            f"Error: Expected list format for music captions in {old_file_path}, got {type(data)}. Skipping."
        )
        return

    music_library_dir = os.path.join(data_dir, "music_library")
    if not os.path.exists(music_library_dir):
        print(f"Warning: music_library directory not found in {data_dir}. Skipping.")
        return

    existing_files = os.listdir(music_library_dir)

    new_data = []
    filename_map = {}

    for item in data:
        filename = item.get("filename")
        if not filename:
            print(f"  [Warning] Item missing filename: {item}")
            continue

        match = find_file_fuzzy(filename, existing_files)

        if match:
            safe_filename = slugify_filename(filename)
            filename_map[filename] = safe_filename
            if match != filename:
                print(
                    f"  [Fuzzy Match] {filename[:20]:<20} -> {match[:20]:<20} -> {safe_filename[:20]:<20}"
                )

            # item["filename"] = safe_filename
            item["filename"] = match
            new_data.append(item)
            # os.rename(
            #     os.path.join(music_library_dir, match),
            #     os.path.join(music_library_dir, safe_filename),
            # )
        else:
            filename_map[filename] = filename
            print(
                f"  [Warning] File not found: '{filename}' in {music_library_dir}. Skipping."
            )

    new_data.sort(key=lambda x: x["filename"])
    caption_path = os.path.join(data_dir, "music_captions.json")
    filename_map_path = os.path.join(data_dir, "music_filename_map.json")
    if not dry_run:
        with open(caption_path, "w") as f:
            json.dump(new_data, f, indent=2)
        with open(filename_map_path, "w") as f:
            json.dump(filename_map, f, indent=2)
        print(f"Wrote {len(new_data)} items to {caption_path}")
    else:
        print(f"[Dry Run] Would write {len(new_data)} items to {caption_path}")


# def process_video_split()


def main():
    parser = argparse.ArgumentParser(
        description="Organize data JSONs and match with files."
    )
    parser.add_argument(
        "dirs",
        nargs="*",
        help="Directories to process (e.g. small, medium). Defaults to scanning current dir.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Do not write files, just simulate."
    )

    args = parser.parse_args()

    dirs_to_process = args.dirs
    if not dirs_to_process:
        # Default to checking subdirectories 'small' and 'medium' if they exist in current dir
        current_dir = Path.cwd()
        candidates = ["small", "medium"]
        dirs_to_process = [d for d in candidates if (current_dir / d).is_dir()]

        if not dirs_to_process:
            print(
                "No directories provided and 'small'/'medium' not found. searching all subdirectories..."
            )
            dirs_to_process = [x.name for x in current_dir.iterdir() if x.is_dir()]

    for d in dirs_to_process:
        d_path = os.path.abspath(d)
        if not os.path.exists(d_path):
            print(f"Directory {d} does not exist. Skipping.")
            continue

        print(f"=== Processing Directory: {d} ===")
        process_video_captions(d_path, dry_run=args.dry_run)
        process_music_captions(d_path, dry_run=args.dry_run)
        print("\n")


if __name__ == "__main__":
    main()
