import json

# Load titles from txt file
with open("filter", "r") as f:
    valid_titles = set(line.strip() for line in f if line.strip())

# Load JSON list
with open("simplified_videos.json", "r") as f:
    data = json.load(f)

# Filter items where title is in the txt list
filtered = [item for item in data if item.get("title") in valid_titles]

# Write output
with open("filtered.json", "w", encoding="utf-8") as f:
    json.dump(filtered, f, indent=4, ensure_ascii=False)
