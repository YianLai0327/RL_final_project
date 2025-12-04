import json
import os

simplify_json = []
for filename in os.listdir("output"):
    if filename.endswith(".json"):
        with open(os.path.join("output", filename), "r", encoding="utf-8") as f:
            data = json.load(f)
        
        for item in data:
            if item["duration_minutes"] >= 10 and item["viewCount"] >= 1000000:
                simplify_json.append({
                    "videoId": item["videoId"],
                    "title": item["title"]
                })
print(f"Total videos with duration >= 10 minutes: {len(simplify_json)}")
with open("simplified_videos.json", "w", encoding="utf-8") as f:
    json.dump(simplify_json, f, indent=4, ensure_ascii=False)

with open("simplified_videos.txt", "w", encoding="utf-8") as f:
    for item in simplify_json:
        f.write(f"{item['title']}\n")