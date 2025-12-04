import json
import os

english_ytr = ["drewbinsky.json", "KaraandNate.json", "YesTheory.json"]
simplify_json = []
dir_path = "json/lot_bgm"
for filename in os.listdir(dir_path):
    if filename.endswith(".json"):
        with open(os.path.join(dir_path, filename), "r", encoding="utf-8") as f:
            data = json.load(f)
        
        for item in data:
            if item["duration_minutes"] >= 10 and item["viewCount"] >= 500000:
                if filename in english_ytr and item["viewCount"] < 2500000:
                    continue
                simplify_json.append({
                    "videoId": "https://www.youtube.com/watch?v=" + item["videoId"],
                    "title": item["title"],
                    "viewCount": item["viewCount"],
                    "uploader": item["uploader"],
                })
print(f"Total videos with duration >= 10 minutes: {len(simplify_json)}")
with open("simplified_videos.json", "w", encoding="utf-8") as f:
    json.dump(simplify_json, f, indent=4, ensure_ascii=False)

with open("simplified_videos.txt", "w", encoding="utf-8") as f:
    for item in simplify_json:
        f.write(f"{item['title']}\n")