import re

def is_travel_video(meta):
    title = meta.get("title","").lower()
    desc = meta.get("description","").lower()
    tags = [t.lower() for t in meta.get("tags",[])]
    category = meta.get("categoryId","")
    duration = meta.get("duration_minutes",0)

    # 1. 類別直接命中 Travel
    if category == "19":
        return True

    # 2. 旅遊關鍵字
    travel_keywords = [
        "travel", "vlog", "trip", "journey", "explore", "adventure",
        "旅遊", "旅行", "出國", "環島", "自由行", "行程", "遊記", "觀光"
    ]

    if any(kw in title for kw in travel_keywords):
        return True
    if any(kw in desc for kw in travel_keywords):
        return True

    # 3. tags 檢查
    if any(kw in tags for kw in ["travel", "travel vlog", "旅遊", "旅行"]):
        return True

    # 4. 地名檢測（簡化版本，可加入完整世界地名庫）
    locations = [
        "japan", "tokyo", "osaka", "korea", "seoul", "bangkok",
        "taiwan","台灣","台北","日本","韓國","泰國","香港"
    ]
    if any(loc in title for loc in locations):
        return True

    # 5. 片長過短（短影片通常非旅遊）
    if duration < 2:
        return False

    return False
