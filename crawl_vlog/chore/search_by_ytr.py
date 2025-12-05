from googleapiclient.discovery import build
import json
import isodate  # pip install isodate

API_KEY = "AIzaSyBnu4VMIzon6sZoJfJI0WbArEfJqrSSIss"
youtube = build("youtube", "v3", developerKey=API_KEY)

############################################
# CONFIG
############################################
CHANNEL_QUERY = "@TheDoDoMen"
REGION_CODE = "TW"
IGNORE_SHORTS = True  # 忽略 shorts (<60秒)
############################################


############################################
# TRAVEL DETECTOR（旅遊判斷器）
############################################

def is_travel_video(meta):
    title = meta.get("title", "").lower()
    desc = meta.get("description", "").lower()
    tags = [t.lower() for t in meta.get("tags", [])]
    category = meta.get("categoryId", "")
    duration = meta.get("duration_minutes", 0)

    # 0. Shorts 以外才可能是旅遊
    if duration < 10:
        return False

    # 1. YouTube 旅遊分類
    if category == "19":
        return True
    
    return False

    # 2. 旅遊主題常見字詞
    travel_keywords = [
        "travel", "trip", "journey", "explore", "adventure", "vlog",
        "旅遊", "旅行", "出國", "遊記", "觀光", "行程", "自由行"
    ]

    if any(kw in title for kw in travel_keywords):
        return True
    if any(kw in desc for kw in travel_keywords):
        return True

    # 3. Tags
    if any(kw in tags for kw in ["travel", "travel vlog", "旅遊", "旅行"]):
        return True

    # 4. 地名（簡易版，可擴充成全球城市）
    locations = [
        "japan", "tokyo", "osaka", "kyoto",
        "korea", "seoul", "busan",
        "thailand", "bangkok", "chiang mai",
        "taiwan", "台灣", "台北", "台中", "高雄",
        "香港", "澳門", "新加坡"
    ]

    if any(loc in title for loc in locations):
        return True



    return False

def main(CHANNEL_QUERY, TOP_K):
    ############################################
    # SCRAPER
    ############################################

    # Step 1: 搜尋 Channel ID
    search_resp = youtube.search().list(
        part="snippet",
        q=CHANNEL_QUERY,
        type="channel",
        maxResults=1
    ).execute()

    channel_id = search_resp["items"][0]["id"]["channelId"]
    print("Channel ID:", channel_id)

    # Step 2: 拿 uploads playlistId
    channel_info = youtube.channels().list(
        part="contentDetails,snippet",
        id=channel_id
    ).execute()

    uploads_playlist = channel_info["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]
    uploader_name = channel_info["items"][0]["snippet"]["title"]
    print("Uploads playlist:", uploads_playlist)

    # Step 3: 抓所有影片 ID
    video_ids = []
    next_page = None

    while True:
        resp = youtube.playlistItems().list(
            part="contentDetails",
            playlistId=uploads_playlist,
            maxResults=50,
            pageToken=next_page
        ).execute()

        for it in resp["items"]:
            video_ids.append(it["contentDetails"]["videoId"])

        next_page = resp.get("nextPageToken")
        if not next_page:
            break

    print(f"Total videos found in playlist: {len(video_ids)}")


    # Step 4: 抓影片內容（title, stats, description, tags, duration）
    video_stats = []

    for i in range(0, len(video_ids), 50):
        batch = video_ids[i:i+50]

        resp = youtube.videos().list(
            part="snippet,contentDetails,statistics",
            id=",".join(batch),
            maxResults=50
        ).execute()

        for v in resp["items"]:
            snippet = v["snippet"]
            stats = v["statistics"]
            detail = v["contentDetails"]

            # duration ISO → 秒數
            try:
                td = isodate.parse_duration(detail.get("duration", "PT0S"))
                duration_seconds = td.total_seconds()
            except:
                duration_seconds = 0

            duration_minutes = duration_seconds / 60

            # 忽略 Shorts
            if IGNORE_SHORTS and duration_seconds < 60:
                continue

            item = {
                "videoId": v["id"],
                "title": snippet.get("title"),
                "viewCount": int(stats.get("viewCount", 0)),
                "uploader": snippet.get("channelTitle"),
                "publishedAt": snippet.get("publishedAt"),

                "videoLanguage": snippet.get("defaultAudioLanguage") or
                                snippet.get("defaultLanguage") or
                                "unknown",

                "description": snippet.get("description", ""),
                # "tags": snippet.get("tags", []),
                "duration_minutes": duration_minutes,
                "duration_raw": detail.get("duration"),
                "categoryId": snippet.get("categoryId", "0")
            }

            # 加入旅遊分類
            # item["is_travel"] = is_travel_video(item)
            if not is_travel_video(item):
                continue

            video_stats.append(item)


    # Step 5: 排序並取 Top K
    video_stats.sort(key=lambda x: x["viewCount"], reverse=True)
    top_videos = video_stats[:TOP_K]

    return top_videos
'''
# Step 6: 輸出所有影片（Top K）
with open("top_videos.json", "w", encoding="utf-8") as f:
    json.dump(top_videos, f, indent=4, ensure_ascii=False)

# Step 7: 輸出旅遊影片
travel_videos = [v for v in video_stats if v["is_travel"]]

with open("travel_videos.json", "w", encoding="utf-8") as f:
    json.dump(travel_videos, f, indent=4, ensure_ascii=False)

print("Done! Saved top_videos.json and travel_videos.json")
'''