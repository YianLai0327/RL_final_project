from googleapiclient.discovery import build
import json

API_KEY = "AIzaSyBnu4VMIzon6sZoJfJI0WbArEfJqrSSIss"
youtube = build("youtube", "v3", developerKey=API_KEY)

request = youtube.search().list(
    part="snippet",
    eventType="completed", # prevent streaming
    q="vlog taiwan",        
    type="video",
    videoDuration="medium", # 4-20 minutes 
    maxResults=25,
    regionCode="TW",
    relevanceLanguage="zh-Hant",
    videoCategoryId="19", # Travel & Events
    order="viewCount"
)

response = request.execute()

json_data = []

for item in response["items"]:
    title = item["snippet"]["title"]
    video_id = item["id"]["videoId"]
    url = f"https://www.youtube.com/watch?v={video_id}"
    json_data.append({
        "title": title,
        "url" : url
    })
    print(title, url)
with open('output.json', 'w') as f:
    json.dump(json_data, f, indent=4)