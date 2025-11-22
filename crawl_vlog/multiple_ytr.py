from search_by_ytr import main
import json


TOP_K = 5
ytr_list = ["@TheDoDoMen", 
            "@happyariel", 
            "@SpiceTravel", 
            "@bensadventure", 
            "@elephantgogo",
            "@LenaPatrickTaiwan",
            "@chinatravel5971",
            "@kuanglu91",
            "@KaraandNate",
            "@drewbinsky",
            "@baldandbankrupt",
            "@YesTheory",
            "@HaraldBaldr",
            "@MarkWiens",
            "@fearlessandfar",
            "@GabrielTravelerVideos",
            "@evazubeck",
            "@IndigoTraveller",
            "@lostleblanc",
            "@SimonWilson12",
            "@KurtCaz"
            ]

json_list = []
for ytr in ytr_list:
    print(f"Processing YouTuber: {ytr}")
    json_list = json_list + main(ytr, TOP_K)

# Step 6: 輸出所有影片（Top K）
with open("top_videos.json", "w", encoding="utf-8") as f:
    json.dump(json_list, f, indent=4, ensure_ascii=False)