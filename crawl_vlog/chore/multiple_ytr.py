from search_by_ytr import main
import json


TOP_K = 5
ytr_list = ["@TheDoDoMen",            # lot of bgm 
            "@happyariel",            # without bgm, discard
            "@SpiceTravel",           # partial without bgm
            "@bensadventure",         # lot of bgm
            "@elephantgogo",          # lot of bgm
            "@LenaPatrickTaiwan",     # lot of bgm
            "@chinatravel5971",       # no video found, discard
            "@kuanglu91",             # without bgm, discard
            "@KaraandNate",           # lot of bgm
            "@drewbinsky",            # lot of bgm
            "@baldandbankrupt",       # without bgm, discard
            "@YesTheory",             # lot of bgm
            "@HaraldBaldr",           # without bgm, discard
            "@MarkWiens",             # partial without bgm
            "@fearlessandfar",        # partial without bgm
            "@GabrielTravelerVideos", # partial without bgm
            "@evazubeck",             # partial without bgm
            "@IndigoTraveller",       # without bgm, discard
            "@lostleblanc",           # partial without bgm
            "@SimonWilson12",         # without bgm, discard
            "@KurtCaz"                # without bgm, discard
            ]

json_list = []
for ytr in ytr_list:
    print(f"Processing YouTuber: {ytr}")
    json_list = json_list + main(ytr, TOP_K)

# Step 6: 輸出所有影片（Top K）
with open("top_videos.json", "w", encoding="utf-8") as f:
    json.dump(json_list, f, indent=4, ensure_ascii=False)