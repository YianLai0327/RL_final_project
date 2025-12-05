# 使用說明 — 使用 `download_from_json.py` 與 `crawl_v3.json` 下載 YouTube 影片

下面說明如何使用工作目錄中的 `download_from_json.py` 搭配 `crawl_v3.json` 批次下載 YouTube 影片。

**先決條件**
- 安裝 Python 3.8+。
- 建議使用虛擬環境（`venv` / `conda`）。
- 安裝必要套件：

```bash
pip install pytubefix tqdm
```

（如果你已經有 `pytube` 或其他 fork，請以你環境需求調整。）

- 若要將下載的 MP4 轉成 MP3，可使用倉庫中的 `convert_mp4_to_mp3.sh`，該腳本需要 `ffmpeg` 安裝在系統中。

**檔案說明**
- `download_from_json.py`：主下載腳本，會讀取同目錄下的 `crawl_v3.json`，逐一下載 `videoId` 欄位指定的影片到 `downloads_v3` 目錄。
- `crawl_v3.json`：JSON 陣列格式，每個項目為物件，範例欄位如下：
  - `videoId`: 可為完整 YouTube 影片 URL（例如 `https://www.youtube.com/watch?v=...`），或影片 ID（視程式處理，當前檔案中為完整 URL）。
  - 其他欄位（`title`, `viewCount`, `uploader`）為資訊欄位，程式不使用它們作為下載來源。

**如何執行**
1. 切到專案目錄：

```bash
cd /path/to/crawl_vlog
```

2. 建議先建立並啟用虛擬環境：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install pytubefix tqdm
```

3. 執行下載腳本：

```bash
python download_from_json.py
```

- 下載的影片會存放在 `downloads_v3` 目錄（若不存在會自動建立）。
- 腳本會讀取 `crawl_v3.json` 的 `videoId` 欄位，並使用 `pytubefix.YouTube` 取得最高解析度的串流下載。
- 預設採用多執行緒下載（`max_workers = 4`），你可在 `download_from_json.py` 中修改 `max_workers` 值以調整併發數量。

**範例：只下載前 10 支（快速測試）**
可以暫時修改 `download_from_json.py` 中的 `urls` 列表或在 `main()` 裡面替換：

```python
urls = [item["videoId"] for item in data if "videoId" in item][:10]
```

然後執行 `python download_from_json.py`。

**常見問題與排錯**
- 下載失敗顯示例外：
  - 檢查網路連線。
  - 若 YouTube 有反爬或地區限制，考慮使用 VPN 或等候一段時間重試。
  - 若 `pytubefix` 出錯，確認已安裝相容版本，或改用其他 youtube 下載工具（例如 `yt-dlp`），但需改寫腳本。
- 如果看到很多同時顯示的進度條造成亂碼，可將 `max_workers` 調小，或移除 `tqdm` 的 `position=idx` 參數以改用單一進度顯示。
- 若需要把 MP4 轉成 MP3：

```bash
# 先確保 ffmpeg 已安裝
# 範例使用倉庫提供的腳本（convert_mp4_to_mp3.sh）
chmod +x convert_mp4_to_mp3.sh
./convert_mp4_to_mp3.sh downloads_v3
```

（請先閱讀並確認 `convert_mp4_to_mp3.sh` 的內容符合你的需求，該腳本會依目錄內的檔案進行轉檔。）

**可自訂的地方**
- `max_workers`：控制同時下載數量（預設 4）。
- `output_path`：在 `download_from_json.py` 的 `stream.download(output_path="downloads_v3")` 可改為其他路徑。
- `videoId` 欄位格式：目前 `crawl_v3.json` 使用完整 URL；若你想用影片 ID（像 `fl5HmGz6Kwo`），請在 `download_from_json.py` 改為適當地組成 URL（例如 `https://www.youtube.com/watch?v={id}`）。

**授權與注意事項**
- 請確認你下載影片的使用符合 YouTube 的服務條款及影片版權授權。

---

若你要我改寫腳本改用 `yt-dlp`（更穩定、支援更多情況）或把轉檔整合到流程中，我可以幫你實作並更新 `README.md` 範例命令。