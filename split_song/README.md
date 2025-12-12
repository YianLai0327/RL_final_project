# split_song

## 建置環境 (Build env) ✅
建議使用 Python 3.10

使用 pip 安裝相關套件：

```bash
# 建議先建立隔離環境，例如 conda 或 venv
conda create -n split-song python=3.10 -y
conda activate split-song

# 安裝 TensorFlow，若要 GPU 支援請確認系統上已安裝對應 CUDA / cuDNN
pip install "tensorflow[and-cuda]"

# 另外安裝音訊 / 特徵分析與依賴套件
pip install openl3 ruptures soundfile numpy
```

💡 注意：TensorFlow 的 GPU 支援需要你先在作業系統安裝正確版本的 CUDA 與 cuDNN，以及符合的 GPU 驅動程式。

---

## 資料 (Data) 📂
資料位於 `split_song/split.json`。 這個 JSON 的格式為：

- key：影片標題（或檔名）
- value：一個陣列，陣列中每個物件包含：
  - `time`：偵測到的時間點（秒）
  - `score`：對「換歌（switch song）」的信心分數（數值越大表示越有可能發生換歌）

範例（摘錄）：
```json
{
  "VIDEO_TITLE": [
    {"time": 19.0, "score": 0.012068629264831543},
    {"time": 38.0, "score": 0.019410908222198486}
  ]
}
```

- 分數越大（score 大）代表系統越有信心該時間點是換歌的邊界。
- 根據我實際聽原始音訊的經驗，設一個閾值（threshold）能比較穩定檢測換歌：`0.02`（也就是 `score >= 0.02` 視為換歌）。
  - 這個值不是萬無一失，僅作為一個比較好的起始門檻（empirical）。
  - 如果你想提高精準度，建議在不同影片上做抽樣驗證並微調閾值，或加入後處理規則（例如最小距離間隔、平滑化、或多條件判斷）。

---