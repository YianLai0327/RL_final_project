import openl3
import soundfile as sf
import ruptures as rpt
import numpy as np
from scipy.spatial.distance import cosine
import json
import os

PENALTY = 0.3  # 分段懲罰參數，數值越大切點越少

def get_confidence_score(embeddings, cut_index, window_frames=4):
    """
    計算切點前後的特徵距離作為信心分數
    window_frames: 4 代表前後各看 4 個 frame (若 hop_size=0.5s，就是前後各 2 秒)
    """
    start_pre = max(0, cut_index - window_frames)
    end_post = min(len(embeddings), cut_index + window_frames)
    
    # 取得切點「前」與「後」的平均特徵向量
    # 使用平均值是為了忽略瞬間的雜訊，抓出該段落的整體風格
    vec_pre = np.mean(embeddings[start_pre:cut_index], axis=0)
    vec_post = np.mean(embeddings[cut_index:end_post], axis=0)
    
    # 計算餘弦距離 (0~2, 但通常在 0~1 之間)
    # 距離越大 = 差異越大 = 信心越高
    score = cosine(vec_pre, vec_post)
    return score

# --- 主流程 ---
def analyze_bgm_with_confidence(audio_path):
    # 1. 讀取與提取特徵 (同上)
    audio, sr = sf.read(audio_path)
    if len(audio.shape) > 1: audio = np.mean(audio, axis=1)
    
    # hop_size=0.5 代表每 0.5 秒一格
    emb, ts = openl3.get_audio_embedding(audio, sr, content_type="music", input_repr="mel256", embedding_size=512, hop_size=0.5)

    # 2. 偵測切點
    # penalty 設低一點 (例如 2或3)，故意讓它 "Over-segment" (切多一點)，我們再用分數來濾
    model = rpt.KernelCPD(kernel="cosine", min_size=5).fit(emb)
    change_points = model.predict(pen=PENALTY) 
    
    results = []

    print(f"\n{'時間 (分:秒)':<15} | {'信心/距離分數':<15} | {'判斷結果'}")
    print("-" * 50)

    for cp in change_points:
        if cp >= len(emb): continue # 忽略最後結束的點
        
        # 計算信心分數
        score = get_confidence_score(emb, cp, window_frames=6) # 前後看 3秒
        
        # 轉換時間
        time_sec = cp * 0.5
        m, s = divmod(time_sec, 60)
        time_str = f"{int(m):02d}:{s:04.1f}"
            
        print(f"{time_sec:<15} | {score:.4f}")
        
        results.append({"time": time_sec.item(), "score": score.item()})
        
    return results

directory_path = 'eval_cut_song'
# 執行
data = {}
for filename in os.listdir(directory_path):
        
    # Check if the entry is actually a file (optional, but prevents errors)
    full_path = os.path.join(directory_path, filename)
    file_name, _ = os.path.splitext(filename)
    data[file_name] = analyze_bgm_with_confidence(full_path)
with open('split_eval.json', 'w', encoding='utf-8') as file:
    # 3. 使用 json.dump() 將資料寫入檔案
    json.dump(data, file, ensure_ascii=False, indent=4)
