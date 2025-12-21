import json
import os
import numpy as np
import soundfile as sf
import openl3
import ruptures as rpt
from scipy.spatial.distance import cosine

# =========================
# 參數設定
# =========================
INPUT_BASE_DIR = 'crawl_vlog/vlog_video_dataset/separated'
CUT_DURATION_SECONDS = 300
PENALTY = 0.3
HOP_SIZE = 0.5  # openl3 每個 frame = 0.5 秒

# =========================
# 信心分數計算
# =========================
def get_confidence_score(embeddings, cut_index, window_frames=6):
    start_pre = max(0, cut_index - window_frames)
    end_post = min(len(embeddings), cut_index + window_frames)

    vec_pre = np.mean(embeddings[start_pre:cut_index], axis=0)
    vec_post = np.mean(embeddings[cut_index:end_post], axis=0)

    return cosine(vec_pre, vec_post)

# =========================
# 單一音檔分析（不存檔）
# =========================
def analyze_audio_array(audio, sr):
    # 轉 mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # openl3 embedding
    emb, _ = openl3.get_audio_embedding(
        audio,
        sr,
        content_type="music",
        input_repr="mel256",
        embedding_size=512,
        hop_size=HOP_SIZE
    )

    # 變化點偵測
    model = rpt.KernelCPD(kernel="cosine", min_size=5).fit(emb)
    change_points = model.predict(pen=PENALTY)

    results = []
    for cp in change_points:
        if cp >= len(emb):
            continue

        score = get_confidence_score(emb, cp)
        time_sec = cp * HOP_SIZE

        results.append({
            "time": float(time_sec),
            "score": float(score)
        })

    return results

# =========================
# 主流程：讀 JSON → 切音 → 分段 → 輸出 JSON
# =========================
def process_and_analyze(json_filepath, output_json):
    with open(json_filepath, 'r', encoding='utf-8') as f:
        mapping = json.load(f)

    final_results = {}

    for key, value in mapping.items():
        audio_path = os.path.join(INPUT_BASE_DIR, key, 'no_vocals.mp3')
        print(f"\nProcessing: {audio_path}")

        if not os.path.exists(audio_path):
            print(f"❌ 找不到檔案：{audio_path}")
            continue

        try:
            audio, sr = sf.read(audio_path)

            # ===== 直接用 numpy 切前 300 秒 =====
            max_samples = int(CUT_DURATION_SECONDS * sr)
            audio = audio[:max_samples]

            # ===== 分段分析 =====
            segments = analyze_audio_array(audio, sr)

            final_results[value] = segments
            print(f"✅ 完成 {value}，切點數量：{len(segments)}")

        except Exception as e:
            print(f"❌ 處理失敗：{e}")

    # ===== 輸出 JSON =====
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=4)

    print(f"\n🎉 分析完成，輸出檔案：{output_json}")

# =========================
# 執行
# =========================
process_and_analyze(
    json_filepath="translation_log_5.json",
    output_json="split_5.json"
)
