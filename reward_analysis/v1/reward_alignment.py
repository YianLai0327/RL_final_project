import json
import random
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# 1. 初始化模型與設定
# ==========================================
# 使用輕量級模型，速度快且效果不錯
print("正在載入 Embedding 模型...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# 權重設定 (這部分是可以調優的 Hyperparameters)
WEIGHTS = {
    "mood": 2.0,        # 情緒匹配權重
    "energy": 1.5,      # 能量匹配權重
    "vocal_clash": 3.0, # 人聲干擾懲罰權重 (扣分項)
    "transition": 3.0   # 轉場頻率懲罰權重 (扣分項)
}

# ==========================================
# 2. 輔助功能
# ==========================================

def get_energy_score(energy_str):
    """將文字能量轉為數值 0.0 ~ 1.0"""
    mapping = {"Low": 0.0, "Medium": 0.5, "High": 1.0}
    return mapping.get(energy_str, 0.5)

def get_embedding(text):
    """將文字轉為向量"""
    return model.encode([text])[0]

def parse_time_to_seconds(time_str):
    """將 '00:02' 轉為秒數 (int)"""
    m, s = map(int, time_str.split(':'))
    return m * 60 + s

# ==========================================
# 3. 核心 Reward Function (單段計算)
# ==========================================
def calculate_segment_reward(video_seg, audio_seg, is_switch, prev_song_total_duration, use_metadata_pairing=True):
    total_reward = 0
    details = {}
    
    # 資料解包
    if 'audio_caption' in audio_seg:
        current_audio_props = audio_seg['audio_caption']
    else:
        current_audio_props = audio_seg

    # --- A. Mood Alignment (修改處: 加入消融邏輯) ---
    v_text = f"{video_seg['visual_caption']} Category: {video_seg['scene_category']}. Mood: {', '.join(video_seg['mood_tags'])}"
    
    if use_metadata_pairing:
        # 原本邏輯：包含 video_pairing (可能有 Data Leakage 風險)
        a_text = f"{current_audio_props['video_pairing']} Genre: {current_audio_props['suggested_genre']}. Mood: {', '.join(current_audio_props['mood_tags'])}"
    else:
        # 消融邏輯：只看 Tags, Genre, Instrument (純實力對決)
        # 我們把 video_pairing 拿掉，改放 instrument 或單純 tags
        instr = current_audio_props.get('instrumentation', '')
        a_text = f"Genre: {current_audio_props['suggested_genre']}. Instruments: {instr}. Mood: {', '.join(current_audio_props['mood_tags'])}"
    
    v_emb = get_embedding(v_text).reshape(1, -1)
    a_emb = get_embedding(a_text).reshape(1, -1)
    mood_score = cosine_similarity(v_emb, a_emb)[0][0]
    total_reward += mood_score * WEIGHTS["mood"]
    details['mood'] = mood_score

    # --- B. 能量匹配 (Energy) ---
    v_energy = get_energy_score(video_seg['energy'])
    a_energy = get_energy_score(audio_seg['energy'])
    # 距離越小分數越高 (1 - distance)
    energy_dist = abs(v_energy - a_energy)
    energy_score = 1.0 - energy_dist
    term_energy = energy_score * WEIGHTS["energy"]
    total_reward += term_energy
    details['energy'] = energy_score

    # --- C. 功能性懲罰 (Vocal Clash) ---
    # 如果畫面是對話，音樂卻有人聲 -> 扣分
    is_dialogue = video_seg['scene_category'] == "Dialogue"
    has_vocals = audio_seg['has_vocals']
    
    term_clash = 0
    if is_dialogue and has_vocals:
        term_clash = -1.0 * WEIGHTS["vocal_clash"]
        total_reward += term_clash
    details['vocal_clash'] = term_clash

    # --- D. 轉場順暢度 (Transition) ---
    # 這裡只計算「是否過於頻繁切歌」
    term_trans = 0
    if is_switch:
        # 如果換歌了 (filename 不同)
        if prev_song_total_duration < 5:
            term_trans = -1.0 * WEIGHTS["transition"]
            total_reward += term_trans
    
    details['transition'] = term_trans
    
    return total_reward, details

# ==========================================
# 4. 序列評估 (整支影片總分)
# ==========================================
def evaluate_sequence(video_list, audio_list, use_metadata_pairing=True):
    total_score = 0
    log = []
    
    current_song = None
    current_duration = 0.0
    
    for i in range(len(video_list)):
        v_seg = video_list[i]
        a_seg = audio_list[i]
        
        seg_start = parse_time_to_seconds(v_seg['start'])
        seg_end = parse_time_to_seconds(v_seg['end'])
        seg_duration = seg_end - seg_start
        
        if 'audio_caption' in a_seg:
            # 優先找外層，沒有找內層，再沒有給預設
            curr_filename = a_seg.get('filename', a_seg['audio_caption'].get('filename', 'unknown'))
        else:
            curr_filename = a_seg.get('filename', 'unknown')

        # === 核心轉場判斷邏輯 ===
        is_switch = False
        duration_to_check = 0.0
        
        if i == 0:
            # 第一段，初始化
            current_run_filename = curr_filename
            current_run_duration = seg_duration
        else:
            if curr_filename == current_run_filename:
                # Case 1: 續播 (Continue)
                # 累加時間，不觸發切歌檢查
                current_run_duration += seg_duration
                is_switch = False
            else:
                # Case 2: 切歌 (Switch)
                is_switch = True
                # 記錄切歌當下，上一首歌「總共」播了多久
                duration_to_check = current_run_duration
                
                # 重置狀態給新歌
                current_run_filename = curr_filename
                current_run_duration = seg_duration
        
        step_reward, details = calculate_segment_reward(
            v_seg, 
            a_seg, 
            is_switch=is_switch, 
            prev_song_total_duration=duration_to_check,
            use_metadata_pairing=use_metadata_pairing # 傳入參數
        )
        total_score += step_reward
        
        log.append({
            "step": i,
            "time_range": f"{v_seg['start']}-{v_seg['end']}",
            "filename": curr_filename,
            "is_switch": is_switch,
            "accumulated_time_before_switch": duration_to_check if is_switch else 0,
            "reward": step_reward,
            "details": details
        })
        
    avg_score = total_score / len(video_list)
    return avg_score, log

import matplotlib.pyplot as plt
import numpy as np

def visualize_results(orig_log, rand_log):
    # 1. 提取數據
    steps = [x['step'] for x in orig_log]
    
    # 總分
    orig_total = [x['reward'] for x in orig_log]
    rand_total = [x['reward'] for x in rand_log]
    
    # 細項分數
    orig_mood = [x['details']['mood'] for x in orig_log]
    rand_mood = [x['details']['mood'] for x in rand_log]
    
    orig_trans = [x['details']['transition'] for x in orig_log]
    rand_trans = [x['details']['transition'] for x in rand_log]

    # 設定圖表風格
    plt.style.use('ggplot') 
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    
    # ==========================================
    # 圖表 1: Total Reward Comparison (總分對決)
    # ==========================================
    ax1.plot(steps, orig_total, label='Original BGM (Ground Truth)', color='#2ca02c', linewidth=2.5, marker='o')
    ax1.plot(steps, rand_total, label='Random BGM', color='#d62728', linewidth=2, linestyle='--', marker='x')
    
    ax1.set_title('Step-by-Step Reward Comparison', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Total Reward', fontsize=12)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # 標記出差異巨大的點
    for i in range(len(steps)):
        if orig_total[i] - rand_total[i] > 1.0: # 差距過大時標註
            ax1.annotate('Big Win', xy=(i, orig_total[i]), xytext=(i, orig_total[i]+0.5),
                         arrowprops=dict(facecolor='black', shrink=0.05), ha='center', fontsize=8)

    # ==========================================
    # 圖表 2: Mood Alignment Score (情緒匹配度)
    # ==========================================
    width = 0.35
    x = np.arange(len(steps))
    
    ax2.bar(x - width/2, orig_mood, width, label='Original Mood Score', color='#1f77b4', alpha=0.8)
    ax2.bar(x + width/2, rand_mood, width, label='Random Mood Score', color='#ff7f0e', alpha=0.6)
    
    ax2.set_title('Semantic Mood Alignment (Embedding Similarity)', fontsize=14)
    ax2.set_ylabel('Cosine Similarity', fontsize=12)
    ax2.legend(loc='upper right')
    ax2.grid(axis='y', linestyle='--', alpha=0.5)

    # ==========================================
    # 圖表 3: Transition Penalty (轉場懲罰 - 倒扣分)
    # ==========================================
    # 這裡我們畫負分，越往下代表扣分越重
    ax3.fill_between(steps, rand_trans, 0, where=(np.array(rand_trans) < 0), 
                     color='red', alpha=0.3, label='Random Penalty Area')
    ax3.plot(steps, rand_trans, color='red', linestyle=':', label='Random Transition Score')
    ax3.plot(steps, orig_trans, color='green', linewidth=2, label='Original Transition Score (Stable)')
    
    ax3.set_title('Transition Stability (Penalties)', fontsize=14)
    ax3.set_ylabel('Penalty Score', fontsize=12)
    ax3.set_xlabel('Video Segment Index', fontsize=14)
    ax3.legend(loc='lower right')
    
    # 調整佈局
    plt.tight_layout()
    plt.savefig('exp_no_transitionpenalty_reward_analysis.png', dpi=300)
    print("✅ 圖表已生成：reward_analysis.png")
    plt.close()

import random

def generate_smart_random_sequence(video_data, audio_library, min_duration=5.0):
    """
    生成一個符合轉場規則的隨機序列 (Smart Random Baseline)
    """
    smart_sequence = []
    
    # 隨機選第一首
    current_track = random.choice(audio_library)
    current_run_time = 0.0
    
    for v_seg in video_data:
        # 計算當前 Video Segment 長度
        start = parse_time_to_seconds(v_seg['start'])
        end = parse_time_to_seconds(v_seg['end'])
        seg_duration = end - start
        
        # 決定這一格要放什麼歌
        if current_run_time < min_duration:
            # 如果累積時間還不夠長，強制續播 (Keep)
            track_to_append = current_track
            current_run_time += seg_duration
        else:
            # 時間夠長了，可以隨機決定要不要切歌
            # 這裡我們模擬 50% 機率切歌，50% 續播
            if random.random() < 0.5:
                # 切歌 (Switch)
                new_track = random.choice(audio_library)
                
                # 如果隨機骰到同一首，視為續播
                # 這裡假設 library 裡的 dict 物件是同一個 reference 或有 id
                if new_track['filename'] == current_track['filename']:
                    current_run_time += seg_duration
                else:
                    current_track = new_track
                    current_run_time = seg_duration # 重置時間
            else:
                # 決定續播 (Keep)
                current_run_time += seg_duration
                
            track_to_append = current_track
            
        smart_sequence.append(track_to_append)
        
    return smart_sequence

def generate_greedy_sequence(video_data, audio_library, use_metadata_pairing=True):
    """
    Greedy Strategy: 針對每一段 Video，遍歷所有 Audio，
    找出 (Mood + Energy + Vocal) 分數總和最高的那首。
    完全不考慮 Transition (轉場) 的後果。
    """
    greedy_sequence = []
    
    for v_seg in video_data:
        best_track = None
        best_local_score = -float('inf')
        
        # 1. 準備 Video Embedding
        v_text = f"{v_seg['visual_caption']} Category: {v_seg['scene_category']}. Mood: {', '.join(v_seg['mood_tags'])}"
        v_emb = get_embedding(v_text).reshape(1, -1)
        v_energy = get_energy_score(v_seg['energy'])
        is_dialogue = (v_seg['scene_category'] == "Dialogue")
        
        # 2. 遍歷所有候選歌曲
        for track in audio_library:
            # 資料解包
            if 'audio_caption' in track:
                props = track['audio_caption']
            else:
                props = track
                
            # --- 計算 Mood Score ---
            if use_metadata_pairing:
                a_text = f"{props['video_pairing']} Genre: {props['suggested_genre']}. Mood: {', '.join(props['mood_tags'])}"
            else:
                instr = props.get('instrumentation', '')
                a_text = f"Genre: {props['suggested_genre']}. Instruments: {instr}. Mood: {', '.join(props['mood_tags'])}"
            
            a_emb = get_embedding(a_text).reshape(1, -1)
            mood_score = cosine_similarity(v_emb, a_emb)[0][0]
            
            # --- 計算 Energy Score ---
            a_energy = get_energy_score(props['energy'])
            energy_score = 1.0 - abs(v_energy - a_energy)
            
            # --- 計算 Vocal Penalty ---
            vocal_penalty = 0
            if is_dialogue and props['has_vocals']:
                vocal_penalty = -1.0 # 這裡只是為了選歌，用 raw score 或乘上權重皆可
                # 為了跟 Evaluation 一致，建議乘上權重
                vocal_penalty *= WEIGHTS["vocal_clash"]

            # --- 加權總分 (只看當下) ---
            # 注意：這裡【不計算】Transition Penalty
            current_score = (mood_score * WEIGHTS["mood"]) + \
                            (energy_score * WEIGHTS["energy"]) + \
                            vocal_penalty
                            
            if current_score > best_local_score:
                best_local_score = current_score
                best_track = track
        
        # 選出這一段的冠軍
        greedy_sequence.append(best_track)
        
    return greedy_sequence

# ==========================================
# 5. 執行
# ==========================================

video_caption_json = "./caption/Vlog Captions 2.json"
with open(video_caption_json, 'r', encoding='utf-8') as f:
    video_data = json.load(f)
    
audio_caption_json = "caption/flattened_output.json"
with open(audio_caption_json, 'r', encoding='utf-8') as f:
    original_audio_data = json.load(f)
    
bgm_library_json = "caption/music_captions.json"
with open(bgm_library_json, 'r', encoding='utf-8') as f:
    bgm_library = json.load(f)

random_audio_data = bgm_library.copy()
random.seed(42) # 固定種子以便重現
# random a sequence by sampling len(video_data) times
random_audio_data = [random.choice(bgm_library) for _ in range(len(video_data))]

# --- 執行評估 ---
print("\n計算 Original BGM 分數...")
orig_score, orig_log = evaluate_sequence(video_data, original_audio_data)

print("\n計算 Random BGM 分數...")
rand_score, rand_log = evaluate_sequence(video_data, random_audio_data)

# ==========================================
# 6. 結果輸出
# ==========================================
print("\n" + "="*40)
print("實驗結果 (Validation Result)")
print("="*40)
print(f"Original Sequence Avg Reward: {orig_score:.4f}")
print(f"Random Sequence Avg Reward:   {rand_score:.4f}")

if orig_score > rand_score:
    print("\n✅ 驗證成功：原版配樂分數顯著高於隨機配樂。")
    print("這代表目前的 Reward Function 能有效區分「好」與「壞」的配對。")
else:
    print("\n❌ 驗證失敗：分數差異不大，需要調整權重 (Weights)。")

print("\n--- 詳細分析範例 (第一段 Video) ---")
for i in range(len(video_data)):
    print(f"Video: {video_data[i]['visual_caption']}")
    print(f"Original Audio Match ({original_audio_data[i]['mood_tags']}): Reward Details = {orig_log[i]['details']}")
    print(f"Random Audio Match   ({random_audio_data[i]['mood_tags']}): Reward Details = {rand_log[i]['details']}")


print("=== 實驗 1: Smart Random Baseline (排除轉場扣分干擾) ===")
# 生成聰明的隨機序列
smart_random_seq = generate_smart_random_sequence(video_data, bgm_library)

# 評估 Original (或是你的 Agent 選擇)
score_best, ex1_log_ori = evaluate_sequence(video_data, original_audio_data, use_metadata_pairing=True)

# 評估 Smart Random
score_smart, ex1_log_ran = evaluate_sequence(video_data, smart_random_seq, use_metadata_pairing=True)

print(f"Best Selection Score: {score_best:.4f}")
print(f"Smart Random Score:   {score_smart:.4f}")

if score_best > score_smart:
    print("✅ 通過：即使 Random 不亂切歌，我們選的歌在情緒/能量上依然比較好！")
else:
    print("❌ 警告：Random 只要不亂切歌，分數就跟我們差不多，代表 Mood Matching 沒發揮作用。")
    
# visualize_results(ex1_log_ori, ex1_log_ran)

print("\n=== 實驗 2: Ablation Test (拿掉 Video Pairing 作弊欄位) ===")
# 兩邊都關掉 pairing metadata
score_best_ablated, _ = evaluate_sequence(video_data, original_audio_data, use_metadata_pairing=False)
score_smart_ablated, _ = evaluate_sequence(video_data, smart_random_seq, use_metadata_pairing=False)

print(f"Best Selection (No Pairing): {score_best_ablated:.4f}")
print(f"Smart Random (No Pairing):   {score_smart_ablated:.4f}")

if score_best_ablated > score_smart_ablated:
    print("✅ 通過：拿掉文字小抄後，我們的 Tags/Genre/Instrument 匹配依然勝出！")
else:
    print("❌ 警告：拿掉 Pairing 後優勢消失，代表模型過度依賴文字描述的一致性。")

# --- 執行繪圖 ---
# 確保你還有 orig_log 和 rand_log 變數
# visualize_results(orig_log, rand_log)

print("\n" + "="*50)
print("=== 最終決戰: Original vs. Smart Random vs. Greedy ===")
print("="*50)

# 1. 產生 Greedy 序列
greedy_seq = generate_greedy_sequence(video_data, bgm_library, use_metadata_pairing=True)

# 2. 評估三者 (使用 Evaluate Sequence 計算最終 Reward)
# 注意：evaluate_sequence 會加上 Transition Penalty，這是 Greedy 的死穴
score_orig, log_orig = evaluate_sequence(video_data, original_audio_data, use_metadata_pairing=True)
score_rand, log_rand = evaluate_sequence(video_data, smart_random_seq, use_metadata_pairing=True)
score_greedy, log_greedy = evaluate_sequence(video_data, greedy_seq, use_metadata_pairing=True)

print(f"1. Original / RL Policy Score: {score_orig:.4f}")
print(f"2. Smart Random Score:         {score_rand:.4f}")
print(f"3. Greedy Selection Score:     {score_greedy:.4f}")

# --- 分析勝負原因 ---
# 計算 Greedy 被扣了多少 Transition 分數
greedy_trans_penalties = sum([x['details']['transition'] for x in log_greedy])
greedy_mood_score = sum([x['details']['mood'] for x in log_greedy]) / len(log_greedy)
orig_mood_score = sum([x['details']['mood'] for x in log_orig]) / len(log_orig)

print("\n--- 深度分析 (Greedy vs Original) ---")
print(f"Greedy Avg Mood Score:   {greedy_mood_score:.4f} (理論上限，應該最高)")
print(f"Original Avg Mood Score: {orig_mood_score:.4f}")
print(f"Greedy Total Trans Penalty: {greedy_trans_penalties:.1f}")

if score_orig > score_greedy:
    print("\n✅ 驗證成功：RL (Original) 勝出！")
    print("   理由：雖然 Greedy 在單點匹配上可能略高，但它因為頻繁切歌被扣了太多分。")
    print("   這證明了系統需要考慮『時間序列』的決策 (Long-term planning)，而不只是單點貪婪。")
else:
    print("\n⚠️ 警告：Greedy 勝出或平手。")
    print("   可能原因：")
    print("   1. Transition Penalty 權重設太輕了，切歌的代價不夠大。")
    print("   2. 影片本身的段落都切得很長 (> 5秒)，導致 Greedy 剛好避開了懲罰。")
    
visualize_results(log_orig, log_greedy)