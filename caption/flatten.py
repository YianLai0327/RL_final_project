import json
import os

def flatten_data(input_file, output_file):
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        flattened_list = []
        
        # 確保 data 是一個 list
        if isinstance(data, dict): 
            data = [data] # 如果只有單筆資料，轉成 list
            
        for index, item in enumerate(data):
            # 複製一份以免改到原始資料
            new_item = item.copy()
            
            # 檢查是否有 'audio_caption' 這個巢狀 key
            if 'audio_caption' in new_item:
                # 取出內部的 dictionary
                audio_props = new_item.pop('audio_caption')
                # 將內部的 key-value 更新到最外層
                new_item.update(audio_props)
            
            # 【重要補丁】
            # 你的 RL Code 需要 'filename' 來判斷轉場
            # 如果資料裡原本沒有 filename，我們幫原版影片補一個固定的名字
            if 'filename' not in new_item:
                new_item['filename'] = "original_video_track.mp3"
                
            flattened_list.append(new_item)
            
        # 寫入新檔案
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(flattened_list, f, indent=2, ensure_ascii=False)
            
        print(f"✅ 轉換完成！已輸出至 {output_file}")
        print(f"總共處理了 {len(flattened_list)} 筆資料。")
        
    except FileNotFoundError:
        print("❌ 找不到 input.json，請確認檔案名稱。")
    except Exception as e:
        print(f"❌ 發生錯誤: {e}")

# --- 使用方式 ---
# 1. 把你的原始資料存成 'input.json' (和此腳本放在同目錄)
# 2. 執行此腳本
if __name__ == "__main__":
    if not os.path.exists('"./caption/audio_captions_22.json'):
        print("❌ 找不到 './caption/audio_captions_22.json'，請確認檔案路徑。")
    flatten_data('./caption/audio_captions_22.json', 'flattened_output.json')