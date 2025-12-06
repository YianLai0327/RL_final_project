#!/bin/bash

# Google Drive 檔案 ID
FILE_ID="1obxgQoYLuPggcJUH8yHAnTgJPxswQsa4"

# 下載檔案
echo "正在下載檔案..."
gdown "https://drive.google.com/uc?id=$FILE_ID" -O separated.zip

# 檢查下載是否成功
if [ ! -f separated.zip ]; then
    echo "下載失敗"
    exit 1
fi

# 解壓縮
echo "正在解壓縮..."
unzip -q separated.zip

# 重命名資料夾為 separated (如果已存在則移除舊的)
if [ -d separated ]; then
    rm -rf separated_old
    mv separated separated_old
fi

# 如果解壓出來的資料夾名稱不是 separated，則進行重命名
EXTRACTED_DIR=$(unzip -l separated.zip | head -2 | tail -1 | awk '{print $NF}' | cut -d'/' -f1)

if [ -n "$EXTRACTED_DIR" ] && [ "$EXTRACTED_DIR" != "separated" ]; then
    mv "$EXTRACTED_DIR" separated
fi

# 清理 zip 檔案
rm separated.zip

echo "完成！檔案已下載並解壓縮到 'separated' 資料夾"