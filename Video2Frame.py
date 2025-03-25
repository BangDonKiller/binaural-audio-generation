import cv2
import os
import numpy as np
import subprocess

def create_video_frames(video_path, output_path="data/frames/"):
    # 參數設定
    output_dir = output_path
    fps = 10  # 目標 FPS

    # 建立輸出資料夾
    os.makedirs(output_dir, exist_ok=True)

    # 讀取影片
    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)  # 取得原始 FPS
    frame_interval = int(original_fps / fps)  # 計算要擷取的幀間隔
    frame_count = 0
    saved_count = 1  # 從 1 開始命名

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            filename = os.path.join(output_dir, f"{saved_count}.jpg")
            cv2.imwrite(filename, frame)
            saved_count += 1
        frame_count += 1

    cap.release()
    

def create_mask_video(video_path, output_path):
    input_video = video_path
    temp_video = "output_black.mp4"
    output_video = output_path + ".mp4"

    # 開啟影片
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print("無法開啟影片")
        exit()

    # 取得影片資訊
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # 建立輸出影片 (解析度與輸入相同，但畫面全黑)
    out = cv2.VideoWriter(temp_video, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 產生全黑畫面
        black_frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # 寫入全黑影像
        out.write(black_frame)

    # 釋放資源
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # 使用 FFmpeg 將原始音訊加入黑畫面影片
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-i", temp_video, "-i", input_video,
        "-c:v", "libx264", "-preset", "slow", "-crf", "23", 
        "-c:a", "mp3", "-b:a", "192k", "-map", "0:v:0", "-map", "1:a:0", output_video
    ]
    subprocess.run(ffmpeg_cmd)

    print("處理完成，影片已儲存為 output_black.mp4，可在 Windows Media Player 播放。")


video_path = "D:/Dataset/FAIR-Play/videos"
frame_path = "D:/Dataset/FAIR-Play/frames"
mask_video_path = "D:/Dataset/FAIR-Play/mask_videos"
mask_video_frame_path = "D:/Dataset/FAIR-Play/mask_frames"


for video in os.listdir(video_path):
    # create_mask_video(os.path.join(video_path, video), os.path.join(mask_video_path, video.split(".")[0]))
    create_video_frames(os.path.join(mask_video_path, video), os.path.join(mask_video_frame_path, video.split(".")[0]))