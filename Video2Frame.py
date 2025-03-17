import cv2
import os

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
    
video_path = "D:/Dataset/FAIR-Play/videos"
frame_path = "D:/Dataset/FAIR-Play/frames"
for video in os.listdir(video_path):
    create_video_frames(os.path.join(video_path, video), os.path.join(frame_path, video.split(".")[0]))