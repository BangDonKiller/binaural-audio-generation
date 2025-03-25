import os
import librosa
import numpy as np
import scipy.signal

target_folder = "eval_demo/stereo/model-best/split5_100"

def STFT_Distance(input_audio, predicted_audio):
    # 讀取音訊檔案（保留 stereo）
    y1, sr1 = librosa.load(input_audio, sr=None, mono=False)
    y2, sr2 = librosa.load(predicted_audio, sr=None, mono=False)

    # 確保取樣率一致
    if sr1 != sr2:
        raise ValueError(f"Sampling rates do not match: {sr1} vs {sr2}")

    # 確保 shape 一致（stereo）
    if y1.shape != y2.shape:
        min_len = min(y1.shape[1], y2.shape[1])  # 取最短長度
        y1, y2 = y1[:, :min_len], y2[:, :min_len]

    # 計算每個 channel 的 STFT
    stft1_L = librosa.stft(y1[0])
    stft1_R = librosa.stft(y1[1])
    stft2_L = librosa.stft(y2[0])
    stft2_R = librosa.stft(y2[1])

    # 計算 STFT 距離（對左、右 channel 分別計算 MSE，然後取平均）
    distance_L = np.mean(np.abs(stft1_L - stft2_L) ** 2)
    distance_R = np.mean(np.abs(stft1_R - stft2_R) ** 2)

    return (distance_L + distance_R) / 2  # 平均左右聲道距離

def Envelope_Distance(input_audio, predicted_audio):
    # 讀取音訊檔案（保留 stereo）
    y1, sr1 = librosa.load(input_audio, sr=None, mono=False)
    y2, sr2 = librosa.load(predicted_audio, sr=None, mono=False)

    # 確保取樣率一致
    if sr1 != sr2:
        raise ValueError(f"Sampling rates do not match: {sr1} vs {sr2}")

    # 確保 shape 一致（stereo）
    if y1.shape != y2.shape:
        min_len = min(y1.shape[1], y2.shape[1])  # 取最短長度
        y1, y2 = y1[:, :min_len], y2[:, :min_len]

    # 計算每個 channel 的包絡線
    env1_L = np.abs(scipy.signal.hilbert(y1[0]))
    env1_R = np.abs(scipy.signal.hilbert(y1[1]))
    env2_L = np.abs(scipy.signal.hilbert(y2[0]))
    env2_R = np.abs(scipy.signal.hilbert(y2[1]))

    # 計算 Envelope 距離（MSE）
    distance_L = np.mean((env1_L - env2_L) ** 2)
    distance_R = np.mean((env1_R - env2_R) ** 2)

    return (distance_L + distance_R) / 2  # 平均左右聲道距離

def main():
    total_STFT_dis = 0
    total_env_dis = 0
    count = 0

    for root, _, files in os.walk(target_folder):
        input_audio = None
        predicted_audio = None

        # 找到對應的 input 和 predicted 檔案
        for file in files:
            if file.startswith("predicted_binaural") and file.endswith(".wav"):
                predicted_audio = os.path.join(root, file)
            elif file.startswith("input_binaural") and file.endswith(".wav"):
                input_audio = os.path.join(root, file)

        # 計算 STFT 距離 & Envelope 距離
        if input_audio and predicted_audio:
            total_STFT_dis += STFT_Distance(input_audio, predicted_audio)
            total_env_dis += Envelope_Distance(input_audio, predicted_audio)
            count += 1

    # 計算平均距離
    if count > 0:
        avg_STFT_dis = total_STFT_dis / count
        avg_env_dis = total_env_dis / count
        print("Average STFT Distance:", avg_STFT_dis)
        print("Average Envelope Distance:", avg_env_dis)
    else:
        print("No valid input-predicted audio pairs found.")

if __name__ == "__main__":
    main()
