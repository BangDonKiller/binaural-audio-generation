# import h5py

# # 打開 .h5 檔案（以唯讀模式）
# file_path = "D:/Dataset/FAIR-Play/splits/split8/train.h5"
# with h5py.File(file_path, "r") as f:

#     def print_structure(name, obj):
#         print(name, "->", obj)

#     print("HDF5 檔案結構:")
#     f.visititems(print_structure)

# with h5py.File(file_path, "r") as f:
#     dataset_name = "audio"  # 替換成實際的 dataset 名稱
#     data = f[dataset_name][0:20]
#     print("Dataset 內容:", data)

import librosa
from sklearn.metrics.pairwise import cosine_similarity

file1 = "predicted_binaural1.wav"
file2 = "predicted_binaural2.wav"

def mfcc_similarity(file1, file2):
    y1, sr1 = librosa.load(file1)
    y2, sr2 = librosa.load(file2)

    # 提取 MFCC 特徵
    mfcc1 = librosa.feature.mfcc(y=y1, sr=sr1)
    mfcc2 = librosa.feature.mfcc(y=y2, sr=sr2)

    # 確保相同形狀
    min_len = min(mfcc1.shape[1], mfcc2.shape[1])
    mfcc1, mfcc2 = mfcc1[:, :min_len], mfcc2[:, :min_len]

    # 計算餘弦相似度
    similarity = cosine_similarity(mfcc1.T, mfcc2.T)
    return similarity.mean()

print("MFCC Cosine Similarity:", mfcc_similarity(file1, file2))
