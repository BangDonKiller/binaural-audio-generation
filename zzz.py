import librosa
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

file1 = "predicted_binaural1.wav"
file2 = "predicted_binaural2.wav"
file3 = "predicted_binaural3.wav"

y1, sr1 = librosa.load(file1, mono=False)
y2, sr2 = librosa.load(file2, mono=False)
y3, sr3 = librosa.load(file3, mono=False)

# 只擷取前1秒的內容
y1 = y1[:, :int(sr1*0.05)]
y2 = y2[:, :int(sr2*0.05)]
y3 = y3[:, :int(sr3*0.05)]


def plot_ENV_left(y1, y2, y3): # Pass sample rates to the function
    # 轉為包絡線頻譜圖
    env1_L = np.abs(scipy.signal.hilbert(y1[0]))
    env2_L = np.abs(scipy.signal.hilbert(y2[0]))
    env3_L = np.abs(scipy.signal.hilbert(y3[0]))

    # 繪製包絡線頻譜圖
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.plot(y1[0], label="wave", color="blue")
    plt.plot(env1_L, label="ENV", color="red")
    plt.title("Pretrain Left")

    plt.subplot(132)
    plt.plot(y2[0], label="wave", color="blue")
    plt.plot(env2_L, label="ENV", color="red")
    plt.title("Res 100 Left")

    plt.subplot(133)
    plt.plot(y3[0], label="wave", color="blue")
    plt.plot(env3_L, label="ENV", color="red")
    plt.title("TCN 100 Left")

    plt.tight_layout()
    plt.show()
    
def plot_ENV_right(y1, y2, y3): # Pass sample rates to the function
    # 轉為包絡線頻譜圖
    env1_R = np.abs(scipy.signal.hilbert(y1[1]))
    env2_R = np.abs(scipy.signal.hilbert(y2[1]))
    env3_R = np.abs(scipy.signal.hilbert(y3[1]))
    
    # 繪製包絡線頻譜圖
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.plot(env1_R)
    plt.title("Pretrain Right")
    
    plt.subplot(132)
    plt.plot(env2_R)
    plt.title("Res 100 Right")
    
    plt.subplot(133)
    plt.plot(env3_R)
    plt.title("TCN 100 Right")
    
    plt.tight_layout()
    plt.show()
    


def plot_STFT(y1, y2, y3, sr1, sr2, sr3): # Pass sample rates to the function
    # 轉為STFT頻譜圖
    D1 = librosa.amplitude_to_db(np.abs(librosa.stft(y1[0])), ref=np.max) if y1.ndim > 1 and y1.shape[1] > 0 else np.zeros((1025, 10)) # Handle empty audio
    D2 = librosa.amplitude_to_db(np.abs(librosa.stft(y2[0])), ref=np.max) if y2.ndim > 1 and y2.shape[1] > 0 else np.zeros((1025, 10)) # Handle empty audio
    D3 = librosa.amplitude_to_db(np.abs(librosa.stft(y3[0])), ref=np.max) if y3.ndim > 1 and y3.shape[1] > 0 else np.zeros((1025, 10)) # Handle empty audio


    # 繪製頻譜圖
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    librosa.display.specshow(D1, sr=sr1, x_axis="time", y_axis="log")
    plt.colorbar(format="%+2.0f dB")
    plt.title("Predicted 1")

    plt.subplot(132)
    librosa.display.specshow(D2, sr=sr2, x_axis="time", y_axis="log")
    plt.colorbar(format="%+2.0f dB")
    plt.title("Predicted 2")

    plt.subplot(133)
    librosa.display.specshow(D3, sr=sr3, x_axis="time", y_axis="log")
    plt.colorbar(format="%+2.0f dB")
    plt.title("Predicted 3")

    plt.tight_layout()
    plt.show()

def plot_waveform_left(y1, y2, y3, sr1, sr2, sr3): # Pass sample rates to the function
    # 繪製波形圖（左聲道）
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    if y1.ndim > 1 and y1.shape[1] > 0: # Handle empty audio
        librosa.display.waveshow(y1[0], sr=sr1)
    plt.title("Pretrain Left")

    plt.subplot(132)
    if y2.ndim > 1 and y2.shape[1] > 0: # Handle empty audio
        librosa.display.waveshow(y2[0], sr=sr2)
    plt.title("Res Train 100 Left")

    plt.subplot(133)
    if y3.ndim > 1 and y3.shape[1] > 0: # Handle empty audio
        librosa.display.waveshow(y3[0], sr=sr3)
    plt.title("TCN Train 100 Left")

    plt.tight_layout()
    plt.show()

def plot_waveform_right(y1, y2, y3, sr1, sr2, sr3): # Pass sample rates to the function
    # 繪製波形圖（右聲道）
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    if y1.ndim > 1 and y1.shape[1] > 1: # Handle mono or empty audio
        librosa.display.waveshow(y1[1], sr=sr1)
    plt.title("Pretrain Right")

    plt.subplot(132)
    if y2.ndim > 1 and y2.shape[1] > 1: # Handle mono or empty audio
        librosa.display.waveshow(y2[1], sr=sr2)
    plt.title("Res Train 100 Right")

    plt.subplot(133)
    if y3.ndim > 1 and y3.shape[1] > 1: # Handle mono or empty audio
        librosa.display.waveshow(y3[1], sr=sr3)
    plt.title("TCN Train 100 Right")

    plt.tight_layout()
    plt.show()



plot_ENV_left(y1, y2, y3)
plot_ENV_right(y1, y2, y3)
# plot_waveform_left(y1, y2, y3, sr1, sr2, sr3)
# plot_waveform_right(y1, y2, y3, sr1, sr2, sr3)
# plot_STFT(y1, y2, y3, sr1, sr2, sr3)