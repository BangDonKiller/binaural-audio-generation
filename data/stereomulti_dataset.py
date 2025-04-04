import os.path
import librosa
import h5py
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageEnhance
import torchvision.transforms as transforms

def normalize(samples, desired_rms = 0.1, eps = 1e-4):
  rms = np.maximum(eps, np.sqrt(np.mean(samples**2)))
  samples = samples * (desired_rms / rms)
  return samples

def generate_spectrogram(audio):
    spectro = librosa.core.stft(audio, n_fft=512, hop_length=160, win_length=400, center=True)
    real = np.expand_dims(np.real(spectro), axis=0)
    imag = np.expand_dims(np.imag(spectro), axis=0)
    spectro_two_channel = np.concatenate((real, imag), axis=0)
    return spectro_two_channel

def process_image(image, augment):
    image = image.resize((480,240))
    w,h = image.size
    w_offset = w - 448
    h_offset = h - 224
    left = random.randrange(0, w_offset + 1)
    upper = random.randrange(0, h_offset + 1)
    image = image.crop((left, upper, left+448, upper+224))

    if augment:
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(random.random()*0.6 + 0.7)
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(random.random()*0.6 + 0.7)
    return image


class StereoDataset(Dataset):
    def __init__(self, opt):
        Dataset.__init__(self)

        self.opt = opt
        self.audios = []

        #load hdf5 file here
        h5f_path = os.path.join(opt.hdf5FolderPath, opt.mode+".h5")
        h5f = h5py.File(h5f_path, 'r')
        self.audios = h5f['audio'][:]
        self.audios = [bytes.decode(_path) for _path in self.audios]

        for i in range(len(self.audios)):
            self.audios[i] = self.audios[i].replace(
            "/private/home/rhgao/datasets/BINAURAL_MUSIC_ROOM/binaural16k/",
            "D:/Dataset/FAIR-Play/binaural_audios/",
        )


        normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
        )
        vision_transform_list = [transforms.ToTensor(), normalize]
        self.vision_transform = transforms.Compose(vision_transform_list)

    def _get_stereo_item(self, index):
        # 載入音訊
        audio, audio_rate = librosa.load(self.audios[index].strip(), sr=self.opt.audio_sampling_rate, mono=False)

        # 隨機選擇音訊片段
        audio_start_time = random.uniform(0, 9.9 - self.opt.audio_length)
        audio_end_time = audio_start_time + self.opt.audio_length
        audio_start = int(audio_start_time * self.opt.audio_sampling_rate)
        audio_end = audio_start + int(self.opt.audio_length * self.opt.audio_sampling_rate)
        audio = audio[:, audio_start:audio_end]
        audio = normalize(audio)

        # 隨機翻轉音訊通道
        x = random.randint(1, 10)
        if x < 6:
            audio_channel1 = audio[0, :]
            audio_channel2 = audio[1, :]
            flag = np.float32(1.0)
        else:
            audio_channel2 = audio[0, :]
            audio_channel1 = audio[1, :]
            flag = np.float32(0.0)

        # 影像處理
        # video_path = self.audios[index].replace("binaural_audios", "videos")[:-4] + ".mp4"
        frame_path = self.audios[index].replace("binaural_audios", "mask_frames")[:-4]
        # print(frame_path)

        # 選擇最接近音訊時間的影格
        frame_index = int(round(((audio_start_time + audio_end_time) / 2.0 + 0.05) * 10))  # 每秒擷取 10 張影格
        frame = process_image(Image.open(os.path.join(frame_path, str(frame_index) + '.jpg')).convert('RGB'), self.opt.enable_data_augmentation)
        frame = self.vision_transform(frame)

        # 計算頻譜圖
        audio_diff_spec = torch.FloatTensor(generate_spectrogram(audio[0, :] - audio[1, :]))
        audio_mix_spec = torch.FloatTensor(generate_spectrogram(audio_channel1 + audio_channel2))
        left = torch.FloatTensor(generate_spectrogram(np.asfortranarray(audio_channel1)))
        right = torch.FloatTensor(generate_spectrogram(np.asfortranarray(audio_channel2)))

        return {'frame': frame, 'audio_diff_spec': audio_diff_spec, 'audio_mix_spec': audio_mix_spec, 'left': left, 'right': right, 'flag': flag}

    def __getitem__(self, index):
        result = self._get_stereo_item(index)
        return result


    def __len__(self):
        return len(self.audios)

    def name(self):
        return 'StereoDataset'

