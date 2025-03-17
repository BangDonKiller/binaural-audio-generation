import argparse
import torch

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--hdf5FolderPath', default="D:/Dataset/FAIR-Play/splits/split8", help='path to the folder that contains train.h5, val.h5 and test.h5')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--name', type=str, default='spatialAudioVisual', help='name of the experiment. It decides where to store models')
        self.parser.add_argument('--checkpoints_dir', type=str, default='checkpoints/', help='models are saved here')
        self.parser.add_argument('--audio_model', type=str, default='AudioNet', help='audio model type')
        self.parser.add_argument('--visual_model', type=str, choices=['VisualNet', 'VisualNetDilated'], default='VisualNet', help='visual model type')
        self.parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
        self.parser.add_argument('--nThreads', default=1, type=int, help='# threads for loading data')
        self.parser.add_argument('--audio_sampling_rate', default=16000, type=int, help='audio sampling rate')
        self.parser.add_argument('--audio_length', default=0.63, type=float, help='audio length, default 0.63s')
        self.parser.add_argument('--norm_mode', type=str, choices=['syncbn', 'bn', 'in'], default='syncbn', help='norm mode')
        self.enable_data_augmentation = True
        self.initialized = True

    def parse(self):
        if not self.initialized:
                self.initialize()
        self.opt = self.parser.parse_args()

        self.opt.mode = self.mode
        self.opt.isTrain = self.isTrain
        self.opt.enable_data_augmentation = self.enable_data_augmentation

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
                id = int(str_id)
                if id >= 0:
                        self.opt.gpu_ids.append(id)

        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
                torch.cuda.set_device(self.opt.gpu_ids[0])


        #I should process the opt here, like gpu ids, etc.
        # print('------------ Options -------------')
        # for k, v in sorted(args.items()):
        #         print('%s: %s' % (str(k), str(v)))
        # print('-------------- End ----------------')


        # save to the disk
        return self.opt
