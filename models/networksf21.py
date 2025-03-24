import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm
from .sync_batchnorm import SynchronizedBatchNorm2d


def unet_conv(input_nc, output_nc, norm_layer=nn.BatchNorm2d):
    downconv = nn.Conv2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1)
    downrelu = nn.LeakyReLU(0.2, True)
    downnorm = norm_layer(output_nc)
    return nn.Sequential(*[downconv, downnorm, downrelu])


def unet_upconv(input_nc, output_nc, outermost=False, norm_layer=nn.BatchNorm2d):
    upconv = nn.ConvTranspose2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1)
    uprelu = nn.ReLU(True)
    upnorm = norm_layer(output_nc)
    if not outermost:
        return nn.Sequential(*[upconv, upnorm, uprelu])
    else:
        return nn.Sequential(*[upconv, nn.Sigmoid()])


def create_conv(
    input_channels,
    output_channels,
    kernel,
    paddings,
    batch_norm=True,
    Relu=True,
    stride=1,
):
    # model = [
    #     nn.Conv2d(
    #         input_channels, output_channels, kernel, stride=stride, padding=paddings
    #     )
    # ]
    model = [
        nn.Conv1d(
            input_channels, output_channels, kernel, stride=stride, padding=paddings
        )
    ]
    
    if batch_norm:
        # model.append(nn.BatchNorm2d(output_channels))
        model.append(nn.BatchNorm1d(output_channels))
    if Relu:
        model.append(nn.ReLU())
    return nn.Sequential(*model)


def create_conv_sig(
    input_channels,
    output_channels,
    kernel,
    paddings,
    batch_norm=True,
    Sigmoid=True,
    stride=1,
):
    # model = [
    #     nn.Conv2d(
    #         input_channels, output_channels, kernel, stride=stride, padding=paddings
    #     )
    # ]
    model = [
        nn.Conv1d(
            input_channels, output_channels, kernel, stride=stride, padding=paddings
        )
    ]
    if batch_norm:
        # model.append(nn.BatchNorm2d(output_channels))
        model.append(nn.BatchNorm1d(output_channels))
    if Sigmoid:
        model.append(nn.Sigmoid())
    return nn.Sequential(*model)


def _get_spectrogram(mask_prediction, audio_mix):
    spec_diff_real = (
        audio_mix[:, 0, :-1, :] * mask_prediction[:, 0, :, :]
        - audio_mix[:, 1, :-1, :] * mask_prediction[:, 1, :, :]
    )
    spec_diff_img = (
        audio_mix[:, 0, :-1, :] * mask_prediction[:, 1, :, :]
        + audio_mix[:, 1, :-1, :] * mask_prediction[:, 0, :, :]
    )
    binaural_spectrogram = torch.cat(
        (spec_diff_real.unsqueeze(1), spec_diff_img.unsqueeze(1)), 1
    )

    return binaural_spectrogram


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find("Linear") != -1:
        m.weight.data.normal_(0.0, 0.02)


class AVFusionBlock(nn.Module):
    def __init__(self, audio_channel, vision_channel=512):
        super().__init__()
        self.channel_mapping_conv_w = nn.Conv1d(
            vision_channel, audio_channel, kernel_size=1
        )
        self.activation = nn.ReLU()

    def forward(self, audiomap, visionmap):
        visionmap = visionmap.view(visionmap.size(0), visionmap.size(1), -1)
        vision_W = self.channel_mapping_conv_w(visionmap)
        vision_W = self.activation(vision_W)
        (bz, c, wh) = vision_W.size()
        vision_W = vision_W.view(bz, c, wh)
        vision_W = vision_W.transpose(2, 1)
        audio_size = audiomap.size()
        output = torch.bmm(vision_W, audiomap.view(bz, audio_size[1], -1)).view(
            bz, wh, *audio_size[2:]
        )
        return output


class VisualNet(nn.Module):
    def __init__(self, original_resnet):
        super().__init__()
        layers = list(original_resnet.children())[0:-2]
        self.feature_extraction = nn.Sequential(*layers)  # features before conv1x1

    def forward(self, x):
        x = self.feature_extraction(x)
        return x


class VisualNetDilated(nn.Module):
    def __init__(self, orig_resnet):
        super().__init__()
        from functools import partial

        fc_dim = 512
        dilate_scale = 16
        conv_size = 3

        if dilate_scale == 8:
            orig_resnet.layer3.apply(partial(self._nostride_dilate, dilate=2))
            orig_resnet.layer4.apply(partial(self._nostride_dilate, dilate=4))
        elif dilate_scale == 16:
            orig_resnet.layer4.apply(partial(self._nostride_dilate, dilate=2))

        self.features = nn.Sequential(*list(orig_resnet.children())[:-2])

        self.fc = nn.Conv2d(512, 512, kernel_size=conv_size, padding=conv_size // 2)

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        x = self.features(x)
        x = F.avg_pool2d(x, kernel_size=2, stride=2)

        return x


class AudioNet1(nn.Module):
    def __init__(self, ngf=64, input_nc=2, output_nc=2, norm_mode="syncbn"):
        super().__init__()
        # initialize layers
        if norm_mode == "syncbn":
            norm_layer = SynchronizedBatchNorm2d
        else:
            norm_layer = nn.BatchNorm2d
        self.audionet_convlayer1 = unet_conv(input_nc, ngf, norm_layer=norm_layer)
        self.audionet_convlayer2 = unet_conv(ngf, ngf * 2, norm_layer=norm_layer)
        self.audionet_convlayer3 = unet_conv(ngf * 2, ngf * 4, norm_layer=norm_layer)
        self.audionet_convlayer4 = unet_conv(ngf * 4, ngf * 8, norm_layer=norm_layer)
        self.audionet_convlayer5 = unet_conv(ngf * 8, ngf * 8, norm_layer=norm_layer)
        self.audionet_upconvlayer1 = unet_upconv(
            1296, ngf * 8, norm_layer=norm_layer
        )  # 1296 (audio-visual feature) = 784 (visual feature) + 512 (audio feature)
        self.audionet_upconvlayer2 = unet_upconv(
            ngf * 16, ngf * 4, norm_layer=norm_layer
        )
        self.audionet_upconvlayer3 = unet_upconv(
            ngf * 8, ngf * 2, norm_layer=norm_layer
        )
        self.audionet_upconvlayer4 = unet_upconv(ngf * 4, ngf, norm_layer=norm_layer)
        self.audionet_upconvlayer5 = unet_upconv(
            ngf * 2, output_nc, outermost=True, norm_layer=norm_layer
        )
        self.conv1x1 = create_conv(
            512, 8, 1, 0
        )  # reduce dimension of extracted visual features

    def forward(self, audio_diff, audio_mix, visual_feat, return_upfeatures=False):
        audio_conv1feature = self.audionet_convlayer1(audio_mix)
        audio_conv2feature = self.audionet_convlayer2(audio_conv1feature)
        audio_conv3feature = self.audionet_convlayer3(audio_conv2feature)
        audio_conv4feature = self.audionet_convlayer4(audio_conv3feature)
        audio_conv5feature = self.audionet_convlayer5(audio_conv4feature)

        visual_feat = self.conv1x1(visual_feat)
        visual_feat = visual_feat.view(
            visual_feat.shape[0], -1, 1, 1
        )  # flatten visual feature
        visual_feat = visual_feat.repeat(
            1, 1, audio_conv5feature.shape[-2], audio_conv5feature.shape[-1]
        )  # tile visual feature

        audioVisual_feature = torch.cat((visual_feat, audio_conv5feature), dim=1)

        audio_upconv1feature = self.audionet_upconvlayer1(audioVisual_feature)
        audio_upconv2feature = self.audionet_upconvlayer2(
            torch.cat((audio_upconv1feature, audio_conv4feature), dim=1)
        )
        audio_upconv3feature = self.audionet_upconvlayer3(
            torch.cat((audio_upconv2feature, audio_conv3feature), dim=1)
        )
        audio_upconv4feature = self.audionet_upconvlayer4(
            torch.cat((audio_upconv3feature, audio_conv2feature), dim=1)
        )
        upfeatures = [
            audio_upconv1feature,
            audio_upconv2feature,
            audio_upconv3feature,
            audio_upconv4feature,
        ]

        mask_prediction = (
            self.audionet_upconvlayer5(
                torch.cat((audio_upconv4feature, audio_conv1feature), dim=1)
            )
            * 2
            - 1
        )
        binaural_spectrogram = _get_spectrogram(mask_prediction, audio_mix)
        output = {
            "mask_prediction": mask_prediction,
            "binaural_spectrogram": binaural_spectrogram,
            "audio_gt": audio_diff[:, :, :-1, :],
        }

        if return_upfeatures:
            return upfeatures, output
        else:
            return output


class Attention(nn.Module):
    def __init__(self, input_nc=512, output_nc=512, kernel_size=1, norm_mode="syncbn"):
        super().__init__()
        if norm_mode == "syncbn":
            norm_layer = SynchronizedBatchNorm2d
        else:
            norm_layer = nn.BatchNorm2d

        self.conv1x1_1 = create_conv(input_nc, output_nc, kernel_size, 0)
        self.conv1xa_2 = create_conv_sig(output_nc, input_nc, kernel_size, 0)

    def forward(self, v):
        input_v = v
        v = self.conv1x1_1(v)
        v = self.conv1xa_2(v)
        v = v * input_v
        return v


class Stereo(nn.Module):
    """
    音訊翻轉任務編碼器
    輸入為左右聲道的音訊
    """

    def __init__(self, ngf=64, input_nc=4, output_nc=2, norm_mode="syncbn"):
        super().__init__()
        # initialize layers
        if norm_mode == "syncbn":
            norm_layer = SynchronizedBatchNorm2d
        else:
            norm_layer = nn.BatchNorm2d
        self.audionet_convlayer1 = unet_conv(input_nc, ngf, norm_layer=norm_layer)
        self.audionet_convlayer2 = unet_conv(ngf, ngf * 2, norm_layer=norm_layer)
        self.audionet_convlayer3 = unet_conv(ngf * 2, ngf * 4, norm_layer=norm_layer)
        self.audionet_convlayer4 = unet_conv(ngf * 4, ngf * 8, norm_layer=norm_layer)
        self.audionet_convlayer5 = unet_conv(ngf * 8, ngf * 8, norm_layer=norm_layer)

    def forward(self, left, right):
        audio = torch.cat((left, right), dim=1)
        audio_conv1feature = self.audionet_convlayer1(audio)
        audio_conv2feature = self.audionet_convlayer2(audio_conv1feature)
        audio_conv3feature = self.audionet_convlayer3(audio_conv2feature)
        audio_conv4feature = self.audionet_convlayer4(audio_conv3feature)
        audio_conv5feature = self.audionet_convlayer5(audio_conv4feature)
        return audio_conv5feature


class netD3(nn.Module):
    """
    音訊翻轉任務的輸出層
    輸入為音訊特徵和視覺特徵
    """

    def __init__(self, ndf=1, nc=2320, nb_label=2):

        super(netD3, self).__init__()
        self.conv1x1 = create_conv(512, 8, 1, 0)
        self.conv1x11 = create_conv(1296, 512, 1, 0)
        self.disc_linear = nn.Linear(ndf * 1, 1)
        self.sigmoid = nn.Sigmoid()
        self.ndf = ndf

    def forward(self, input, v):

        v = self.conv1x1(v)
        v = v.view(v.shape[0], -1, 1, 1)  # flatten visual feature
        v = v.repeat(1, 1, 8, 2)
        input = torch.cat((v, input), dim=1)
        
        input = input.view(input.size(0), input.size(1), -1)
        x = self.conv1x11(input)
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        # x = x.view(-1, self.ndf * 1)
        x = x.view(x.size(0), -1)
        
        s = self.disc_linear(x)
        s = self.sigmoid(s)
        return s


class AssoConv(nn.Module):
    def __init__(
        self,
        ngf=64,
        output_nc=2,
        visual_feat_size=7 * 14,
        vision_channel=512,
        norm_mode="syncbn",
    ):
        super().__init__()

        if norm_mode == "syncbn":
            norm_layer = SynchronizedBatchNorm2d
        else:
            norm_layer = nn.BatchNorm2d
        self.fusion = AVFusionBlock(ngf, vision_channel)
        self.lastconv_left = unet_upconv(
            visual_feat_size, output_nc, outermost=True, norm_layer=norm_layer
        )
        self.lastconv_right = unet_upconv(
            visual_feat_size, output_nc, outermost=True, norm_layer=norm_layer
        )

    def forward(self, audio_mix, visual_feat, upfeatures):
        audio_upconv4feature = upfeatures[-1]
        AVfusion_feature = self.fusion(audio_upconv4feature, visual_feat)
        pred_left_mask = self.lastconv_left(AVfusion_feature) * 2 - 1
        pred_right_mask = self.lastconv_right(AVfusion_feature) * 2 - 1

        left_spectrogram = _get_spectrogram(pred_left_mask, audio_mix)
        right_spectrogram = _get_spectrogram(pred_right_mask, audio_mix)
        output = {"pred_left": left_spectrogram, "pred_right": right_spectrogram}

        return output


class APNet1(nn.Module):
    def __init__(
        self,
        ngf=64,
        output_nc=2,
        visual_feat_size=7 * 14,
        vision_channel=512,
        norm_mode="syncbn",
    ):
        super().__init__()

        if norm_mode == "syncbn":
            norm_layer = SynchronizedBatchNorm2d
        else:
            norm_layer = nn.BatchNorm2d
        self.fusion1 = AVFusionBlock(ngf * 8, vision_channel)
        self.fusion2 = AVFusionBlock(ngf * 4, vision_channel)
        self.fusion3 = AVFusionBlock(ngf * 2, vision_channel)
        self.fusion4 = AVFusionBlock(ngf * 1, vision_channel)

        self.fusion_upconv1 = unet_upconv(
            visual_feat_size, visual_feat_size, norm_layer=norm_layer
        )
        self.fusion_upconv2 = unet_upconv(
            visual_feat_size * 2, visual_feat_size, norm_layer=norm_layer
        )
        self.fusion_upconv3 = unet_upconv(
            visual_feat_size * 2, visual_feat_size, norm_layer=norm_layer
        )
        self.lastconv_left = unet_upconv(
            visual_feat_size * 2, output_nc, outermost=True, norm_layer=norm_layer
        )
        self.lastconv_right = unet_upconv(
            visual_feat_size * 2, output_nc, outermost=True, norm_layer=norm_layer
        )

    def forward(self, audio_mix, visual_feat, upfeatures):
        (
            audio_upconv1feature,
            audio_upconv2feature,
            audio_upconv3feature,
            audio_upconv4feature,
        ) = upfeatures
        AVfusion_feature1 = self.fusion1(audio_upconv1feature, visual_feat)
        AVfusion_feature1 = self.fusion_upconv1(AVfusion_feature1)
        AVfusion_feature2 = self.fusion2(audio_upconv2feature, visual_feat)
        AVfusion_feature2 = self.fusion_upconv2(
            torch.cat((AVfusion_feature2, AVfusion_feature1), dim=1)
        )
        AVfusion_feature3 = self.fusion3(audio_upconv3feature, visual_feat)
        AVfusion_feature3 = self.fusion_upconv3(
            torch.cat((AVfusion_feature3, AVfusion_feature2), dim=1)
        )
        AVfusion_feature4 = self.fusion4(audio_upconv4feature, visual_feat)
        AVfusion_feature4 = torch.cat((AVfusion_feature4, AVfusion_feature3), dim=1)

        pred_left_mask = self.lastconv_left(AVfusion_feature4) * 2 - 1
        pred_right_mask = self.lastconv_right(AVfusion_feature4) * 2 - 1
        left_spectrogram = _get_spectrogram(pred_left_mask, audio_mix)
        right_spectrogram = _get_spectrogram(pred_right_mask, audio_mix)
        output = {"pred_left": left_spectrogram, "pred_right": right_spectrogram}
        return output


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


# TODO: fix this module
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
    
    
class B4_TCN(nn.Module):
    def __init__(self, num_channels=3, kernel_size=2):
        super(B4_TCN, self).__init__()
        
        self.conv_layers = nn.Sequential(
            
        # Layer 1: 輸入 224x448
        nn.Conv2d(num_channels, 32, kernel_size=3, padding=1), # 保持尺寸
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2), # 輸出 112x224

        # Layer 2: 輸入 112x224
        nn.Conv2d(32, 64, kernel_size=3, padding=1), # 保持尺寸
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2), # 輸出 56x112

        # Layer 3: 輸入 56x112
        nn.Conv2d(64, 128, kernel_size=3, padding=1), # 保持尺寸
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2), # 輸出 28x56

        # Layer 4: 輸入 28x56
        nn.Conv2d(128, 256, kernel_size=3, padding=1), # 保持尺寸
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2), # 輸出 14x28

        # Layer 5: 輸入 14x28
        nn.Conv2d(256, 512, kernel_size=3, padding=1), # 保持尺寸
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2), # 輸出 7x14
        )
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), x.size(1), -1)
        return x