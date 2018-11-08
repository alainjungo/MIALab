import torch
import torch.nn as nn
import torch.nn.functional as F

import mialab.configuration.config as cfg
import mialab.model.base as base


MODEL_UNET_2D = 'unet2d'


class ConvDONormReLu2D(nn.Module):

    def __init__(self, in_ch, out_ch, dropout_p=None, norm: str= 'bn'):
        super().__init__()

        self.sequential = nn.Sequential()
        self.sequential.add_module('conv', nn.Conv2d(in_ch, out_ch, 3, padding=1))
        if dropout_p is not None:
            self.sequential.add_module('dropout', nn.Dropout2d(p=dropout_p))  # todo: in_place?
        if norm == 'bn':
            self.sequential.add_module(norm, nn.BatchNorm2d(out_ch))

        self.sequential.add_module('relu', nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.sequential(x)
        return x


class DownConv2D(nn.Module):
    def __init__(self, in_ch, out_ch, dropout_p=None, norm: str= 'bn'):
        super().__init__()

        self.double_conv = nn.Sequential(ConvDONormReLu2D(in_ch, out_ch, dropout_p, norm),
                                         ConvDONormReLu2D(out_ch, out_ch, dropout_p, norm))
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        skip_x = self.double_conv(x)
        x = self.pool(skip_x)
        return x, skip_x


class UpConv2D(nn.Module):
    def __init__(self, in_ch, out_ch, dropout_p=None, norm: str= 'bn'):
        super().__init__()

        self.upconv = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv2x = nn.Sequential(ConvDONormReLu2D(2 * out_ch, out_ch, dropout_p, norm),
                                    ConvDONormReLu2D(out_ch, out_ch, dropout_p, norm))

    def forward(self, x, skip_x):
        up = self.upconv(x)

        # todo: verify if speed or accuracy issue
        up_shape, skip_shape = up.size()[-2:], skip_x.size()[-2:]
        if up_shape < skip_shape:
            x_diff = skip_shape[-1] - up_shape[-1]
            y_diff = skip_shape[-2] - up_shape[-1]
            x_pad = (x_diff // 2, x_diff // 2 + (x_diff % 2))
            y_pad = (y_diff // 2, y_diff // 2 + (y_diff % 2))
            up = F.pad(up, y_pad + x_pad)

        x = torch.cat((up, skip_x), 1)
        x = self.conv2x(x)
        return x


class UNetModel(nn.Module):

    def __init__(self, in_ch: int, out_ch: int, n_channels: int=32, n_pooling: int=3,
                 dropout_p: float=0.2, norm: str= 'bn'):
        super().__init__()

        n_classes = out_ch
        out_ch = n_channels

        self.down_convs = nn.ModuleList()
        for i in range(n_pooling):
            down_conv = DownConv2D(in_ch, out_ch, dropout_p, norm)
            self.down_convs.append(down_conv)
            in_ch = out_ch
            out_ch *= 2

        self.bottleneck = nn.Sequential(ConvDONormReLu2D(in_ch, out_ch, dropout_p, norm),
                                        ConvDONormReLu2D(out_ch, out_ch, dropout_p, norm))

        self.up_convs = nn.ModuleList()
        for i in range(n_pooling, 0, -1):
            in_ch = out_ch
            out_ch = in_ch // 2
            up_conv = UpConv2D(in_ch, out_ch, dropout_p, norm)
            self.up_convs.append(up_conv)

        in_ch = out_ch
        self.conv_cls = nn.Conv2d(in_ch, n_classes, 1)

    def forward(self, x):
        skip_xs = []
        for down_conv in self.down_convs:
            x, skip_x = down_conv(x)
            skip_xs.append(skip_x)

        x = self.bottleneck(x)

        for inv_depth, up_conv in enumerate(self.up_convs, 1):
            skip_x = skip_xs[-inv_depth]
            x = up_conv(x, skip_x)

        logits = self.conv_cls(x)
        return logits


class UNet2D(base.TorchMRFModel):

    IS_VOLUMETRIC = False

    def __init__(self, sample: dict, config: cfg.Configuration):
        super().__init__(sample, config, UNetModel)
