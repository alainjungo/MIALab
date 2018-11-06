import torch
import torch.nn as nn
import torch.nn.functional as F
import typing as t
import torch.utils.data.dataloader as loader

import mialab.configuration.config as cfg
import mialab.model.base as base


MODEL_UNET = 'unet'


class DoubleConv2d(nn.Module):
    """(Dropout => conv 3x3 => BN => ReLU) * 2"""

    def __init__(self, in_ch, out_ch, groups=1):
        super(DoubleConv2d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, groups=groups),
            nn.Dropout2d(0.2, inplace=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, groups=groups),
            nn.Dropout2d(0.2, inplace=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Down2d(nn.Module):
    """(MP 2x2 => DoubleConv)"""

    def __init__(self, in_ch, out_ch, groups=1, pooling: t.Union[int, tuple] = 2):
        super(Down2d, self).__init__()
        self.mp_double_conv = nn.Sequential(
            nn.MaxPool2d(pooling),
            DoubleConv2d(in_ch, out_ch, groups=groups)
        )

    def forward(self, x):
        x = self.mp_double_conv(x)
        return x


class Up2d(nn.Module):
    """(trans-conv 2x2 => copy-crop => DoubleConv)"""

    def __init__(self, in_ch, out_ch, groups=1, stride: t.Union[int, tuple] = 2):
        super(Up2d, self).__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, stride, stride=stride, groups=groups)
        self.double_conv = DoubleConv2d(in_ch, out_ch, groups=groups)

    def forward(self, feature_map, feed_forward):
        feature_map = self.up(feature_map)

        # copy, crop and concatenate
        fm_shape, ff_shape = feature_map.size()[-2:], feed_forward.size()[-2:]
        if fm_shape < ff_shape:
            d_h = ff_shape[-2] - fm_shape[-2]
            d_w = ff_shape[-1] - fm_shape[-1]
            h_pad = (d_h // 2, d_h // 2 + (d_h % 2))
            w_pad = (d_w // 2, d_w // 2 + (d_w % 2))
            feature_map = F.pad(feature_map, w_pad + h_pad)

        x = torch.cat([feed_forward, feature_map], dim=1)
        x = self.double_conv(x)
        return x


class UNetModel(nn.Module):
    def __init__(self, in_ch, out_ch, n_channels=64, n_pooling: int=2):
        super(UNetModel, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.feat = n_channels

        self.inc = DoubleConv2d(in_ch, n_channels)
        self.down1 = Down2d(n_channels, 2 * n_channels, pooling=n_pooling)
        self.down2 = Down2d(2 * n_channels, 4 * n_channels, pooling=n_pooling)
        self.down3 = Down2d(4 * n_channels, 8 * n_channels, pooling=n_pooling)
        self.down4 = Down2d(8 * n_channels, 16 * n_channels, pooling=n_pooling)
        self.up1 = Up2d(16 * n_channels, 8 * n_channels, stride=n_pooling)
        self.up2 = Up2d(8 * n_channels, 4 * n_channels, stride=n_pooling)
        self.up3 = Up2d(4 * n_channels, 2 * n_channels, stride=n_pooling)
        self.up4 = Up2d(2 * n_channels, n_channels, stride=n_pooling)
        self.outc = nn.Conv2d(n_channels, out_ch, 1)  # conv 1x1

    def __str__(self):
        return 'Unet_F{}'.format(self.feat)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.down4(x4)
        x = self.up1(x, x4)
        del x4
        x = self.up2(x, x3)
        del x3
        x = self.up3(x, x2)
        del x2
        x = self.up4(x, x1)
        del x1
        x = self.outc(x)
        return x


class UNET(base.TorchMRFModel):

    def __init__(self, sample: dict, config: cfg.Configuration):
        super().__init__(sample, config, UNetModel)
