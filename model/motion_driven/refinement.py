import functools
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.motion_driven.base_function import FineADAINResBlock2d, FineEncoder, FineDecoderV2, LayerNorm2d
from model.stylegan2.model import ConvLayer, EqualConv2d, EqualLinear, ResBlock, ScaledLeakyReLU
from model.stylegan2.op import FusedLeakyReLU


class LinearNet(nn.Module):
    def __init__(self, coeff_nc, descriptor_nc, layer):
        super(LinearNet, self).__init__()

        self.layer = layer
        nonlinearity = nn.LeakyReLU(0.1)

        self.first = nn.Linear(in_features=coeff_nc, out_features=descriptor_nc, bias=True)

        for i in range(layer):
            net = nn.Sequential(nonlinearity,
                                nn.Linear(descriptor_nc, descriptor_nc, bias=True)
                                )
            setattr(self, 'encoder' + str(i), net)

        # self.pooling = nn.AdaptiveAvgPool1d(1)
        self.output_nc = descriptor_nc

    def forward(self, input_3dmm):
        # b, c
        out = self.first(input_3dmm)
        for i in range(self.layer):
            model = getattr(self, 'encoder' + str(i))
            out = model(out) + out
        # out = self.pooling(out)
        return out


class ConvUpLayer(nn.Module):
    """Conv Up Layer. Bilinear upsample + Conv.
    Args:
        in_channels (int): Channel number of the input.
        out_channels (int): Channel number of the output.
        kernel_size (int): Size of the convolving kernel.
        stride (int): Stride of the convolution. Default: 1
        padding (int): Zero-padding added to both sides of the input.
            Default: 0.
        bias (bool): If ``True``, adds a learnable bias to the output.
            Default: ``True``.
        bias_init_val (float): Bias initialized value. Default: 0.
        activate (bool): Whether use activateion. Default: True.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 bias=True,
                 bias_init_val=0,
                 activate=True):
        super(ConvUpLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.scale = 1 / math.sqrt(in_channels * kernel_size**2)

        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))

        if bias and not activate:
            self.bias = nn.Parameter(torch.zeros(out_channels).fill_(bias_init_val))
        else:
            self.register_parameter('bias', None)

        # activation
        if activate:
            if bias:
                self.activation = FusedLeakyReLU(out_channels)
            else:
                self.activation = ScaledLeakyReLU(0.2)
        else:
            self.activation = None

    def forward(self, x):
        # bilinear upsample
        out = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        # conv
        out = F.conv2d(
            out,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )
        # activation
        if self.activation is not None:
            out = self.activation(out)
        return out


class ResUpBlock(nn.Module):
    """Residual block with upsampling.
    Args:
        in_channels (int): Channel number of the input.
        out_channels (int): Channel number of the output.
    """

    def __init__(self, in_channels, out_channels):
        super(ResUpBlock, self).__init__()

        self.conv1 = ConvLayer(in_channels, in_channels, 3, bias=True, activate=True)
        self.conv2 = ConvUpLayer(in_channels, out_channels, 3, stride=1, padding=1, bias=True, activate=True)
        self.skip = ConvUpLayer(in_channels, out_channels, 1, bias=False, activate=False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        skip = self.skip(x)
        out = (out + skip) / math.sqrt(2)
        return out


class Refinement(nn.Module):

    def __init__(
        self,
        out_size,
        input_channel,
        num_style_feat=512,
        channel_multiplier=1,
        resample_kernel=(1, 3, 3, 1),
        # for stylegan decoder
        narrow=1
    ):

        super(Refinement, self).__init__()
        self.num_style_feat = num_style_feat

        unet_narrow = narrow * 0.5
        channels = {
            '4': int(512 * unet_narrow),
            '8': int(512 * unet_narrow),
            '16': int(512 * unet_narrow),
            '32': int(512 * unet_narrow),
            '64': int(256 * channel_multiplier * unet_narrow),
            '128': int(128 * channel_multiplier * unet_narrow),
            '256': int(64 * channel_multiplier * unet_narrow),
            '512': int(32 * channel_multiplier * unet_narrow),
            '1024': int(16 * channel_multiplier * unet_narrow)
        }

        self.log_size = int(math.log(out_size, 2))
        first_out_size = 2**(int(math.log(out_size, 2)))

        self.conv_body_first = ConvLayer(input_channel, channels[f'{first_out_size}'], 1, bias=True, activate=True)

        # downsample
        in_channels = channels[f'{first_out_size}']
        self.conv_body_down = nn.ModuleList()
        for i in range(self.log_size, 2, -1):
            out_channels = channels[f'{2**(i - 1)}']
            self.conv_body_down.append(ResBlock(in_channels, out_channels, resample_kernel))
            in_channels = out_channels

        self.final_conv = ConvLayer(in_channels, channels['4'], 3, bias=True, activate=True)

        # upsample
        in_channels = channels['4']
        self.conv_body_up = nn.ModuleList()
        for i in range(3, self.log_size + 1):
            out_channels = channels[f'{2**i}']
            self.conv_body_up.append(ResUpBlock(in_channels, out_channels))
            in_channels = out_channels

        self.linear_3dmm = nn.Sequential(
            nn.Linear(in_features=256, out_features=256),
            nn.LeakyReLU(0.1),
            nn.Linear(in_features=256, out_features=256),
        )
        self.inject_3dmm = nn.ModuleList()
        self.condition_scale = nn.ModuleList()
        self.condition_shift = nn.ModuleList()
        for i in range(3, self.log_size + 1):
            out_channels = channels[f'{2**i}']
            sft_out_channels = out_channels * 2
            self.inject_3dmm.append(
                FineADAINResBlock2d(input_nc=out_channels, feature_nc=out_channels)
                # FineADAINResBlock2d(input_nc=out_channels, feature_nc=out_channels, z_feature_nc=256)
            )
            self.condition_scale.append(
                nn.Sequential(
                    EqualConv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=True),
                    ScaledLeakyReLU(0.2),
                    EqualConv2d(out_channels, sft_out_channels, 3, stride=1, padding=1, bias=True)))
            self.condition_shift.append(
                nn.Sequential(
                    EqualConv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=True),
                    ScaledLeakyReLU(0.2),
                    EqualConv2d(out_channels, sft_out_channels, 3, stride=1, padding=1, bias=True)))

    def forward(
        self,
        x,
        z,  # 3dmm
    ):
        conditions = []
        unet_skips = []

        # encoder
        feat = self.conv_body_first(x) # torch.Size([10, 256, 64, 64])
        for i in range(self.log_size - 2):
            feat = self.conv_body_down[i](feat)
            unet_skips.insert(0, feat)

        feat = self.final_conv(feat) # torch.Size([10, 256, 4, 4])

        z = z.squeeze(-1)
        z = self.linear_3dmm(z.squeeze(-1)) # torch.Size([10, 256])
        # decode
        for i in range(self.log_size - 2):
            # print(feat.shape, unet_skips[i].shape)
            # add unet skip
            feat = feat + unet_skips[i]
            # ResUpLayer
            feat = self.conv_body_up[i](feat)
            feat = self.inject_3dmm[i](feat, z) # torch.Size([10, 256, 8, 8])
            # generate scale and shift for SFT layer
            scale = self.condition_scale[i](feat)
            conditions.append(scale.clone())
            shift = self.condition_shift[i](feat)
            conditions.append(shift.clone())

        return conditions

