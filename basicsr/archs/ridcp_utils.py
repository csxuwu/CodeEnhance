import torch
from torch.nn import functional as F
from torch import nn as nn

class NormLayer(nn.Module):
    """Normalization Layers.
    ------------
    # Arguments
        - channels: input channels, for batch norm and instance norm.
        - input_size: input shape without batch size, for layer norm.
    """
    def __init__(self, channels, norm_type='bn'):
        super(NormLayer, self).__init__()
        norm_type = norm_type.lower()
        self.norm_type = norm_type
        self.channels = channels
        if norm_type == 'bn':
            self.norm = nn.BatchNorm2d(channels, affine=True)
        elif norm_type == 'in':
            self.norm = nn.InstanceNorm2d(channels, affine=False)
        elif norm_type == 'gn':
            self.norm = nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6, affine=True)
        elif norm_type == 'none':
            self.norm = lambda x: x*1.0
        else:
            assert 1==0, 'Norm type {} not support.'.format(norm_type)

    def forward(self, x):
        return self.norm(x)


class ActLayer(nn.Module):
    """activation layer.
    ------------
    # Arguments
        - relu type: type of relu layer, candidates are
            - ReLU
            - LeakyReLU: default relu slope 0.2
            - PRelu 
            - SELU
            - none: direct pass
    """
    def __init__(self, channels, relu_type='leakyrelu'):
        super(ActLayer, self).__init__()
        relu_type = relu_type.lower()
        if relu_type == 'relu':
            self.func = nn.ReLU(True)
        elif relu_type == 'leakyrelu':
            self.func = nn.LeakyReLU(0.2, inplace=True)
        elif relu_type == 'prelu':
            self.func = nn.PReLU(channels)
        elif relu_type == 'none':
            self.func = lambda x: x*1.0
        elif relu_type == 'silu':
            self.func = nn.SiLU(True)
        elif relu_type == 'gelu':
            self.func = nn.GELU()
        else:
            assert 1==0, 'activation type {} not support.'.format(relu_type)

    def forward(self, x):
        return self.func(x)


class ResBlock(nn.Module):
    """
    Use preactivation version of residual block, the same as taming
    """
    def __init__(self, in_channel, out_channel, norm_type='gn', act_type='leakyrelu'):
        super(ResBlock, self).__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel

        self.conv = nn.Sequential(
            NormLayer(in_channel, norm_type),
            ActLayer(in_channel, act_type),
            nn.Conv2d(in_channel, out_channel, 3, stride=1, padding=1),
            NormLayer(out_channel, norm_type),
            ActLayer(out_channel, act_type),
            nn.Conv2d(out_channel, out_channel, 3, stride=1, padding=1),
        )
        if self.in_channel != self.out_channel:
            self.conv_out = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0)

    def forward(self, input):

        # 设置 torch.backends.cudnn:
        # 1.enabled=True：说明设置为使用非确定性算法
        # 2.deterministic=True：由于计算中有随机性，每次网络前馈结果略有差异。如果想要避免这种结果波动
        # 3.benchmark=True 将会让程序在开始时花费一点额外时间，为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速。
        # 设置这个 flag 可以让内置的 cuDNN 的 auto-tuner 自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题
        with torch.backends.cudnn.flags(enabled=True, deterministic=True, benchmark=False):
            # torch.backends.cudnn.enabled = False
            res = self.conv(input)

        if self.in_channel != self.out_channel:
            input = self.conv_out(input)
        out = res + input
        return out


class CombineQuantBlock(nn.Module):
    def __init__(self, in_ch1, in_ch2, out_channel):
        super().__init__()
        self.conv = nn.Conv2d(in_ch1 + in_ch2, out_channel, 3, 1, 1)

    def forward(self, input1, input2=None):
        if input2 is not None:
            input2 = F.interpolate(input2, input1.shape[2:])
            input = torch.cat((input1, input2), dim=1)
        else:
            input = input1
        out = self.conv(input)
        return out


