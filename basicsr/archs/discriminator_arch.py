from basicsr.utils.registry import ARCH_REGISTRY
from torch import nn as nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm


@ARCH_REGISTRY.register()
class UNetDiscriminatorSN(nn.Module):
    """Defines a U-Net discriminator with spectral normalization (SN)

    It is used in Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    Arg:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_feat (int): Channel number of base intermediate features. Default: 64.
        skip_connection (bool): Whether to use skip connections between U-Net. Default: True.
    """

    def __init__(self, num_in_ch, num_feat=64, skip_connection=True):
        super(UNetDiscriminatorSN, self).__init__()
        self.skip_connection = skip_connection
        norm = spectral_norm
        # the first convolution
        self.conv0 = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1)
        # downsample
        self.conv1 = norm(nn.Conv2d(num_feat, num_feat * 2, 4, 2, 1, bias=False))
        self.conv2 = norm(nn.Conv2d(num_feat * 2, num_feat * 4, 4, 2, 1, bias=False))
        self.conv3 = norm(nn.Conv2d(num_feat * 4, num_feat * 8, 4, 2, 1, bias=False))
        # upsample
        self.conv4 = norm(nn.Conv2d(num_feat * 8, num_feat * 4, 3, 1, 1, bias=False))
        self.conv5 = norm(nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1, bias=False))
        self.conv6 = norm(nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1, bias=False))
        # extra convolutions
        self.conv7 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv8 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv9 = nn.Conv2d(num_feat, 1, 3, 1, 1)

    def forward(self, x):
        # downsample
        x0 = F.leaky_relu(self.conv0(x), negative_slope=0.2, inplace=True)
        x1 = F.leaky_relu(self.conv1(x0), negative_slope=0.2, inplace=True)
        x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2, inplace=True)
        x3 = F.leaky_relu(self.conv3(x2), negative_slope=0.2, inplace=True)

        # upsample
        x3 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
        x4 = F.leaky_relu(self.conv4(x3), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x4 = x4 + x2
        x4 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=False)
        x5 = F.leaky_relu(self.conv5(x4), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x5 = x5 + x1
        x5 = F.interpolate(x5, scale_factor=2, mode='bilinear', align_corners=False)
        x6 = F.leaky_relu(self.conv6(x5), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x6 = x6 + x0

        # extra convolutions
        out = F.leaky_relu(self.conv7(x6), negative_slope=0.2, inplace=True)
        out = F.leaky_relu(self.conv8(out), negative_slope=0.2, inplace=True)
        out = self.conv9(out)

        return out


@ARCH_REGISTRY.register()
class UNetDiscriminatorSN_Non_Prior(nn.Module):
    """Defines a U-Net discriminator with spectral normalization (SN)

    It is used in Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    Arg:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_feat (int): Channel number of base intermediate features. Default: 64.
        skip_connection (bool): Whether to use skip connections between U-Net. Default: True.
    """

    def __init__(self, num_in_ch, num_feat=64, skip_connection=True):
        super(UNetDiscriminatorSN_Non_Prior, self).__init__()
        self.skip_connection = skip_connection
        norm = spectral_norm
        # the first convolution
        self.conv0 = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1)
        # downsample
        self.conv1 = norm(nn.Conv2d(num_feat, num_feat * 2, 4, 2, 1, bias=False))
        self.conv2 = norm(nn.Conv2d(num_feat * 2, num_feat * 4, 4, 2, 1, bias=False))
        self.conv3 = norm(nn.Conv2d(num_feat * 4, num_feat * 8, 4, 2, 1, bias=False))
        # upsample
        self.conv4 = norm(nn.Conv2d(num_feat * 8, num_feat * 4, 3, 1, 1, bias=False))
        self.conv5 = norm(nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1, bias=False))
        self.conv6 = norm(nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1, bias=False))
        # extra convolutions
        self.conv7 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv8 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv9 = nn.Conv2d(num_feat, 1, 3, 1, 1)

    def forward(self, x):
        # downsample
        x0 = F.leaky_relu(self.conv0(x), negative_slope=0.2, inplace=True)
        x1 = F.leaky_relu(self.conv1(x0), negative_slope=0.2, inplace=True)
        x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2, inplace=True)
        x3 = F.leaky_relu(self.conv3(x2), negative_slope=0.2, inplace=True)

        # upsample
        x3 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
        x4 = F.leaky_relu(self.conv4(x3), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x4 = x4 + x2
        x4 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=False)
        x5 = F.leaky_relu(self.conv5(x4), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x5 = x5 + x1
        x5 = F.interpolate(x5, scale_factor=2, mode='bilinear', align_corners=False)
        x6 = F.leaky_relu(self.conv6(x5), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x6 = x6 + x0

        # extra convolutions
        out = F.leaky_relu(self.conv7(x6), negative_slope=0.2, inplace=True)
        out = F.leaky_relu(self.conv8(out), negative_slope=0.2, inplace=True)
        out = self.conv9(out)

        return out


class Conv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride=1,
                 padding=0, dilation=1, norm='none', activation='relu', pad_type='zero', bias=True):
        super(Conv2dBlock, self).__init__()
        self.use_bias = bias
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        # else:
        # assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, dilation=dilation, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

@ARCH_REGISTRY.register()
class DomainDiscriminator(nn.Module):

    # 对特征属于哪一个domain进行判别

    def __init__(self, num_in_ch):
        super(DomainDiscriminator, self).__init__()
        # FCN classification layer
        self.feature = nn.Sequential(
            Conv2dBlock(num_in_ch, 64, 6, stride=2, padding=2, norm='none', activation='lrelu', bias=False),
            Conv2dBlock(64, 128, 4, stride=2, padding=1, norm='in', activation='lrelu', bias=False),
            Conv2dBlock(128, 256, 4, stride=2, padding=1, norm='in', activation='lrelu', bias=False),
            Conv2dBlock(256, 512, 4, stride=2, padding=1, norm='in', activation='lrelu', bias=False),
            nn.Conv2d(512, 1, 1, padding=0),
            # nn.Sigmoid()
        )
        self.global_pooling = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.feature(x)
        # x = self.global_pooling(x).view(-1)
        return x

@ARCH_REGISTRY.register()
class DomainDiscriminator2(nn.Module):

    # 对特征属于哪一个domain进行判别

    def __init__(self, num_in_ch):
        super(DomainDiscriminator2, self).__init__()
        # FCN classification layer
        self.feature = nn.Sequential(
            Conv2dBlock(num_in_ch, 64, 6, stride=2, padding=2, norm='none', activation='lrelu', bias=False),
            Conv2dBlock(64, 128, 4, stride=2, padding=1, norm='in', activation='lrelu', bias=False),
            Conv2dBlock(128, 256, 4, stride=2, padding=1, norm='in', activation='lrelu', bias=False),
            Conv2dBlock(256, 512, 4, stride=2, padding=1, norm='in', activation='lrelu', bias=False),
            nn.Conv2d(512, 1, 1, padding=0),
            nn.Sigmoid()
        )
        self.global_pooling = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.feature(x)
        # x = self.global_pooling(x).view(-1)
        return x


class LightPassFilter_conv1(nn.Module):
    def __init__(self, inputsize):
        super(LightPassFilter_conv1, self).__init__()
        
        self.hidden = nn.Linear(inputsize, inputsize//2)
        self.hidden2 = nn.Linear(inputsize//2, inputsize//4)
        self.output = nn.Linear(inputsize//4, 64)
        self.leakyrelu = nn.LeakyReLU()
 
    def forward(self, x):
        x = self.hidden(x)
        x = self.leakyrelu(x)
        x = self.hidden2(x)
        x = self.leakyrelu(x)
        x = self.output(x)

        return x

class LightPassFilter_res1(nn.Module):
    def __init__(self, inputsize):
        super(LightPassFilter_res1, self).__init__()
        
        self.hidden = nn.Linear(inputsize, inputsize//8)
        self.output = nn.Linear(inputsize//8, 64)  # 2 
        self.leakyrelu = nn.LeakyReLU()
        
    def forward(self, x):
        x = self.hidden(x)
        x = self.leakyrelu(x)
        x = self.output(x)
        
        return x
