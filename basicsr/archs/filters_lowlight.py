

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math

# ------------------------------
# 20211223
# 借鉴：Exposure: 《A White-Box Photo Post-Processing Framework》
# 《Image-Adaptive YOLO for Object Detection in Adverse Weather Conditions》
# ------------------------------


class RGB_HSV(nn.Module):
    '''
    Pytorch implementation of RGB convert to HSV, and HSV convert to RGB,
    RGB or HSV's shape: (B * C * H * W)
    RGB or HSV's range: [0, 1)
    https://blog.csdn.net/Brikie/article/details/115086835
    '''
    def __init__(self, eps=1e-8):
        super(RGB_HSV, self).__init__()
        self.eps = eps

    def rgb_to_hsv(self, img):
        hue = torch.Tensor(img.shape[0], img.shape[2], img.shape[3]).to(img.device)

        hue[img[:, 2] == img.max(1)[0]] = 4.0 + ((img[:, 0] - img[:, 1]) / (img.max(1)[0] - img.min(1)[0] + self.eps))[
            img[:, 2] == img.max(1)[0]]
        hue[img[:, 1] == img.max(1)[0]] = 2.0 + ((img[:, 2] - img[:, 0]) / (img.max(1)[0] - img.min(1)[0] + self.eps))[
            img[:, 1] == img.max(1)[0]]
        hue[img[:, 0] == img.max(1)[0]] = (0.0 + ((img[:, 1] - img[:, 2]) / (img.max(1)[0] - img.min(1)[0] + self.eps))[
            img[:, 0] == img.max(1)[0]]) % 6

        hue[img.min(1)[0] == img.max(1)[0]] = 0.0
        hue = hue / 6

        saturation = (img.max(1)[0] - img.min(1)[0]) / (img.max(1)[0] + self.eps)
        saturation[img.max(1)[0] == 0] = 0

        value = img.max(1)[0]

        hue = hue.unsqueeze(1)
        saturation = saturation.unsqueeze(1)
        value = value.unsqueeze(1)
        hsv = torch.cat([hue, saturation, value], dim=1)
        return hsv

    def hsv_to_rgb(self, hsv):
        h, s, v = hsv[:, 0, :, :], hsv[:, 1, :, :], hsv[:, 2, :, :]
        # 对出界值的处理
        h = h % 1
        s = torch.clamp(s, 0, 1)
        v = torch.clamp(v, 0, 1)

        r = torch.zeros_like(h)
        g = torch.zeros_like(h)
        b = torch.zeros_like(h)

        hi = torch.floor(h * 6)
        f = h * 6 - hi
        p = v * (1 - s)
        q = v * (1 - (f * s))
        t = v * (1 - ((1 - f) * s))

        hi0 = hi == 0
        hi1 = hi == 1
        hi2 = hi == 2
        hi3 = hi == 3
        hi4 = hi == 4
        hi5 = hi == 5

        r[hi0] = v[hi0]
        g[hi0] = t[hi0]
        b[hi0] = p[hi0]

        r[hi1] = q[hi1]
        g[hi1] = v[hi1]
        b[hi1] = p[hi1]

        r[hi2] = p[hi2]
        g[hi2] = v[hi2]
        b[hi2] = t[hi2]

        r[hi3] = p[hi3]
        g[hi3] = q[hi3]
        b[hi3] = v[hi3]

        r[hi4] = t[hi4]
        g[hi4] = p[hi4]
        b[hi4] = v[hi4]

        r[hi5] = v[hi5]
        g[hi5] = p[hi5]
        b[hi5] = q[hi5]

        r = r.unsqueeze(1)
        g = g.unsqueeze(1)
        b = b.unsqueeze(1)
        rgb = torch.cat([r, g, b], dim=1)
        return rgb


def rgb_to_hsv(img, eps=1e-8):
    hue = torch.Tensor(img.shape[0], img.shape[2], img.shape[3]).to(img.device)

    hue[img[:, 2] == img.max(1)[0]] = 4.0 + ((img[:, 0] - img[:, 1]) / (img.max(1)[0] - img.min(1)[0] + eps))[
        img[:, 2] == img.max(1)[0]]
    hue[img[:, 1] == img.max(1)[0]] = 2.0 + ((img[:, 2] - img[:, 0]) / (img.max(1)[0] - img.min(1)[0] + eps))[
        img[:, 1] == img.max(1)[0]]
    hue[img[:, 0] == img.max(1)[0]] = (0.0 + ((img[:, 1] - img[:, 2]) / (img.max(1)[0] - img.min(1)[0] + eps))[
        img[:, 0] == img.max(1)[0]]) % 6

    hue[img.min(1)[0] == img.max(1)[0]] = 0.0
    hue = hue / 6

    saturation = (img.max(1)[0] - img.min(1)[0]) / (img.max(1)[0] + eps)
    saturation[img.max(1)[0] == 0] = 0

    value = img.max(1)[0]

    hue = hue.unsqueeze(1)
    saturation = saturation.unsqueeze(1)
    value = value.unsqueeze(1)
    hsv = torch.cat([hue, saturation, value], dim=1)
    return hsv

def rgb_to_hsv2(im, eps=1e-8):
    # https://github.com/odegeasslbc/Differentiable-RGB-to-HSV-convertion-pytorch/blob/master/pytorch_hsv.py
    img = im * 0.5 + 0.5
    hue = torch.Tensor(im.shape[0], im.shape[2], im.shape[3]).to(im.device)

    hue[img[:, 2] == img.max(1)[0]] = 4.0 + ((img[:, 0] - img[:, 1]) / (img.max(1)[0] - img.min(1)[0] + eps))[
        img[:, 2] == img.max(1)[0]]
    hue[img[:, 1] == img.max(1)[0]] = 2.0 + ((img[:, 2] - img[:, 0]) / (img.max(1)[0] - img.min(1)[0] + eps))[
        img[:, 1] == img.max(1)[0]]
    hue[img[:, 0] == img.max(1)[0]] = (0.0 + ((img[:, 1] - img[:, 2]) / (img.max(1)[0] - img.min(1)[0] + eps))[
        img[:, 0] == img.max(1)[0]]) % 6

    hue[img.min(1)[0] == img.max(1)[0]] = 0.0
    hue = hue / 6

    saturation = (img.max(1)[0] - img.min(1)[0]) / (img.max(1)[0] + eps)
    saturation[img.max(1)[0] == 0] = 0

    value = img.max(1)[0]
    return hue, saturation, value


def hsv_to_rgb(hsv, eps=1e-8):
    h, s, v = hsv[:, 0, :, :], hsv[:, 1, :, :], hsv[:, 2, :, :]
    # 对出界值的处理
    h = h % 1
    s = torch.clamp(s, 0, 1)
    v = torch.clamp(v, 0, 1)

    r = torch.zeros_like(h)
    g = torch.zeros_like(h)
    b = torch.zeros_like(h)

    hi = torch.floor(h * 6)
    f = h * 6 - hi
    p = v * (1 - s)
    q = v * (1 - (f * s))
    t = v * (1 - ((1 - f) * s))

    hi0 = hi == 0
    hi1 = hi == 1
    hi2 = hi == 2
    hi3 = hi == 3
    hi4 = hi == 4
    hi5 = hi == 5

    r[hi0] = v[hi0]
    g[hi0] = t[hi0]
    b[hi0] = p[hi0]

    r[hi1] = q[hi1]
    g[hi1] = v[hi1]
    b[hi1] = p[hi1]

    r[hi2] = p[hi2]
    g[hi2] = v[hi2]
    b[hi2] = t[hi2]

    r[hi3] = p[hi3]
    g[hi3] = q[hi3]
    b[hi3] = v[hi3]

    r[hi4] = t[hi4]
    g[hi4] = p[hi4]
    b[hi4] = v[hi4]

    r[hi5] = v[hi5]
    g[hi5] = p[hi5]
    b[hi5] = q[hi5]

    r = r.unsqueeze(1)
    g = g.unsqueeze(1)
    b = b.unsqueeze(1)
    rgb = torch.cat([r, g, b], dim=1)
    return rgb


def hsv_to_rgb2(img_hsv):
    C = img_hsv[2] * img_hsv[1]
    X = C * (1 - abs((img_hsv[0] * 6) % 2 - 1))
    m = img_hsv[2] - C

    if img_hsv[0] < 1 / 6:
        R_hat, G_hat, B_hat = C, X, 0
    elif img_hsv[0] < 2 / 6:
        R_hat, G_hat, B_hat = X, C, 0
    elif img_hsv[0] < 3 / 6:
        R_hat, G_hat, B_hat = 0, C, X
    elif img_hsv[0] < 4 / 6:
        R_hat, G_hat, B_hat = 0, X, C
    elif img_hsv[0] < 5 / 6:
        R_hat, G_hat, B_hat = X, 0, C
    elif img_hsv[0] <= 6 / 6:
        R_hat, G_hat, B_hat = C, 0, X

    R, G, B = (R_hat + m), (G_hat + m), (B_hat + m)

    return R, G, B


def rgb2lum(image):
  image = 0.27 * image[:, 0, :, :] + 0.67 * image[:, 1, :, :] + 0.06 * image[:, 2, :, :]
  return image[:, :, :, None]


def linear_inter(a, b, l):
    '''
    linear interpolation.
    :param a:
    :param b:
    :param l:
    :return:
    '''
    return (1 - l) * a + l * b


def tanh01(x):
    return F.tanh(x) * 0.5 + 0.5

def tanh_range(l, r, initial=None):
    def get_activation(left, right, initial):

        def activation(x):
            if initial is not None:
                bias = math.atanh(2 * (initial - left) / (right - left) - 1)
            else:
                bias = 0
            return tanh01(x + bias) * (right - left) + left

        return activation

    return get_activation(l, r, initial)


def luminance(image):
    b, c, h, w = image.size()
    a = torch.rand((b, h, w))
    down = torch.zeros_like(a).cuda()  # 下限
    top = torch.ones_like(a).cuda()  # 上限
    l = 0.27 * image[:, 0, :, :] + 0.67 * image[:, 1, :, :] + 0.06 * image[:, 2, :, :]

    luminance = torch.where(l < torch.as_tensor(0.0).cuda(), down, l)  # 去除越界的
    luminance = torch.where(l > torch.as_tensor(1.0).cuda(), top, luminance)

    return luminance[:, None, :, :]

def luminance_y(image):
    b, c, h, w = image.size()
    a = torch.rand((b, h, w))
    down = torch.zeros_like(a).cuda()  # 下限
    top = torch.ones_like(a).cuda()  # 上限
    l = 0.257 * image[:, 0, :, :] + 0.564 * image[:, 1, :, :] + 0.098 * image[:, 2, :, :]

    luminance = torch.where(l < torch.as_tensor(0.0).cuda(), down, l)  # 去除越界的
    luminance = torch.where(l > torch.as_tensor(1.0).cuda(), top, luminance)

    return luminance[:, None, :, :]

def clip(x, down=0.0, down_value = 0.0, top=1.0, top_value=1.0):
    '''
    去除超过上下限的部分
    :param x:
    :param down: 下限
    :param down_value: 超过下限的部分，将其替换成down_value
    :param top: 上限
    :param top_value: 超过上限的部分，将其替换成top_value
    :return:
    '''
    b, c, h, w = x.size()
    a = torch.rand((b, c, h, w))
    down_t = torch.zeros_like(a).cuda()  # 下限
    top_t = torch.zeros_like(a).cuda()  # 上限

    down_t += down_value
    top_t += top_value

    x = torch.where(x < torch.as_tensor(down).cuda(), down_t, x)  # 去除越界的
    x = torch.where(x > torch.as_tensor(top).cuda(), top_t, x)

    return x

# ====================================================================

def saturation_filter(img, param):
    '''
    enhancing saturation.
    :param img:
    :param img_hsv:
    :param param:
    :return:
    '''
    img_hsv = rgb_to_hsv(img)
    s = img_hsv[:, 1:2, :, :]
    v = img_hsv[:, 2:3, :, :]

    enhanced_s = s + (1-s) * (0.5 - torch.abs(0.5 - v)) * 0.8
    hsv = torch.cat((img_hsv[:, 0:1, :, :], enhanced_s, img_hsv[:, 2:3, :, :]), 1)
    # enhanced_img = hsv_to_rgb(hsv)
    enhanced_img = hsv_to_rgb(hsv)

    if len(param.size()) == 4:
        out = linear_inter(img, enhanced_img, param)
    else:
        out = linear_inter(img, enhanced_img, param[:, :, None, None])

    return out


def WNB_filer(img, param):
    '''
    :param img:
    :param param:
    :return:
    '''
    param = F.sigmoid(param)    # 经过sigmoid处理
    luminace = rgb2lum(img)

    if len(param.size()) == 4:
        out = linear_inter(img, luminace, param)
    else:
        out = linear_inter(img, luminace, param[:, :, None, None])

    return out


def exposure_filter(img, param, exposure_range):
    '''
    Enhancing light of image.
    :param img:
    :param param:
    :return:
    '''
    param = tanh_range(-exposure_range, exposure_range, 0)(param)

    if len(param.size()) == 4:
        out = img * torch.exp_(param * np.log(2))
    else:
        out = img * torch.exp_(param[:,:,None, None] * np.log(2))

    return out


def gamma_filter(img, param, gamma_range):
    '''
    gamma correction.
    :param img:
    :param param:
    :param gamma_range:
    :return:
    '''
    log_gamma_range = np.log(gamma_range)
    param = torch.exp(tanh_range(-log_gamma_range, log_gamma_range)(param))

    down = torch.zeros_like(img).cuda()  # 下限
    down = down + 0.001
    img = torch.where(img < torch.as_tensor(0.001).cuda(), down, img)

    if len(param.size()) == 2:
        out = torch.pow(img, param[:, :, None, None])   # param 是 2D tensor
    else:
        out = torch.pow(img, param)  # param 是4D tensor

    return out


def s_curve2(x, param):
    '''
    对 x 的细节进行优化
    :param x:
    :return:
    '''
    param = F.tanh(param)
    if len(param.size()) == 4:
        out = x + param * x * (1 - x)
    else:
        out = x + param[:,:,None,None] * x * (1 - x)

    return out


def s_curve( x, phs, phh):
    '''
    S-curve  f(x) = x + s * f_1(x) - h * f_1(1-x)
    :param x:
    :param phs:
    :param phh:
    :return:
    '''
    phs = F.tanh(phs)
    phh = F.tanh(phh)
    if len(phs.size()) == 4:
        s_x = x + phs * alpha_delta(x) - phh * alpha_delta(1 - x)
    else:
        s_x = x + phs[:,:,None,None] * alpha_delta(x) - phh[:,:,None,None] * alpha_delta(1 - x)
    return s_x


def alpha_delta( x):
    '''
    f(t) 函数
    :param x:
    :return:
    '''
    return torch.mul(torch.mul(x, 5), torch.exp(torch.mul(torch.pow(x, 1.6), -14)))


def contrast_filter(luminance, x, param):
    param = F.tanh(param)
    contrast_lum = -torch.cos(math.pi * luminance) * 0.5 + 0.5
    contrast_image = x / (luminance + 1e-6) * contrast_lum
    # return self.lerp(x, contrast_image, param[:, :, None, None])	# tf: NHWC   pytorch: NCHW
    # return linear_inter(x, contrast_image, param)  # tf: NHWC   pytorch: NCHW
    if len(param.size()) == 4:
        return linear_inter(x, contrast_image, param)  # tf: NHWC   pytorch: NCHW
    else:
        return linear_inter(x, contrast_image, param[:, :, None, None])  # tf: NHWC   pytorch: NCHW


def contrast_filter_clip(luminance, x, param):
    param = F.tanh(param)
    contrast_lum = -torch.cos(math.pi * luminance) * 0.5 + 0.5
    contrast_image = x / (luminance + 1e-6) * contrast_lum      # 这函数的输出可能大于1
    # t1 = torch.max(contrast_image, 2)[0]
    # t2 = torch.max(contrast_image, 3)[0]
    # max_value = torch.maximum(torch.max(contrast_image, 2)[0], torch.max(contrast_image, 3)[0])
    # max_value1 = torch.max(contrast_image, 2, keepdim=True)[0]
    # max_value2 = torch.max(contrast_image, 3, keepdim=True)[0]
    # max_value1 = max_value2[:,:,:, ]
    contrast_image = clip(contrast_image, down_value=0.0, top_value=0.9)

    if len(param.size()) == 4:
        return linear_inter(x, contrast_image, param)  # tf: NHWC   pytorch: NCHW
    else:
        return linear_inter(x, contrast_image, param[:, :, None, None])  # tf: NHWC   pytorch: NCHW


def WB_filter(img, param):
    '''
    White balance.
    :param img:
    :param param:
    :return:
    '''
    log_wb_range = 0.5
    mask = np.array(((0, 1, 1)), dtype=np.float32).reshape(1, 3)
    # mask = np.array(((1, 0, 1)), dtype=np.float32).reshape(1, 3)

    print(mask.shape)
    assert mask.shape == (1, 3)
    features = param * torch.from_numpy(mask).cuda()
    color_scaling = torch.exp(tanh_range(-log_wb_range, log_wb_range)(features))
    # There will be no division by zero here unless the WB range lower bound is 0
    # normalize by luminance
    color_scaling *= 1.0 / (1e-5 + 0.27 * color_scaling[:, 0, :, :] + 0.67 * color_scaling[:, 1, :, :] + 0.06 * color_scaling[:, 2, :, :])

    out = img * color_scaling

    return out


def color_filter(img, param, color_curve_range=(0.90, 1.10), curve_steps=4):

    '''
    调整color，采用的是一个分段函数（separate function）
    :param img:
    :param param:
    :param color_curve_range: 调整范围
    :param curve_steps: 分段函数的子函数数量
    :return:
    '''
    # 处理param
    # param = torch.reshape(param, shape=(-1, 3, curve_steps))
    # param = param.view(param.size(0), 1, curve_steps)
    param = param[:, None, None, None, :]
    param = tanh_range(*color_curve_range, initial=1)(param)

    param_sum = torch.sum(param, dim=4) + 1e-30
    total_img = img * 0

    for i in range(curve_steps):
        total_img += clip(x=img - 1.0 * i / curve_steps, down=0.0, down_value=0.0, top=1.0 / curve_steps, top_value=1.0 / curve_steps) * param[:, :, :, :, i]

    total_img *= curve_steps / param_sum

    return total_img


# 提取 low-level的信息
class Low_Level_Filter_conv(nn.Module):
    '''
    输入是 gram matrix
    '''
    def __init__(self, input_channels, output_features):
        super(Low_Level_Filter_conv, self).__init__()
        
        # self.hidden = nn.Linear(inputsize, inputsize//2)
        # self.hidden2 = nn.Linear(inputsize//2, inputsize//4)
        # self.output = nn.Linear(inputsize//4, 64)
        # self.leakyrelu = nn.LeakyReLU()

        # self.hidden = nn.Conv2d(inputsize, inputsize // 2, 1)
        # self.hidden2 = nn.Conv2d(inputsize // 2, inputsize // 4, 1)
        # self.output = nn.Conv2d(inputsize // 4, output_size, 1)


        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, output_features)
        )
        
    def forward(self, x):
        # x = self.hidden(x)
        # x = self.leakyrelu(x)
        # x = self.hidden2(x)
        # x = self.leakyrelu(x)
        # x = self.output(x)

        # x = torch.mean(x, dim=[2, 3])
        self.conv_layers(x)

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


class Low_Level_Filter2_conv(nn.Module):
    def __init__(self,num_in=384, num_out=1):
        super(Low_Level_Filter2_conv, self).__init__()
        
        # 3x3 convolution with 128 output channels
        self.conv1 = nn.Conv2d(num_in, 128, kernel_size=3, padding=1)
        self.silu1 = nn.SiLU()  # SiLU (Sigmoid Linear Unit) activation function
        
        # 3x3 convolution with 32 output channels
        self.conv2 = nn.Conv2d(128, 32, kernel_size=3, padding=1)
        self.silu2 = nn.SiLU()  # SiLU (Sigmoid Linear Unit) activation function

        self.AAP = nn.AdaptiveAvgPool2d((16, 16))

        # MLP with 32 input features and 3 output features
        self.mlp = nn.Sequential(
            nn.Linear(32 * 16 * 16, 128),  # Assuming input image size is 64x64
            nn.SiLU(),  # SiLU (Sigmoid Linear Unit) activation function
            nn.Linear(128, num_out)
        )

    def forward(self, x):

        # Apply conv1 and silu1
        x = self.conv1(x)
        x = self.silu1(x)
        
        # Apply conv2 and silu2
        x = self.conv2(x)
        x = self.silu2(x)
        
        # Flatten the output before passing it to MLP
        # x = x.view(x.size(0), -1)
        x = self.AAP(x)
        x = x.view(x.size(0), -1)
        
        # Apply MLP
        x = self.mlp(x)
        
        return x




















