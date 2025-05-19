
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# import pytorch_colors as colors
import numpy as np
from basicsr.archs import filters_lowlight


# ---------------------------
# 2022.04.4
# 相比branch h1，不对输入进行降采样
# ---------------------------
class Filter_Branch_Img(nn.Module):

    def __init__(self,
                 filters='ContrastClip_Saturation',
                 filters_param_ch={'Color': None, 'Contrast': 3, 'Saturation': 3, 'WB': None},):
        super(Filter_Branch_Img, self).__init__()

        self.filters = filters
        self.filters_param_ch = filters_param_ch
        self.relu = nn.ReLU(inplace=True)
        out_ch = 0
        for key in self.filters_param_ch:
            if self.filters_param_ch[key] is not None:
                out_ch += self.filters_param_ch[key]

        number_f = 32
        self.e_conv1 = nn.Conv2d(3 ,number_f ,3 ,1 ,1 ,bias=True)
        self.e_conv2 = nn.Conv2d(number_f ,number_f ,3 ,1 ,1 ,bias=True)
        self.e_conv3 = nn.Conv2d(number_f ,number_f ,3 ,1 ,1 ,bias=True)
        self.e_conv4 = nn.Conv2d(number_f ,number_f ,3 ,1 ,1 ,bias=True)
        self.e_conv5 = nn.Conv2d(number_f *2 ,number_f ,3 ,1 ,1 ,bias=True)
        self.e_conv6 = nn.Conv2d(number_f *2 ,number_f ,3 ,1 ,1 ,bias=True)
        self.e_conv7 = nn.Conv2d(number_f *2 ,out_ch ,3 ,1 ,1 ,bias=True)

        self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

        self.tanh = nn.Tanh()

    def forward(self, input1, input2, luminance=None):
        '''
        从原始输入图像input1中学习各个filter的参数
        将input2送入各个filter中处理，input2是经过网络生成的，增强了照度
        '''
        # print(input1)

        n,c,h,w = input1.size()
        if luminance is None:
            luminance = filters_lowlight.luminance(input1)

        # 将输入图像固定在 64*64 尺寸，是的filter 参数的学习是独立于原始输入分辨率的，且计算量小
        # input1 = F.interpolate(input1, (64, 64))
        x1 = self.relu(self.e_conv1(input1))
        # p1 = self.maxpool(x1)
        x2 = self.relu(self.e_conv2(x1))
        # p2 = self.maxpool(x2)
        x3 = self.relu(self.e_conv3(x2))
        # p3 = self.maxpool(x3)
        x4 = self.relu(self.e_conv4(x3))

        x5 = self.relu(self.e_conv5(torch.cat([x3 ,x4] ,1)))
        # x5 = self.upsample(x5)
        x6 = self.relu(self.e_conv6(torch.cat([x2 ,x5] ,1)))

        params = self.e_conv7(torch.cat([x1 ,x6] ,1))
        if (params.shape[2], params.shape[3]) != (h, w):
            params = F.interpolate(params, (h,w))


        # ---------------------------------------------
        out_dict = {}
        out_dict['coeffs_out'] = params

        if self.filters == 'Contrast_WB' or self.filters == 'Contrast_Saturation' or \
                self.filters == 'ContrastClip_Saturation':
            params1 = params[:, 0:3, :, :]
            params2 = params[:, 3:6, :, :]
            out1 = 1
            out2 = 2
            if self.filters == 'Contrast_WB':
                out1 = filters_lowlight.contrast_filter(luminance=luminance, x=input2, param=params1)
                out2 = filters_lowlight.WB_filter(out1, params2)
            elif self.filters == 'Contrast_Saturation':
                out1 = filters_lowlight.contrast_filter(luminance=luminance, x=input2, param=params1)
                out2 = filters_lowlight.saturation_filter(out1, params2)
            elif self.filters == 'ContrastClip_Saturation':
                out1 = filters_lowlight.contrast_filter_clip(luminance=luminance, x=input2, param=params1)
                out2 = filters_lowlight.saturation_filter(out1, params2)

            key1, key2 = self.filters.split('_')
            out_dict['param_' + key1] = params1
            out_dict['param_' + key2] = params2
            out_dict['out_' + key1] = out1
            out_dict['out_' + key2] = out2
            out_dict['img_enhance2'] = out2


        elif self.filters == 'Contrast_Color':
            params1 = params[:, 0:3, :, :]
            params_color = params[:,3:15,:,:]
            params_color1 = params[:, 3:6, :, :]
            params_color2 = params[:, 6:9, :, :]
            params_color3 = params[:, 9:12, :, :]
            params_color4 = params[:, 12:15, :, :]

            out1 = filters_lowlight.contrast_filter(luminance=luminance, x=input2, param=params1)
            out2 = filters_lowlight.color_filter(out1, params_color, color_curve_range=(0.90, 1.10))

            out_dict['param_contrast'] = params1
            out_dict['params_color1'] = params_color1
            out_dict['params_color2'] = params_color2
            out_dict['params_color3'] = params_color3
            out_dict['params_color4'] = params_color4
            out_dict['out_contrast'] = out1
            out_dict['out_WB'] = out2
            out_dict['img_enhance2'] = out2

        return out_dict


class Filter_Branch_Feat(nn.Module):

    def __init__(self,
                 filters='ContrastClip_Saturation',
                 filters_param_ch={'Color': None, 'Contrast': 3, 'Saturation': 3, 'WB': None},):
        super(Filter_Branch_Feat, self).__init__()

        self.filters = filters
        self.filters_param_ch = filters_param_ch
        self.relu = nn.ReLU(inplace=True)
        out_ch = 0
        for key in self.filters_param_ch:
            if self.filters_param_ch[key] is not None:
                out_ch += self.filters_param_ch[key]

        number_f = 128
        self.e_conv1 = nn.Conv2d(3 ,number_f ,3 ,1 ,1 ,bias=True)
        self.e_conv2 = nn.Conv2d(number_f ,number_f ,3 ,1 ,1 ,bias=True)
        self.e_conv3 = nn.Conv2d(number_f ,number_f ,3 ,1 ,1 ,bias=True)
        self.e_conv4 = nn.Conv2d(number_f ,number_f ,3 ,1 ,1 ,bias=True)
        self.e_conv5 = nn.Conv2d(number_f *2 ,number_f ,3 ,1 ,1 ,bias=True)
        self.e_conv6 = nn.Conv2d(number_f *2 ,number_f ,3 ,1 ,1 ,bias=True)
        self.e_conv7 = nn.Conv2d(number_f *2 ,out_ch ,3 ,1 ,1 ,bias=True)

        self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

        self.tanh = nn.Tanh()

    def forward(self, input1, input2, luminance=None):
        '''
        从原始输入图像input1中学习各个filter的参数
        将input2送入各个filter中处理，input2是经过网络生成的，增强了照度
        '''

        n,c,h,w = input1.size()

        # 将输入图像固定在 64*64 尺寸，是的filter 参数的学习是独立于原始输入分辨率的，且计算量小
        input1 = F.interpolate(input1, (64, 64))
        x1 = self.relu(self.e_conv1(input1))
        # p1 = self.maxpool(x1)
        x2 = self.relu(self.e_conv2(x1))
        # p2 = self.maxpool(x2)
        x3 = self.relu(self.e_conv3(x2))
        # p3 = self.maxpool(x3)
        x4 = self.relu(self.e_conv4(x3))

        x5 = self.relu(self.e_conv5(torch.cat([x3 ,x4] ,1)))
        # x5 = self.upsample(x5)
        x6 = self.relu(self.e_conv6(torch.cat([x2 ,x5] ,1)))

        params = self.e_conv7(torch.cat([x1 ,x6] ,1))
        if (params.shape[2], params.shape[3]) != (h, w):
            params = F.interpolate(params, (h,w))


        # ---------------------------------------------
        out_dict = {}
        out_dict['coeffs_out'] = params

        if self.filters == 'Contrast_WB' or self.filters == 'Contrast_Saturation' or \
                self.filters == 'ContrastClip_Saturation':
            params1 = params[:, 0:3, :, :]
            params2 = params[:, 3:6, :, :]
            out1 = 1
            out2 = 2
            if self.filters == 'Contrast_WB':
                out1 = filters_lowlight.contrast_filter(luminance=luminance, x=input2, param=params1)
                out2 = filters_lowlight.WB_filter(out1, params2)
            elif self.filters == 'Contrast_Saturation':
                out1 = filters_lowlight.contrast_filter(luminance=luminance, x=input2, param=params1)
                out2 = filters_lowlight.saturation_filter(out1, params2)
            elif self.filters == 'ContrastClip_Saturation':
                out1 = filters_lowlight.contrast_filter_clip(luminance=luminance, x=input2, param=params1)
                out2 = filters_lowlight.saturation_filter(out1, params2)

            key1, key2 = self.filters.split('_')
            out_dict['param_' + key1] = params1
            out_dict['param_' + key2] = params2
            out_dict['out_' + key1] = out1
            out_dict['out_' + key2] = out2
            out_dict['img_enhance2'] = out2


        elif self.filters == 'Contrast_Color':
            params1 = params[:, 0:3, :, :]
            params_color = params[:,3:15,:,:]
            params_color1 = params[:, 3:6, :, :]
            params_color2 = params[:, 6:9, :, :]
            params_color3 = params[:, 9:12, :, :]
            params_color4 = params[:, 12:15, :, :]

            out1 = filters_lowlight.contrast_filter(luminance=luminance, x=input2, param=params1)
            out2 = filters_lowlight.color_filter(out1, params_color, color_curve_range=(0.90, 1.10))

            out_dict['param_contrast'] = params1
            out_dict['params_color1'] = params_color1
            out_dict['params_color2'] = params_color2
            out_dict['params_color3'] = params_color3
            out_dict['params_color4'] = params_color4
            out_dict['out_contrast'] = out1
            out_dict['out_WB'] = out2
            out_dict['img_enhance2'] = out2




        return out_dict

