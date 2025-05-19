
import torch
import torch.nn as nn
from basicsr.utils.registry import LOSS_REGISTRY


# -----------------------------
# 图像经过某种变换后计算损失
# -----------------------------
@LOSS_REGISTRY.register()
class FrequencyLoss(nn.Module):
    '''
    FT，傅里叶变换
    '''
    def __init__(self):
        super(FrequencyLoss, self).__init__()

    def forward(self, x, target):
        b, c, h, w = x.size()
        x = x.contiguous().view(-1, h, w)
        target = target.contiguous().view(-1, h, w)
        x_fft = torch.rfft(x, signal_ndim=2, normalized=False, onesided=True)
        target_fft = torch.rfft(target, signal_ndim=2, normalized=False, onesided=True)

        _, h, w, f = x_fft.size()
        x_fft = x_fft.view(b, c, h, w, f)
        target_fft = target_fft.view(b, c, h, w, f)
        diff = x_fft - target_fft
        return torch.mean(torch.sum(diff ** 2, (1, 2, 3, 4)))

# class WTLoss(nn.Module):
#     '''
#     小波变换
#     '''
#     def __init__(self):
#         super(FrequencyLoss, self).__init__()
#
#     def forward(self, x, target):
#         b, c, h, w = x.size()
#         x = x.contiguous().view(-1, h, w)
#         target = target.contiguous().view(-1, h, w)
#         x_fft = torch.rfft(x, signal_ndim=2, normalized=False, onesided=True)
#         target_fft = torch.rfft(target, signal_ndim=2, normalized=False, onesided=True)
#
#         _, h, w, f = x_fft.size()
#         x_fft = x_fft.view(b, c, h, w, f)
#         target_fft = target_fft.view(b, c, h, w, f)
#         diff = x_fft - target_fft
#         return torch.mean(torch.sum(diff ** 2, (1, 2, 3, 4)))











