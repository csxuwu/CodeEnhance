
import torch
import torch.nn as nn

from basicsr.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class Seg_Loss(nn.Module):
    '''
    利用标签图像的分割结果，计算各个区域内部的照度一致性
    增强图像的区域强度：
        逐像素：应该与标签图像保持一直
        全局：区域强度的均值也应该与标签图像保持一直
        自己：自己跟自己的约束，区域每个像素强度应该趋向于区域均值
    '''
    def __init__(self):
        super(Seg_Loss, self).__init__()

    def forward(self, x, y, mask_y):
        '''

        :param x: enhanced image
        :param y: ground truth
        :param mask_y: mask of ground truth
        :return:
        '''

        mask_y_arg = torch.argmax(mask_y, dim=1, keepdim=True).to(torch.float)
        labels = torch.unique(mask_y_arg)       # 分割图中包含的类别个数

        b, c, h, w = x.shape
        x_avg = torch.mean(x, 1, keepdim=True)  # 通道维度求平均 [B, 1, H, W]
        a1 = torch.zeros(b, h, w).cuda()

        loss1 = 0
        loss2 = 0
        loss3 = 0

        for i in enumerate(labels):

            # x 的区域
            x_region = torch.where(labels == i, x_avg, a1).cuda()
            x_region_avg = torch.mean(x_region)
            x_region = torch.where(x_region == 0, x_region_avg, x_region).cuda()

            # y 的区域
            y_region = torch.where(labels == i, x_avg, a1).cuda()
            y_region_avg = torch.mean(y_region)
            y_region = torch.where(y_region == 0, y_region_avg, y_region).cuda()

            # 逐像素：区域强度应该与标签图像保持一直
            loss1 = loss1 + torch.mean(torch.pow(x_region, y_region))

            # 全局：区域强度的均值也应该与标签图像保持一直
            loss2 = loss2 + torch.mean(torch.pow(torch.FloatTensor([x_region_avg]).cuda(), torch.FloatTensor([y_region_avg]).cuda()))

            # 自己：自己跟自己的约束，区域每个像素强度应该趋向于区域均值
            loss3 = loss3 + torch.mean(torch.pow(x_region - torch.FloatTensor([x_region_avg]).cuda(), 2))

        return loss1, loss2, loss3






