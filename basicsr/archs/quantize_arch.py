

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import einsum
from einops import rearrange
import numpy as np

from basicsr.utils.registry import ARCH_REGISTRY

# --------------------------------------------------
# 2023.07.17
# 量化特征
# --------------------------------------------------


class CodebookWeight(nn.Module):
    '''
    参考CBAM中的空间注意力机制设计的
    得到一张 空间注意力图
    将空间注意力图拉平至2D，长度为codesize，维度自适应，对应于codebook中每个离散特征的权重
    '''
    def __init__(self, kernel_size=7, e_dim=512):
        super(CodebookWeight, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.e_dim = e_dim

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(320, 320 // 16, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(320 // 16, 1024, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, codesize=1024):

        b, c, _, _ = x.size()

        avg_pool = self.avg_pool(x).view(b,c)
        y = self.fc(avg_pool)
        y = y.permute(1, 0).contiguous()
        y = torch.mean(y, dim=1, keepdim=True)
        y = self.sigmoid(y)

        return y


@ARCH_REGISTRY.register()
class VectorQuantizer_WeightCodebook(nn.Module):
    """
    对codebook进行微调了
    see https://github.com/MishaLaskin/vqvae/blob/d761a999e2267766400dc646d82d3ac3657771d4/models/quantizer.py
    ____________________________________________
    Discretization bottleneck part of the VQ-VAE.
    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    _____________________________________________
    """

    def __init__(self,
                 n_e,
                 e_dim,
                 weight_path=r'/home/wuxu/codes/RIDCP/pretrain_networks/weight_for_matching_dehazing_Flickr.pth',
                 beta=0.25,
                 LQ_stage=False,
                 use_weight=True,
                 weight_alpha=1.0,
                 weight_codebook_type=None):
        super().__init__()
        self.codebook_size = int(n_e)
        self.e_dim = int(e_dim)
        self.LQ_stage = LQ_stage
        self.beta = beta
        self.use_weight = use_weight
        self.weight_alpha = weight_alpha
        if self.use_weight:
            self.weight = nn.Parameter(torch.load(weight_path))
            self.weight.requires_grad = False
        self.embedding = nn.Embedding(self.codebook_size, self.e_dim)

        # weight_codebook 长度 == codebook_size，确保每个离散特征都有不同的加权，这里可以参考 channel attention的那种算法
        if weight_codebook_type is not None:
            self.weight_codebook_flag = True
            # self.weight_codebook = CodebookWeight(ch_in=320, ch_out=1024)
            if weight_codebook_type == 'wc_ss':
                # 利用语义特征来计算 微调codebook 的权重
                self.weight_codebook = CodebookWeight()
            elif weight_codebook_type == 'wc_params':
                # 直接初始化一个可学习参数来 微调codebook 的权重
                self.weight_codebook = nn.Parameter(torch.normal(mean=0, std=1, size=(self.codebook_size, 1)))
            else:
                self.weight_codebook_flag = False


    def dist(self, x, y):
        '''
        计算 x，y之间距离，必须用 (x-y)^2的展开式，因为计算的是 x 与 y 每一项的距离，需要 x * y^T 的操作
        Args:
            x:
            y:

        Returns:

        '''
        if x.shape == y.shape:
            return (x - y) ** 2
        else:
            return torch.sum(x ** 2, dim=1, keepdim=True) + \
                   torch.sum( y**2, dim=1) - 2 * \
                   torch.matmul(x, y.t())
        # return torch.sum(x ** 2, dim=1, keepdim=True) + \
        #        torch.sum(y ** 2, dim=1) - 2 * \
        #        torch.matmul(x, y.t())

    def gram_loss(self, x, y):
        b, h, w, c = x.shape
        x = x.reshape(b, h* w, c)
        y = y.reshape(b, h * w, c)

        gmx = x.transpose(1, 2) @ x / (h * w)
        gmy = y.transpose(1, 2) @ y / (h * w)

        return (gmx - gmy).square().mean()

    def forward(self, z, gt_indices=None, seg_feat=None, current_iter=None, weight_alpha=None, is_train=True):
        """
        Args:
            z: input features to be quantized, z (continuous) -> z_q (discrete)
               z.shape = (batch, channel, height, width)
            gt_indices: feature map of given indices, used for visualization.
        """
        # -------------------------------------------------------
        # 拉平 z，获得codebook
        # reshape z -> (batch, height, width, channel) and flatten
        # -------------------------------------------------------
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)

        codebook = self.embedding.weight

        if gt_indices is None and self.weight_codebook_flag and seg_feat is not None and is_train:
            # gt_indices 为 None, 此时用于获得 gt_indices
            # 在Stage II中，使用到的针对具体任务的训练集，该训练集与训练Stage I 的存在一定的差异（场景丰富度，类别数量等等，一般用ImageNet训练，那么图像内容简单）
            # --> codebook需要做适当迁移，finetune
            # y = self.weight_codebook(seg_feat)
            # print(y.size())
            # print(codebook.size())

            codebook = self.weight_codebook(seg_feat) * codebook

        # -------------------------------------------------------
        # 计算 特征与codebook中量化特征之间的距离
        # -------------------------------------------------------
        d = self.dist(z_flattened, codebook)  # d: [16383, 512]

        # -------------------------------------------------------
        # CHM : Controllable HQPs Matching
        # 仅在测试时使用
        # -------------------------------------------------------
        if self.use_weight and self.LQ_stage:
            if weight_alpha is not None:
                self.weight_alpha = weight_alpha
            d = d * torch.exp(self.weight_alpha * self.weight)

        # -------------------------------------------------------
        # 根据计算的距离，获得z对应量化特征的下标 index
        # find closest encodings
        # -------------------------------------------------------
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)  #
        min_encodings = torch.zeros(min_encoding_indices.shape[0], codebook.shape[0]).to(z)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # -------------------------------------------------------
        # 获得 GT 的 量化特征，用来计算 codebook loss
        # -------------------------------------------------------
        if gt_indices is not None:
            gt_indices = gt_indices.reshape(-1)

            gt_min_indices = gt_indices.reshape_as(min_encoding_indices)
            gt_min_onehot = torch.zeros(gt_min_indices.shape[0], codebook.shape[0]).to(z)
            gt_min_onehot.scatter_(1, gt_min_indices, 1)

            z_q_gt = torch.matmul(gt_min_onehot, codebook)
            z_q_gt = z_q_gt.view(z.shape)

        # -------------------------------------------------------
        # 获得输入图像的量化特征，get quantized latent vectors
        # -------------------------------------------------------
        z_q = torch.matmul(min_encodings, codebook)
        z_q = z_q.view(z.shape)

        # -------------------------------------------------------
        # 训练 Stage I使用到的 特征损失
        # -------------------------------------------------------
        e_latent_loss = torch.mean((z_q.detach() - z) ** 2)
        q_latent_loss = torch.mean((z_q - z.detach()) ** 2)

        if self.LQ_stage and gt_indices is not None:
            # 训练 Stage II
            # codebook_loss = self.dist(z_q, z_q_gt.detach()).mean() \
            # + self.beta * self.dist(z_q_gt.detach(), z)

            codebook_loss = self.beta * self.dist(z_q_gt.detach(), z)
            texture_loss = self.gram_loss(z, z_q_gt.detach())
            # print("codebook loss:", codebook_loss.mean(), "\ntexture_loss: ", texture_loss.mean())
            codebook_loss = codebook_loss + texture_loss
        else:
            # 训练 Stage I
            codebook_loss = q_latent_loss + e_latent_loss * self.beta

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q, codebook_loss, min_encoding_indices.reshape(z_q.shape[0], 1, z_q.shape[2], z_q.shape[3])

    def get_codebook_entry(self, indices):
        b, _, h, w = indices.shape

        indices = indices.flatten().to(self.embedding.weight.device)
        min_encodings = torch.zeros(indices.shape[0], self.codebook_size).to(indices)
        min_encodings.scatter_(1, indices[:, None], 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings.float(), self.embedding.weight)
        z_q = z_q.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        return z_q


@ARCH_REGISTRY.register()
class VectorQuantizer_WeightCodebookLoss(nn.Module):
    """
    对codebook进行微调了
    see https://github.com/MishaLaskin/vqvae/blob/d761a999e2267766400dc646d82d3ac3657771d4/models/quantizer.py
    ____________________________________________
    Discretization bottleneck part of the VQ-VAE.
    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    _____________________________________________
    """

    def __init__(self,
                 codebook_emb_num,
                 codebook_emb_dim,
                 weight_path=r'/home/wuxu/codes/RIDCP/pretrain_networks/weight_for_matching_dehazing_Flickr.pth',
                 beta=0.25,
                 LQ_stage=False,
                 use_weight=True,
                 weight_alpha=1.0,
                 weight_codebook_type=None):
        super().__init__()
        self.codebook_emb_num = int(codebook_emb_num)
        self.codebook_emb_dim = int(codebook_emb_dim)
        self.LQ_stage = LQ_stage
        self.beta = beta
        self.use_weight = use_weight
        self.weight_alpha = weight_alpha
        if self.use_weight:
            self.weight = nn.Parameter(torch.load(weight_path))
            self.weight.requires_grad = False
        self.embedding = nn.Embedding(self.codebook_emb_num, self.codebook_emb_dim)

        self.log_sm = torch.nn.LogSoftmax(dim=1)
        self.sm = torch.nn.Softmax(dim=1)

        # weight_codebook 长度 == codebook_size，确保每个离散特征都有不同的加权，这里可以参考 channel attention的那种算法
        self.weight_codebook_type = weight_codebook_type
        self.weight_codebook_flag = False
        if weight_codebook_type is not None:
            self.weight_codebook_flag = True
            # self.weight_codebook = CodebookWeight(ch_in=320, ch_out=1024)
            if 'wc_ss' in weight_codebook_type:
                # 利用语义特征来计算 微调codebook 的权重
                self.weight_codebook = CodebookWeight()
            elif weight_codebook_type == 'wc_params':
                # 直接初始化一个可学习参数来 微调codebook 的权重
                self.weight_codebook = nn.Parameter(torch.normal(mean=0, std=1, size=(self.codebook_emb_num, 1)))   # 应该初始化为 0
            # else:
            #     self.weight_codebook_flag = False


    def dist(self, x, y):
        '''
        计算 x，y之间距离，必须用 (x-y)^2的展开式，因为计算的是 x 与 y 每一项的距离，需要 x * y^T 的操作
        Args:
            x:
            y:

        Returns:

        '''
        if x.shape == y.shape:
            return (x - y) ** 2
        else:
            return torch.sum(x ** 2, dim=1, keepdim=True) + \
                   torch.sum( y**2, dim=1) - 2 * \
                   torch.matmul(x, y.t())
        # return torch.sum(x ** 2, dim=1, keepdim=True) + \
        #        torch.sum(y ** 2, dim=1) - 2 * \
        #        torch.matmul(x, y.t())

    def codebook_loss_stage2(self, x, y, weight):
        # x : b,h,w,c --> b, c , h, w
        x = x.permute(0, 3, 1, 2).contiguous()
        y = y.permute(0, 3, 1, 2).contiguous()
        loss = F.mse_loss(x, y)
        loss = torch.mean(weight * loss)

        # print(weight.shape)
        # print(loss.shape)
        # print(loss)
        # print(loss)

        return loss

    def gram_loss(self, x, y):
        b, h, w, c = x.shape
        x = x.reshape(b, h* w, c)
        y = y.reshape(b, h * w, c)

        gmx = x.transpose(1, 2) @ x / (h * w)
        gmy = y.transpose(1, 2) @ y / (h * w)

        return (gmx - gmy).square().mean()

    def forward(self, z, gt_indices=None, seg_feat=None, seg_feat_gt=None, current_iter=None, weight_alpha=None, is_train=True):
        """
        Args:
            z: input features to be quantized, z (continuous) -> z_q (discrete)
               z.shape = (batch, channel, height, width)
            gt_indices: feature map of given indices, used for visualization.
        """
        # -------------------------------------------------------
        # 拉平 z，获得codebook
        # reshape z -> (batch, height, width, channel) and flatten
        # -------------------------------------------------------
        exp_variance = self.beta
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.codebook_emb_dim)

        codebook = self.embedding.weight

        # 计算 GT indices 时，对code book做微调
        if gt_indices is None and self.weight_codebook_flag and seg_feat is not None and is_train:
            # gt_indices 为 None, 此时用于获得 gt_indices
            # 在Stage II中，使用到的针对具体任务的训练集，该训练集与训练Stage I 的存在一定的差异（场景丰富度，类别数量等等，一般用ImageNet训练，那么图像内容简单）
            # --> codebook需要做适当迁移，finetune
            # y = self.weight_codebook(seg_feat)
            # print(y.size())
            # print(codebook.size())
            if self.weight_codebook_type == 'wc_params':
                codebook = self.weight_codebook * codebook + codebook

            elif self.weight_codebook == 'wc_ss':
                codebook = self.weight_codebook(seg_feat) * codebook + codebook
            elif self.weight_codebook == 'wc_ss_gt':
                codebook = self.weight_codebook(seg_feat_gt) * codebook + codebook

        # 计算 stage II 时，用于微调 codebook loss
        if seg_feat is not None and seg_feat_gt is not None and gt_indices is not None and is_train:
            # kl_loss = F.kl_div(seg_feat.softmax(dim=-1).log(), seg_feat_gt.softmax(dim=-1), reduction='sum')

            kl_distance = nn.KLDivLoss(reduction='none')

            variance = torch.sum(kl_distance(self.log_sm(seg_feat), self.sm(seg_feat_gt)), dim=1)
            exp_variance = torch.exp(-variance)

            # print(variance.shape)
            # print('variance mean: %.4f' % torch.mean(exp_variance[:]))
            # print('variance min: %.4f' % torch.min(exp_variance[:]))
            # print('variance max: %.4f' % torch.max(exp_variance[:]))
            # print('000')


        # -------------------------------------------------------
        # 计算 特征与codebook中量化特征之间的距离
        # -------------------------------------------------------
        d = self.dist(z_flattened, codebook)  # d: [16383, 512]

        # -------------------------------------------------------
        # CHM : Controllable HQPs Matching
        # 仅在测试时使用
        # -------------------------------------------------------
        if self.use_weight and self.LQ_stage:
            if weight_alpha is not None:
                self.weight_alpha = weight_alpha
            d = d * torch.exp(self.weight_alpha * self.weight)

        # -------------------------------------------------------
        # 根据计算的距离，获得z对应量化特征的下标 index
        # find closest encodings
        # -------------------------------------------------------
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)  # 索引，相当于分类中的索引
        min_encodings = torch.zeros(min_encoding_indices.shape[0], codebook.shape[0]).to(z)     # [b, codesize]
        min_encodings.scatter_(1, min_encoding_indices, 1)          # 将索引进行 one-hot 编码

        # -------------------------------------------------------
        # 获得 GT 的 量化特征，用来计算 codebook loss
        # -------------------------------------------------------
        if gt_indices is not None:
            gt_indices = gt_indices.reshape(-1)

            gt_min_indices = gt_indices.reshape_as(min_encoding_indices)
            gt_min_onehot = torch.zeros(gt_min_indices.shape[0], codebook.shape[0]).to(z)
            gt_min_onehot.scatter_(1, gt_min_indices, 1)

            z_q_gt = torch.matmul(gt_min_onehot, codebook)
            z_q_gt = z_q_gt.view(z.shape)

        # -------------------------------------------------------
        # 获得输入图像的量化特征，get quantized latent vectors
        # -------------------------------------------------------
        z_q = torch.matmul(min_encodings, codebook)
        z_q = z_q.view(z.shape)

        # -------------------------------------------------------
        # 训练 Stage I使用到的 特征损失
        # -------------------------------------------------------
        e_latent_loss = torch.mean((z_q.detach() - z) ** 2)
        q_latent_loss = torch.mean((z_q - z.detach()) ** 2)

        if self.LQ_stage and gt_indices is not None:
            # 训练 Stage II
            # codebook_loss = self.dist(z_q, z_q_gt.detach()).mean() \
            # + self.beta * self.dist(z_q_gt.detach(), z)

            # codebook_loss = self.beta * self.dist(z_q_gt.detach(), z)
            # print(exp_variance)
            # print(z.shape)
            # codebook_loss = exp_variance * self.dist(z_q_gt.detach(), z)
            codebook_loss = self.codebook_loss_stage2(z_q_gt.detach(), z, exp_variance)
            texture_loss = self.gram_loss(z, z_q_gt.detach())
            # print("codebook loss:", codebook_loss.mean(), "\ntexture_loss: ", texture_loss.mean())
            codebook_loss = codebook_loss + texture_loss
        else:
            # 训练 Stage I
            codebook_loss = q_latent_loss + e_latent_loss * self.beta

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q, codebook_loss, min_encoding_indices.reshape(z_q.shape[0], 1, z_q.shape[2], z_q.shape[3])

    def get_codebook_entry(self, indices):
        b, _, h, w = indices.shape

        indices = indices.flatten().to(self.embedding.weight.device)
        min_encodings = torch.zeros(indices.shape[0], self.codebook_emb_num).to(indices)
        min_encodings.scatter_(1, indices[:, None], 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings.float(), self.embedding.weight)
        z_q = z_q.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()

        return z_q

@ARCH_REGISTRY.register()
class VectorQuantizer_WeightCodebookLoss2(nn.Module):
    """
    对codebook进行微调了
    相比 VectorQuantizer_WeightCodebookLoss 调整了下代码编写
    see https://github.com/MishaLaskin/vqvae/blob/d761a999e2267766400dc646d82d3ac3657771d4/models/quantizer.py
    ____________________________________________
    Discretization bottleneck part of the VQ-VAE.
    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    _____________________________________________
    """

    def __init__(self,
                 codebook_emb_num,
                 codebook_emb_dim,
                 weight_path=r'/home/wuxu/codes/RIDCP/pretrain_networks/weight_for_matching_dehazing_Flickr.pth',
                 beta=0.25,
                 LQ_stage=False,
                 use_weight=True,
                 weight_alpha=1.0,
                 weight_codebook_type=None,
                 weight_codebook_loss_type=None):
        super().__init__()
        self.codebook_emb_num = int(codebook_emb_num)
        self.codebook_emb_dim = int(codebook_emb_dim)
        self.LQ_stage = LQ_stage
        self.beta = beta
        self.use_weight = use_weight
        self.weight_alpha = weight_alpha
        if self.use_weight:
            self.weight = nn.Parameter(torch.load(weight_path))
            self.weight.requires_grad = False
        self.embedding = nn.Embedding(self.codebook_emb_num, self.codebook_emb_dim)

        self.log_sm = torch.nn.LogSoftmax(dim=1)
        self.sm = torch.nn.Softmax(dim=1)

        self.weight_codebook_loss_type = weight_codebook_loss_type
        self.weight_codebook_loss_flag = True if weight_codebook_loss_type is not None else False

        # weight_codebook 长度 == codebook_size，确保每个离散特征都有不同的加权，这里可以参考 channel attention的那种算法
        self.weight_codebook_flag = True if weight_codebook_type is not None else False
        self.weight_codebook_type = weight_codebook_type

        if weight_codebook_type is not None:
            if 'wc_ss' in weight_codebook_type:
                # 利用语义特征来计算 微调codebook 的权重
                self.weight_codebook = CodebookWeight()
            elif weight_codebook_type == 'wc_params':
                # 直接初始化一个可学习参数来 微调codebook 的权重
                # self.weight_codebook = nn.Parameter(torch.normal(mean=0, std=1, size=(self.codebook_emb_num, 1)))   # 应该初始化为 0
                self.weight_codebook = nn.Parameter(torch.zeros(size=(self.codebook_emb_num, 1)))   # 应该初始化为 0
            elif weight_codebook_type == 'wc_nc':
                self.nc_beta = 0.1
                self.new_embedding = nn.Embedding(self.codebook_emb_num, self.codebook_emb_dim)


    def dist(self, x, y):
        '''
        计算 x，y之间距离，必须用 (x-y)^2的展开式，因为计算的是 x 与 y 每一项的距离，需要 x * y^T 的操作
        Args:
            x:
            y:

        Returns:

        '''
        if x.shape == y.shape:
            return (x - y) ** 2
        else:
            return torch.sum(x ** 2, dim=1, keepdim=True) + \
                   torch.sum( y**2, dim=1) - 2 * \
                   torch.matmul(x, y.t())
        # return torch.sum(x ** 2, dim=1, keepdim=True) + \
        #        torch.sum(y ** 2, dim=1) - 2 * \
        #        torch.matmul(x, y.t())

    def codebook_loss_stage2(self, x, y, weight):
        # x : b,h,w,c --> b, c , h, w
        x = x.permute(0, 3, 1, 2).contiguous()
        y = y.permute(0, 3, 1, 2).contiguous()
        loss = F.mse_loss(x, y)
        loss = torch.mean(weight * loss)

        # print(weight.shape)
        # print(loss.shape)
        # print(loss)
        # print(loss)

        return loss

    def gram_loss(self, x, y):
        b, h, w, c = x.shape
        x = x.reshape(b, h* w, c)
        y = y.reshape(b, h * w, c)

        gmx = x.transpose(1, 2) @ x / (h * w)
        gmy = y.transpose(1, 2) @ y / (h * w)

        return (gmx - gmy).square().mean()


    def get_z_quant(self, z):

        # -------------------------------------------------------
        # 拉平 z，获得codebook
        # reshape z -> (batch, height, width, channel) and flatten
        # -------------------------------------------------------
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.codebook_emb_dim)

        codebook = self.embedding.weight

        # -------------------------------------------------------
        # 计算 特征与codebook中量化特征之间的距离
        # -------------------------------------------------------
        d = self.dist(z_flattened, codebook)  # d: [16383, 512]


        # -------------------------------------------------------
        # 根据计算的距离，获得z对应量化特征的下标 index
        # find closest encodings
        # -------------------------------------------------------
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)  #
        min_encodings = torch.zeros(min_encoding_indices.shape[0], codebook.shape[0]).to(z)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # -------------------------------------------------------
        # 获得输入图像的量化特征，get quantized latent vectors
        # -------------------------------------------------------
        z_q = torch.matmul(min_encodings, codebook)
        z_q = z_q.view(z.shape)


        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q, min_encoding_indices.reshape(z_q.shape[0], 1, z_q.shape[2], z_q.shape[3])


    def forward(self, z, gt_indices=None,  seg_feat=None, seg_feat_gt=None, vgg_feat=None, vgg_feat_gt=None, current_iter=None, weight_alpha=None, is_train=True):
        """
        Args:
            z: input features to be quantized, z (continuous) -> z_q (discrete)
               z.shape = (batch, channel, height, width)
            gt_indices: feature map of given indices, used for visualization.
        """
        # -------------------------------------------------------
        # 拉平 z，获得codebook
        # reshape z -> (batch, height, width, channel) and flatten
        # -------------------------------------------------------
        exp_variance = self.beta
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.codebook_emb_dim)

        codebook = self.embedding.weight

        # -------------------------------------------------------
        # Stage II：计算微调 codebook 的权重计算
        # LQ stage == True, GT indices == None
        # -------------------------------------------------------
        if self.LQ_stage and gt_indices is None and self.weight_codebook_flag:
            # gt_indices 为 None, 此时用于获得 gt_indices
            # 在Stage II中，使用到的针对具体任务的训练集，该训练集与训练Stage I 的存在一定的差异（场景丰富度，类别数量等等，一般用ImageNet训练，那么图像内容简单）
            # --> codebook需要做适当迁移，finetune
            # y = self.weight_codebook(seg_feat)
            if self.weight_codebook_type == 'wc_params':
                codebook = self.weight_codebook * codebook + codebook
            elif self.weight_codebook_type == 'wc_ss':
                codebook = self.weight_codebook(seg_feat) * codebook + codebook
            elif self.weight_codebook_type == 'wc_ss_gt':
                codebook = self.weight_codebook(seg_feat_gt) * codebook + codebook
            elif self.weight_codebook_type == 'wc_nc':
                codebook = self.beta * self.new_embedding.weight + codebook

        # -------------------------------------------------------
        # Stage II: 计算 codebook loss 的权重
        # -------------------------------------------------------
        if self.LQ_stage and gt_indices is not None and self.weight_codebook_loss_flag:

            kl_distance = nn.KLDivLoss(reduction='none')
            if self.weight_codebook_loss_type == 'seg_feat':
                if seg_feat is not None and seg_feat_gt is not None:
                    variance = torch.sum(kl_distance(self.log_sm(seg_feat), self.sm(seg_feat_gt)), dim=1)
                else:
                    print('seg_feat or seg_feat_gt is None!')
            elif self.weight_codebook_loss_type == 'vgg_feat':
                if vgg_feat is not None and vgg_feat_gt is not None:
                    variance = torch.sum(kl_distance(self.log_sm(seg_feat), self.sm(seg_feat_gt)), dim=1)
                else:
                    print('vgg_feat or vgg_feat_gt is None!')
            exp_variance = torch.exp(-variance) #　exp(1/KL)

        # -------------------------------------------------------
        # 计算 特征与codebook中量化特征之间的距离
        # -------------------------------------------------------
        d = self.dist(z_flattened, codebook)  # d: [16383, 512]

        # -------------------------------------------------------
        # CHM : Controllable HQPs Matching
        # 仅在测试时使用
        # -------------------------------------------------------
        if self.use_weight and self.LQ_stage:
            if weight_alpha is not None:
                self.weight_alpha = weight_alpha
            d = d * torch.exp(self.weight_alpha * self.weight)

        # -------------------------------------------------------
        # 根据计算的距离，获得z对应量化特征的下标 index
        # find closest encodings
        # -------------------------------------------------------
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)  #
        min_encodings = torch.zeros(min_encoding_indices.shape[0], codebook.shape[0]).to(z)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # -------------------------------------------------------
        # 获得 GT 的 量化特征，用来计算 codebook loss
        # -------------------------------------------------------
        if gt_indices is not None:

            gt_indices = gt_indices.reshape(-1)
            gt_min_indices = gt_indices.reshape_as(min_encoding_indices)
            gt_min_onehot = torch.zeros(gt_min_indices.shape[0], codebook.shape[0]).to(z)
            gt_min_onehot.scatter_(1, gt_min_indices, 1)

            z_q_gt = torch.matmul(gt_min_onehot, codebook)
            z_q_gt = z_q_gt.view(z.shape)

        # -------------------------------------------------------
        # 获得输入图像的量化特征，get quantized latent vectors
        # -------------------------------------------------------
        z_q = torch.matmul(min_encodings, codebook)
        z_q = z_q.view(z.shape)

        # -------------------------------------------------------
        # 训练 Stage I使用到的 特征损失
        # -------------------------------------------------------
        e_latent_loss = torch.mean((z_q.detach() - z) ** 2)
        q_latent_loss = torch.mean((z_q - z.detach()) ** 2)

        if self.LQ_stage and gt_indices is not None:
            # 训练 Stage II
            if self.weight_codebook_loss_flag:
                codebook_loss = self.codebook_loss_stage2(z_q_gt.detach(), z, exp_variance)
            else:
                codebook_loss = self.dist(z_q_gt.detach(), z)
            texture_loss = self.gram_loss(z, z_q_gt.detach())
            # print("codebook loss:", codebook_loss.mean(), "\ntexture_loss: ", texture_loss.mean())
            codebook_loss = codebook_loss + texture_loss
        else:
            # 训练 Stage I
            codebook_loss = q_latent_loss + e_latent_loss * self.beta

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q, codebook_loss, min_encoding_indices.reshape(z_q.shape[0], 1, z_q.shape[2], z_q.shape[3])


    # def get_codebook_entry(self, indices):
    #     b, _, h, w = indices.shape
    #
    #     indices = indices.flatten().to(self.embedding.weight.device)
    #     min_encodings = torch.zeros(indices.shape[0], self.codebook_emb_num).to(indices)
    #     min_encodings.scatter_(1, indices[:, None], 1)
    #
    #     # get quantized latent vectors
    #     z_q = torch.matmul(min_encodings.float(), self.embedding.weight)
    #     z_q = z_q.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
    #
    #     return z_q
    def get_codebook_feat(self, indices, shape):
        '''
        codeformer 中采用的
        :param indices:
        :param shape:
        :return:
        '''
        # input indices: batch*token_num -> (batch*token_num)*1
        # shape: batch, height, width, channel
        indices = indices.view(-1,1)
        min_encodings = torch.zeros(indices.shape[0], self.codebook_emb_num).to(indices)
        min_encodings.scatter_(1, indices, 1)
        # get quantized latent vectors
        z_q = torch.matmul(min_encodings.float(), self.embedding.weight)

        if shape is not None:  # reshape back to match original input shape
            z_q = z_q.view(shape).permute(0, 3, 1, 2).contiguous()

        return z_q

class VectorQuantizer(nn.Module):
    """
    see https://github.com/MishaLaskin/vqvae/blob/d761a999e2267766400dc646d82d3ac3657771d4/models/quantizer.py
    ____________________________________________
    Discretization bottleneck part of the VQ-VAE.
    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    _____________________________________________
    """

    def __init__(self,
                 n_e,
                 e_dim,
                 weight_path=r'/home/wuxu/codes/RIDCP/pretrain_networks/weight_for_matching_dehazing_Flickr.pth',
                 beta=0.25,
                 LQ_stage=False,
                 use_weight=True,
                 weight_alpha=1.0,
                 weight_codebook_flag=False):
        super().__init__()
        self.codebook_size = int(n_e)
        self.e_dim = int(e_dim)
        self.LQ_stage = LQ_stage
        self.beta = beta
        self.use_weight = use_weight
        self.weight_alpha = weight_alpha
        if self.use_weight:
            self.weight = nn.Parameter(torch.load(weight_path))
            self.weight.requires_grad = False
        self.embedding = nn.Embedding(self.codebook_size, self.e_dim)


        # weight_codebook 长度 == codebook_size，确保每个离散特征都有不同的加权，这里可以参考 channel attention的那种算法
        if weight_codebook_flag:
            self.weight_codebook_flag = weight_codebook_flag
            # self.weight_codebook = CodebookWeight(ch_in=320, ch_out=1024)
            self.weight_codebook = CodebookWeight()

    def dist(self, x, y):
        '''
        计算 x，y之间距离，必须用 (x-y)^2的展开式，因为计算的是 x 与 y 每一项的距离，需要 x * y^T 的操作
        Args:
            x:
            y:

        Returns:

        '''
        if x.shape == y.shape:
            return (x - y) ** 2
        else:
            return torch.sum(x ** 2, dim=1, keepdim=True) + \
                   torch.sum( y**2, dim=1) - 2 * \
                   torch.matmul(x, y.t())
        # return torch.sum(x ** 2, dim=1, keepdim=True) + \
        #        torch.sum(y ** 2, dim=1) - 2 * \
        #        torch.matmul(x, y.t())

    def gram_loss(self, x, y):
        b, h, w, c = x.shape
        x = x.reshape(b, h* w, c)
        y = y.reshape(b, h * w, c)

        gmx = x.transpose(1, 2) @ x / (h * w)
        gmy = y.transpose(1, 2) @ y / (h * w)

        return (gmx - gmy).square().mean()

    def forward(self, z, gt_indices=None, seg_feat=None, current_iter=None, weight_alpha=None, is_train=True):
        """
        Args:
            z: input features to be quantized, z (continuous) -> z_q (discrete)
               z.shape = (batch, channel, height, width)
            gt_indices: feature map of given indices, used for visualization.
        """
        # -------------------------------------------------------
        # 拉平 z，获得codebook
        # reshape z -> (batch, height, width, channel) and flatten
        # -------------------------------------------------------
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)

        codebook = self.embedding.weight

        # -------------------------------------------------------
        # 计算 特征与codebook中量化特征之间的距离
        # -------------------------------------------------------
        d = self.dist(z_flattened, codebook)  # d: [16383, 512]

        # -------------------------------------------------------
        # CHM : Controllable HQPs Matching
        # 仅在测试时使用
        # -------------------------------------------------------
        if self.use_weight and self.LQ_stage:
            if weight_alpha is not None:
                self.weight_alpha = weight_alpha
            d = d * torch.exp(self.weight_alpha * self.weight)

        # -------------------------------------------------------
        # 根据计算的距离，获得z对应量化特征的下标 index
        # find closest encodings
        # -------------------------------------------------------
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)  #
        min_encodings = torch.zeros(min_encoding_indices.shape[0], codebook.shape[0]).to(z)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # -------------------------------------------------------
        # 获得 GT 的 量化特征，用来计算 codebook loss
        # -------------------------------------------------------
        if gt_indices is not None:
            gt_indices = gt_indices.reshape(-1)

            gt_min_indices = gt_indices.reshape_as(min_encoding_indices)
            gt_min_onehot = torch.zeros(gt_min_indices.shape[0], codebook.shape[0]).to(z)
            gt_min_onehot.scatter_(1, gt_min_indices, 1)

            z_q_gt = torch.matmul(gt_min_onehot, codebook)
            z_q_gt = z_q_gt.view(z.shape)

        # -------------------------------------------------------
        # 获得输入图像的量化特征，get quantized latent vectors
        # -------------------------------------------------------
        z_q = torch.matmul(min_encodings, codebook)
        z_q = z_q.view(z.shape)

        # -------------------------------------------------------
        # 训练 Stage I使用到的 特征损失
        # -------------------------------------------------------
        e_latent_loss = torch.mean((z_q.detach() - z) ** 2)
        q_latent_loss = torch.mean((z_q - z.detach()) ** 2)

        if self.LQ_stage and gt_indices is not None:
            # 训练 Stage II
            # codebook_loss = self.dist(z_q, z_q_gt.detach()).mean() \
            # + self.beta * self.dist(z_q_gt.detach(), z)

            codebook_loss = self.beta * self.dist(z_q_gt.detach(), z)
            texture_loss = self.gram_loss(z, z_q_gt.detach())
            # print("codebook loss:", codebook_loss.mean(), "\ntexture_loss: ", texture_loss.mean())
            codebook_loss = codebook_loss + texture_loss
        else:
            # 训练 Stage I
            codebook_loss = q_latent_loss + e_latent_loss * self.beta

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q, codebook_loss, min_encoding_indices.reshape(z_q.shape[0], 1, z_q.shape[2], z_q.shape[3])

    def get_codebook_entry(self, indices):
        b, _, h, w = indices.shape

        print(f'indice size: {indices.shape}')

        indices = indices.flatten().to(self.embedding.weight.device)
        min_encodings = torch.zeros(indices.shape[0], self.codebook_size).to(indices)
        min_encodings.scatter_(1, indices[:, None], 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings.float(), self.embedding.weight)
        z_q = z_q.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        return z_q


class GumbelQuantize(nn.Module):
    """
    credit to @karpathy: https://github.com/karpathy/deep-vector-quantization/blob/main/model.py (thanks!)
    Gumbel Softmax trick quantizer
    Categorical Reparameterization with Gumbel-Softmax, Jang et al. 2016
    https://arxiv.org/abs/1611.01144
    """
    def __init__(self, num_hiddens, embedding_dim, n_embed, straight_through=True,
                 kl_weight=5e-4, temp_init=1.0, use_vqinterface=True,
                 remap=None, unknown_index="random"):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.n_embed = n_embed

        self.straight_through = straight_through
        self.temperature = temp_init
        self.kl_weight = kl_weight

        self.proj = nn.Conv2d(num_hiddens, n_embed, 1)
        self.embed = nn.Embedding(n_embed, embedding_dim)

        self.use_vqinterface = use_vqinterface

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed+1
            print(f"Remapping {self.n_embed} indices to {self.re_embed} indices. "
                  f"Using {self.unknown_index} for unknown indices.")
        else:
            self.re_embed = n_embed

    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        match = (inds[:,:,None]==used[None,None,...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2)<1
        if self.unknown_index == "random":
            new[unknown]=torch.randint(0,self.re_embed,size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]: # extra token
            inds[inds>=self.used.shape[0]] = 0 # simply set to zero
        back=torch.gather(used[None,:][inds.shape[0]*[0],:], 1, inds)
        return back.reshape(ishape)

    def forward(self, z, temp=None, return_logits=False):
        # force hard = True when we are in eval mode, as we must quantize. actually, always true seems to work
        hard = self.straight_through if self.training else True
        temp = self.temperature if temp is None else temp

        logits = self.proj(z)
        if self.remap is not None:
            # continue only with used logits
            full_zeros = torch.zeros_like(logits)
            logits = logits[:,self.used,...]

        soft_one_hot = F.gumbel_softmax(logits, tau=temp, dim=1, hard=hard)
        if self.remap is not None:
            # go back to all entries but unused set to zero
            full_zeros[:,self.used,...] = soft_one_hot
            soft_one_hot = full_zeros
        z_q = einsum('b n h w, n d -> b d h w', soft_one_hot, self.embed.weight)

        # + kl divergence to the prior loss
        qy = F.softmax(logits, dim=1)
        diff = self.kl_weight * torch.sum(qy * torch.log(qy * self.n_embed + 1e-10), dim=1).mean()

        ind = soft_one_hot.argmax(dim=1)
        if self.remap is not None:
            ind = self.remap_to_used(ind)
        if self.use_vqinterface:
            if return_logits:
                return z_q, diff, (None, None, ind), logits
            return z_q, diff, (None, None, ind)
        return z_q, diff, ind

    def get_codebook_entry(self, indices, shape):
        b, h, w, c = shape
        assert b*h*w == indices.shape[0]
        indices = rearrange(indices, '(b h w) -> b h w', b=b, h=h, w=w)
        if self.remap is not None:
            indices = self.unmap_to_all(indices)
        one_hot = F.one_hot(indices, num_classes=self.n_embed).permute(0, 3, 1, 2).float()
        z_q = einsum('b n h w, n d -> b d h w', one_hot, self.embed.weight)
        return z_q


class VectorQuantizer_org_VQGAN(nn.Module):
    """
    原始 VQGAN里面用到的
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly
    avoids costly matrix multiplications and allows for post-hoc remapping of indices.
    """
    # NOTE: due to a bug the beta term was applied to the wrong term. for
    # backwards compatibility we use the buggy version by default, but you can
    # specify legacy=False to fix it.
    def __init__(self, n_e, e_dim, beta, remap=None, unknown_index="random",
                 sane_index_shape=False, legacy=True):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.legacy = legacy

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed+1
            print(f"Remapping {self.n_e} indices to {self.re_embed} indices. "
                  f"Using {self.unknown_index} for unknown indices.")
        else:
            self.re_embed = n_e

        self.sane_index_shape = sane_index_shape

    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        match = (inds[:,:,None]==used[None,None,...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2)<1
        if self.unknown_index == "random":
            new[unknown]=torch.randint(0,self.re_embed,size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]: # extra token
            inds[inds>=self.used.shape[0]] = 0 # simply set to zero
        back=torch.gather(used[None,:][inds.shape[0]*[0],:], 1, inds)
        return back.reshape(ishape)

    def forward(self, z, temp=None, rescale_logits=False, return_logits=False, gt_indices=None,):
        assert temp is None or temp==1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits==False, "Only for interface compatible with Gumbel"
        assert return_logits==False, "Only for interface compatible with Gumbel"
        # reshape z -> (batch, height, width, channel) and flatten
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(self.embedding.weight, 'n d -> d n'))

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        perplexity = None
        min_encodings = None

        # compute loss for embedding
        if gt_indices is not None:
            gt_indices = gt_indices.reshape(-1)

            # codebook = self.embedding.weight
            gt_min_indices = gt_indices.reshape_as(min_encoding_indices)
            z_q_gt = self.embedding(gt_min_indices).view(z.shape)

            loss = torch.mean((z_q_gt.detach() - z) ** 2) + self.beta * \
                   torch.mean((z_q_gt - z.detach()) ** 2)
        else:
            if not self.legacy:
                loss = self.beta * torch.mean((z_q.detach()-z)**2) + \
                       torch.mean((z_q - z.detach()) ** 2)
            else:
                loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
                       torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()

        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(z.shape[0],-1) # add batch axis
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1,1) # flatten

        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(
                z_q.shape[0], z_q.shape[2], z_q.shape[3])

        z_indices =  min_encoding_indices.reshape(z_q.shape[0], 1, z_q.shape[2], z_q.shape[3])

        return z_q, loss, (perplexity, min_encodings, min_encoding_indices, z_indices)

    def get_codebook_entry(self, indices, shape):
        # shape specifying (batch, height, width, channel)
        if self.remap is not None:
            indices = indices.reshape(shape[0],-1) # add batch axis
            indices = self.unmap_to_all(indices)
            indices = indices.reshape(-1) # flatten again

        # get quantized latent vectors
        z_q = self.embedding(indices)

        if shape is not None:
            z_q = z_q.view(shape)
            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q