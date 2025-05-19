import torch
import torch.nn.functional as F
from torch import nn as nn
import numpy as np
import math

from basicsr.utils.registry import ARCH_REGISTRY

from basicsr.archs.network_swinir import RSTB
from basicsr.archs.ridcp_utils import ResBlock, CombineQuantBlock
from basicsr.archs.vgg_arch import VGGFeatureExtractor
from basicsr.archs.Segs.seg_arg import SegFeatureExtractor


# --------------------------------------------------------
# 2023.09.14
# 基于 LLIE_Prior_OS_Refer_Sem_Skip423_exp3_arch
# 改：
# 1. 引入语义信息：encoder提取了浅层信息后，与语义信息融合，然后送入 swin transformer中处理
# 2. 设置专门用于学习 gt indices 的函数
# 3. 将 encoder 的各层输出通过跳跃连接送入到 decoder中,
#       纹理控制：采用codeformer中的融合方式： Fuse_aft_block，计算均值，方差,均值 + 线性插值
#           将 enc feat 与 dec feat 融合后，用两个cnn layers 计算方差、均值
#       风格（照度、对比度）控制：计算方差，方差 * decoder feat。
#           计算方差时，可以选择输入图像：低照度图像，或者是高质量图像（类似于风格迁移）
#           将 reference image feat 与 decoder feat 融合后，计算方差，均值
#           x = self.encode_enc2(torch.cat([enc_feat, x], dim=1))
#         style_feat_mean, style_feat_std = calc_mean_std(x)
#         residual2 = w2 * (dec_feat * style_feat_std + style_feat_mean)
# 4. 添加了 特征损失的
# 5. 将 test 的推理代码与训练的解偶
# --------------------------------------------------------


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

class Fuse_aft_block(nn.Module):

    '''
    跳跃连接的特征融合
    -1 cat decoder 、 encoder特征
    -2 根据cat的特征，获得 scale,shift 两个参数
    -3 y = a*b + c的形式获得融合特征： y = scale * dec feat + shift
    '''

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.encode_enc = ResBlock(2*in_ch, out_ch)
        self.encode_enc2 = ResBlock(2*in_ch, out_ch)

        self.scale = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1))

        self.shift = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1))

    def forward(self, enc_feat, style_feat=None, dec_feat=None, w1=1, w2=1):
        '''

        :param enc_feat: low level feats,弥补纹理信息
        :param dec_feat: decoder feats
        :param style_feat: it is used for computing variance，用于控制输出图像的对比度、亮度（style feats）
        :param w1: 控制纹理信息的影响
        :param w2: 控制风格信息的影响
        :return:
        '''
        enc_feat = self.encode_enc(torch.cat([enc_feat, dec_feat], dim=1))
        scale = self.scale(enc_feat)
        shift = self.shift(enc_feat)
        residual = w1 * (dec_feat * scale + shift)

        if style_feat is not None:
            x = style_feat
        else:
            x = enc_feat

        x = self.encode_enc2(torch.cat([enc_feat, x], dim=1))
        style_feat_mean, style_feat_std = calc_mean_std(x)
        residual2 = w2 * (dec_feat * style_feat_std + style_feat_mean)

        out = dec_feat + residual + residual2

        return out


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
                 n_e,       # codebook 的长度
                 e_dim,     # codebook 每个特征的维度
                 weight_path=r'/home/wuxu/codes/RIDCP/pretrain_networks/weight_for_matching_dehazing_Flickr.pth',
                 beta=0.25,
                 LQ_stage=False,
                 use_weight=True,
                 weight_alpha=1.0):
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


    def dist(self, x, y):
        if x.shape == y.shape:
            return (x - y) ** 2
        else:
            return torch.sum(x ** 2, dim=1, keepdim=True) + \
                    torch.sum(y**2, dim=1) - 2 * \
                    torch.matmul(x, y.t())

    def gram_loss(self, x, y):
        b, h, w, c = x.shape
        x = x.reshape(b, h*w, c)
        y = y.reshape(b, h*w, c)

        gmx = x.transpose(1, 2) @ x / (h*w)
        gmy = y.transpose(1, 2) @ y / (h*w)

        return (gmx - gmy).square().mean()

    def forward(self, z, gt_indices=None, current_iter=None, weight_alpha=None):
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
        # print('000000000000000')
        # print(self.e_dim)
        z_flattened = z.view(-1, self.e_dim)

        codebook = self.embedding.weight
        # print(z_flattened.size())
        # print(codebook.size())

        # -------------------------------------------------------
        # 计算 特征与codebook中量化特征之间的距离
        # -------------------------------------------------------
        d = self.dist(z_flattened, codebook)    # d: [16383, 512]

        # -------------------------------------------------------
        # CHM : Controllable HQPs Matching
        # 仅在测试时使用
        # -------------------------------------------------------
        # if self.use_weight and self.LQ_stage:
        #     if weight_alpha is not None:
        #         self.weight_alpha = weight_alpha
        #     d = d * torch.exp(self.weight_alpha * self.weight)

        # -------------------------------------------------------
        # 根据计算的距离，获得z对应量化特征的下标 index
        # find closest encodings
        # -------------------------------------------------------
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)   #
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

        e_latent_loss = torch.mean((z_q.detach() - z)**2)  
        q_latent_loss = torch.mean((z_q - z.detach())**2)

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
        min_encodings.scatter_(1, indices[:,None], 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings.float(), self.embedding.weight)
        z_q = z_q.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        return z_q


class SwinLayers(nn.Module):
    def __init__(self, input_resolution=(32, 32), embed_dim=256,
                blk_depth=6,
                num_heads=8,
                window_size=8,
                **kwargs):
        super().__init__()
        self.swin_blks = nn.ModuleList()
        for i in range(4):
            layer = RSTB(embed_dim, input_resolution, blk_depth, num_heads, window_size, patch_size=1, **kwargs)
            self.swin_blks.append(layer)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.reshape(b, c, h*w).transpose(1, 2)
        for m in self.swin_blks:
            x = m(x, (h, w))
        x = x.transpose(1, 2).reshape(b, c, h, w)
        return x


class MultiScaleEncoder(nn.Module):
    def __init__(self,
                 in_channel,
                 max_depth,
                 input_res=256,
                 channel_query_dict=None,
                 norm_type='gn',
                 act_type='leakyrelu',
                 LQ_stage=True,
                 **swin_opts,
                 ):
        super().__init__()
        self.LQ_stage = LQ_stage
        ksz = 3

        self.in_conv = nn.Conv2d(in_channel, channel_query_dict[input_res], 4, padding=1)

        self.blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.max_depth = max_depth
        res = input_res
        for i in range(max_depth):
            in_ch, out_ch = channel_query_dict[res], channel_query_dict[res // 2]
            tmp_down_block = [
                nn.Conv2d(in_ch, out_ch, ksz, stride=2, padding=1),
                ResBlock(out_ch, out_ch, norm_type, act_type),
                ResBlock(out_ch, out_ch, norm_type, act_type),
            ]
            self.blocks.append(nn.Sequential(*tmp_down_block))
            res = res // 2

    def forward(self, input):

        x = self.in_conv(input)

        for idx, m in enumerate(self.blocks):
            with torch.backends.cudnn.flags(enabled=False):
                x = m(x)

        return x


class DecoderBlock(nn.Module):

    def __init__(self, in_channel, out_channel, norm_type='gn', act_type='leakyrelu'):
        super().__init__()

        self.block = []
        self.block += [
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channel, out_channel, 3, stride=1, padding=1),
            ResBlock(out_channel, out_channel, norm_type, act_type),
            ResBlock(out_channel, out_channel, norm_type, act_type),
        ]

        self.block = nn.Sequential(*self.block)

    def forward(self, input):
        return self.block(input)


@ARCH_REGISTRY.register()
class CodeEnhance4321_Arch(nn.Module):
    def __init__(self,
                 *,
                 in_channel=3,
                 codebook_params=None,
                 gt_resolution=256,
                 LQ_stage=False,
                 norm_type='gn',
                 act_type='silu',
                 use_quantize=True,
                 use_semantic_loss=False,
                 use_Latent_ContrastLoss=False,
                 use_weight=False,
                 weight_alpha=1.0,
                 seg_cfg=None,
                 weight_texture=1,
                 weight_style=1,
                 **ignore_kwargs):

        super().__init__()

        codebook_params = np.array(codebook_params)

        self.codebook_scale = codebook_params[:, 0]             # 64
        codebook_emb_num = codebook_params[:, 1].astype(int)    # 1024
        codebook_emb_dim = codebook_params[:, 2].astype(int)    # 512

        self.use_quantize = use_quantize
        self.in_channel = in_channel
        self.gt_res = gt_resolution
        self.LQ_stage = LQ_stage
        # self.use_residual = use_residual
        # self.only_residual = only_residual
        self.use_weight = use_weight
        # self.use_warp = use_warp
        self.weight_alpha = weight_alpha
        self.weight_texture = weight_texture
        self.weight_style = weight_style
        self.use_Latent_ContrastLoss = use_Latent_ContrastLoss

        channel_query_dict = {
            8: 256,
            16: 256,
            32: 256,
            64: 256,
            128: 128,
            256: 64,
            512: 32,
        }

        # build encoder
        self.max_depth = int(np.log2(gt_resolution // self.codebook_scale[0]))
        self.multiscale_encoder = MultiScaleEncoder(
                                in_channel,
                                self.max_depth,
                                self.gt_res,
                                channel_query_dict,
                                norm_type, act_type, LQ_stage
                            )

        self.context_module = SwinLayers()

        # build decoder
        self.decoder_group = nn.ModuleList()
        for i in range(self.max_depth):
            res = gt_resolution // 2**self.max_depth * 2**i
            in_ch, out_ch = channel_query_dict[res], channel_query_dict[res * 2]
            self.decoder_group.append(DecoderBlock(in_ch, out_ch, norm_type, act_type))

        self.out_conv = nn.Conv2d(out_ch, 3, 3, 1, 1)
        # self.residual_conv = nn.Conv2d(out_ch, 3, 3, 1, 1)

        # fuse_conv_dict
        self.connect_list = [64, 128]
        self.fuse_convs_dict = nn.ModuleDict()
        for f_size in self.connect_list:
            in_ch = channel_query_dict[f_size]
            self.fuse_convs_dict[str(f_size)] = Fuse_aft_block(in_ch, in_ch)

        # build multi-scale vector quantizers
        self.quantize_group = nn.ModuleList()
        self.before_quant_group = nn.ModuleList()
        self.after_quant_group = nn.ModuleList()

        # 对特征进行量化操作，codebook_params.shape[0] = 1
        for scale in range(0, codebook_params.shape[0]):
            quantize = VectorQuantizer(
                codebook_emb_num[scale],
                codebook_emb_dim[scale],
                LQ_stage=self.LQ_stage,
                use_weight=self.use_weight,
                weight_alpha=self.weight_alpha
            )
            self.quantize_group.append(quantize)

            scale_in_ch = channel_query_dict[self.codebook_scale[scale]]
            if scale == 0:
                quant_conv_in_ch = scale_in_ch
                comb_quant_in_ch1 = codebook_emb_dim[scale]
                comb_quant_in_ch2 = 0
            else:
                quant_conv_in_ch = scale_in_ch * 2
                comb_quant_in_ch1 = codebook_emb_dim[scale - 1]
                comb_quant_in_ch2 = codebook_emb_dim[scale]

            self.before_quant_group.append(nn.Conv2d(quant_conv_in_ch, codebook_emb_dim[scale], 1))
            self.after_quant_group.append(CombineQuantBlock(comb_quant_in_ch1, comb_quant_in_ch2, scale_in_ch))

        # semantic loss for HQ pretrain stage
        self.use_semantic_loss = use_semantic_loss
        self.conv_semantic = nn.Sequential(nn.Conv2d(512, 512, 3, 2, 1),
                                           nn.ReLU(), )
        self.vgg_feat_layer = 'relu4_4'
        self.vgg_feat_extractor = VGGFeatureExtractor([self.vgg_feat_layer])

        self.seg_feat_extractor = SegFeatureExtractor(cfg=seg_cfg)
        self.resize_semantic_feat_conv = nn.Conv2d(320, 256, 3, 1, 1)
        self.fuse_semantic_encode = nn.Sequential(nn.Conv2d(512, 256, 3, 1, 1),
                                                  nn.ReLU())


    def _embed_semantic_before_context(self, encode_feat, semantic_feat):
        '''
        将语义信息与encoder的输入融合
        :param x:
        :param y:
        :return:
        '''
        feat_size = encode_feat.size()[2:]

        semantic_feat = F.interpolate(semantic_feat, size=feat_size)
        semantic_feat = self.resize_semantic_feat_conv(semantic_feat)

        fuse_info = self.fuse_semantic_encode(torch.cat((encode_feat,semantic_feat), 1))

        return fuse_info


    def _encode_reference_feats(self, input, net_hq):

        enc_feat_dict = {}

        x= net_hq.multiscale_encoder.in_conv(input)
        for idx, m in enumerate(net_hq.multiscale_encoder.blocks):
            cur_res = self.gt_res // 2 ** self.max_depth * 2 ** (1-idx)
            with torch.backends.cudnn.flags(enabled=False):
                x = m(x)
                enc_feat_dict[str(cur_res)] = x.clone()
        # print(enc_feat_dict)

        return enc_feat_dict


    def _get_semantic_info(self, x):
        '''
        利用现有的网络，获得图像的语义信息
        :param x:
        :return:
        '''

        out = {}
        out['vgg_feats'] = None
        out['seg_feats'] = None

        # vgg
        if x is not None:
            with torch.no_grad():
                vgg_feats = self.vgg_feat_extractor(x)[self.vgg_feat_layer]
                out['vgg_feats'] = vgg_feats
                # mobilnet
                #   deeplabv3plus: seg_feat = {'high_level_features': 'out', 'low_level_features': 'low_level'}
                #   deeplabv3: seg_feat = {'high_level_features': 'out'}
                # resnet
                #   deeplabv3plus: seg_feat = {'layer4': 'out', 'layer1': 'low_level'}
                #   deeplabv3: seg_feat = {'layer4': 'out'}
                seg_feats = self.seg_feat_extractor(x)
                out['seg_feats'] = seg_feats['out']

        return out


    def encode_indices(self, input, weight_alpha=None):

        '''
        用于获得 input 的 量化特征 的 indices，这部分只能用到 VQGAN stage1 中训练到的结构！
        :param input:
        :return:
        '''

        indices_list = []
        enc_feats = self.multiscale_encoder(input)

        feat_to_quant = self.before_quant_group[0](enc_feats)

        # 量化特征：获得量化特征、计算codebook loss
        if weight_alpha is not None:
            self.weight_alpha = weight_alpha
        z_quant, codebook_loss, indices = self.quantize_group[0](feat_to_quant, weight_alpha=self.weight_alpha)

        after_quant_feat = self.after_quant_group[0](z_quant, None)

        indices_list.append(indices)

        return indices_list, z_quant, feat_to_quant, after_quant_feat


    def decode_indices(self, indices):
        assert len(indices.shape) == 4, f'shape of indices must be (b, 1, h, w), but got {indices.shape}'

        z_quant = self.quantize_group[0].get_codebook_entry(indices)
        x = self.after_quant_group[0](z_quant)

        for m in self.decoder_group:
            x = m(x)
        out_img = self.out_conv(x)

        return out_img

    @torch.no_grad()
    def _inference(self, input, reference_img=None, weight_alpha=None, net_hq=None):
        '''
        仅用于测试
        :param input:
        :param reference_img:
        :param weight_alpha:
        :param net_hq:
        :return:
        '''
        # --------------------------------------------
        # encoder：获得图像特征
        # --------------------------------------------

        high_level_feats_input = self._get_semantic_info(input)
        vgg_feat = high_level_feats_input['vgg_feats']
        semantic_feat = high_level_feats_input['seg_feats']

        # enc_feats = self.multiscale_encoder(input)
        enc_feat_dict = {}
        enc_feats_refer_dict = {}

        x = self.multiscale_encoder.in_conv(input)
        for idx, m in enumerate(self.multiscale_encoder.blocks):
            cur_res = self.gt_res // 2 ** self.max_depth * 2 ** (1 - idx)
            with torch.backends.cudnn.flags(enabled=False):
                x = m(x)
                enc_feat_dict[str(cur_res)] = x.clone()

        enc_feats = x

        if reference_img is not None and net_hq is not None:
            if reference_img.shape[:2] != input.shape[:2]:
                reference_img = F.interpolate(reference_img, input.shape[:2])
            enc_feats_refer_dict = self._encode_reference_feats(reference_img, net_hq)
        else:
            enc_feats_refer_dict = enc_feat_dict

        # --------------------------------------------
        # 引入语义信息,计算上下文关系
        # --------------------------------------------
        enc_feats_semantic = self._embed_semantic_before_context(encode_feat=enc_feats,
                                                                 semantic_feat=semantic_feat)
        enc_feats_context = self.context_module(enc_feats_semantic)

        codebook_loss_list = []
        indices_list = []
        semantic_loss_list = []
        code_decoder_output = []

        quant_idx = 0
        prev_dec_feat = None
        prev_quant_feat = None
        out_img = None

        # --------------------------------------------
        # Decoder：VQ-GAN  stage I 训练好的 decoder 重构图像
        # 输入：为encoder的特征，以及经过量化的特征。Note：将多个level的特征量化后送入相应Decoder层
        # 输出：重构图像
        # --------------------------------------------
        x = enc_feats_context
        # print(f'enc_feats_context: {enc_feats_context.shape}')
        for i in range(self.max_depth):
            cur_res = self.gt_res // 2 ** self.max_depth * 2 ** i

            # 如果此时输入特征的长度在 [64, 1024, 512] 中，则将该特征量化
            if cur_res in self.codebook_scale:  # needs to perform quantize

                # 获得用于量化的特征：将输入特征
                if prev_dec_feat is not None:
                    # 将decoer上一个level的输入特征与当前输入特征做拼接，相当于一个跳跃连接了。
                    before_quant_feat = torch.cat((x, prev_dec_feat), dim=1)
                else:
                    before_quant_feat = x
                feat_to_quant = self.before_quant_group[quant_idx](before_quant_feat)

                # 量化特征：获得量化特征、计算codebook loss
                if weight_alpha is not None:
                    self.weight_alpha = weight_alpha
                z_quant, codebook_loss, indices = self.quantize_group[quant_idx](feat_to_quant,
                                                                                 weight_alpha=self.weight_alpha)

                if not self.use_quantize:
                    z_quant = feat_to_quant

                after_quant_feat = self.after_quant_group[quant_idx](z_quant, prev_quant_feat)

                quant_idx += 1
                prev_quant_feat = z_quant
                x = after_quant_feat

            # 跳跃连接
            x = self.fuse_convs_dict[str(cur_res)](enc_feat_dict[str(cur_res)].detach(),
                                                   enc_feats_refer_dict[str(cur_res)].detach(),
                                                   x,
                                                   self.weight_texture,
                                                   self.weight_style)

            x = self.decoder_group[i](x)
            code_decoder_output.append(x)
            prev_dec_feat = x

        out_img = self.out_conv(x)

        return out_img

    @torch.no_grad()
    def test_tile(self, input, tile_size=240, tile_pad=16):
        # return self.test(input)
        """It will first crop input images to tiles, and then process each tile.
        Finally, all the processed tiles are merged into one images.
        Modified from: https://github.com/xinntao/Real-ESRGAN/blob/master/realesrgan/utils.py
        """
        batch, channel, height, width = input.shape
        output_height = height
        output_width = width
        output_shape = (batch, channel, output_height, output_width)

        # start with black image
        output = input.new_zeros(output_shape)
        tiles_x = math.ceil(width / tile_size)
        tiles_y = math.ceil(height / tile_size)

        # loop over all tiles
        for y in range(tiles_y):
            for x in range(tiles_x):
                # extract tile from input image
                ofs_x = x * tile_size
                ofs_y = y * tile_size
                # input tile area on total image
                input_start_x = ofs_x
                input_end_x = min(ofs_x + tile_size, width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + tile_size, height)

                # input tile area on total image with padding
                input_start_x_pad = max(input_start_x - tile_pad, 0)
                input_end_x_pad = min(input_end_x + tile_pad, width)
                input_start_y_pad = max(input_start_y - tile_pad, 0)
                input_end_y_pad = min(input_end_y + tile_pad, height)

                # input tile dimensions
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y
                tile_idx = y * tiles_x + x + 1
                input_tile = input[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

                # upscale tile
                output_tile, _ = self.test(input_tile)

                # output tile area on total image
                output_start_x = input_start_x
                output_end_x = input_end_x
                output_start_y = input_start_y
                output_end_y = input_end_y

                # output tile area without padding
                output_start_x_tile = (input_start_x - input_start_x_pad)
                output_end_x_tile = output_start_x_tile + input_tile_width
                output_start_y_tile = (input_start_y - input_start_y_pad)
                output_end_y_tile = output_start_y_tile + input_tile_height

                # put tile into output image
                output[:, :, output_start_y:output_end_y,
                       output_start_x:output_end_x] = output_tile[:, :, output_start_y_tile:output_end_y_tile,
                                                                  output_start_x_tile:output_end_x_tile]
        return output


    @torch.no_grad()
    def test(self, input, reference, net_hq, weight_alpha=None):
        org_use_semantic_loss = self.use_semantic_loss
        self.use_semantic_loss = False

        # padding to multiple of window_size * 8
        wsz = 32
        _, _, h_old, w_old = input.shape
        h_pad = (h_old // wsz + 1) * wsz - h_old
        w_pad = (w_old // wsz + 1) * wsz - w_old
        input = torch.cat([input, torch.flip(input, [2])], 2)[:, :, :h_old + h_pad, :]
        input = torch.cat([input, torch.flip(input, [3])], 3)[:, :, :, :w_old + w_pad]

        if reference is not None:
            reference = torch.cat([reference, torch.flip(reference, [2])], 2)[:, :, :h_old + h_pad, :]
            reference = torch.cat([reference, torch.flip(reference, [3])], 3)[:, :, :, :w_old + w_pad]

        output= self._inference(input=input,
                                                    reference_img=reference,
                                                    net_hq=net_hq,
                                                    weight_alpha=weight_alpha)

        if output is not None:
            output = output[..., :h_old, :w_old]
        self.use_semantic_loss = org_use_semantic_loss

        return output


    def encode_and_decode(self, input, gt_img=None, reference_img=None, gt_indices=None, weight_alpha=None, net_hq=None):

        # --------------------------------------------
        # encoder：获得图像特征
        # --------------------------------------------

        high_level_feats_input = self._get_semantic_info(input)
        vgg_feat = high_level_feats_input['vgg_feats']
        semantic_feat = high_level_feats_input['seg_feats']

        # enc_feats = self.multiscale_encoder(input)
        enc_feat_dict = {}
        enc_feats_refer_dict = {}

        x= self.multiscale_encoder.in_conv(input)
        for idx, m in enumerate(self.multiscale_encoder.blocks):
            cur_res = self.gt_res // 2 ** self.max_depth * 2 ** (1-idx)
            with torch.backends.cudnn.flags(enabled=False):
                x = m(x)
                enc_feat_dict[str(cur_res)] = x.clone()

        enc_feats = x

        if reference_img is not None and net_hq is not None:
            if reference_img.shape[:2] != input.shape[:2]:
                reference_img = F.interpolate(reference_img, input.shape[:2])
            enc_feats_refer_dict = self._encode_reference_feats(reference_img, net_hq)
        else:
            enc_feats_refer_dict = enc_feat_dict

        # --------------------------------------------
        # 引入语义信息,计算上下文关系
        # --------------------------------------------
        enc_feats_semantic = self._embed_semantic_before_context(encode_feat=enc_feats,
                                                                 semantic_feat=semantic_feat)
        enc_feats_context = self.context_module(enc_feats_semantic)

        codebook_loss_list = []
        indices_list = []
        semantic_loss_list = []
        code_decoder_output = []

        quant_idx = 0
        prev_dec_feat = None
        prev_quant_feat = None
        out_img = None

        # --------------------------------------------
        # Decoder：VQ-GAN  stage I 训练好的 decoder 重构图像
        # 输入：为encoder的特征，以及经过量化的特征。Note：将多个level的特征量化后送入相应Decoder层
        # 输出：重构图像
        # --------------------------------------------
        x = enc_feats_context
        # print(f'enc_feats_context: {enc_feats_context.shape}')
        for i in range(self.max_depth):
            cur_res = self.gt_res // 2**self.max_depth * 2**i

            # 如果此时输入特征的长度在 [64, 1024, 512] 中，则将该特征量化
            if cur_res in self.codebook_scale:  # needs to perform quantize

                # 获得用于量化的特征：将输入特征
                if prev_dec_feat is not None:
                    # 将decoer上一个level的输入特征与当前输入特征做拼接，相当于一个跳跃连接了。
                    before_quant_feat = torch.cat((x, prev_dec_feat), dim=1)
                else:
                    before_quant_feat = x
                feat_to_quant = self.before_quant_group[quant_idx](before_quant_feat)

                # 量化特征：获得量化特征、计算codebook loss
                if weight_alpha is not None:
                    self.weight_alpha = weight_alpha
                if gt_indices is not None:
                    z_quant, codebook_loss, indices = self.quantize_group[quant_idx](feat_to_quant, gt_indices[quant_idx], weight_alpha=self.weight_alpha)
                else:
                    z_quant, codebook_loss, indices = self.quantize_group[quant_idx](feat_to_quant, weight_alpha=self.weight_alpha)

                # 语义损失，在中间特征层中计算
                if self.use_semantic_loss:
                    semantic_z_quant = self.conv_semantic(z_quant)
                    semantic_loss = F.mse_loss(semantic_z_quant, vgg_feat)
                    semantic_loss_list.append(semantic_loss)

                if not self.use_quantize:
                    z_quant = feat_to_quant

                after_quant_feat = self.after_quant_group[quant_idx](z_quant, prev_quant_feat)

                codebook_loss_list.append(codebook_loss)
                indices_list.append(indices)

                quant_idx += 1
                prev_quant_feat = z_quant
                x = after_quant_feat

            # 跳跃连接
            x = self.fuse_convs_dict[str(cur_res)](enc_feat_dict[str(cur_res)].detach(),
                                                   enc_feats_refer_dict[str(cur_res)].detach(),
                                                   x,
                                                   self.weight_texture,
                                                   self.weight_style)


            x = self.decoder_group[i](x)
            code_decoder_output.append(x)
            prev_dec_feat = x

        out_img = self.out_conv(x)

        if len(codebook_loss_list) > 0:
            codebook_loss = sum(codebook_loss_list)
        else:
            codebook_loss = 0
        semantic_loss = sum(semantic_loss_list) if len(semantic_loss_list) else codebook_loss * 0

        
        out_dict = {}
        out_dict['out_img'] = out_img
        out_dict['codebook_loss'] = codebook_loss
        out_dict['semantic_loss'] = semantic_loss
        out_dict['feat_to_quant'] = feat_to_quant
        out_dict['after_quant_feat'] = after_quant_feat
        out_dict['z_quant'] = z_quant
        out_dict['indices_list'] = indices_list

        return out_dict



    def forward(self, input, gt_img=None, reference_img=None, gt_indices=None, weight_alpha=None, net_hq=None):

        # --------------------------------------------------------------
        # 训练 stage II，获得gt的量化特征下标，用于特征量化的监督
        # in LQ training stage, need to pass GT indices for supervise.
        # --------------------------------------------------------------
        # if gt_indices is not None:
        outdict = self.encode_and_decode(input=input,
            gt_img=gt_img,
            reference_img=reference_img,
            gt_indices=gt_indices,
            weight_alpha=weight_alpha,
            net_hq=net_hq)

        # --------------------------------------------------------------
        # 测试阶段
        # in HQ stage, or LQ test stage, no GT indices needed.
        # --------------------------------------------------------------
        # else:
        #     dec, codebook_loss, semantic_loss, quant_before_feature, quant_after_feature, indices = self.encode_and_decode(input, reference_img, weight_alpha=weight_alpha,net_hq=net_hq)

        return outdict
