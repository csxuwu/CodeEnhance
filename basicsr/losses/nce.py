import torch
from torch import nn
from packaging import version
from basicsr.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class PatchNCELoss(nn.Module):
    def __init__(self,
                 batch_size,
                 T=0.07,
                 shuffle_y=True):
        super().__init__()
        # self.opt = opt
        self.batch_size = batch_size
        self.T = T
        self.shuffle_y = shuffle_y
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool

    def forward(self, feat_q, feat_k):
        batchSize = feat_q.shape[0]
        dim = feat_q.shape[1]
        feat_k = feat_k.detach()

        # pos logit for input to output
        # torch.bmm 矩阵乘法
        l_pos = torch.bmm(feat_q.view(batchSize, 1, -1), feat_k.view(batchSize, -1, 1))
        l_pos = l_pos.view(batchSize, 1)

        # neg logit

        # Should the negatives from the other samples of a minibatch be utilized?
        # In CUT and FastCUT, we found that it's best to only include negatives
        # from the same image. Therefore, we set
        # --nce_includes_all_negatives_from_minibatch as False
        # However, for single-image translation, the minibatch consists of
        # crops from the "same" high-resolution image.
        # Therefore, we will include the negatives from the entire minibatch.

        batch_dim_for_bmm = self.batch_size

        # reshape features to batch size
        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

        # diagonal entries are similarity between same features, and hence meaningless.
        # just fill the diagonal with very small number, which is exp(-10) and almost zero
        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)

        out = torch.cat((l_pos, l_neg), dim=1) / self.T
        idx = torch.randperm(out.size(1), dtype=torch.long, device=feat_q.device)

        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long, device=feat_q.device)) \
            if self.shuffle_y else \
            self.cross_entropy_loss(out[idx], idx)

        return loss

@LOSS_REGISTRY.register()
class PatchStyleNCELoss(nn.Module):
    # def __init__(self, opt):
    def __init__(self,SHUFFLE_Y=True, T=0.07):
        super().__init__()
        # self.opt = opt
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool
        self.SHUFFLE_Y = SHUFFLE_Y
        self.T = T

    def forward(self, feat_q, feat_k, feat_o):
        batchSize = feat_q.shape[0]
        feat_k = feat_k.detach()

        # pos logit
        l_pos = torch.bmm(feat_q.view(batchSize, 1, -1), feat_o.view(batchSize, -1, 1))
        l_pos = l_pos.view(batchSize, 1)

        # neg logit
        l_neg = torch.bmm(feat_q.view(batchSize, 1, -1), feat_k.view(batchSize, -1, 1))
        l_neg = l_neg.view(batchSize, 1)

        # out = torch.cat((l_pos, l_neg), dim=1) / self.opt.MODEL.PATCH.T   #MODEL.PATCH.T = 0.07
        out = torch.cat((l_pos, l_neg), dim=1) / 0.07
        idx = torch.randperm(out.size(1), dtype=torch.long, device=feat_q.device)

        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long, device=feat_q.device)) \
            if self.SHUFFLE_Y else \
            self.cross_entropy_loss(out[idx], idx)
        return loss


