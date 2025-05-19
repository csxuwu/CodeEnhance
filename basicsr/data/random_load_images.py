
import torch
from torchvision import transforms
import os
import random
from PIL import Image
import glob
import cv2
import numpy as np
from basicsr.data.data_util import img2tensor


# ---------------------------------
# 2023.09.08
# 给定图像path list，随机读取B张图像
# ---------------------------------


def random_load_images(path_list, batchsize=8, size_h=384, size_w=384):

    transff = transforms.ToTensor()

    refer_path = random.sample(path_list, batchsize)  # 随机采样2个负样本
    R = []

    for i in range(batchsize):
        # refer_img = Image.open(refer_path[i])
        # refer_img = refer_img.resize((size_h, size_w), Image.ANTIALIAS)

        refer_img = cv2.imread(refer_path[i]).astype(np.float32) / 255.0
        refer_img = cv2.resize(refer_img, dsize=(size_w, size_h))

        refer_img = img2tensor(refer_img)
        R.append(refer_img)
    R = torch.stack(R, 0)
    R = R.cuda()

    return R