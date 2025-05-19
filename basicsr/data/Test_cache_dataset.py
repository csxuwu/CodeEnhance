import os
import cv2
import random
import numpy as np
from torch.utils import data as data
from scipy import ndimage
import scipy
import scipy.stats as ss
from scipy.interpolate import interp2d
from scipy.linalg import orth
import glob

from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY

from .data_util import make_dataset


# -----------------------------
# 添加 cache
# -----------------------------


def uint2single(img):
    return np.float32(img/255.)

def single2uint(img):
    return np.uint8((img.clip(0, 1)*255.).round())

def random_resize(img, scale_factor=1.):
    return cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)


cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
@DATASET_REGISTRY.register()
class Test_cache_Dataset(data.Dataset):
    """
    真实低照度图像，非成对的
    """

    def __init__(self, opt):
        super(Test_cache_Dataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.is_noise = opt['is_noise']
        self.is_reference = opt['is_reference']
        self.imgs_refer = None
        self.cache_img = []

        self.imgs_ll = glob.glob(os.path.join(opt['dataroot'], '*.*'))
        if self.is_reference:
            self.imgs_refer = glob.glob(os.path.join(opt['reference_path'], '*.*'))

        self.imgs_ll.sort()

        self.img_ll_list = []
        self.img_gt_list = []
        self.lq_path_list = []
        self.gt_path_list = []
        self.img_refer_list = []

        self.__cache()


    def __cache(self):

        for i, p in enumerate(self.imgs_ll):
            img_lq = cv2.imread(p)
            # img_lq = cv2.imread(p).astype(np.float32) / 255.0
            h, w, c = img_lq.shape

            resize = self.opt.get('resize', None)
            if resize:
                img_lq = cv2.resize(img_lq, (resize, resize))

            # TODO: color space transform
            # BGR to RGB, HWC to CHW, numpy to tensor
            img_lq = img2tensor(img_lq, bgr2rgb=True, float32=True)

            self.img_ll_list.append(img_lq)
            self.img_gt_list.append(img_lq)
            self.lq_path_list.append(p)
            self.gt_path_list.append(p)

        if self.imgs_refer is not None:
            for index in range(len(self.imgs_refer)):
                imgs_refer_path = self.imgs_refer[index]  # 低照度，返回下标为index的低照度图片路径
                # img_refer = cv2.imread(imgs_refer_path)
                # cv2.imwrite('te.jpg', img_refer)
                img_refer = cv2.imread(imgs_refer_path).astype(np.float32) / 255.0

                if resize:
                    img_refer = cv2.resize(img_refer, (resize, resize))
                else:
                    img_refer = cv2.resize(img_refer, (w, h))
                img_refer = img2tensor(img_refer, bgr2rgb=True, float32=True)
                self.img_refer_list.append(img_refer)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)


        img_lq = self.img_ll_list[index]

        if self.imgs_refer is not None:

            img_refer = self.img_refer_list[index]
            return {
                'lq': img_lq,
                'gt': img_lq,
                'refer': img_refer,
                'lq_path': self.imgs_ll[index],
                'gt_path': self.imgs_ll[index]
            }
        else:
            return {
                'lq': img_lq,
                'gt': img_lq,
                'lq_path': self.imgs_ll[index],
                'gt_path': self.imgs_ll[index]
            }


    # def set_use_cache(self, use_cache):
    #
    #     if use_cache:
    #         x_img = tuple()


    def __len__(self):
        return len(self.imgs_ll)
