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

def uint2single(img):
    return np.float32(img/255.)

def single2uint(img):
    return np.uint8((img.clip(0, 1)*255.).round())

def random_resize(img, scale_factor=1.):
    return cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)


cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
@DATASET_REGISTRY.register()
class Test_Caltech256_Dataset(data.Dataset):
    """
    caltech256 数据集，有二级文件夹的
    """

    def __init__(self, opt):
        super(Test_Caltech256_Dataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.is_noise = opt['is_noise']
        self.is_reference = opt['is_reference']
        self.imgs_refer = None

        self.imgs_ll = glob.glob(os.path.join(opt['dataroot'], '*.*'))
        if self.is_reference:
            self.imgs_refer = glob.glob(os.path.join(opt['reference_path'], '*.*'))

        self.imgs_ll.sort()


    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        imgs_ll_path = self.imgs_ll[index]  # 低照度，返回下标为index的低照度图片路径
        img_lq = cv2.imread(imgs_ll_path).astype(np.float32) / 255.0
        # cv2.imwrite('lq.jpg', img_lq)
        h, w, c = img_lq.shape

        resize = self.opt.get('resize', None)
        if resize:
            img_lq = cv2.resize(img_lq, (resize, resize))

        # TODO: color space transform
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_lq = img2tensor(img_lq, bgr2rgb=True, float32=True)

        if self.imgs_refer is not None:
            imgs_refer_path = self.imgs_refer[index]  # 低照度，返回下标为index的低照度图片路径
            # img_refer = cv2.imread(imgs_refer_path)
            # cv2.imwrite('te.jpg', img_refer)
            img_refer = cv2.imread(imgs_refer_path).astype(np.float32) / 255.0

            if resize:
                img_refer = cv2.resize(img_refer, (resize, resize))
            else:
                img_refer = cv2.resize(img_refer, (w, h))
            img_refer = img2tensor(img_refer, bgr2rgb=True, float32=True)

            return {
                'lq': img_lq,
                'gt': img_lq,
                'refer': img_refer,
                'lq_path': imgs_ll_path,
                'gt_path': imgs_ll_path
            }
        else:
            return {
                'lq': img_lq,
                'gt': img_lq,
                'lq_path': imgs_ll_path,
                'gt_path': imgs_ll_path
            }


    def set_use_cache(self, use_cache):

        if use_cache:
            x_img = tuple()


    def __len__(self):
        return len(self.imgs_ll)
