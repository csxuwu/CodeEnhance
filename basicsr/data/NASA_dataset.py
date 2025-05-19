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


@DATASET_REGISTRY.register()
class NASA_Dataset(data.Dataset):
    """Paired image dataset for image restoration.

    """

    def __init__(self, opt):
        super(NASA_Dataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.is_noise = opt['is_noise']

        self.imgs_ll = glob.glob(os.path.join(opt['dataroot'], '*.*'))
        # gt_path = opt['dataroot'].replace('NASA', 'NASA_high')
        # self.imgs_gt = glob.glob(os.path.join(gt_path, '*.*'))
        self.img_size = opt.get('gt_size', 512)
        self.imgs_ll.sort()
        # self.imgs_gt.sort()


    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        #  scale = self.opt['scale']

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        # gt_path = self.gt_paths[index]

        imgs_ll_path = self.imgs_ll[index]  # 低照度，返回下标为index的低照度图片路径
        gt_path = imgs_ll_path.replace('/NASA', '/NASA-high')
        gt_path = gt_path.replace('.jpg', '-rtx00.jpg')

        img_lq = cv2.imread(imgs_ll_path).astype(np.float32) / 255.0
        img_gt = cv2.imread(gt_path).astype(np.float32) / 255.0

        resize = self.opt.get('resize', None)
        if resize:
            img_lq = cv2.resize(img_lq, (resize, resize))
            img_gt = cv2.resize(img_gt, (resize, resize))

        # TODO: color space transform
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': gt_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.imgs_ll)
