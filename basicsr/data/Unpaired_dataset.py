import os
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

from basicsr.data.transforms import augment, paired_random_crop, mod_crop
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
class Unpaired_Dataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(Unpaired_Dataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.is_noise = opt['is_noise']


        self.imgs_ll = glob.glob(os.path.join(opt['dataroot'], '*.*'))

        self.img_size = None
        self.imgs_ll.sort()


    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        imgs_ll_path = self.imgs_ll[index]  # 低照度，返回下标为index的低照度图片路径
        # print(self.imgs_ll_path)
        # import sys
        # sys.exit()
            
        img_lq = cv2.imread(imgs_ll_path).astype(np.float32) / 255.0
        
        resize = self.opt.get('resize', None)
        if resize:
            img_lq = cv2.resize(img_lq, (resize, resize))
            
        img_lq = img2tensor(img_lq, bgr2rgb=True, float32=True)

        return {
        'lq': img_lq,
        'gt': img_lq,
        'lq_path': imgs_ll_path,
        'gt_path': imgs_ll_path
        }



    def __len__(self):
        return len(self.imgs_ll)
