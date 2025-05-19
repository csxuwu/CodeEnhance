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
import json

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


# ----------------------------------
# 2024.1.12
# 读取图像
# 读取图像的文本描述
# ----------------------------------


@DATASET_REGISTRY.register()
class AGLLNet_v21_Dataset(data.Dataset):
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
        super(AGLLNet_v21_Dataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.is_noise = opt['is_noise']
        # self.is_enhanced = 

        if self.is_noise:
            self.train_input = 'train_lowlight'
            self.test_input = 'test_lowlight'
        else:
            self.train_input = 'train_dark'
            self.test_input = 'test_dark'


        if self.opt['phase'] == 'train':
            self.imgs_ll = glob.glob(os.path.join(opt['dataroot'], self.train_input, '*.*'))

            # if opt['is_enhanced']:   # 对比度增强后的图像
            if self.opt.get('is_enhanced', None):
                print('We use the enhanced gt to train the model.')
                self.imgs_gt = glob.glob(os.path.join(opt['dataroot'], 'train_enhance', '*.*'))
            else:
                self.imgs_gt = glob.glob(os.path.join(opt['dataroot'], 'train', '*.*'))
            self.img_size = opt.get('gt_size', 512)
            # self.img_refer = glob.glob(os.path.join(opt['reference_path'], 'train_enhance','*'))
            if self.opt.get('reference_path', None):
                self.img_refer = glob.glob(os.path.join(opt['reference_path'], 'train_enhance','*'))
            else:
                self.img_refer = self.imgs_gt
            self.img_refer.sort()
            # self.img_refer.reverse()
        else:
            self.imgs_ll = glob.glob(os.path.join(opt['dataroot'], self.test_input, '*.*'))
            if self.opt.get('is_enhanced', None):
                self.imgs_gt = glob.glob(os.path.join(opt['dataroot'], 'test_enhance', '*.*'))
            else:
                self.imgs_gt = glob.glob(os.path.join(opt['dataroot'], 'test', '*.*'))

            self.img_size = opt.get('gt_size', 512)

            # 测试集：输入图像的描述
            self.test_input_captions_path = os.path.join(opt['dataroot'], f'{self.test_input}_captions.json')
            self.test_input_captions = json.load(open(self.test_input_captions_path))
            # 标签图像的描述
            self.test_target_captions_path = os.path.join(opt['dataroot'], f'test_captions.json')
            self.test_target_captions = json.load(open(self.test_target_captions_path))

        self.imgs_ll.sort()
        self.imgs_gt.sort()


    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)


        # augmentation for training
        if self.opt['phase'] == 'train':

            imgs_ll_path = self.imgs_ll[index]  # 低照度，返回下标为index的低照度图片路径
            if self.opt.get('is_enhanced', None):
                gt_path = imgs_ll_path.replace('/'+self.train_input, '/train_enhance')
            else:
                gt_path = imgs_ll_path.replace('/'+self.train_input, '/train')
                gt_path = gt_path.replace('.png', '.jpg')

            img_lq = cv2.imread(imgs_ll_path).astype(np.float32) / 255.0
            img_gt = cv2.imread(gt_path).astype(np.float32) / 255.0

            img_refer_path = self.img_refer[index]
            img_refer = cv2.imread(img_refer_path).astype(np.float32) / 255.0

            input_gt_size = np.min(img_gt.shape[:2])
            input_lq_size = np.min(img_lq.shape[:2])
            input_refer_size = np.min(img_refer.shape[:2])
            scale = input_gt_size // input_lq_size
            gt_size = self.opt['gt_size']

            if self.opt['use_resize_crop']:
                # random resize
                if input_gt_size > gt_size:
                    input_gt_random_size = random.randint(gt_size, input_gt_size)
                    input_gt_random_size = input_gt_random_size - input_gt_random_size % scale # make sure divisible by scale 
                    resize_factor = input_gt_random_size / input_gt_size
                else:
                    resize_factor = (gt_size+1) / input_gt_size
                img_gt = random_resize(img_gt, resize_factor)
                img_lq = random_resize(img_lq, resize_factor)
                img_refer = random_resize(img_refer, resize_factor)

                # random crop
                img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, input_gt_size // input_lq_size, gt_path)
                img_refer = cv2.resize(img_refer, (gt_size, gt_size))

            # flip, rotation
            img_gt, img_lq, img_refer = augment([img_gt, img_lq, img_refer], self.opt['use_flip'],self.opt['use_rot'])
            img_gt, img_lq, img_refer = img2tensor([img_gt, img_lq, img_refer], bgr2rgb=True, float32=True)

            return {
                'lq': img_lq,
                'gt': img_gt,
                'refer': img_refer,
                'lq_path': gt_path,
                'gt_path': gt_path
            }

        elif self.opt['phase'] == 'val':

            imgs_ll_path = self.imgs_ll[index]  # 低照度，返回下标为index的低照度图片路径
            if self.opt.get('is_enhanced', None):
                gt_path = imgs_ll_path.replace('/' + self.test_input, '/test_enhance')
            else:
                gt_path = imgs_ll_path.replace('/' + self.test_input, '/test')
                gt_path = gt_path.replace('.png', '.jpg')

            img_lq = cv2.imread(imgs_ll_path).astype(np.float32) / 255.0
            img_gt = cv2.imread(gt_path).astype(np.float32) / 255.0

            name = imgs_ll_path.split('/')[-1].split('.')[0]
            gt_caption = self.test_target_captions[name]

            resize = self.opt.get('resize', None)
            if resize:
                img_lq = cv2.resize(img_lq, (resize, resize))
                img_gt = cv2.resize(img_gt, (resize, resize))
            img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)

            return {
                'lq': img_lq,
                'gt': img_gt,
                'lq_path': gt_path,
                'gt_path': gt_path,
                'gt_caption': gt_caption
            }


    def __len__(self):
        return len(self.imgs_ll)
