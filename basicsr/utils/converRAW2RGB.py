

import cv2
import numpy as np
import os
import glob
from tqdm import tqdm

import rawpy
import imageio
import matplotlib.pyplot as plt


# ------------------------------
# 20230929
# RAW 格式 转 jpg、png
# ------------------------------

def func1(path):
	# 利用 cv2 处理
    h = 2304
    w = 4096
    c = 1

    # read raw img

    raw_img = np.fromfile(path, dtype=np.uint16)

    shape = raw_img.shape
    print(shape)

    raw_img = raw_img.reshape(h, w, c)
    raw_img = raw_img.astype(np.uint16)

    # demosaic
    rgb_img = cv2.cvtColor(raw_img, cv2.COLOR_BayerBGGR2RGB)

    # normalize&save
    rgb_img_norm = cv2.normalize(rgb_img, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite('test_rawDM_1.png', rgb_img_norm)


def func2(path, save_path, no_auto_bright=False):
	# 利用 rawpy 包处理

    raw = rawpy.imread(path)
    # rgb = raw
    rgb = raw.postprocess(use_camera_wb=True, # 自动白平衡，不执行会偏色
    	half_size=False,		# 是否图像尺寸减半
    	no_auto_bright=no_auto_bright, 	# 不自动调整亮度
    	output_bps=8)			# bit 数据，8 or 16
    # rgb = raw.postprocess(use_camera_wb=True)
    raw.close()

    # print(rgb.dtype, rgb.shape)
    imageio.imsave(save_path, rgb)		# uint = 8
    # plt.imshow(rgb)
    # plt.pause(1)


if __name__ == '__main__':
	# .ARW为索尼Sony相机RAW格式
	# .CR2为佳能canon相机RAW格式

	# 拜耳阵列（Bayer pattern）分为GBRG、GRBG、BGGR、RGGB四种模式

	filename = 'short'			# short
	save_img_type = 'jpg'		# jpg，png

	# 路径设置
	root = r'/home/wuxu/datasets/LL/SID/Fuji/'
	root_RAW = os.path.join(root, 'Fuji', filename)

	save_path = os.path.join(root, 'Fuji_RGB', filename)
	
	if not os.path.exists(save_path):
		os.makedirs(save_path)

	# 图片格式
	if 'Sony' in root_RAW:
		RAW_type = 'ARW'
	elif 'Fuji' in root_RAW:
		# RAW_type = 'CR2'
		RAW_type = 'RAF'

	image_list = glob.glob(os.path.join(root_RAW, '*.*'))
	image_list.sort()

	no_auto_bright = False
	if filename == 'short':
		no_auto_bright = True

	for i in tqdm(range(len(image_list))):

		# if i == 4:
		# 	break

    	# path = "/home/wuxu/datasets/LL/SID/Sony/Sony/long/20211_00_10s.ARW"
		path = image_list[i]
		image_name = os.path.basename(path)
		image_name = image_name.replace(RAW_type, save_img_type)
		rgb_img_save_path = os.path.join(save_path, image_name)
		func2(path, rgb_img_save_path, no_auto_bright)











