


import os
from PIL import Image
import shutil

import os
from tqdm import tqdm

# import (../)

# ---------------------------------
# 2024-08-13
# 将模型最佳结果转移到一个文件中
# ---------------------------------
def get_directories(path):
    """
    获取指定路径下的所有目录（文件夹）名称。

    参数:
    path: 要检查的目录的路径。

    返回:
    directories: 一个包含所有目录名称的列表。
    """
    directories = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    return directories

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == '__main__':

    test_name = 'LOLv2_syn'
    best_epoch = 15000
    model_name = 'LLIE_Prior_PromptV2_5_Arch_LOLv2_Syn_Dataset_20240811_105714'
    file_path = f'/data/xuwu/codes/low_light/PromptEnhance/experiments/{model_name}/visualization'
    save_path = f'/data/xuwu/codes/low_light/PromptEnhance/experiments/{model_name}/results_{best_epoch}/{test_name}'
    create_dir(save_path)

    file_name_list = get_directories(file_path)

    for img_name in tqdm(file_name_list):

        img_full_name = f'{img_name}_{best_epoch}.png'
        source_path = f'{file_path}/{img_name}/{img_full_name}'
        target_path = f'{save_path}/{img_name}.png'
        shutil.copy2(source_path, target_path)









