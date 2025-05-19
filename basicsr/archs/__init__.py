import importlib
import os
from copy import deepcopy
from os import path as osp

from basicsr.utils import get_root_logger, scandir
from basicsr.utils.registry import ARCH_REGISTRY

__all__ = ['build_network']


def build_network(opt):
    opt = deepcopy(opt)
    network_type = opt.pop('type')
    net = ARCH_REGISTRY.get(network_type)(**opt)
    logger = get_root_logger()
    logger.info(f'Network [{net.__class__.__name__}] is created.')
    return net

def get_modules(imort_path='basicsr.archs', arch_folder=''):

    '''
    将path路径下的模型import
    :param path:
    :return:
    '''

    arch_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(arch_folder) if v.endswith('_arch.py')]
    # import all the arch modules
    arch_modules = [importlib.import_module(f'{imort_path}.{file_name}') for file_name in arch_filenames]
    return arch_modules


# automatically scan and import arch modules for registry
# scan all the files under the 'archs' folder and collect files ending with
# '_arch.py'
arch_folder = osp.dirname(osp.abspath(__file__))
arch_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(arch_folder) if v.endswith('_arch.py')]
# import all the arch modules
_arch_modules = [importlib.import_module(f'basicsr.archs.{file_name}') for file_name in arch_filenames]


# 二级目录下的model
for item in os.scandir(arch_folder):
    if item.is_dir() and item.name != '__pycache__':
        arch_name = item.name
        archs = get_modules(imort_path=f'basicsr.archs.{arch_name}', arch_folder=f'{arch_folder}/{arch_name}')
        _arch_modules.append(archs)
        # try:
        #     archs = get_modules(imort_path=f'basicsr.archs.{arch_name}', arch_folder=f'{arch_folder}/{arch_name}')
        #     _arch_modules.append(archs)
        # except:
        #     raise KeyError(f"Object named basicsr.archs.'{arch_name}' has registry!")







