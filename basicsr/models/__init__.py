
import os
import importlib
from copy import deepcopy
from os import path as osp


from basicsr.utils import get_root_logger, scandir
from basicsr.utils.registry import MODEL_REGISTRY

__all__ = ['build_model']


def build_model(opt):
    """Build model from options.

    Args:
        opt (dict): Configuration. It must contain:
            model_type (str): Model type.
    """
    opt = deepcopy(opt)
    model = MODEL_REGISTRY.get(opt['model_type'])(opt)
    logger = get_root_logger()
    logger.info(f'Model [{model.__class__.__name__}] is created.')
    return model


def get_modules(imort_path='basicsr.archs', arch_folder=''):

    '''
    将path路径下的模型import
    :param path:
    :return:
    '''

    filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(arch_folder) if v.endswith('_model.py')]

    # import all the arch modules
    modules = [importlib.import_module(f'{imort_path}.{file_name}') for file_name in filenames]

    return modules

# automatically scan and import model modules for registry
# scan all the files under the 'models' folder and collect files ending with
# '_model.py'
model_folder = osp.dirname(osp.abspath(__file__))
model_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(model_folder) if v.endswith('_model.py')]
# import all the model modules
_model_modules = [importlib.import_module(f'basicsr.models.{file_name}') for file_name in model_filenames]

# 二级目录下的model
for item in os.scandir(model_folder):
    if item.is_dir() and item.name != '__pycache__':
        dir_model_name = item.name
        models = get_modules(imort_path=f'basicsr.models.{dir_model_name}', arch_folder=f'{model_folder}/{dir_model_name}')
        _model_modules.append(models)


