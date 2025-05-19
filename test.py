import logging
from os import path as osp
import os
import torch
# 这行代码必须放在 os、torch 之间
os.environ['CUDA_VISIBLE_DEVICES'] = '7'


from basicsr.data import build_dataloader, build_dataset
from basicsr.models import build_model
from basicsr.utils import get_env_info, get_root_logger, get_time_str, make_exp_dirs
from basicsr.utils.options import dict2str, parse_options


def test_pipeline(root_path, opt_path, is_sh):
    # parse options, set distributed setting, set ramdom seed
    opt, _ = parse_options(root_path, opt_path, is_train=False, is_sh=is_sh)

    torch.backends.cudnn.benchmark = True

    # mkdir and initialize loggers
    make_exp_dirs(opt)

    log_file_name =  f"{opt['name']}_{get_time_str()}.log"
    log_path = opt['path']['log']
    log_file = f'{log_path}/{log_file_name}'

    # log_file = osp.join(opt['path']['log'], f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # create test dataset and dataloader
    test_loaders = []
    for _, dataset_opt in sorted(opt['datasets'].items()):
        test_set = build_dataset(dataset_opt)
        test_loader = build_dataloader(
            test_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
        logger.info(f"Number of test images in {dataset_opt['name']}: {len(test_set)}")
        test_loaders.append(test_loader)

    # create model
    model = build_model(opt)

    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt['name']
        logger.info(f'Testing {test_set_name}...')
        model.validation(test_loader, current_iter=opt['name'], tb_logger=None, save_img=opt['val']['save_img'])


if __name__ == '__main__':

    torch.set_num_threads(2)  
    is_sh = False

    current_file = os.path.abspath(__file__)
    root_path = os.path.dirname(current_file)
    opt_path = r'options/exploring/test_ops/Unpaired_codeenhance.yaml'
    test_pipeline(root_path, opt_path, is_sh)







