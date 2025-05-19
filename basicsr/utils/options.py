import argparse
import random
import torch
import yaml
from collections import OrderedDict
from os import path as osp
import os

from basicsr.utils import set_random_seed
from basicsr.utils.dist_util import get_dist_info, init_dist, master_only
import time
from ruamel import yaml as ryml
from ruamel.yaml import YAML

def ordered_yaml():
    """Support OrderedDict for yaml.

    Returns:
        yaml Loader and Dumper.
    """
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data): 
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


def dict2str(opt, indent_level=1):
    """dict to string for printing options.

    Args:
        opt (dict): Option dict.
        indent_level (int): Indent level. Default: 1.

    Return:
        (str): Option string for printing.
    """
    msg = '\n'
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_level * 2) + k + ':['
            msg += dict2str(v, indent_level + 1)
            msg += ' ' * (indent_level * 2) + ']\n'
        else:
            msg += ' ' * (indent_level * 2) + k + ': ' + str(v) + '\n'
    return msg


def _postprocess_yml_value(value):
    # None
    if value == '~' or value.lower() == 'none':
        return None
    # bool
    if value.lower() == 'true':
        return True
    elif value.lower() == 'false':
        return False
    # !!float number
    if value.startswith('!!float'):
        return float(value.replace('!!float', ''))
    # number
    if value.isdigit():
        return int(value)
    elif value.replace('.', '', 1).isdigit() and value.count('.') < 2:
        return float(value)
    # list
    if value.startswith('['):
        return eval(value)
    # str
    return value

def get_time_str():
    return time.strftime('%Y%m%d_%H%M%S', time.localtime())


def deep_update_dict(res_dict, in_dict):
    for key in in_dict.keys():
        # if key in res_dict and isinstance(in_dict[key], dict) and isinstance(res_dict[key], dict) and \
        #         'name' in in_dict[key].keys() and 'kwargs' in in_dict[key].keys() and \
        #         'name' in res_dict[key].keys() and 'kwargs' in res_dict[key].keys() and \
        #         in_dict[key]['name'] == res_dict[key]['name']:
        #     deep_update_dict(res_dict[key]['kwargs'], in_dict[key]['kwargs'])
        if key in res_dict and isinstance(in_dict[key], dict) and isinstance(res_dict[key], dict):
            deep_update_dict(res_dict[key]['kwargs'], in_dict[key]['kwargs'])
        else:
            res_dict[key] = in_dict[key]

def parse_yaml(yaml_file_path):
    res = {}
    abs_yaml_file_path = os.path.join(os.path.dirname(os.getcwd()), yaml_file_path)
    with open(abs_yaml_file_path, 'r') as yaml_file:
        f = yaml.load(yaml_file, Loader=yaml.FullLoader)
        deep_update_dict(res, f)
    return res

def parse_options(root_path, opt_path, is_train=True, is_sh=False, is_LPF=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=opt_path, help='Path to option YAML file.')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none', help='job launcher')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--weight_decay_g', type=float, default=0.0)            # default = 0.0
    parser.add_argument('--pretrain_network_g', type=str, default=None)         # default = None
    parser.add_argument('--weight_texture', type=float, default=1)
    parser.add_argument('--weight_style', type=float, default=1)
    parser.add_argument('--weight_light', type=float, default=1)

    # parser.add_argument('--debug', type=bool, default=False) 
    # parser.add_argument('--debug', action='store_true')
    parser.add_argument('--local_rank', type=int, default=0)
    # parser.add_argument('--is_sh', type=bool, default=False)

    # if not is_train and is_sh:
    #     parser.add_argument('--weight_texture', type=float, default=1)
    #     parser.add_argument('--weight_style', type=float, default=1)

    parser.add_argument(
        '--force_yml', nargs='+', default=None, help='Force to update yml files. Examples: train:ema_decay=0.999')
    args = parser.parse_args()

    # parse yml to dict
    # with open(args.opt, mode='r') as f:
    #     opt = yaml.load(f, Loader=ordered_yaml()[0])
        # opt = yaml.load(f, Loader=yaml.FullLoader)
    # opt = parse_yaml(args.opt)
    with open(args.opt, mode='r') as f:
        opt = yaml.load(f, Loader=ordered_yaml()[0])
        # opt = ryml.round_trip_load(f)


    # if not is_train and is_sh:
    #     opt['network_g']['weight_texture'] = float(args.weight_texture)
    #     opt['network_g']['weight_style'] = float(args.weight_style)

    if args.weight_decay_g != 0.0:
        opt['train']['optim_g']['weight_decay'] = args.weight_decay_g

    if args.pretrain_network_g is not None:
        opt['path']['pretrain_network_g'] = args.pretrain_network_g

    if args.weight_texture !=1:
        opt['network_g']['weight_texture'] = args.weight_texture

    if args.weight_style !=1:
        opt['network_g']['weight_style'] = args.weight_style

    if args.weight_light !=1:
        opt['network_g']['weight_light'] = args.weight_light

    # ----------------------------------------------------
    # 分布时训练设置
    # distributed settings
    # ----------------------------------------------------
    if args.launcher == 'none':
        opt['dist'] = False
        print('Disable distributed.', flush=True)
    else:
        opt['dist'] = True
        if args.launcher == 'slurm' and 'dist_params' in opt:
            init_dist(args.launcher, **opt['dist_params'])
        else:
            init_dist(args.launcher)
    opt['rank'], opt['world_size'] = get_dist_info()

    # ----------------------------------------------------
    # 随机种子设置
    # random seed
    # ----------------------------------------------------
    seed = opt.get('manual_seed')
    if seed is None:
        seed = random.randint(1, 10000)
        opt['manual_seed'] = seed
    set_random_seed(seed + opt['rank'])

    # ----------------------------------------------------
    # force to update yml options
    # ----------------------------------------------------
    if args.force_yml is not None:
        for entry in args.force_yml:
            # now do not support creating new keys
            keys, value = entry.split('=')
            keys, value = keys.strip(), value.strip()
            value = _postprocess_yml_value(value)
            eval_str = 'opt'
            for key in keys.split(':'):
                eval_str += f'["{key}"]'
            eval_str += '=value'
            # using exec function
            exec(eval_str)

    opt['auto_resume'] = args.auto_resume
    opt['is_train'] = is_train

    # ----------------------------------------------------
    # 模型名称设置
    # ----------------------------------------------------
    # if args.debug and not opt['name'].startswith('debug'):
    if opt['debug'] and not opt['name'].startswith('debug'):
        opt['name'] = 'debug_' + opt['name']

    if opt['path'].get('resume_state', None):
        resume_state_path = opt['path'].get('resume_state')
        opt['name'] = resume_state_path.split("/")[-3]
    elif is_train:
        # opt['name'] = f"{get_time_str()}_{opt['name']}"
        # train_name = opt['datasets']['train']['type']
        train_name = opt['datasets']['train']['name']
        opt['name'] = f"{opt['name']}_{train_name}_{get_time_str()}"

    if is_LPF:
        tp_name = opt['name']
        opt['name'] = f'LPF_{tp_name}'


    if opt['num_gpu'] == 'auto':
        opt['num_gpu'] = torch.cuda.device_count()

    # ----------------------------------------------------
    # 数据集路径设置，datasets
    # ----------------------------------------------------
    for phase, dataset in opt['datasets'].items():
        # for multiple datasets, e.g., val_1, val_2; test_1, test_2
        phase = phase.split('_')[0]
        dataset['phase'] = phase
        if 'scale' in opt:
            dataset['scale'] = opt['scale']
        if dataset.get('dataroot_gt') is not None:
            dataset['dataroot_gt'] = osp.expanduser(dataset['dataroot_gt'])
        if dataset.get('dataroot_lq') is not None:
            dataset['dataroot_lq'] = osp.expanduser(dataset['dataroot_lq'])

    # ----------------------------------------------------
    # 日志路径设置
    # ----------------------------------------------------
    for key, val in opt['path'].items():
        if (val is not None) and ('resume_state' in key or 'pretrain_network' in key):
            opt['path'][key] = osp.expanduser(val)

    if is_train:
        experiments_root = osp.join(root_path, 'experiments', opt['name'])
        opt['path']['experiments_root'] = experiments_root
        opt['path']['models'] = osp.join(experiments_root, 'models')
        opt['path']['training_states'] = osp.join(experiments_root, 'training_states')
        opt['path']['log'] = experiments_root
        opt['path']['visualization'] = osp.join(experiments_root, 'visualization')

        # change some options for debug mode
        if 'debug' in opt['name']:
            if 'val' in opt:
                opt['val']['val_freq'] = 8
            opt['logger']['print_freq'] = 1
            opt['logger']['save_checkpoint_freq'] = 8
    else:  # test
        t = opt['path'].get('pretrain_network_g')
        # n = t.split('/')[-3]
        # results_root = osp.join(root_path, 'results', n)
        results_root = osp.join(root_path, 'results')
        # results_root = osp.join(root_path, 'results', opt['name'])
        opt['path']['results_root'] = results_root
        opt['path']['log'] = results_root
        # opt['path']['visualization'] = osp.join(results_root, 'visualization')
        opt['path']['visualization'] = results_root

    # for key in args.__dict__.keys():
    #     if key not in args_dict.keys():
    #         args_dict[key] = args.__dict__[key]


    return opt, args

def save_dict_to_yaml(dict_value, save_path):
    """dict保存为yaml"""
    # args_yaml_info = yaml.dump(dict_value, sort_keys=False, default_flow_style=None)
    # args_yaml_info = yaml.dump(dict_value, sort_keys=False, default_flow_style=None, allow_unicode=True)
    # args_yaml_info = ryml.dump(dict_value, Dumper)
    yaml = YAML()
    with open(save_path, 'w') as file:
        # yaml = yaml(type='unsafe', pure=True)
        yaml.dump(dict_value, file)
        # ryml.dump(dict_value, file, default_flow_style=False)
        # ryml.round_trip_dump(dict_value, file, default_flow_style=False)

        # file.write(args_yaml_info)
        file.close()

import yaml
@master_only
def copy_opt_file(opt_file, experiments_root, opt=None):
    # copy the yml file to the experiment root
    import sys
    import time
    from shutil import copyfile
    cmd = ' '.join(sys.argv)
    filename = osp.join(experiments_root, osp.basename(opt_file))
    if opt is not None:

        save_dict_to_yaml(opt, filename)
    else:
        copyfile(opt_file, filename)

    with open(filename, 'r+') as f:
        lines = f.readlines()
        lines.insert(0, f'# GENERATE TIME: {time.asctime()}\n# CMD:\n# {cmd}\n\n')
        f.seek(0)
        f.writelines(lines)
