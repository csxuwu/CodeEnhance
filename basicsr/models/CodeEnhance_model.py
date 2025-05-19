import glob
import os.path
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm
import sys
import os
import numpy as np

import torch
import torch.nn as nn
import torchvision.utils as tvu
import copy

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.utils import get_root_logger, imwrite, tensor2img, img2tensor
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.models.base_model import BaseModel
from basicsr.metrics import calculate_metric
from basicsr.data.random_load_images import random_load_images

@MODEL_REGISTRY.register()
class CodeEnhance_Model(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)

        # define network
        if opt.get('seg') is not None:
            opt['network_g']['seg_cfg'] = opt['seg']
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)

        # --------------------------------------------------------------
        # 加载预训练好的 decoder、codebook，并冻结参数
        # load pre-trained HQ ckpt, frozen decoder and codebook
        # --------------------------------------------------------------
        self.LQ_stage = self.opt['network_g'].get('LQ_stage', True)
        if self.LQ_stage:
            load_path = self.opt['path'].get('pretrain_network_hq', None)
            assert load_path is not None, 'Need to specify hq prior model path in LQ stage'

            # 用于生成 GT 量化下标
            hq_opt = self.opt['network_g'].copy()
            hq_opt['LQ_stage'] = False
            self.net_hq = build_network(hq_opt)
            self.net_hq = self.model_to_device(self.net_hq)
            self.load_network(self.net_hq, load_path, self.opt['path']['strict_load'])

            # VQ-GAN 第二阶段训练，加载StageI 的参数，并冻结部分模块的参数 ['quantize', 'decoder_group', 'after_quant_group', 'out_conv']
            self.load_network(self.net_g, load_path, False)
            frozen_module_keywords = self.opt['network_g'].get('frozen_module_keywords', None)
            if frozen_module_keywords is not None:
                for name, module in self.net_g.named_modules():
                    for fkw in frozen_module_keywords:
                        if fkw in name:
                            # for p in module.parameters():
                            #     p.requires_grad = False
                            # print('*' * 100)
                            for p in module.named_parameters():
                                # print(f'{p[0]}  --  {p[1].size()}')
                                if 'shift_codebook' in p[0]:
                                    p[1].requires_grad = True
                                elif 'reweight_codebook' in p[0]:
                                    p[1].requires_grad = True
                                else:
                                    p[1].requires_grad = False

                            # break

        # print('*' * 100)
        # sss = 0
        # for name, module in self.net_g.named_modules():
        #     for p in module.parameters():
        #         if p.requires_grad:
        #             print(f'name: {name}, para: {p.numel()}')
        #             sss += p.numel()

        # print('*' * 100)
        # print(f'{sss/1000**2} M')
        # # self.net_g.
        # print('*' * 100)

        # --------------------------------------------------------------
        # 加载 stage I 训练好的参数，作为 stage II 的初始参数
        # load pretrained models
        # --------------------------------------------------------------
        load_path = self.opt['path'].get('pretrain_network_g', None)
        logger = get_root_logger()
        if load_path is not None:
            logger.info(f'Loading net_g from {load_path}')
            self.load_network(self.net_g, load_path, self.opt['path']['strict_load'])

        if self.is_train:
            self.init_training_settings()
            self.use_dis = (self.opt['train']['gan_opt']['loss_weight'] != 0)
            self.net_d_best = copy.deepcopy(self.net_d)

        self.net_g_best = copy.deepcopy(self.net_g)
        self.total_index_counts = torch.zeros(1024, dtype=torch.int).cuda()


        # 查看参数
        # for p in self.net_g.named_parameters():

        #     print(f'{p[0]}  --  {p[1].size()}')
        #     # self.net_g.named_parameters()
        #     if p[0] == 'quantize_group.0.shift_codebook':
        #         print(p[1])

        # load_net = torch.load(load_path, map_location=lambda storage, loc: storage)
        # load_net = load_net['params']
        # self.wx_print(load_net)

        # for name, module in self.net_g.named_modules():
        #     print(name)

        # sys.exit()



    def init_training_settings(self):
        logger = get_root_logger()
        train_opt = self.opt['train']
        self.net_g.train()

        # define network net_d
        self.net_d = build_network(self.opt['network_d'])
        self.net_d = self.model_to_device(self.net_d)
        # load pretrained d models
        load_path = self.opt['path'].get('pretrain_network_d', None)
        # print(load_path)
        if load_path is not None:
            logger.info(f'Loading net_d from {load_path}')
            self.load_network(self.net_d, load_path, self.opt['path'].get('strict_load_d', True))

        self.net_d.train()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
            self.model_to_device(self.cri_perceptual)
        else:
            self.cri_perceptual = None

        if train_opt.get('latent_contrast_opt'):
            self.cri_latent_contrast = build_loss(train_opt['latent_contrast_opt']).to(self.device)
        else:
            self.cri_latent_contrast = None

        if train_opt.get('content_opt'):
            self.cri_content = nn.MSELoss()
        else:
            self.cri_content = None

        if train_opt.get('gan_opt'):
            self.cri_gan = build_loss(train_opt['gan_opt']).to(self.device)

        self.net_d_iters = train_opt.get('net_d_iters', 1)
        self.net_d_init_iters = train_opt.get('net_d_init_iters', 0)

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()


    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        # print('*' * 100)
        for k, v in self.net_g.named_parameters():

            # if 'quanti' in k:
            #     print(k)

            optim_params.append(v)
            if not v.requires_grad:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        # sys.exit()

        # optimizer g
        optim_type = train_opt['optim_g'].pop('type')
        optim_class = getattr(torch.optim, optim_type)
        self.optimizer_g = optim_class(optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

        # optimizer d
        optim_type = train_opt['optim_d'].pop('type')
        optim_class = getattr(torch.optim, optim_type)
        self.optimizer_d = optim_class(self.net_d.parameters(), **train_opt['optim_d'])
        self.optimizers.append(self.optimizer_d)


    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        self.b,_,_,_ = self.lq.shape
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        else:
            self.gt = None

        if 'refer' in data:
            self.refer = data['refer'].to(self.device)
        else:
            self.refer = None

        self.data = data


    def optimize_parameters(self, current_iter):

        train_opt = self.opt['train']

        # --------------------------------------------------------------
        # Stage II 不训练 判别器
        # --------------------------------------------------------------
        for p in self.net_d.parameters():
            p.requires_grad = False
        self.optimizer_g.zero_grad()

        # --------------------------------------------------------------
        # GT：获得GT 的 重构、量化特征、量化特征索引
        # lq：获得重构输出、损失
        # --------------------------------------------------------------
        with torch.no_grad():
            gt_indices, quant_gt, feat_to_quant, after_quant_feat = self.net_hq.encode_indices(input=self.gt)
        self.lq.requires_grad = True
        self.out_dict = self.net_g(input=self.lq,
                                   gt_img=self.gt,
                                   reference_img=self.refer,
                                   gt_indices=gt_indices,
                                   net_hq=self.net_hq)
        self.output = self.out_dict['out_img']
        l_codebook = self.out_dict['codebook_loss']
        l_semantic = self.out_dict['semantic_loss']
        quant_g = self.out_dict['feat_to_quant']        # 对于LQ， feat_to_quant，相当于 quant
        quant_g_z = self.out_dict['z_quant']
        after_quant_feat_lq = self.out_dict['after_quant_feat']

        # self.output, l_codebook, l_semantic, quant_g, quant_g_z, _ = self.net_g(input=self.lq,
        #                                                                 gt_img=self.gt,
        #                                                                 reference_img=self.refer,
        #                                                                 gt_indices=gt_indices,
        #                                                                 net_hq=self.net_hq)

        l_g_total = 0
        loss_dict = OrderedDict()

        # ===================================================
        # codebook loss
        if train_opt.get('codebook_opt', None):
            l_codebook *= train_opt['codebook_opt']['loss_weight']
            l_g_total += l_codebook.mean()
            loss_dict['l_codebook'] = l_codebook.mean()

        # semantic cluster loss, only for LQ stage!
        if train_opt.get('semantic_opt', None) and isinstance(l_semantic, torch.Tensor):
            l_semantic *= train_opt['semantic_opt']['loss_weight']
            l_semantic = l_semantic.mean()
            l_g_total += l_semantic
            loss_dict['l_semantic'] = l_semantic

        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_g_total += l_pix
            loss_dict['l_pix'] = l_pix

        if self.cri_latent_contrast:
            with torch.no_grad():
                _, quant_ouput, _, _ = self.net_hq.encode_indices(input=self.output)
            l_la_cont = self.cri_latent_contrast(quant_ouput, quant_gt, quant_g_z)
            l_g_total += l_la_cont
            loss_dict['l_la_cont'] = l_la_cont

        if self.cri_content:
            f_output = self.net_g.multiscale_encoder(self.output)      # 是否固定参数，再看实验结果

            # print(f_output.shape)
            # print(quant_g_z.shape)

            l_content = self.cri_content(f_output, quant_g_z)
            l_content *= train_opt['content_opt']['loss_weight']
            l_g_total += l_content
            loss_dict['l_content'] = l_content

        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_g_total += l_percep.mean()
                loss_dict['l_percep'] = l_percep.mean()
            if l_style is not None:
                l_g_total += l_style
                loss_dict['l_style'] = l_style

        # gan loss
        if self.use_dis and current_iter > train_opt['net_d_init_iters']:
            fake_g_pred = self.net_d(quant_g)
            l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
            l_g_total += l_g_gan
            loss_dict['l_g_gan'] = l_g_gan

        # print(l_g_total.requires_grad)
        # if l_g_total.requires_grad:
        l_g_total.mean().backward()
        self.optimizer_g.step()

        # optimize net_d
        self.fixed_disc = self.opt['train'].get('fixed_disc', False)
        if not self.fixed_disc and self.use_dis and current_iter > train_opt['net_d_init_iters']:
            for p in self.net_d.parameters():
                p.requires_grad = True
            self.optimizer_d.zero_grad()
            # real
            real_d_pred = self.net_d(quant_gt)
            l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
            loss_dict['l_d_real'] = l_d_real
            loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
            l_d_real.backward()
            # fake
            fake_d_pred = self.net_d(quant_g.detach())
            l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
            loss_dict['l_d_fake'] = l_d_fake
            loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())
            l_d_fake.backward()
            self.optimizer_d.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)


        if current_iter % 10000 == 0:
            self.save_img_cur(current_iter, self.output)


    def test(self):
        self.net_g.eval()
        net_g = self.get_bare_model(self.net_g)
        min_size = 8000 * 8000  # use smaller min_size with limited GPU memory
        lq_input = self.lq
        _, _, h, w = lq_input.shape
        if h * w < min_size:
            # self.output,_ = net_g.test(self.gt)
            self.output = net_g.test(lq_input, reference=self.refer, net_hq=self.net_hq)
        else:
            self.output = net_g.test_tile(lq_input, reference=self.refer)
        if hasattr(net_g, 'total_index_counts'):
            self.total_index_counts += net_g.total_index_counts

        self.net_g.train()


    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, save_as_dir=None, visual_codebook=False):
        logger = get_root_logger()
        logger.info('Only support single GPU validation.')
        self.nondist_validation(dataloader, current_iter, tb_logger, save_img, save_as_dir)

    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img, save_as_dir, visual_codebook=False):


        dataset_name = dataloader.dataset.opt['name']
        # is_reference = dataloader.dataset.opt['is_reference']

        # 设置 reference image
        is_reference = dataloader.dataset.opt.get('is_reference', False)
        reference_image_root = dataloader.dataset.opt['reference_path']
        reference_image_list = glob.glob(os.path.join(reference_image_root, '*'))
        # print(reference_image_list)
        # sys.exit()

        # 
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
        pbar = tqdm(total=len(dataloader), unit='image')

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)

            # zero self.metric_results
            self.metric_results = {metric: 0 for metric in self.metric_results}
            self.key_metric = self.opt['val'].get('key_metric')

        index = 0
        for idx, val_data in enumerate(dataloader):
            # print(val_data)
            index = idx
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            print(img_name)
            self.feed_data(val_data)
            if is_reference:
                _,_,h, w = self.lq.shape
                self.refer = random_load_images(path_list=reference_image_list, batchsize=self.b, size_h=h, size_w=w)

            self.test()

            sr_img = tensor2img(self.output)
            if not self.gt is None:
                gt_img = tensor2img(self.gt)

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                    # save_img_path_gt = osp.join(self.opt['path']['visualization'], img_name,
                    #                             f'{img_name}_gt.png')
                    # save_img_path_in = osp.join(self.opt['path']['visualization'], img_name,
                    #                             f'{img_name}_lq.png')
                    imwrite(sr_img, save_img_path)
                    # gt_img = tensor2img(self.gt)
                    # in_img = tensor2img(self.lq)
                    # imwrite(gt_img, save_img_path_gt)
                    # imwrite(in_img, save_img_path_in)

                    # tentative for out of GPU memory
                    del self.gt
                    del self.lq
                    del self.output
                    torch.cuda.empty_cache()

                else:
                    weight_texture = self.opt['network_g']['weight_texture']
                    weight_style = self.opt['network_g']['weight_style']
                    weight_light = self.opt['network_g']['weight_light']
                    sub_file_name = f'w1={weight_texture}_w2={weight_style}_w3={weight_light}'
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'],
                                                 dataset_name,
                                                 sub_file_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'],
                                                 dataset_name,
                                                 sub_file_name,
                                                 f'{img_name}.png')
                    save_img_path_in = osp.join(self.opt['path']['visualization'],
                                                 dataset_name,
                                                 sub_file_name,
                                                 f'{img_name}_lq.png')
                    save_img_path_gt = osp.join(self.opt['path']['visualization'],
                                                 dataset_name,
                                                 sub_file_name,
                                                 f'{img_name}_gt.png')
                        # f'{img_name}_{self.opt["name"]}.png')
                    # print(save_img_path)
                    # print(save_img_path)
                    # print(save_img_path)
                    # sys.exit()
                    if not self.gt is None:
                        gt_img = tensor2img(self.gt)
                    in_img = tensor2img(self.lq)
                    # imwrite(in_img, save_img_path_in)
                    # imwrite(gt_img, save_img_path_gt)
                    imwrite(sr_img, save_img_path)

                    del self.lq
                    del self.output

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    metric_data = dict(img1=sr_img, img2=gt_img)
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            pbar.update(1)
            pbar.set_description(f'Test {img_name}')

        # 统计 code 的激活频率
        name_pt = 'LOLv2_real_20230927_005249_LLIE_Prior_OS_Refer_Sem_Skip4231_Arch'
        torch.save(self.total_index_counts, f'/home/wuxu/codes/promptenhance/codebook_active_map/{name_pt}.pt')
        pbar.close()

        if with_metrics and self.opt['is_train']:

            # calculate average metric
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (index + 1)
                # self.metric_results[metric] /= (idx + 1)

            if self.key_metric is not None:
                # If the best metric is updated, update and save best model
                to_update = self._update_best_metric_result(dataset_name, self.key_metric,
                                                            self.metric_results[self.key_metric], current_iter)

                if to_update:
                    for name, opt_ in self.opt['val']['metrics'].items():
                        self._update_metric_result(dataset_name, name, self.metric_results[name], current_iter)
                    self.copy_model(self.net_g, self.net_g_best)
                    self.copy_model(self.net_d, self.net_d_best)
                    self.save_network(self.net_g, 'net_g_best', '')
                    self.save_network(self.net_d, 'net_d_best', '')
            else:
                # update each metric separately
                updated = []
                for name, opt_ in self.opt['val']['metrics'].items():
                    tmp_updated = self._update_best_metric_result(dataset_name, name, self.metric_results[name],
                                                                  current_iter)
                    updated.append(tmp_updated)
                # save best model if any metric is updated
                if sum(updated):
                    self.copy_model(self.net_g, self.net_g_best)
                    self.copy_model(self.net_d, self.net_d_best)
                    self.save_network(self.net_g, 'net_g_best', '')
                    self.save_network(self.net_d, 'net_d_best', '')

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)


    def vis_single_code(self, up_factor=2):
        net_g = self.get_bare_model(self.net_g)
        codenum = self.opt['network_g']['codebook_params'][0][1]
        with torch.no_grad():
            code_idx = torch.arange(codenum).reshape(codenum, 1, 1, 1)
            code_idx = code_idx.repeat(1, 1, up_factor, up_factor)
            output_img = net_g.decode_indices(code_idx)
            output_img = tvu.make_grid(output_img, nrow=32)

        return output_img.unsqueeze(0)


    def save_img_cur(self, current_iter, img):

        img_name = osp.splitext(osp.basename(self.data['lq_path'][0]))[0] + '_' + str(current_iter)
        img = tensor2img(img)

        save_path = os.path.join(self.opt['path']['training_states'], 'img_train_visual')

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        save_img_path_lq = os.path.join(save_path, f'{img_name}_lq.png')
        save_img_path_gt = os.path.join(save_path, f'{img_name}_gt.png')
        save_img_path_en = os.path.join(save_path, f'{img_name}_en.png')

        
        img_lq = tensor2img(self.data['lq'])
        img_gt = tensor2img(self.data['gt'])

        imwrite(img_lq, save_img_path_lq)
        imwrite(img_gt, save_img_path_gt)

        imwrite(img, save_img_path_en)


    def get_current_visuals(self):
        vis_samples = 16
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()[:vis_samples]
        # if self.output != None:
        #     out_dict['result_codebook'] = self.output.detach().cpu()[:vis_samples]
        if self.output != None:
            out_dict['output'] = self.output.detach().cpu()[:vis_samples]
        if not self.LQ_stage:
            out_dict['codebook'] = self.vis_single_code()
        if hasattr(self, 'gt_rec'):
            out_dict['gt_rec'] = self.gt_rec.detach().cpu()[:vis_samples]
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()[:vis_samples]
        return out_dict


    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter)
        self.save_network(self.net_d, 'net_d', current_iter)
        self.save_training_state(epoch, current_iter)



