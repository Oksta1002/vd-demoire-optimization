import functools
import torch
import os.path as osp
import time
from tqdm import tqdm
from collections import OrderedDict
from vd.utils.registry import MODEL_REGISTRY
from vd.archs import build_network
from vd.models import BaseModel
from vd.metrics import calculate_metric
from vd.utils import get_root_logger
from vd.losses import build_loss
from vd.data.data_util import tensor2numpy, imwrite_gt, tensor2numpy_moire, imwrite_moire
from deepspeed.profiling.flops_profiler import get_model_profile
from deepspeed.accelerator import get_accelerator
import wandb

def _match_stats(x, ref, eps=1e-8):
    """x의 채널별 mean/std를 ref에 맞춰 선형 변환 (AdaIN). (B*T, C, H, W)"""
    mean_x = x.mean(dim=(-2, -1), keepdim=True)
    std_x  = x.std(dim=(-2, -1), keepdim=True)
    mean_r = ref.mean(dim=(-2, -1), keepdim=True)
    std_r  = ref.std(dim=(-2, -1), keepdim=True)
    return (x - mean_x) / (std_x + eps) * std_r + mean_r


def _restore_train_mode(method):
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        finally:
            if not hasattr(self, 'net_ema'):
                self.net.train()
    return wrapper


@MODEL_REGISTRY.register()
class MultiFrameVDModel(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)
        self.use_wandb = opt.get('logger', {}).get('wandb', {}).get('use_wandb', True)
        self.net = build_network(opt['network'])
        self.net = self.model_to_device(self.net)
        if opt.get('profile_flops', False):
            self._profile_flops()

        load_path = self.opt['path'].get('pretrain_network', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key', 'params')
            self.load_network(self.net, load_path, self.opt['path'].get('strict_load', True), param_key)
        if self.is_train:
            self.init_training_settings()


    def _profile_flops(self):
        with get_accelerator().device(0):
            flops, macs, params = get_model_profile(
                model=self.net, input_shape=(1, 3, 3, 720, 1280),
                args=None, kwargs=None, print_profile=True, detailed=True,
                module_depth=-1, top_modules=1, warm_up=10, as_string=True,
                output_file=None, ignore_modules=None)
        get_root_logger().info(f'flops: {flops}, macs: {macs}, params: {params}')


    def init_training_settings(self):
        self.net.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            self.net_ema = build_network(self.opt['network']).to(self.device)
            load_path = self.opt['path'].get('pretrain_network', None)
            if load_path is not None:
                self.load_network(self.net_ema, load_path, self.opt['path'].get('strict_load', True), 'params_ema')
            else:
                self.model_ema(0)
            self.net_ema.eval()

        self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device) if train_opt.get('pixel_opt') else None
        self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device) if train_opt.get('perceptual_opt') else None
        self.cri_mid_pix = build_loss(train_opt['mid_pix_opt']).to(self.device) if train_opt.get('mid_pix_opt') else None
        self.cri_mid_perceptual = build_loss(train_opt['mid_perceptual_opt']).to(self.device) if train_opt.get('mid_perceptual_opt') else None
        self.cri_self_recon = build_loss(train_opt['self_recon_opt']).to(self.device) if train_opt.get('self_recon_opt') else None
        self.cri_decomp_contrastive = build_loss(train_opt['decomp_contrastive_opt']).to(self.device) if train_opt.get('decomp_contrastive_opt') else None

        # [변경 2-b] style_weight=0 (모든 YML 확인) → l_style 항상 None (dead code 제거).
        # [LEGACY] l_percep, l_style = self.cri_perceptual(...)
        if self.cri_perceptual:
            _raw = self.cri_perceptual
            self.cri_perceptual = lambda *a: _raw(*a)[0]
        if self.cri_mid_perceptual:
            _raw_mid = self.cri_mid_perceptual
            self.cri_mid_perceptual = lambda *a: _raw_mid(*a)[0]

        self.setup_optimizers()
        self.setup_schedulers()


    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        out_dict['moire'] = self.moire[:, 1].detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict


    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        if 'gt_frames' in data:
            self.gt_frames = data['gt_frames'].to(self.device)  # (B, T, C, H, W)


    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')
        optim_type = train_opt['optim'].pop('type')
        self.optimizer = self.get_optimizer(optim_type, optim_params, **train_opt['optim'])
        self.optimizers.append(self.optimizer)


    def _loss_step(self, cri, name, loss_dict, *args):
        if cri:
            l = cri(*args)
            loss_dict[name] = l
            return l
        return 0


    def optimize_parameters(self, current_iter):
        self.optimizer.zero_grad()
        self.output, self.clean, self.moire = self.net(self.lq)
        B, T, C, H, W = self.clean.shape
        self.lq_recon = self.moire + self.clean  # (B, T, C, H, W)

        lq_flat = self.lq.view(B * T, C, H, W)
        gt_flat = self.gt_frames.view(B * T, C, H, W)
        adain_lq_flat = (
            _match_stats(lq_flat, gt_flat)
            if (self.cri_self_recon or self.cri_decomp_contrastive) else None
        )

        l_total, loss_dict = 0, OrderedDict()
        l_total += self._loss_step(self.cri_pix,            'l_pix',           loss_dict, self.output, self.gt)
        l_total += self._loss_step(self.cri_perceptual,     'l_percep',        loss_dict, self.output, self.gt)
        l_total += self._loss_step(self.cri_self_recon,     'l_self_recon',    loss_dict, self.lq_recon.view(B * T, C, H, W), adain_lq_flat)
        l_total += self._loss_step(self.cri_mid_pix,        'l_mid_pixel',     loss_dict, self.clean.view(B * T, C, H, W), gt_flat)
        l_total += self._loss_step(self.cri_mid_perceptual, 'l_mid_percep',    loss_dict, self.clean.view(B * T, C, H, W), gt_flat)

        if self.cri_decomp_contrastive:
            l_total += self._loss_step(self.cri_decomp_contrastive, 'l_decomp_contrastive',
                                       loss_dict, self.moire, adain_lq_flat.view(B, T, C, H, W), self.gt_frames)

        l_total.backward()
        self.optimizer.step()
        self.log_dict = self.reduce_loss_dict(loss_dict)
        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)


    @_restore_train_mode
    @torch.inference_mode()
    def test(self):
        scale = self.opt.get('scale', 1)
        _, _, _, h_old, w_old = self.lq.size()
        net = self.net_ema if hasattr(self, 'net_ema') else self.net
        net.eval()
        self.output, self.clean, self.moire = net(self.lq)
        self.clean = self.clean[:, :, :, :h_old * scale, :w_old * scale]
        self.output = self.output[:, :, :h_old * scale, :w_old * scale]


    def nondist_validation(self, dataloader, current_iter, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            self._initialize_best_metric_results(dataset_name)
            mid_metric_results = {f'mid_{metric}': 0 for metric in self.opt['val']['metrics'].keys()}
        metric_data = dict()
        mid_metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')
        time_inf_total = 0.
        log_images = []
        is_train = self.opt['is_train']

        for idx, val_data in enumerate(dataloader):
            img_name = val_data['key'][0]
            self.feed_data(val_data)

            torch.cuda.synchronize()
            st = time.perf_counter()
            self.test()
            torch.cuda.synchronize()
            time_inf_total += time.perf_counter() - st

            if is_train:
                metric_data['img'] = self.output.detach()
                mid_metric_data['img'] = self.clean[:, 1].detach()
                if hasattr(self, 'gt'):
                    metric_data['img2'] = self.gt.detach()
                    mid_metric_data['img2'] = self.gt.detach()

                if self.use_wandb and idx % 300 == 0:
                    log_images.append(wandb.Image(
                        self.lq[:,1,:,:,:].detach()[0].float().clamp(0, 1),
                        caption=f"LQ: {img_name}"
                    ))
                    moire_t = self.moire.detach()[0, 1].float()
                    moire_norm = ((moire_t - moire_t.min()) / (moire_t.max() - moire_t.min() + 1e-8)).clamp(0, 1)
                    log_images.append(wandb.Image(moire_norm, caption=f"Moire: {img_name}"))
                    log_images.append(wandb.Image(
                        self.clean.detach()[0, 1].float().clamp(0, 1),
                        caption=f"Clean: {img_name}"
                    ))
                    log_images.append(wandb.Image(
                        self.output.detach()[0].float().clamp(0, 1),
                        caption=f"SR2: {img_name}"
                    ))
                    if hasattr(self, 'gt'):
                        log_images.append(wandb.Image(
                            self.gt.detach()[0].float().clamp(0, 1),
                            caption=f"GT: {img_name}"
                        ))
            else:
                visuals = self.get_current_visuals()
                sr_img = tensor2numpy(visuals['result'][0])
                moire_img = tensor2numpy_moire(visuals['moire'][0])
                metric_data['img'] = sr_img
                mid_metric_data['img'] = tensor2numpy(self.clean.detach()[0, 1])
                if 'gt' in visuals:
                    gt_img = tensor2numpy(visuals['gt'])
                    metric_data['img2'] = mid_metric_data['img2'] = gt_img
                if save_img:
                    save_img_path = osp.join(self.opt['path']['visualization'], dataset_name, 'clean', f'{img_name}.png')
                    imwrite_gt(sr_img, save_img_path)
                    save_moire_path = osp.join(self.opt['path']['visualization'], dataset_name, 'moire', f'{img_name}.png')
                    imwrite_moire(moire_img, save_moire_path)

            del self.lq
            if hasattr(self, 'gt'):
                del self.gt
            if hasattr(self, 'gt_frames'):
                del self.gt_frames
            del self.output
            del self.clean
            del self.moire

            if with_metrics:
                for name, opt_ in self.opt['val']['metrics'].items():
                    if is_train:
                        self.metric_results[name] += calculate_metric(metric_data, opt_).detach().cpu().numpy().sum()
                        mid_metric_results[f'mid_{name}'] += calculate_metric(mid_metric_data, opt_).detach().cpu().numpy().sum()
                    else:
                        self.metric_results[name] += calculate_metric(metric_data, opt_)
                        mid_metric_results[f'mid_{name}'] += calculate_metric(mid_metric_data, opt_)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()
        if self.use_wandb and log_images:
            wandb.log({f"val/{dataset_name}/examples": log_images, "iter": current_iter})
        time_avg = time_inf_total / (idx + 1)
        logger = get_root_logger()
        logger.info('average test time: %.3f, total time: %.3f' % (time_avg, time_inf_total))
        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)
            self._log_validation_metric_values(current_iter, dataset_name)
            for metric in mid_metric_results.keys():
                mid_metric_results[metric] /= (idx + 1)
            log_str = f'Val [{current_iter}] {dataset_name} [mid]\n'
            for metric, value in mid_metric_results.items():
                log_str += f'\t # {metric}: {value:.4f}'
            logger.info(log_str)
            if self.use_wandb and is_train and wandb.run is not None:
                wandb.log({f'val/{dataset_name}/{m}': v for m, v in mid_metric_results.items()} | {'iter': current_iter})


    def _log_validation_metric_values(self, current_iter, dataset_name):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'
        logger = get_root_logger()
        logger.info(log_str)

        if self.use_wandb and self.opt['is_train']:
            wandb_log_dict = {'val/iter': current_iter}
            for metric, value in self.metric_results.items():
                wandb_log_dict[f'val/{dataset_name}/{metric}'] = value
            wandb.log(wandb_log_dict)


    def save(self, epoch, current_iter):
        if hasattr(self, 'net_ema'):
            self.save_network([self.net, self.net_ema], 'net', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net, 'net', current_iter)
        self.save_training_state(epoch, current_iter)
