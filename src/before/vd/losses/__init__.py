import importlib
from copy import deepcopy
from os import path as osp
from vd.utils import get_root_logger, scandir
from vd.utils.registry import LOSS_REGISTRY

loss_folder = osp.dirname(osp.abspath(__file__))
loss_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(loss_folder) if v.endswith('_loss.py')]
_loss_modules = [importlib.import_module(f'vd.losses.{file_name}') for file_name in loss_filenames]

def build_loss(opt):
    opt = deepcopy(opt)
    loss_type = opt.pop('type')
    loss = LOSS_REGISTRY.get(loss_type)(**opt)
    logger = get_root_logger()
    logger.info(f'Loss [{loss.__class__.__name__}] is created.')
    return loss