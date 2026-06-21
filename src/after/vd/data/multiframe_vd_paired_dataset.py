import torch
from torch.utils import data as data

from vd.data.data_util import *
from vd.data import augment, paired_random_crop
from vd.utils import FileClient, imfrombytes, img2tensor
from vd.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class MultiFrameVDPairedImageDataset(data.Dataset):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        if self.opt['phase'] == 'train':
            self.paths = multiframe_paired_paths_from_folders_train([self.lq_folder, self.gt_folder])
        else:
            self.paths = multiframe_paired_paths_from_folders_val([self.lq_folder, self.gt_folder])

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)
        key = self.paths[index]['key']
        img_gt_0 = read_img(self.paths[index]['gt_0_path'])
        img_gt_1 = read_img(self.paths[index]['gt_path'])
        img_gt_2 = read_img(self.paths[index]['gt_2_path'])
        img_lq_0 = read_img(self.paths[index]['lq_0_path'])
        img_lq_1 = read_img(self.paths[index]['lq_1_path'])
        img_lq_2 = read_img(self.paths[index]['lq_2_path'])

        if self.opt['phase'] == 'train':
            img_gt_0, img_gt_1, img_gt_2, img_lq_0, img_lq_1, img_lq_2 = augment(
                [img_gt_0, img_gt_1, img_gt_2, img_lq_0, img_lq_1, img_lq_2],
                self.opt['use_hflip'], self.opt['use_rot']
            )

        img_gt_0, img_gt_1, img_gt_2, img_lq_0, img_lq_1, img_lq_2 = img2tensor(
            [img_gt_0, img_gt_1, img_gt_2, img_lq_0, img_lq_1, img_lq_2], bgr2rgb=True, float32=True
        )
        img_lqs = torch.stack((img_lq_0, img_lq_1, img_lq_2), dim=0)   # (T, C, H, W)
        img_gts = torch.stack((img_gt_0, img_gt_1, img_gt_2), dim=0)   # (T, C, H, W)
        return {'lq': img_lqs, 'gt': img_gt_1, 'gt_frames': img_gts, 'key': key}

    def __len__(self):
        return len(self.paths)