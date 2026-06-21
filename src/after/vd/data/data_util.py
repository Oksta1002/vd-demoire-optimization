import os
import os.path as osp
import cv2
import numpy as np
from vd.utils import scandir

# 영상 내 첫/마지막 프레임은 경계 처리: 자기 자신을 반복 사용
_LAST_FRAME_IDX = 59

def multiframe_paired_paths_from_folders(folders):
    assert len(folders) == 2, (
        'The len of folders should be 2 with [input_folder, gt_folder]. ' f'But got {len(folders)}')
    input_folder, gt_folder = folders
    gt_names = [osp.basename(p).split('.jpg')[0] for p in scandir(gt_folder)]
    paths = []
    for gt_name in gt_names:
        scene_idx = gt_name.split('_')[0]
        lq_1_idx = int(gt_name.split('_')[1])
        if lq_1_idx == 0:
            lq_0_name = gt_name
            lq_2_name = scene_idx + '_' + str(lq_1_idx + 1).zfill(5)
        elif lq_1_idx == _LAST_FRAME_IDX:
            lq_0_name = scene_idx + '_' + str(lq_1_idx - 1).zfill(5)
            lq_2_name = gt_name
        else:
            lq_0_name = scene_idx + '_' + str(lq_1_idx - 1).zfill(5)
            lq_2_name = scene_idx + '_' + str(lq_1_idx + 1).zfill(5)
        paths.append({
            'lq_0_path': osp.join(input_folder, lq_0_name + '.jpg'),
            'lq_1_path': osp.join(input_folder, gt_name + '.jpg'),
            'lq_2_path': osp.join(input_folder, lq_2_name + '.jpg'),
            'gt_path':   osp.join(gt_folder, gt_name + '.jpg'),
            'gt_0_path': osp.join(gt_folder, lq_0_name + '.jpg'),
            'gt_2_path': osp.join(gt_folder, lq_2_name + '.jpg'),
            'key':       gt_name,
        })
    return paths

def multiframe_paired_paths_from_folders_train(folders):
    return multiframe_paired_paths_from_folders(folders)

def multiframe_paired_paths_from_folders_val(folders):
    return multiframe_paired_paths_from_folders(folders)


def tensor2numpy(tensor):
    img_np = tensor.squeeze().cpu().numpy()
    img_np[img_np < 0] = 0
    img_np = np.transpose(img_np, (1, 2, 0))
    return img_np.astype(np.float32)


def tensor2numpy_moire(tensor):
    img_np = tensor.squeeze().cpu().numpy()
    img_np = np.transpose(img_np, (1, 2, 0))
    return img_np.astype(np.float32)


def imwrite_gt(img, img_path, auto_mkdir=True):
    if auto_mkdir:
        dir_name = os.path.abspath(os.path.dirname(img_path))
        os.makedirs(dir_name, exist_ok=True)
    img = img.clip(0, 1.0)
    uint8_image = np.round(img * 255.0).astype(np.uint8)
    uint8_image = cv2.cvtColor(uint8_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(img_path, uint8_image)
    return None


def imwrite_moire(img, img_path, auto_mkdir=True):
    if auto_mkdir:
        dir_name = os.path.abspath(os.path.dirname(img_path))
        os.makedirs(dir_name, exist_ok=True)
    img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)  # [0, 1]
    img_vis = (img_norm * 255).astype(np.uint8)
    img_vis = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)
    cv2.imwrite(img_path, img_vis)
    return None


def float2uint8(img, bgr2rgb=False):
    img = img.clip(0, 1.0)
    img = np.round(img * 255.0).astype(np.uint8)
    if bgr2rgb and img.ndim == 3 and img.shape[2] == 3:
        img = img[:, :, [2, 1, 0]]  # BGR to RGB
    return img


def read_img(img_path):
    img = cv2.imread(img_path, -1)
    return img / 255.
