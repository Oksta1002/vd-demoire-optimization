import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import normalize
from vd.utils.registry import METRIC_REGISTRY

try:
    import lpips as lpips_module
except ImportError:
    print('Please install lpips: pip install lpips')

_lpips_model = None

def _get_lpips_model(device):
    global _lpips_model
    if _lpips_model is None:
        _lpips_model = lpips_module.LPIPS(net='alex', version='0.1').to(device)
        _lpips_model.eval()
    return _lpips_model


@METRIC_REGISTRY.register()
def calculate_lpips(img, img2, **kwargs):
    """
    Calculate LPIPS (Learned Perceptual Image Patch Similarity) using NumPy.

    Used for actual inference with real image files.

    Args:
        img (ndarray): Predicted image with range [0, 1], shape (h, w, 3).
        img2 (ndarray): Ground truth image with range [0, 1], shape (h, w, 3).
        **kwargs: Additional arguments.

    Returns:
        float: LPIPS value.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_fn_lpips = _get_lpips_model(device)
    
    # Convert numpy to tensor
    img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
    img2_tensor = torch.from_numpy(img2.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
    
    # Normalize to [-1, 1]
    mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(device)
    
    img_tensor = (img_tensor - mean) / std
    img2_tensor = (img2_tensor - mean) / std
    
    # Calculate LPIPS
    with torch.no_grad():
        lpips_val = loss_fn_lpips(img_tensor, img2_tensor)
    
    return lpips_val.item()


@METRIC_REGISTRY.register()
def calculate_lpips_pt(img, img2, crop_border=0, **kwargs):
    """
    Calculate LPIPS (Learned Perceptual Image Patch Similarity) using PyTorch.
    
    Used for validation during training with tensor inputs.
    
    Args:
        img (Tensor): Predicted images with range [0, 1], shape (n, 3, h, w).
        img2 (Tensor): Ground truth images with range [0, 1], shape (n, 3, h, w).
        crop_border (int): Cropped pixels in each edge. Default: 0.
        **kwargs: Additional arguments.
    
    Returns:
        Tensor: LPIPS values for each image in batch.
    """
    assert img.shape == img2.shape, f'Image shapes are different: {img.shape}, {img2.shape}.'
    
    if crop_border != 0:
        img = img[:, :, crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[:, :, crop_border:-crop_border, crop_border:-crop_border]
    
    device = img.device
    loss_fn_lpips = _get_lpips_model(device)
    
    # Ensure images are in the correct format
    img = img.to(torch.float32)
    img2 = img2.to(torch.float32)
    
    # LPIPS expects images normalized to [-1, 1]
    mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(device)
    
    img_norm = (img - mean) / std
    img2_norm = (img2 - mean) / std
    
    # Calculate LPIPS
    with torch.no_grad():
        lpips_val = loss_fn_lpips(img_norm, img2_norm)
    
    return lpips_val.squeeze()
