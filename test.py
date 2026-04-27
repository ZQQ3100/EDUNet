import torch
import os
from utils import cv_utils
from opts.test_opt import TestOpt
from data.dataloader import CustomDatasetDataLoader
from networks.networks import NetworksFactory



def rgb_to_yuv_tensor(rgb_tensor):
    """
    将RGB tensor转换为YUV tensor
    Args:
        rgb_tensor: [1, 3, H, W] 范围的tensor
    Returns:
        y_tensor: [1, 1, H, W] 范围的Y分量
        u_tensor: [1, 1, H, W] 范围的U分量
        v_tensor: [1, 1, H, W] 范围的V分量
    """
    rgb_tensor = torch.clamp(rgb_tensor, 0, 1)

    transform_matrix = torch.tensor([
        [0.299, 0.587, 0.114],
        [-0.14713, -0.28886, 0.436],
        [0.615, -0.51499, -0.10001]
    ]).float().to(rgb_tensor.device)

    batch_size, channels, height, width = rgb_tensor.shape
    rgb_flat = rgb_tensor.view(batch_size, channels, -1)

    yuv_flat = torch.bmm(transform_matrix.unsqueeze(0).expand(batch_size, -1, -1),
                         rgb_flat)

    yuv_tensor = yuv_flat.view(batch_size, 3, height, width)

    y_tensor = yuv_tensor[:, 0:1, :, :]
    u_tensor = yuv_tensor[:, 1:2, :, :] + 0.5
    v_tensor = yuv_tensor[:, 2:3, :, :] + 0.5

    y_tensor = torch.clamp(y_tensor, 0, 1)
    u_tensor = torch.clamp(u_tensor, 0, 1)
    v_tensor = torch.clamp(v_tensor, 0, 1)

    return y_tensor, u_tensor, v_tensor


def yuv_to_rgb_tensor(y_tensor, u_tensor, v_tensor):
    """
    将YUV tensor转换回RGB tensor
    Args:
        y_tensor: [1, 1, H, W] 范围的Y分量
        u_tensor: [1, 1, H, W] 范围的U分量
        v_tensor: [1, 1, H, W] 范围的V分量
    Returns:
        rgb_tensor: [1, 3, H, W] 范围的RGB tensor
    """
    y_tensor = torch.clamp(y_tensor, 0, 1)
    u_tensor = torch.clamp(u_tensor, 0, 1)
    v_tensor = torch.clamp(v_tensor, 0, 1)

    u_tensor = u_tensor - 0.5
    v_tensor = v_tensor - 0.5

    transform_matrix = torch.tensor([
        [1.0, 0.0, 1.13983],
        [1.0, -0.39465, -0.58060],
        [1.0, 2.03211, 0.0]
    ]).float().to(y_tensor.device)

    batch_size, _, height, width = y_tensor.shape
    yuv_tensor = torch.cat([y_tensor, u_tensor, v_tensor], dim=1)
    yuv_flat = yuv_tensor.view(batch_size, 3, -1)

    rgb_flat = torch.bmm(transform_matrix.unsqueeze(0).expand(batch_size, -1, -1), yuv_flat)

    rgb_tensor = rgb_flat.view(batch_size, 3, height, width)

    rgb_tensor = torch.clamp(rgb_tensor, 0, 1)

    return rgb_tensor


def process_with_yuv_scheme(model, blurred, event_bins, device):
    """
    使用YUV方案处理图像
    """
    y_tensor, u_tensor, v_tensor = rgb_to_yuv_tensor(blurred)

    y_tensor = y_tensor.to(device)
    u_tensor = u_tensor.to(device)
    v_tensor = v_tensor.to(device)
    event_bins = event_bins.to(device)

    with torch.no_grad():
        outputs_y, _, _ = model(y_tensor, event_bins)
        outputs_y = torch.clamp(outputs_y, 0, 1)
    outputs_rgb = yuv_to_rgb_tensor(outputs_y, u_tensor, v_tensor)

    return outputs_rgb


if __name__ == '__main__':
    opt = TestOpt().parse()
    device = torch.device(opt.cuda)
    if 'cuda' in opt.cuda:
        torch.cuda.set_device(device)

    data_loader_test = CustomDatasetDataLoader(opt, is_for_train=False)
    dataset_test = data_loader_test.load_data()
    dataset_train_size = len(data_loader_test)
    print('# Testing images : %d' % dataset_train_size)

    model = NetworksFactory.get_by_name('our')
    model = torch.load(opt.load_G)
    model = model.to(device)

    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)

    model.eval()

    for i, test_batch in enumerate(dataset_test):
        blurred = test_batch['blurred']
        event_bins = test_batch['event_bins']

        outputs = process_with_yuv_scheme(model, blurred, event_bins, device)

        dataname = test_batch['dataname'][0]
        img_idx = test_batch['img_idx'][0].item()
        path = os.path.join(opt.output_dir, dataname)
        if not os.path.exists(path):
            os.makedirs(path)
        path_i = os.path.join(path, str(img_idx).zfill(4) + '.png')
        cv_utils.debug_save_tensor(outputs, path_i, rela=False, rgb=3)

    print('Finished Testing')