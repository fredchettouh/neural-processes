import torch


def tensor_rbf(x1, x2, l1_scale, simga_f_scale, sigma_noise, dimy=1):
    batch_size = x1.shape[0]
    num_points = x1.shape[1]
    dimx = x1.shape[2]
    x1 = x1.unsqueeze(1)
    x2 = x2.unsqueeze(2)
    diff = x1 - x2
    l1 = torch.ones(batch_size, dimy, dimx) * l1_scale
    sigma_f = torch.ones(batch_size, dimy) * simga_f_scale
    norm = (diff[:, None, :, :, :] / l1[:, :, None, None, :]).pow(2).mean(-1)
    kernel = sigma_f.pow(2)[:, :, None, None] * (-0.5 * norm).pow(2)
    kernel += (sigma_noise ** 2) * torch.eye(num_points)
    return kernel
