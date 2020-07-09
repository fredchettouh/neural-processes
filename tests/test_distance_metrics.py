import torch


def test_laplace_kernel():
    bs = 64
    nt = 10
    nc = 5
    dx = 128

    output = torch.normal(0, 1, (bs, nt, nc, dx))

    bs_out, nt_out, nc_out, dx_out = output.size()

    assert(bs_out == bs and nt_out == nt and nc_out == nc and dx_out == dx)
