import sys
import torch
from torch.autograd import Function
import torch.nn.functional as F


def split_feature(tensor, type="split"):
    """
    type = ["split", "cross"]
    """
    C = tensor.size(1)
    if type == "split":
        return tensor[:, :C // 2, ...], tensor[:, C // 2:, ...]
    elif type == "cross":
        return tensor[:, 0::2, ...], tensor[:, 1::2, ...]


def cat_feature(tensor_a, tensor_b):
    return torch.cat((tensor_a, tensor_b), dim=1)


def dclamp(x, min, max):
    return x - (x - torch.clamp(x, min, max)).detach()


def dfloor(x):
    return x + (torch.floor(x) - x).detach()


def dround(x):
    return x + (torch.round(x) - x).detach()


def quantize_img(x, range=[0, 1]):
    if range == [0, 1]:
        return x - (x - torch.round(torch.clamp(x, 0, 1) * 255.) / 255.).detach()
    elif range == [-1, 1]:
        x_round = (torch.round(torch.clamp((x + 1) * 0.5, 0, 1) * 255.) / 255.) * 2 - 1
        return x - (x - x_round).detach()
    else:
        raise RuntimeError('Invalid image value range, only support [0, 1] and [-1, 1]')



def replace_nan(x):
    x = torch.where(torch.isnan(x), torch.rand_like(x), x)
    return x


def check_nan_inf(x, name='None', shutdown=False):
    if (torch.isnan(x).any() or torch.isinf(x).any()):
        print(f'Some elements in {name} are NaN or Inf')
        if shutdown:
            sys.exit(0)


def set_requires_grad(m, requires_grad=True):
    for p in m.parameters():
        p.requires_grad_(requires_grad)