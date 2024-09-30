import torch
import torch.nn as nn
import torch.nn.functional as F
from fgsm import fgsm

def pgd(model, x, label, k, eps, eps_step, targeted, **kwargs):
    # define the box
    box_min = x - eps
    box_max = x + eps
    # copy x and detach + random initialize
    x_adv = x.clone().detach() + torch.empty_like(x).uniform_(-eps, eps).detach()
    x_adv.clamp_(0, 1)
    # start iteration
    for _ in range(k):
        x_adv = fgsm.fgsm(model, x_adv, label, eps_step, targeted)
        # project
        x_adv = torch.min(box_max, torch.max(box_min, x_adv)).detach()
    return x_adv.clamp_(0, 1)
