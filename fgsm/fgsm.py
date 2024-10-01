import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Implementation of Fast Gradient Sign Method for adversarial Attacks
"""
def fgsm(model, x, t, eps, targeted, **kwargs):
    # define the cross entropy loss
    loss = nn.CrossEntropyLoss()
    # make sure gradient wrt to x is computed
    x.requires_grad = True
    # compute the output of the model
    output = model(x) # (batch_size, n_classes)
    # recast the target to a tensor
    t = torch.tensor(t, torch.int64)
    # compute the loss of the output with respect to the target label
    loss_val = loss(output, t)
    # compute the backwards pass
    loss_val.backward()
    # compute the perturbation
    eta = eps * torch.sign(x.grad)
    # if targeted minimize the loss to maximize the likelihood, else do the opposite
    x = x - eta if targeted else x + eta
    return x.clamp_(0, 1) # this can be done in place

def fgsm_targeted(model, x, t, eps, **kwargs):
    return fgsm(model, x, t, eps, True)
def fgsm_untargeted(model, x, t, eps, **kwargs):
    return fgsm(model, x, t, eps, False)
