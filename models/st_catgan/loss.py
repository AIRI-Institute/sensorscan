import torch

def entropy(p):
    return -(p * torch.log(p + 1e-6)).sum(dim=-1)

def disc_loss(real_pred, fake_pred):
    return -entropy(real_pred.mean(dim=0)) + entropy(real_pred).mean() - entropy(fake_pred).mean()

def gen_loss(fake_pred):
    return -entropy(fake_pred.mean(dim=0)) + entropy(fake_pred).mean()