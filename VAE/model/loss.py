import torch
import torch.nn.functional as F

def loss_fn(x_decoded, x, y_pred, y, mu, logvar, w):

    kl_loss = w * (-0.5) * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    recon_loss = F.mse_loss(x_decoded, x)
    regression_loss = F.mse_loss(y_pred, y)
    return kl_loss + recon_loss + regression_loss, recon_loss, kl_loss, regression_loss