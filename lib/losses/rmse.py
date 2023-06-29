import torch
import torch.nn as nn


class rmse_loss(nn.Module):
    '''
    Params:
        x_pred: (batch_size, dec_steps, loc_dim)
        x_true: (batch_size, dec_steps, loc_dim)
    Returns:
        rmse: scalar
    '''
    def __init__(self):
        super(rmse_loss, self).__init__()
    
    def forward(self, x_pred, x_true):#sum over errors in the last dimension
        L2_diff = torch.sqrt(torch.sum((x_pred - x_true)**2, dim=2))
        # sum over prediction time steps
        L2_all_pred = torch.sum(L2_diff, dim=1)
        # mean of all batches
        L2_mean_pred = torch.mean(L2_all_pred, dim=0)
        return L2_mean_pred