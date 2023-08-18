#!/usr/bin/env python3

import torch
import torch.nn.functional as F



class lossfun(torch.nn.Module):
    def __init__(self):
        super(lossfun, self).__init__()

    def forward(self, gt, oups):
        mean = oups[:,0:1,:,:]
        std = oups[:,1:2,:,:]
        #means = oups[:,:,0:1,:,:]
        #stds = oups[:,:,1:2,:,:]
        #mean = torch.mean(means, dim=0)
        #data_square = torch.mean(torch.square(stds), dim=0) * 2
        #model_square = torch.var(means, dim=0)
        #model_sigma = torch.std(means, dim=0)
        #std = torch.sqrt(data_square + model_square)
        loss = torch.div(torch.abs(gt - mean), std + 1e-6) + torch.log(std + 1e-6)
        #loss = torch.div(torch.abs(gt - mean), std + 1e-6) + torch.log(std + 1e-6) + model_sigma * 30
        loss2 = F.l1_loss(mean, gt)
        #loss2 = loss
        return loss, loss2

