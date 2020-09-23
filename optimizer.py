import torch
import torch.nn.modules.loss
import torch.nn.functional as F


def loss_function(r1,labels):

    loss = torch.nn.BCEWithLogitsLoss()#nn.BCELoss()
    bc=loss(r1,labels)  

    return bc