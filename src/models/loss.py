import torch
from torch import nn


# Cough cough, heavily inspired by adabins
class SILogLoss(nn.Module):
    def __init__(self):
        super(SILogLoss, self).__init__()
        self.name = "SILog"

    def forward(self, input, target):
        eps = 1e-6
        input = torch.maximum(
            torch.zeros(requires_grad=True, size=input.shape, device=input.device), input
        )

        g = torch.log(input + eps) - torch.log(target + eps)

        Dg = torch.var(g) + 0.15 * torch.pow(torch.mean(g), 2)

        loss = 10 * torch.sqrt(Dg)

        if torch.isnan(loss):
            print("Nan SILog loss")
            print("input:", input.shape)
            print("target:", target.shape)
            print("G", torch.sum(torch.isnan(g)))
            print("Input min max", torch.min(input), torch.max(input))
            print("Target min max", torch.min(target), torch.max(target))
            print("Dg", torch.isnan(Dg))
            print("loss", torch.isnan(loss))

        return loss
