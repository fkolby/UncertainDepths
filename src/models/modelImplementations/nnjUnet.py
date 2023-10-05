import nnj
import torch
import src.utility.train_utils as utils
from src.utility.debug_utils import debugxshape, shape_and_print_tensor
from typing import Union, Literal


import torch
import torchvision
from torch import nn, Tensor
import time
from torchinfo import summary
import pdb


class SkipConnect(nnj.AbstractJacobian, nn.Module):
    def __init__(self, *args, add_hooks: bool = False):
        super().__init__()
        self._F = nnj.Sequential(*args, add_hooks=add_hooks)
        self._n_params = 0

    def forward(self, x: Tensor):
        return torch.cat([x, self._F(x)], dim=1)

    @torch.no_grad()
    def jacobian(
        self,
        x: torch.Tensor,
        val: Union[None, torch.Tensor],
        wrt: Literal["input", "weight"] = "input",
    ):
        if val == None:
            return self.forward(x)

        if wrt == "weight":
            # Non parametric layer, so does not have any wrt weight
            return None
        else:
            raise NotImplementedError


class Flatten(nnj.AbstractJacobian, nn.Module):
    # NOT SURE ABOUT INIT: NOT INCLUDED IN MARCO?!
    def __init__(self):
        self._n_params = 0

    def forward(self, x: Tensor):
        return x.flatten(start_dim=1, end_dim=-1)

    @torch.no_grad()
    def jacobian(
        self,
        x: torch.Tensor,
        val: Union[None, torch.Tensor],
        wrt: Literal["input", "weight"] = "input",
    ):
        if val == None:
            return self.forward(x)

        if wrt == "weight":
            # Non parametric layer, so does not have any wrt weight
            return None
        else:
            raise NotImplementedError


class Conv2d(nnj.AbstractJacobian, nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
    ):
        super(self, Conv2d).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )

    @torch.no_grad()
    def jacobian(
        self,
        x: torch.Tensor,
        val: Union[None, torch.Tensor],
        wrt: Literal["input", "weight"] = "input",
    ):
        if val == None:
            return self.forward(x)

        if wrt == "weight":
            # Non parametric layer, so does not have any wrt weight
            return None
        else:
            raise NotImplementedError


pdb.set_trace()


class UNet_stochman_64(torch.nn.Module):
    def __init__(self, in_chan, out_chan):
        super().__init__()

        self.stochastic_net = nnj.Sequential(
            nnj.Conv2d(in_chan, 8, 3, stride=1, padding=1),
            nnj.Tanh(),
            nnj.Conv2d(8, 8, 3, stride=1, padding=1),
            nnj.Tanh(),
            nnj.Flatten(),
            nnj.SkipConnection(
                nnj.Reshape(8, 64, 64),
                nnj.MaxPool2d(2),
                nnj.Conv2d(8, 16, 3, stride=1, padding=1),
                nnj.Tanh(),
                nnj.Conv2d(16, 16, 3, stride=1, padding=1),
                nnj.Tanh(),
                nnj.Flatten(),
                nnj.SkipConnection(
                    nnj.Reshape(16, 32, 32),
                    nnj.MaxPool2d(2),
                    nnj.Conv2d(16, 32, 3, stride=1, padding=1),
                    nnj.Tanh(),
                    nnj.Conv2d(32, 32, 3, stride=1, padding=1),
                    nnj.Tanh(),
                    nnj.Flatten(),
                    nnj.SkipConnection(
                        nnj.Reshape(32, 16, 16),
                        nnj.MaxPool2d(2),
                        nnj.Conv2d(32, 64, 3, stride=1, padding=1),
                        nnj.Tanh(),
                        nnj.Conv2d(64, 64, 3, stride=1, padding=1),
                        nnj.Tanh(),
                        nnj.Flatten(),
                        nnj.SkipConnection(
                            nnj.Reshape(64, 8, 8),
                            nnj.MaxPool2d(2),
                            nnj.Conv2d(64, 128, 3, stride=1, padding=1),
                            nnj.Tanh(),
                            nnj.Conv2d(128, 64, 3, stride=1, padding=1),
                            nnj.Upsample(scale_factor=2),
                            nnj.Tanh(),
                            nnj.Flatten(),
                            add_hooks=True,
                        ),
                        nnj.Reshape(128, 8, 8),
                        nnj.Conv2d(128, 64, 3, stride=1, padding=1),
                        nnj.Tanh(),
                        nnj.Conv2d(64, 32, 3, stride=1, padding=1),
                        nnj.Upsample(scale_factor=2),
                        nnj.Tanh(),
                        nnj.Flatten(),
                        add_hooks=True,
                    ),
                    nnj.Reshape(64, 16, 16),
                    nnj.Conv2d(64, 32, 3, stride=1, padding=1),
                    nnj.Tanh(),
                    nnj.Conv2d(32, 16, 3, stride=1, padding=1),
                    nnj.Upsample(scale_factor=2),
                    nnj.Tanh(),
                    nnj.Flatten(),
                    add_hooks=True,
                ),
                nnj.Reshape(32, 32, 32),
                nnj.Conv2d(32, 16, 3, stride=1, padding=1),
                nnj.Tanh(),
                nnj.Conv2d(16, 8, 3, stride=1, padding=1),
                nnj.Upsample(scale_factor=2),
                nnj.Tanh(),
                nnj.Flatten(),
                add_hooks=True,
            ),
            nnj.Reshape(16, 64, 64),
            nnj.Conv2d(16, 8, 3, stride=1, padding=1),
            nnj.Tanh(),
            nnj.Conv2d(8, 8, 3, stride=1, padding=1),
            nnj.Tanh(),
            nnj.Conv2d(8, out_chan, 1, stride=1, padding=0),
            add_hooks=True,
        )

    def forward(self, x):
        return self.stochastic_net(x)


if __name__ == "__main__":
    a_based_u_net = UNet_stochman_64(in_chan=3, out_chan=1)
    summary(a_based_u_net, (1, 3, 704, 352))

    from torchview import draw_graph

    graph_of_model = draw_graph(a_based_u_net, input_size=(1, 3, 1024, 608))
    graph_of_model.visual_graph

    graph_of_model.visual_graph.view()
