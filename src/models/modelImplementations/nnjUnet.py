import pdb
import time
from typing import Literal, Union

import torch
import torchvision
from omegaconf import OmegaConf
from torch import Tensor, nn
from torchinfo import summary

import nnj
import src.utility.train_utils as utils
from nnj.abstract_diagonal_jacobian import AbstractDiagonalJacobian
from src.utility.debug_utils import debugxshape, shape_and_print_tensor


class Shift_Scale(AbstractDiagonalJacobian, nn.Module):
    """First shift then scales, e.g. if X is distributed N(1,2) (mean,std), shift_scale(-1,0.5) returns X~N(0,1)"""

    def __init__(self, *args, **kwargs):
        scale = kwargs.pop("scale")
        shift = kwargs.pop("shift")
        super().__init__(*args, **kwargs)
        self.scale = scale
        self.shift = shift
        self._n_params = 0
        print(shift, scale)

    def forward(self, x: Tensor):
        return (x + self.shift) * self.scale

    @torch.no_grad()
    def jacobian(
        self,
        x: Tensor,
        val: Union[Tensor, None] = None,
        wrt: Literal["input", "weight"] = "input",
        diag: bool = False,
    ) -> Union[Tensor, None]:
        """Returns the Jacobian matrix"""
        if wrt == "input":
            if val is None:
                val = self.forward(x)
            diag_jacobian = (torch.ones_like(val) * self.scale).reshape(val.shape[0], -1)
            if diag:
                return diag_jacobian
            else:
                return torch.diag_embed(diag_jacobian)
        elif wrt == "weight":
            # non parametric layer has no jacobian with respect to weight
            return None


class Dropout(nn.Dropout2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._n_params = 0


class InputDropout(nn.Dropout):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._n_params = 0


class stochastic_unet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, cfg=None, dropout_version=False):
        super().__init__()
        if cfg:
            multiplication_factor = cfg.neural_net_param_multiplication_factor
            im_height = cfg.dataset_params.input_height
            im_width = cfg.dataset_params.input_width
            self.min_depth = cfg.dataset_params.min_depth
            self.max_depth = cfg.dataset_params.max_depth
            self.cfg = cfg
        else:
            multiplication_factor = 48
            im_height = 352
            im_width = 704
            self.min_depth = 1e-8
            self.max_depth = 80
            if dropout_version:
                self.cfg.models.p_hidden_dropout = 0.2
                self.cfg.models.model_type = "Dropout"
            else:
                self.cfg.models.model_type = "stochastic_unet"

        if self.cfg.models.model_type == "Dropout":
            first_downblock = [
                InputDropout(p=self.cfg.models.p_input_dropout),
                nnj.Conv2d(in_channels, multiplication_factor, 3, stride=1, padding=1),
                Dropout(p=self.cfg.models.p_hidden_dropout),
                nnj.Tanh(),
                nnj.Conv2d(multiplication_factor, multiplication_factor, 3, stride=1, padding=1),
                Dropout(p=self.cfg.models.p_hidden_dropout),
                nnj.Tanh(),
            ]
            last_upblock = [
                nnj.Conv2d(
                    multiplication_factor * 2, multiplication_factor, 3, stride=1, padding=1
                ),
                Dropout(p=self.cfg.models.p_hidden_dropout),
                nnj.Tanh(),
                nnj.Conv2d(
                    int(multiplication_factor), multiplication_factor, 3, stride=1, padding=1
                ),
                nnj.Tanh(),
                nnj.Conv2d(multiplication_factor, out_channels, 3, stride=1, padding=1),
            ]
        else:
            first_downblock = [
                nnj.Conv2d(in_channels, multiplication_factor, 3, stride=1, padding=1),
                nnj.Tanh(),
                nnj.Conv2d(multiplication_factor, multiplication_factor, 3, stride=1, padding=1),
                nnj.Tanh(),
            ]
            last_upblock = [
                nnj.Conv2d(
                    multiplication_factor * 2, multiplication_factor, 3, stride=1, padding=1
                ),
                nnj.Tanh(),
                nnj.Conv2d(
                    int(multiplication_factor), multiplication_factor, 3, stride=1, padding=1
                ),
                nnj.Tanh(),
                nnj.Conv2d(multiplication_factor, out_channels, 3, stride=1, padding=1),
            ]
        downblocks = [
            self.downblock_gen(
                int(2**i * multiplication_factor), int(im_height / 2**i), int(im_width / 2**i)
            )
            for i in [0, 1, 2, 3]
        ]
        upblocks = [
            self.upblock_gen(
                int(2 * 2**i * multiplication_factor),
                int(im_height / 2**i),
                int(im_width / 2**i),
            )
            for i in [1, 2, 3, 4]
        ]

        self.stochastic_net = nnj.Sequential(
            *first_downblock,
            nnj.SkipConnection(
                *downblocks[0],
                nnj.SkipConnection(
                    *downblocks[1],
                    nnj.SkipConnection(
                        *downblocks[2],
                        nnj.SkipConnection(
                            *self.middleblock_gen(
                                int(2**3 * multiplication_factor),
                                int(im_height / 2**3),
                                int(im_width / 2**3),
                            ),
                            add_hooks=True,
                        ),
                        *upblocks[2],
                        add_hooks=True,
                    ),
                    *upblocks[1],
                    add_hooks=True,
                ),
                *upblocks[0],
                add_hooks=True,
            ),
            *last_upblock,
            nnj.Sigmoid(),
            Shift_Scale(
                shift=self.min_depth / (self.max_depth - self.min_depth),
                scale=self.max_depth - self.min_depth,
            ),
            add_hooks=True,
        )

    def downblock_gen(self, in_channels, im_height, im_width):
        if self.cfg.models.model_type == "Dropout":
            downblock = [
                nnj.MaxPool2d(2),
                nnj.Conv2d(in_channels, in_channels * 2, 3, stride=1, padding=1),
                nnj.Tanh(),
                Dropout(p=self.cfg.models.p_hidden_dropout),
                nnj.Conv2d(in_channels * 2, in_channels * 2, 3, stride=1, padding=1),
                nnj.Tanh(),
                Dropout(p=self.cfg.models.p_hidden_dropout),
            ]
        else:
            downblock = [
                nnj.MaxPool2d(2),
                nnj.Conv2d(in_channels, in_channels * 2, 3, stride=1, padding=1),
                nnj.Tanh(),
                nnj.Conv2d(in_channels * 2, in_channels * 2, 3, stride=1, padding=1),
                nnj.Tanh(),
            ]
        return downblock

    def upblock_gen(self, in_channels, im_height, im_width):
        if self.cfg.models.model_type == "Dropout":
            upblock = [
                nnj.Conv2d(in_channels, int(in_channels / 2), 3, stride=1, padding=1),
                nnj.Tanh(),
                Dropout(p=self.cfg.models.p_hidden_dropout),
                nnj.Conv2d(int(in_channels / 2), int(in_channels / 4), 3, stride=1, padding=1),
                nnj.Tanh(),
                Dropout(p=self.cfg.models.p_hidden_dropout),
                nnj.Upsample(scale_factor=2),
            ]
        else:
            upblock = [
                nnj.Conv2d(in_channels, int(in_channels / 2), 3, stride=1, padding=1),
                nnj.Tanh(),
                nnj.Conv2d(int(in_channels / 2), int(in_channels / 4), 3, stride=1, padding=1),
                nnj.Tanh(),
                nnj.Upsample(scale_factor=2),
            ]
        return upblock

    def middleblock_gen(self, in_channels, im_height, im_width):
        if self.cfg.models.model_type == "Dropout":
            midblock = [
                nnj.MaxPool2d(2),
                nnj.Conv2d(in_channels, in_channels * 2, 3, stride=1, padding=1),
                nnj.Tanh(),
                Dropout(p=self.cfg.models.p_hidden_dropout),
                nnj.Conv2d(in_channels * 2, in_channels, 3, stride=1, padding=1),
                nnj.Tanh(),
                Dropout(p=self.cfg.models.p_hidden_dropout),
                nnj.Upsample(scale_factor=2),
            ]
        else:
            midblock = [
                nnj.MaxPool2d(2),
                nnj.Conv2d(in_channels, in_channels * 2, 3, stride=1, padding=1),
                nnj.Tanh(),
                nnj.Conv2d(in_channels * 2, in_channels, 3, stride=1, padding=1),
                nnj.Tanh(),
                nnj.Upsample(scale_factor=2),
            ]
        return midblock

    def forward(self, x):
        x = self.stochastic_net(x)
        return x


if __name__ == "__main__":
    # u_net = stochastic_unet(in_channels=3, out_channels=1)
    print("GPU: ", torch.cuda.is_available())
    print("nGPUs:", torch.cuda.device_count())
    print("CurrDevice:", torch.cuda.current_device())
    print("Device name: ", torch.cuda.get_device_name(0))
    print(torch.cuda.memory_summary())
    cfg = OmegaConf.create(
        {
            "neural_net_param_multiplication_factor": 96,
            "dataset_params": {
                "input_height": 352,
                "input_width": 1216,
                "min_depth": 1e-4,
                "max_depth": 80,
            },
            "models": {"model_type": "Dropout", "p_hidden_dropout": 0.5, "p_input_dropout": 0.2},
        }
    )
    cfg_2 = OmegaConf.create(
        {
            "neural_net_param_multiplication_factor": 96,
            "dataset_params": {
                "input_height": 352,
                "input_width": 1216,
                "min_depth": 1e-4,
                "max_depth": 80,
            },
            "models": {"model_type": "Dropout", "p_hidden_dropout": 0.5, "p_input_dropout": 0.2},
        }
    )
    cfg_3 = OmegaConf.create(
        {
            "neural_net_param_multiplication_factor": 64,
            "dataset_params": {
                "input_height": 352,
                "input_width": 1216,
                "min_depth": 1e-4,
                "max_depth": 80,
            },
            "models": {"model_type": "Dropout", "p_hidden_dropout": 0.5, "p_input_dropout": 0.2},
        }
    )
    u_net = stochastic_unet(in_channels=3, out_channels=1, cfg=cfg)
    u_net2 = stochastic_unet(in_channels=3, out_channels=1, cfg=cfg_2)
    u_net3 = stochastic_unet(in_channels=3, out_channels=1, cfg=cfg_3)
    # u_net.to(device="cuda")
    print(u_net.stochastic_net)
    t = torch.rand((8, 3, 352, 704), device="cuda")
    summary(u_net, (8, 3, 352, 704), depth=300)
    summary(u_net2, (8, 3, 352, 1216), depth=300)
    summary(u_net3, (8, 3, 352, 1216), depth=300)
    a = u_net(t)

    print(torch.max(u_net(t)))
    print(torch.min(u_net(t)))

#    from torchview import draw_graph

#    graph_of_model = draw_graph(a_based_u_net, input_size=(1, 3, 1024, 608))
#    graph_of_model.visual_graph

#    graph_of_model.visual_graph.view()
