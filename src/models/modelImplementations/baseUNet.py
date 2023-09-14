import src.utility.train_utils as utils
from src.utility.debug_utils import debugxshape, shape_and_print_tensor


@utils.retry(2)
def lightning_imports():
    import pydantic
    import lightning

    return 0


lightning_imports()
import torch
import torchvision
from torch import nn
import time
from torchinfo import summary
import decorator
import hydra
from omegaconf import DictConfig, OmegaConf
import logging


log = logging.getLogger(__name__)
in_debug = True


# BatchXColorXHeightXWidth
class DoubleConv(nn.Sequential):
    # no_padding is original way, but we prefer to include padding.
    def __init__(self, in_chan, out_chan, include_padding=False):
        super().__init__(
            nn.Conv2d(
                in_chan, out_chan, kernel_size=3, padding=int(include_padding)
            ),  # up to outchannels in first layer.
            nn.ReLU(),
            nn.Conv2d(out_chan, out_chan, kernel_size=3, padding=int(include_padding)),
            nn.ReLU(),
        )


class DownSample(nn.Sequential):
    def __init__(self):
        super().__init__(nn.MaxPool2d(2))


class UpSample(nn.Sequential):
    def __init__(self, in_chan, out_chan):
        super().__init__(
            nn.ConvTranspose2d(in_channels=in_chan, out_channels=out_chan, kernel_size=2, stride=2)
        )  # not sure i understand this module


class CropAndConcatenateOp(nn.Module):
    def forward(self, x: torch.Tensor, contracting_x: torch.Tensor):
        contracting_x = torchvision.transforms.functional.center_crop(
            contracting_x, (x.shape[2], x.shape[3])
        )
        x = torch.cat([x, contracting_x], dim=1)
        return x


class BaseUNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.down_conv = nn.ModuleList(
            [DoubleConv(i, o) for (i, o) in [(in_channels, 64), (64, 128), (128, 256), (256, 512)]]
        )
        self.down_sample = nn.ModuleList([DownSample() for _ in range(4)])
        self.middle_conv = DoubleConv(512, 1024)

        self.up_sample = nn.ModuleList(
            [UpSample(i, o) for (i, o) in [(1024, 512), (512, 256), (256, 128), (128, 64)]]
        )
        self.up_conv = nn.ModuleList(
            [DoubleConv(i, o) for (i, o) in [(1024, 512), (512, 256), (256, 128), (128, 64)]]
        )
        self.concat = nn.ModuleList([CropAndConcatenateOp() for _ in range(4)])
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor):
        pass_through = []
        debugxshape(x)

        for i in range(len(self.down_conv)):  # contract
            x = self.down_conv[i](x)
            debugxshape(x)
            pass_through.append(x)
            debugxshape(x)
            x = self.down_sample[i](x)
            debugxshape(x)

        x = self.middle_conv(x)
        debugxshape(x)

        for i in range(len(self.up_conv)):  # upsample and scale
            x = self.up_sample[i](x)
            debugxshape(x)
            x = self.concat[i](x, contracting_x=pass_through.pop())
            debugxshape(x)
            x = self.up_conv[i](x)
            debugxshape(x)

        x = self.final_conv(x)
        debugxshape(x)

        return x


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def save_config(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    save_config()
    x = torch.rand(1, 3, 572, 572)
    x = BaseUNet(3, 2)(x)
    debugxshape(x)

    a_based_u_net = BaseUNet(3, 2)
    summary(a_based_u_net, (1, 3, 572, 572))

    from torchview import draw_graph

    graph_of_model = draw_graph(a_based_u_net, input_size=(1, 3, 572, 572))
    graph_of_model.visual_graph

    graph_of_model.visual_graph.view()

    shape_and_print_tensor(x)
