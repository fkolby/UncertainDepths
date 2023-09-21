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
import logging


# log = logging.getLogger(__name__)
in_debug = True


# BatchXColorXHeightXWidth
class DoubleConv(nn.Sequential):
    # no_padding is original way, but we prefer to include padding.
    def __init__(self, in_chan, out_chan, include_padding=True):
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
        # contracting_x = torchvision.transforms.CenterCrop(
        #     (x.shape[2], x.shape[3])
        # )(contracting_x)
        x = torch.cat([x, contracting_x], dim=1)
        return x


class BaseUNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, in_debug: bool = False):
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
        self.in_debug = in_debug

    def forward(self, x: torch.Tensor):
        pass_through = []
        debugxshape(x, in_debug=self.in_debug)

        for i in range(len(self.down_conv)):  # contract
            x = self.down_conv[i](x)
            debugxshape(x, in_debug=self.in_debug)
            pass_through.append(x)
            debugxshape(x, in_debug=self.in_debug)
            x = self.down_sample[i](x)
            debugxshape(x, in_debug=self.in_debug)

        x = self.middle_conv(x)
        debugxshape(x, in_debug=self.in_debug)

        for i in range(len(self.up_conv)):  # upsample and scale
            x = self.up_sample[i](x)
            debugxshape(x, in_debug=self.in_debug)
            x = self.concat[i](x, contracting_x=pass_through.pop())
            debugxshape(x, in_debug=self.in_debug)
            x = self.up_conv[i](x)
            debugxshape(x, in_debug=self.in_debug)

        x = self.final_conv(x)
        debugxshape(x, in_debug=self.in_debug)

        return x


if __name__ == "__main__":
    a_based_u_net = BaseUNet(3, 1, in_debug=True)
    summary(a_based_u_net, (1, 3, 1024, 608))

    from torchview import draw_graph

    graph_of_model = draw_graph(a_based_u_net, input_size=(1, 3, 1024, 608))
    graph_of_model.visual_graph

    graph_of_model.visual_graph.view()
