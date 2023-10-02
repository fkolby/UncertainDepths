from stochman import nnj
import torch


class UNet_stochman_64(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.stochastic_net = nnj.Sequential(
            nnj.Conv2d(3, 8, 3, stride=1, padding=1),
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
            nnj.Conv2d(8, 1, 1, stride=1, padding=0),
            add_hooks=True,
        )

    def forward(self, x):
        return self.stochastic_net(x)
