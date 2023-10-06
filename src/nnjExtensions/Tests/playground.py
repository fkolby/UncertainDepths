import nnj

import pdb
import torch


from src.nnjExtensions import Conv2d, SkipConnection, Flatten, Upsample, MaxPool2d

x = torch.rand((2, 3, 4, 5))


# pdb.set_trace()
conv = Conv2d.Conv2d(3, 2, 1)
skp = SkipConnection.SkipConnection(conv)
flt = Flatten.Flatten()
mx2 = MaxPool2d.MaxPool2d(kernel_size =2)
Upsample = Upsample.Upsample(scale_factor=2)

conv(x)
skp(x)
flt(x)
pdb.set_trace()
