import nnj

import pdb
import torch


from nnj.nnjExtensions import Conv2d, SkipConnection, Flatten

x = torch.rand((2, 3, 4, 5))


# pdb.set_trace()
conv = Conv2d.Conv2d(3, 2, 1)
skp = SkipConnection.SkipConnection(conv)
flt = Flatten.Flatten()

conv(x)
skp(x)
flt(x)
pdb.set_trace()
