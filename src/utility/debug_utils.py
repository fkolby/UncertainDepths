import torch

def shape_and_print_tensor(x: torch.Tensor,log):
    log.info(f"The shape of the tensor is {x.shape}")
    log.info(f"It prints like: \n {x}")
    return 0


def debugxshape(x: torch.Tensor, log, in_debug = True):
    if in_debug:
        log.info(x.shape)
