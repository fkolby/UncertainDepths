import torch
import time


def shape_and_print_tensor(x: torch.Tensor, log):
    log.info(f"The shape of the tensor is {x.shape}")
    log.info(f"It prints like: \n {x}")
    return 0


def debugxshape(x: torch.Tensor, in_debug=False):
    if in_debug:
        print(x.shape)


def time_since_previous_log(prev_time: float, function_ran=""):
    timenow = time.time()
    print(
        "time passed since timer func last called,",
        prev_time - timenow,
        "ran function",
        function_ran,
    )
    return timenow
