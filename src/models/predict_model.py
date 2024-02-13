import numpy
import torch
import torchvision


# heavily inspired by zoedepth (https://github.com/isl-org/ZoeDepth/blob/main/evaluate.py).
@torch.no_grad()
def infer(model, images: torch.Tesnor, **kwargs) -> torch.Tensor:
    """Inference with flip augmentation"""

    # images.shape = N, C, H, W
    def get_depth_from_prediction(pred):
        if isinstance(pred, torch.Tensor):
            pred = pred  # pass
        elif isinstance(pred, dict):
            pred = pred["metric_depth"]
        else:
            raise NotImplementedError(f"Unknown output type {type(pred)}")
        return pred

    pred1 = model(images, **kwargs)
    pred1 = get_depth_from_prediction(pred1)

    pred2 = model(torch.flip(images, [3]), **kwargs)
    pred2 = get_depth_from_prediction(pred2)
    pred2 = torch.flip(pred2, [3])

    mean_pred = 0.5 * (pred1 + pred2)

    return mean_pred
