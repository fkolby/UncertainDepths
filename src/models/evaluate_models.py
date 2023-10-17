import torch
from tqdm import tqdm
import pdb
from src.models.predict_model import infer
from src.utility.other_utils import RunningAverageDict
from src.utility.eval_utils import compute_metrics

@torch.no_grad()
def eval_model(model, test_loader, config, round_vals=True, round_precision=3):
    model.eval()
    metrics = RunningAverageDict()
    for i, sample in tqdm(enumerate(test_loader), total=len(test_loader)):
        if 'has_valid_depth' in sample:
            if not sample['has_valid_depth']:
                continue
        image, _,  depth = sample
        depth = depth
        pred = infer(model, image)
        # Save image, depth, pred for visualization
        if config.save_images and (i==0 or i==(len(sample)-1)):
            import os
            # print("Saving images ...")
            from PIL import Image
            import torchvision.transforms as transforms
            from src.utility.viz_utils import colorize

            os.makedirs(save_images_path, exist_ok=True)
            d = colorize(depth.cpu().numpy(), 0, 80)
            p = colorize(pred.cpu().numpy(), 0, 80)
            im = transforms.ToPILImage()(image.cpu())
            im.save(os.path.join(save_images_path, f"{i}_img.png"))
            Image.fromarray(d).save(os.path.join(save_images_path, f"{i}_depth.png"))
            Image.fromarray(p).save(os.path.join(save_images_path, f"{i}_pred.png"))



        # print(depth.shape, pred.shape)
        metrics.update(compute_metrics(depth, pred, config=config))
    if round_vals:
        def r(m): return torch.round(m,decimals= round_precision)
    else:
        def r(m): return m
    metrics = {k: r(v) for k, v in metrics.get_value().items()}
    return metrics