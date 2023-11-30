import pytorch_lightning as pl

from src.utility.viz_utils import log_images, log_loss_metrics

class Base_module(pl.LightningModule):
    def __init__(
        self,
        model,
        loss_function,
        steps_per_epoch: int,
        cfg,
    ):
        super().__init__()
        self.model = model
        self.loss_function = loss_function
        self.lr = cfg.hyperparameters.learning_rate
        self.tstep = 0
        self.min_depth = cfg.dataset_params.min_depth
        self.max_depth = cfg.dataset_params.max_depth
        self.input_height = cfg.dataset_params.input_height
        self.input_width = cfg.dataset_params.input_width
        self.batch_size = cfg.hyperparameters.batch_size
        self.epochs = cfg.trainer_args.max_epochs
        self.steps_per_epoch = steps_per_epoch

    def forward(self, inputs):
        try:
            assert (x[:, 0, :, :].shape == y[:, 0, :, :].shape) & (
                x[0, 0, :, :].shape == torch.Size((self.input_height, self.input_width))
            )

        except:
            pdb.set_trace()
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        x, y, fullsize_targets = batch

        preds = self(x)
        print(f"TRAIN:  x: {x.shape} y: {y.shape}, pred: {preds.shape}, tstep: {self.tstep}")
        if self.tstep % 10 == 0:
            log_images(
                img=x[0, :, :, :].detach(),
                depth=y[0, :, :, :].detach(),
                pred=preds[0, :, :, :].detach(),
                vmin=self.min_depth,
                vmax=self.max_depth,
                step=self.tstep,
            )
        mask = torch.logical_and(
            y > self.min_depth, y < self.max_depth
        )  # perhaps also punish above maxdepth during training?
        loss = self.loss_function(preds * mask, y * mask)

        self.log("train_loss", loss)
        wandb.log(
            {"train_loss": loss, "learning_rate": self.lr_schedulers().get_last_lr()[0]},
            step=self.tstep,
        )

        fullsize_mask = torch.logical_and(
            fullsize_targets > self.min_depth, fullsize_targets < self.max_depth
        )

        masked_full_size_targets = fullsize_targets[fullsize_mask]

        resized_preds = nn.functional.interpolate(
            preds, fullsize_targets.shape[-2:], mode="bilinear", align_corners=True
        )
        masked_resized_preds = resized_preds[fullsize_mask]
        log_loss_metrics(
            preds=masked_resized_preds.detach(),
            targets=masked_full_size_targets.detach(),
            tstep=self.tstep,
            loss_prefix="train_fullsize",
        )
        self.tstep += 1
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, fullsize_targets = batch

        assert (x[:, 0, :, :].shape == y[:, 0, :, :].shape) & (
            x[0, 0, :, :].shape == torch.Size((self.input_height, self.input_width))
        )
        preds = self(x)
        print(f"VALIDATION: x: {x.shape} y: {y.shape}, pred: {preds.shape}")
        mask = torch.logical_and(y > self.min_depth, y < self.max_depth)
        loss = self.loss_function(preds * mask, y * mask)

        wandb.log({"val_loss": loss}, step=self.tstep)
        self.log("validation_loss", loss)

        fullsize_mask = torch.logical_and(
            fullsize_targets > self.min_depth, fullsize_targets < self.max_depth
        )

        masked_full_size_targets = fullsize_targets[fullsize_mask]

        resized_preds = nn.functional.interpolate(
            preds, fullsize_targets.shape[-2:], mode="bilinear", align_corners=True
        )

        masked_resized_preds = resized_preds[fullsize_mask]
        log_loss_metrics(
            preds=masked_resized_preds.detach(),
            targets=masked_full_size_targets.detach(),
            tstep=self.tstep,
            loss_prefix="val",
        )
        return loss
s

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.lr, epochs=self.epochs, steps_per_epoch=self.steps_per_epoch
        )
        opt_dict = {
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
            "optimizer": optimizer,
        }
        return opt_dict

