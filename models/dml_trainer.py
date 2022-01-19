import numpy as np
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
from utils.auxiliaries import instantiate_from_config
from criteria import add_criterion_optim_params


class DML_Model(pl.LightningModule):
    def __init__(self, config, ckpt_path=None, ignore_keys=[]):
        super().__init__()
        config = OmegaConf.to_container(config)

        ## Init optimizer hyperparamters
        self.type_optimizer = None
        self.weight_decay = 0
        self.gamma = 0
        self.tau = 0

        ## Load model using config
        self.model = instantiate_from_config(config["Architecture"])
        self.config_arch = config["Architecture"]

        ## Init loss
        batchminer = instantiate_from_config(config["Batchmining"]) if "Batchmining" in config.keys() else None
        config["Loss"]["params"]['batchminer'] = batchminer
        self.loss = instantiate_from_config(config["Loss"])

        ## Init constom log scripts
        self.custom_logs = instantiate_from_config(config["CustomLogs"])

        ### Init metric computer
        self.metric_computer = instantiate_from_config(config["Evaluation"])

        if ckpt_path is not None:
            print("Loading model from {}".format(ckpt_path))
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list()):
        ## Load from checkpoint
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)

    def forward(self, x):
        out = self.model(x)
        if len(out) == 5:
            x, vqloss = out['embeds'], out['vq_loss'] # {'embeds': z, 'avg_features': y, 'features': x, 'extra_embeds': prepool_y, 'vq_loss': vq_loss}
            return x, vqloss
        else:
            x = out['embeds']
            return x

    def training_step(self, batch, batch_idx):
        ## Define one training step, the loss returned will be optimized
        inputs = batch[0]
        labels = batch[1]
        output = self.model(inputs)

        loss = self.loss(output['embeds'], labels, global_step=self.global_step, split="train") ## Change inputs to loss

        if self.model.VQ:
            vq_loss = output['vq_loss']
            self.log("DML_Loss", loss, prog_bar=True, logger=True, on_step=False, on_epoch=True) ## Add to progressbar
            self.log("VQ_Loss", vq_loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        else:
            self.log("Loss", loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)  ## Add to progressbar

        # compute gradient magnitude
        mean_gradient_magnitude = 0.
        if self.global_step > 0:
            for name, param in self.model.named_parameters():
                if (param.requires_grad) and ("bias" not in name) and param.grad is not None:
                    mean_gradient_magnitude += param.grad.abs().mean().cpu().detach().numpy()
        if len(output) == 5:
            return {"loss": loss+vq_loss, "av_grad_mag": mean_gradient_magnitude}
        else:
            return {"loss": loss, "av_grad_mag": mean_gradient_magnitude}

    def training_epoch_end(self, outputs):
        grad_mag_avs = np.mean([x["av_grad_mag"] for x in outputs])

        # log results
        log_data = {f"grad_mag_avs": grad_mag_avs}

        if self.loss.REQUIRES_LOGGING:
            loss_log_data = self.loss.get_log_data()
            log_data = {**log_data, **loss_log_data}

        self.log_dict(log_data, prog_bar=False, logger=True, on_step=False, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        inputs = batch[0]
        labels = batch[1]

        with torch.no_grad():
            out = self.model(inputs)
            embeds = out['embeds']  # {'embeds': z, 'avg_features': y, 'features': x, 'extra_embeds': prepool_y}
            features = out['features']


        return {"embeds": embeds, "labels": labels, "features": features}

    def validation_epoch_end(self, outputs):
        embeds = torch.cat([x["embeds"] for x in outputs]).cpu().detach()
        labels = torch.cat([x["labels"] for x in outputs]).cpu().detach()

        # perform validation
        computed_metrics = self.metric_computer.compute_standard(embeds, labels, self.device)

        # log validation results
        log_data = {"epoch": self.current_epoch}
        for k, v in computed_metrics.items():
            log_data[f"val/{k}"] = v

        print(f"\nEpoch {self.current_epoch} validation results:")
        for k,v in computed_metrics.items():
            print(f"{k}: {v}")

        self.log_dict(log_data, prog_bar=False, logger=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        to_optim = [{'params': self.model.parameters(), 'lr': self.learning_rate, 'weight_decay': self.weight_decay}]
        to_optim = add_criterion_optim_params(self.loss, to_optim)

        if self.type_optim == 'adam':
            optimizer = torch.optim.Adam(to_optim)
        elif self.type_optim == 'adamW':
            optimizer = torch.optim.AdamW(to_optim)
        else:
            raise Exception(f'[{self.type_optim}] is an unknown/not supported optimizer. Currently supported optimizer [Adam, AdamW].')

        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.tau, gamma=self.gamma)

        return [optimizer], [scheduler]

    def get_progress_bar_dict(self):
        ## Drop version name in progressbar
        tqdm_dict = super().get_progress_bar_dict()
        tqdm_dict.pop("v_num", None)
        return tqdm_dict
