import numpy as np
from itertools import chain
import torch, wandb
import pytorch_lightning as pl
from omegaconf import OmegaConf
from utils.auxiliaries import instantiate_from_config, extract_features
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

        if self.model.VQ and self.model.e_init == 'feature_clustering':
            ## init dataloader:
            tmp_dataloader = instantiate_from_config(config["Optional_dataloader"]).val_dataloader()

            ## extract features
            features = extract_features(self.model.cuda(), tmp_dataloader, self.model.VectorQuantizer.k_e)

            ## init VQ using feature clustering
            self.model.VectorQuantizer.init_codebook_by_clustering(features)

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

        if 'vq_loss' in out.keys():
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
        self.log("DML_Loss", loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)  ## Add to progressbar

        if 'vq_loss' in output.keys():
            vq_loss = output['vq_loss']
            self.log("VQ_Loss", vq_loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
            loss = loss + vq_loss

        # compute gradient magnitude
        mean_gradient_magnitude = 0.
        if self.global_step > 0:
            for name, param in self.model.named_parameters():
                if (param.requires_grad) and ("bias" not in name) and param.grad is not None:
                    mean_gradient_magnitude += param.grad.abs().mean().cpu().detach().numpy()

        out_dict = {"loss": loss, "av_grad_mag": mean_gradient_magnitude}
        if 'vq_perplexity' in output.keys():
            vq_perp = float(output['vq_perplexity'].cpu().detach().numpy())
            self.log("vq_perp", vq_perp, prog_bar=True, logger=False, on_step=False, on_epoch=True)
            out_dict['vq_perplexity'] = vq_perp

        if 'vq_cluster_use' in output.keys():
            vq_clust_use = float(output['vq_cluster_use'].cpu().detach().numpy())
            self.log("vq_cl_use", vq_clust_use, prog_bar=True, logger=False, on_step=False, on_epoch=True)
            out_dict['vq_cluster_use'] = vq_clust_use

        if 'vq_indices' in output.keys():
            vq_indices = output['vq_indices'].cpu().detach().numpy()
            out_dict['vq_indices'] = vq_indices

        return out_dict

    def training_epoch_end(self, outputs):
        grad_mag_avs = np.mean([x["av_grad_mag"] for x in outputs])

        # log results
        log_data = {f"grad_mag_avs": grad_mag_avs}

        if 'vq_perplexity' in outputs[0].keys():
            vq_perplexity_avs = np.mean([x["vq_perplexity"] for x in outputs])
            log_data = {**log_data, f"vq_perplexity": vq_perplexity_avs}

        if 'vq_cluster_use' in outputs[0].keys():
            vq_cluster_use_avs = np.mean([x["vq_cluster_use"] for x in outputs])
            log_data = {**log_data, f"vq_cluster_use": vq_cluster_use_avs}

        if 'vq_indices' in outputs[0].keys():
            vq_indices = [x["vq_indices"].tolist() for x in outputs]
            vq_indices = list(chain.from_iterable(vq_indices))
            data = [[i, vq_indices.count(i)] for i in range(self.model.n_e)]
            table = wandb.Table(data=data, columns=["codeword", "frequency"])
            # wandb.log({'histogram-codebook_usage': wandb.plot.histogram(table, "codeword")})
            wandb.log({"freq_per_codeword": wandb.plot.line(table, "codeword", "frequency", title="codeword frequency")})

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
