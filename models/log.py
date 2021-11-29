import wandb
import numpy as np
import torch.nn as nn
import torch

#ref https://docs.wandb.ai/library/log

class custom_logging:
    def __init__(self):
        self.batch = []
        self.output = []
        self.mode = ""
        self.others = []

    def update(self, batch, output, mode = "train", others = []):
        self.batch = batch
        self.output = output
        self.mode = mode
        self.others = others

    def image_prediction(self, n_ims=10):
        images = self.batch[0].clone().detach().cpu().numpy()[:n_ims]
        predictions = self.batch[1].clone().detach().cpu().numpy()[:n_ims]
        out = [wandb.Image(img[0], caption="Prediction: " + str(pred)) for (img,pred) in zip(images, predictions)]
        return {"Image Predictions": out}

    def accuracy(self):
        sm = nn.LogSoftmax()
        labels = self.batch[1]
        correct = torch.argmax(sm(self.output), dim=1) == labels
        accuracy = correct.sum() / labels.size()[0]
        return {"{}/accuracy".format(self.mode): accuracy.detach().mean()}