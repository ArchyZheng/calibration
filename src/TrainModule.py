import pytorch_lightning as pl
import torch.optim.optimizer
import torch.nn as nn

class TrainModule(pl.LightningModule):
    def __init__(self, loss, model):
        super().__init__()
        self.model = model
        self.loss = loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        location, target = batch
        prediction = self.model(location[0])
        loss = self.loss(prediction, target)
        self.log("train_loss:", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        location, target = batch
        prediction = self.model(location[0])
        loss = self.loss(prediction, target)
        self.log("validation_loss:", loss, on_epoch=True)
        return loss
