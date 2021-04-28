import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.utils.data


class CharacterClassifier(pl.LightningModule):
    def __init__(self, model_cls, lr: float = 0.01, kernel_size_1: int = 5,
                 kernel_size_2: int = 5):
        super().__init__()

        self.save_hyperparameters()

        self.model = model_cls(self.hparams).float()

        self.train_acc = pl.metrics.Accuracy()
        self.valid_acc = pl.metrics.Accuracy()

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.flatten()
        output = self.forward(x)
        preds = torch.argmax(output, dim=1)

        loss = nn.CrossEntropyLoss()(output, y)

        self.log('train_loss',  loss, on_epoch=True)
        self.log('train_acc', self.train_acc(preds, y),  prog_bar=True)

        return loss

    def training_epoch_end(self, outs):
        print(outs)
        #self.train_acc.compute()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.flatten()
        output = self.forward(x)
        preds = torch.argmax(output, dim=1)

        loss = nn.CrossEntropyLoss()(output, y)

        self.log('valid_loss', loss,  prog_bar=True)
        self.log('valid_acc', self.valid_acc(preds, y),  prog_bar=True)

        return {'loss': loss}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y = y.flatten()
        output = self.forward(x)
        loss = nn.CrossEntropyLoss()(output, y)

        self.log('test_loss', loss, prog_bar=True)

        return {'loss': loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams['lr'])
        return optimizer
