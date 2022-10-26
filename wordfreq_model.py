import torch
from torch import nn, optim
from pytorch_lightning import LightningModule, Trainer

# TODO: change when we actually have GPUs
GPUS = 0

class WordFreqModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
                nn.Linear(57, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 1))
        self.loss_fn = nn.BCEWithLogitsLoss()
    
    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return optim.SGD(self.parameters(), lr=1e-3)

    def training_step(self, batch, batch_idx):
        features, labels = batch
        predicted_logits = self.model(features).view(-1)
        loss = self.loss_fn(predicted_logits, labels.float())
        accuracy = ((predicted_logits >= 0.5).int() == labels).float().mean()
        self.log('acc', accuracy, prog_bar=True)
        return loss

class WordFreqLinearModel(WordFreqModel):
    def __init__(self):
        super().__init__()
        self.model = nn.Linear(57, 1)
