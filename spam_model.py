import torch
from torch import nn, optim
from pytorch_lightning import LightningModule, Trainer

from transformers import AdamW, AutoTokenizer, AutoModelForSequenceClassification

class SpamModel(LightningModule):
    def __init__(self, modelname='distilbert-base-uncased'):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(modelname)
        self.model = AutoModelForSequenceClassification.from_pretrained(modelname)

    def forward(self, sequences):
        batch = self.tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
        logits = self.model(**batch).logits
        probs = torch.softmax(logits, dim=-1)
        return probs[:, 1].detach()

    def configure_optimizers(self):
        return AdamW(self.parameters())

    def training_step(self, batch, batch_idx):
        sequences, labels = batch
        batch = self.tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
        batch['labels'] = labels
        # m = self.model(**batch)
        loss = self.model(**batch).loss

        # accuracy = ((predicted_logits >= 0.5).int() == labels).float().mean()
        # self.log('acc', accuracy, prog_bar=True)
        return loss