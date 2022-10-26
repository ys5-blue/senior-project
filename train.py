from wordfreq_model import WordFreqModel, WordFreqLinearModel
from datasets import get_spambase_dataloader
from pytorch_lightning import Trainer

GPUS=0
def train():
    model = WordFreqModel()
    # model = WordFreqLinearModel()
    dataloader = get_spambase_dataloader()

    trainer = Trainer(gpus=GPUS, max_epochs=100)
    trainer.fit(model, dataloader)


if __name__ == '__main__':
    from argparse import ArgumentParser

    p = ArgumentParser()
    p.add_argument('--train', action='store_true')
    args = p.parse_args()
    
    if args.train:
        train()
    else:
        p.print_usage()
