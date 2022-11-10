from wordfreq_model import WordFreqModel, WordFreqLinearModel
from spam_model import SpamModel
from datasets import get_spambase_dataloader, get_enron_dataloader
from pytorch_lightning import Trainer

GPUS = 0

def train(modelname):
    if modelname == 'wordfreq':
        model = WordFreqModel()
        # model = WordFreqLinearModel()
        dataloader = get_spambase_dataloader()
        max_epochs = 100

    if modelname == 'spamformer':
        model = SpamModel()
        dataloader = get_enron_dataloader()
        max_epochs = 10

    trainer = Trainer(gpus=GPUS, max_epochs=max_epochs)
    trainer.fit(model, dataloader)

def infer(modelname, checkpoint, sequences):
    from os.path import exists

    if modelname == 'spamformer':
        if checkpoint is not None and exists(checkpoint):
            model = SpamModel.load_from_checkpoint(checkpoint)
        else:
            model = SpamModel()
        result = model.forward(sequences)
        print(result)


if __name__ == '__main__':
    from argparse import ArgumentParser

    p = ArgumentParser()
    p.add_argument('--train', action='store_true')
    p.add_argument('--infer', action='store_true')
    p.add_argument('--model', type=str)
    p.add_argument('--checkpoint', type=str, nargs='?')
    p.add_argument('--text', type=str, nargs='?')
    args = p.parse_args()
    
    if args.train:
        train(args.model)
    elif args.infer:
        infer(args.model, args.checkpoint, [args.text])
    else:
        p.print_usage()
