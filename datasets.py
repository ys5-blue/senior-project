import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pandas as pd

class SpambaseDataset(Dataset):
    def __init__(self, path='data/spambase/spambase.data'):
        super().__init__()
        dataframe = pd.read_csv(path, header=None)
        self.xs = torch.tensor(dataframe.iloc[:, :-1].values, dtype=torch.float32)
        self.ys = torch.tensor(dataframe.iloc[:, -1].values, dtype=torch.int64)

    def __len__(self):
        return self.xs.size(0)

    def __getitem__(self, index):
        return self.xs[index], self.ys[index]

def get_spambase_dataloader():
    return DataLoader(SpambaseDataset(),
                      batch_size=64,
                      shuffle=True)



class EnronDataset(Dataset):
    def __init__(self, path='data/enron1'):
        super().__init__()
        path = Path(path)
        spams = (path / 'spam').glob('*.txt')
        hams = (path / 'ham').glob('*.txt')

        self.files = ([(ham, 0) for ham in hams] +
                      [(spam, 1) for spam in spams])
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        filename, label = self.files[index]
        with open(filename, 'r', encoding='latin1') as f:
            text = f.read()
        return text, label

def get_email_dataloader():
    return DataLoader(EnronDataset(),
                      batch_size=16,
                      shuffle=True)

class MalwareDataset(Dataset):
    def __init__(self, path='data/big15'):
        super().__init__()
        self.path = Path(path)
        dataframe = pd.read_csv(self.path / 'trainLabels.csv')
        self.labels = dataframe['Class']
        self.hashes = dataframe['Id']

    def __len__(self):
        return len(self.hashes)

    def __getitem__(self, index):
        filename = self.path / 'train' / (self.hashes[index] + '.bytes')
        with open(filename) as f:
            bytestr = ''
            for line in f:
                if '?' in line: continue
                # Trim off the address part
                bytestr += line[9:]
        buffer = bytes.fromhex(bytestr)
        if len(buffer) == 0:
            return self.__getitem__(index - 1)
        tensor = torch.frombuffer(buffer, dtype=torch.uint8)
        return tensor.long(), self.labels[index] - 1

def get_malware_dataloader():
    return DataLoader(MalwareDataset(),
                      batch_size=1,
                      shuffle=True)
