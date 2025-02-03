import sys
import os
from os import path
import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader

USE_GPU = torch.cuda.is_available()

class RNA_Dataset(Dataset):
    def __init__(self,data):
        self.data=data
        self.tokens={nt:i for i,nt in enumerate('ACGU')}

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sequence=[self.tokens[nt] for nt in self.data.loc[idx,'sequence']]
        sequence=np.array(sequence)
        sequence=torch.tensor(sequence)

        return {'sequence':sequence}

seq = sys.argv[1]
test_dataset=RNA_Dataset(pd.DataFrame([{'sequence': seq}]))

sys.path.append(os.environ['RIBONANZANET_PATH'])

from Network import *
import yaml

class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)
        self.entries=entries

    def print(self):
        print(self.entries)

def load_config_from_yaml(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return Config(**config)

model = RibonanzaNet(load_config_from_yaml(path.join(os.environ['RIBONANZANET_PATH'], 'configs/pairwise.yaml')))
if USE_GPU:
    model = model.cuda()
model.load_state_dict(torch.load(path.join(os.environ['RIBONANZANET_WEIGHTS_PATH'], 'RibonanzaNet.pt'), map_location='cpu'))

from tqdm import tqdm

test_preds=[]
model.eval()
for i in tqdm(range(len(test_dataset))):
    example=test_dataset[i]
    sequence=example['sequence']
    if USE_GPU:
        sequence = sequence.cuda()
    sequence = sequence.unsqueeze(0)

    with torch.no_grad():
        pred = model(sequence,torch.ones_like(sequence)).squeeze()
        if USE_GPU:
            pred = pred.cpu()
        test_preds.append(pred.numpy())

pred_2a3 = test_preds[0][:,0]
pred_dms = test_preds[0][:,1]

print('2a3:' + ','.join(str(x) for x in pred_2a3.tolist()))
print('dms:' +','.join(str(x) for x in pred_dms.tolist()))
