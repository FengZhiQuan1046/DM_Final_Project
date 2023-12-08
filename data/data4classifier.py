import pandas as pd
# from torch.utils.data import Dataset
import json
from tqdm import tqdm
from copy import copy
import re


class data4classifier:
    def __init__(self, dir, preprocess=True) -> None:
        with open(dir, 'r') as f:
            self.data = json.load(f)

        if preprocess:
            self.preprocess()

    def preprocess(self):
        pass
            # print()

    # def __getitem__(self, index):
    #     return self.data[index]
    
    # def __len__(self):
    #     return len(self.data)

if __name__ == '__main__':
    # from params import *
    dc = data4classifier("./dataset/features.json")