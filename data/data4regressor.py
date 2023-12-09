import pandas as pd
# from torch.utils.data import Dataset
import json
from logging import getLogger
from tqdm import tqdm
from copy import copy
import re
from catboost import Pool
from sklearn.model_selection import train_test_split
logger = getLogger(__name__)
cat_feats = ['id']
class Data4Regressor:
    def __init__(self, dir, preprocess=True) -> None:
        with open(dir, 'r') as f:
            self.data = json.load(f)
        self.data = {k: [self.data[i][k] for i in range(len(self.data))] for k in self.data[0]}
        self.data = pd.DataFrame(self.data)
        if preprocess:
            self.preprocess()

    def preprocess(self):
        pass
    
    def pack_to_catboost(self):

        global cat_feats
        def split_xy(data):
            y = data.label
            x = data.drop(labels="label", axis=1)
            x = x.drop(labels="text", axis=1)
            return x, y
        train, val = train_test_split(self.data, test_size=0.1, random_state=114514)
        trainx, trainy = split_xy(train)
        valx, valy = split_xy(val)
        # if self.split in ["train", "trainval"]:
        train = Pool(data=trainx,
                    label=trainy,
                    cat_features=cat_feats)
        
        val = Pool(data=valx,
                    # label=valy,
                    cat_features=cat_feats)
        logger.info(msg=f"Data has been packed.")

        return train, {'data':val, 'label':valy}

if __name__ == '__main__':
    # from params import *
    dc = Data4Regressor("./dataset/features.json")
    a, b = dc.pack_to_catboost()
    print(a, b)