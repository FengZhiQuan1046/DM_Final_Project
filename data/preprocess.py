import json
from tqdm import tqdm
from multiprocessing import Pool
import math
import os
import pandas as pd

def key2val(k):
    return data.iloc[k].to_dict()

if __name__ == '__main__':
    # from params import *
    data = None
    root = "./dataset/original"
    save = "./dataset/processed"
    for each in tqdm(os.listdir(root)):
        data = pd.read_csv(os.path.join(root, each))#.transpose().to_dict()
        with Pool(1) as p:
            data = p.map(key2val, list(range(data.shape[0])))
        # data = [data[k] for k in data]
        fname = os.path.splitext(each)[0]
        # if not os.path.exists(os.path.join(save, fname)):
        #     os.mkdir(os.path.join(save, fname))
        # for i in range(32):
        with open(os.path.join(save, fname+f".json"), 'w') as f:
            json.dump(data, f)


    with open("./dataset/processed/train_logs.json", 'r') as f:
        data = json.load(f)
    with open("./dataset/processed/train_scores.json", 'r') as f:
        labels = json.load(f)
    labels = {each['id'] : each['score'] for each in labels}
    data = {k : [data[i][k] for i in tqdm(range(len(data)))] for k in data[0]}
    #[{k: self.data[i][k] for k in self.data[i]} for i in tqdm(range(len(self.data)))]
    # for i in tqdm(range(len(self.data['id']))):
    data['score'] = [labels[data['id'][i]] for i in tqdm(range(len(data['id'])))]
    df = pd.DataFrame(data)
    df.to_csv('./dataset/train.csv', index=False)