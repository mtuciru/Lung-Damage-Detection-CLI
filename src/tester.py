import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import (
    jaccard_score,
    f1_score, 
    precision_score, 
    recall_score
)


class Tester:

    def __init__(self, model, device, dataloader, batch_size: int):
        self.model = model
        self.device = device
        self.dataloader = dataloader
        self.bs = batch_size

    @staticmethod
    def _dice(inp, tar):
        smooth = 1
        intersection = (inp * tar).sum()
        res = (2. * intersection + smooth) / (inp.sum() + tar.sum() + smooth)
        return res

    def run(self):
        self.model.eval()
        labels = np.array([])
        preds = np.array([])
        with torch.no_grad():
            for x, y in tqdm(self.dataloader):
                for i in range(self.bs):
                    x_ = x.to(self.device)
                    pred = self.model(x_)        
                    pred = torch.where(pred > 0.2, 1.0, 0.0)
                    preds = np.concatenate((preds, torch.flatten(pred[i]).to('cpu').numpy()))
                    labels = np.concatenate((labels, torch.flatten(y[i]).to('cpu').numpy()))
        prec = precision_score(labels, preds)
        recall = recall_score(labels, preds)
        f1 = f1_score(labels, preds)
        jac = jaccard_score(labels, preds)
        dice = self._dice(labels, preds)
        return prec, recall, f1, jac, dice
