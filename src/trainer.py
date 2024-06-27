import torch
from torch.autograd import Variable
from tqdm import tqdm


class Trainer:

    class EarlyStopper:

        def __init__(self, patience=1, min_delta=0):
            self.patience = patience
            self.min_delta = min_delta
            self.counter = 0
            self.min_validation_loss = float('inf')

        def early_stop(self, validation_loss):
            if validation_loss < self.min_validation_loss:
                self.min_validation_loss = validation_loss
                self.counter = 0
            elif validation_loss > (self.min_validation_loss + self.min_delta):
                self.counter += 1
                if self.counter >= self.patience:
                    return True
            return False

    def __init__(
            self, 
            model, 
            device, 
            train_dataloader, 
            val_dataloader,
            epochs: int,
            save_path: str,
            lr: float
        ):
        self.model = model
        self.device = device
        self.optim = torch.optim.Adam(lr=lr, params=model.parameters())
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optim, 
            mode='max', 
            factor=0.1, 
            min_lr=1e-6, 
            patience=10, 
            verbose=1, 
            cooldown=2
        )
        self.early_stopper = self.EarlyStopper(patience=30)
        self.dl_train = train_dataloader
        self.dl_val = val_dataloader
        self.epochs = epochs
        self.save_path = save_path

    @staticmethod
    def _tversky(y_true, y_pred, smooth=1, alpha=0.7):
        y_true_pos = torch.flatten(y_true)
        y_pred_pos = torch.flatten(y_pred)
        true_pos = torch.sum(y_true_pos * y_pred_pos)
        false_neg = torch.sum(y_true_pos * (1 - y_pred_pos))
        false_pos = torch.sum((1 - y_true_pos) * y_pred_pos)
        return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)
    
    def run(self):
        for epoch in tqdm(range(self.epochs)):
            self.model.train()
            for x, y in self.dl_train:
                self.optim.zero_grad()
                y = y.to(self.device)
                x = x.to(self.device)
                pred = self.model(x)
                
                pred = torch.where(pred > 0.2, 1.0, 0.0)

                l = Variable(self._tversky(y, pred), requires_grad=True)
                l.backward()
                self.optim.step()

            self.model.eval()

            for x, y in self.dl_val:
                y = y.to(self.device)
                x = x.to(self.device)
                pred = self.model(x)
                pred = torch.where(pred > 0.2, 1.0, 0.0)

                l = Variable(self._tversky(y, pred), requires_grad=True)
                l.backward()

            self.scheduler.step(l)

            if self.early_stopper.early_stop(l):
                break
            torch.save(self.model.state_dict(), self.save_path)
