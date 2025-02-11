import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score

from model_development.utils import logger

class EarlyStopping:
    """
    Stops training if the validation metric doesn't improve after a given patience.
    """
    def __init__(self, patience=5, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict()
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                logger.info("EarlyStopping counter: %d out of %d", self.counter, self.patience)
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if self.verbose:
                logger.info("Validation score improved (%.4f --> %.4f).", self.best_score, score)
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.counter = 0

def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return f1_score(all_labels, all_preds, average='macro')