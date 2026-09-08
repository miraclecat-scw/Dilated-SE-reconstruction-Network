"""
    @Author: SiyuChen
    @Date: 2024-12-30
    @Description: Early stopping for training with torch.
    @Reference: https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
    @License: MIT
    @File: early_stopping.py
"""

import torch
import os
import numpy as np

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, val_loss, model_state, epoch, save_path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self._save_checkpoint(val_loss, model_state, epoch, save_path)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self._save_checkpoint(val_loss, model_state, epoch, save_path)
            self.counter = 0

    def _save_checkpoint(self, val_loss, model_state, epoch, save_path):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model_state, os.path.join(save_path, f"checkpoint_{epoch}_final_{val_loss:.6f}.pth.tar"))
        self.val_loss_min = val_loss