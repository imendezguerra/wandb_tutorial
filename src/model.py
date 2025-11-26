"""Network"""

import random
import numpy as np
import torch
from torch import nn
from contextlib import contextmanager

class MLP(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, output_dim=2, dropout=0.0):
        super().__init__()
        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = torch.relu(self.l1(x))
        x = self.dropout(x)
        x = torch.relu(self.l2(x))
        x = self.dropout(x)
        return self.l3(x)


def get_device():
    # Get device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    return device

def set_random_seed(rand_seed):
    """Set random seeds in places that use random generators."""
    # Python random
    random.seed(rand_seed)
    # NumPy
    np.random.seed(rand_seed)
    # PyTorch CPU
    torch.manual_seed(rand_seed)
    # PyTorch GPU (CUDA)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(rand_seed)
        torch.cuda.manual_seed_all(rand_seed)


@contextmanager
def mode_check(mode):
    if mode == "train":
        with torch.enable_grad():
            yield
    else:  # "eval", "test", "validation"
        with torch.no_grad():
            yield


def run_one_epoch(model, loader, optimizer, epoch, mode):
    # Set model mode
    if mode:
        model.train()
    else:
        model.eval()
    
    # Initialise variables
    total_loss = 0.0
    correct = 0
    total = 0
    aux = {
        'loss_batch': [],
        'batch_epoch': [],
        'batch_step': [],
    }

    with mode_check(mode):
        for batch_idx, (X_batch, y_batch) in enumerate(loader):

            # Run model for every batch
            logits = model(X_batch)
            loss = nn.functional.cross_entropy(logits, y_batch)

            # Compute gradients and update model
            if mode == 'train':
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Store loss batch
                if batch_idx % 10 == 0:
                    aux['loss_batch'].append(loss.item())
                    aux['batch_epoch'].append(epoch)
                    aux['batch_step'].append(epoch * len(loader) + batch_idx)

            # Compute loss
            total_loss += loss.item() * X_batch.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += X_batch.size(0)

    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc, aux