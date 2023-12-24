import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import copy

from torch.utils.data import Dataset


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.2):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, 2 * hidden_size)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(2 * hidden_size, 1)

        # Weight initialization
        init.xavier_uniform_(self.fc1.weight)
        init.xavier_uniform_(self.fc2.weight)
        
        #initialize bias with zeros
        init.zeros_(self.fc1.bias)
        init.zeros_(self.fc2.bias)

        # weight decay
        self.fc1.weight_decay = 0.1

    def forward(self, x):
        x = F.relu(self.dropout1(self.fc1(x)))
        x = F.relu(self.dropout2(self.fc2(x)))
        x = self.fc3(x)
        return x


class EarlyStopper:
    def __init__(
        self, patience: int = 1, min_delta: float = 0.0, verbose: bool = False
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_validation_loss = float("inf")
        self.best_model_weights = None
        self.verbose = verbose

    def early_stop(self, validation_loss, model):
        if self.verbose:
            print(f"Early Stopping counter: {self.counter} out of {self.patience}")
        if validation_loss < self.best_validation_loss - self.min_delta:
            self.best_validation_loss = validation_loss
            self.counter = 0
            self.save_best_weights(model)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

    def save_best_weights(self, model):
        self.best_model_weights = copy.deepcopy(model.state_dict())

    def restore_best_weights(self, model):
        model.load_state_dict(self.best_model_weights)


class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return (self.x[index], self.y[index])

    def __len__(self):
        return len(self.x)


def train_regression(model, train_loader, optimizer, criterion, device):
    model.to(device).train()
    train_batch_loss = []

    for x_train, y_train in train_loader:
        x_train, y_train = x_train.to(device), y_train.to(device)

        optimizer.zero_grad()

        y_hat = model(x_train)
        loss = criterion(y_hat, y_train)
        loss.backward()

        optimizer.step()

        train_batch_loss.append(loss.item())

    train_loss = sum(train_batch_loss) / len(train_batch_loss)

    return train_loss


def validate_regression(model, val_loader, criterion, device):
    model.to(device).eval()
    val_batch_loss = []

    with torch.no_grad():
        for x_val, y_val in val_loader:
            x_val, y_val = x_val.to(device), y_val.to(device)

            y_hat_val = model(x_val)
            loss_val = criterion(y_hat_val, y_val)

            val_batch_loss.append(loss_val.item())

    val_loss = sum(val_batch_loss) / len(val_batch_loss)

    return val_loss


def evaluate(model, data_loader, device):
    model.eval()
    predictions = []

    with torch.no_grad():
        for x_test, y_test in data_loader:
            x_test, y_test = x_test.to(device), y_test.to(device)
            y_hat = model(x_test)
            predictions.append(y_hat)

    return predictions


def predict(model, data_loader, device):
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in data_loader:
            x_batch = batch.to(device)
            y_hat = model(x_batch)
            predictions.append(y_hat.cpu())

    predictions = torch.cat(predictions, dim=0).squeeze().numpy()
    return predictions
