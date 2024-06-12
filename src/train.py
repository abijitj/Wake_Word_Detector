"""
File that trains a wake word detection model that detects the word 
"Arise" using MFCCs as input. 
Sources: 
- https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files
- https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html
- https://pytorch.org/tutorials/beginner/saving_loading_models.html
"""

import torch 
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import constants as const
import pandas as pd 
from typing import Tuple
import statistics as stats
import sys

device = 'cuda' if torch.cuda.is_available() else 'cpu'

"""
Hyperparameters
"""
epochs = 1000
eval_interval = 100
batch_size = 16
lr = 0.0001
dropout = 0.5

def evaluate(model, dataloader, criterion) -> Tuple[float, float, float]: 
    """
    Evaluates model on given dataloader. Returns the loss and accuracy.  
    """
    model.eval()
    with torch.no_grad():
        correct, total = 0, 0 
        losses = []
        for X, y in dataloader:
            # print(X.shape, y.shape)
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = criterion(logits, y.unsqueeze(1).float()) # BCELoss expects 2D tensor (batch_size, 1) for y

            total += y.size(0) # increment by batch size
            predictions = (logits > const.THRESHOLD).float()
            correct += (predictions == y.unsqueeze(1)).sum().item()
            losses.append(loss.item())

        accuracy = correct / total
        # print(correct, total, accuracy)
    model.train()
    return (round(stats.mean(losses), 5), 
            round(accuracy * 100, 5))

class MFCCDataset(Dataset):
    """
    Custom PyTorch Dataset that loads MFCCs and labels from a pandas dataframe.
    """
    def __init__(self, dir_path):
        self.data = pd.read_pickle(dir_path)
        self.inputs = self.data['mfcc']
        self.labels = self.data['label']
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]

class WakeWord(nn.Module): 
    """
    A basic Wake word detection model that uses MFCCs as input and outputs a 
    binary classification.
    """
    def __init__(self, in_dim, dropout_rate=0) -> None:
        super().__init__()
        # self.flatten = nn.Flatten() # Flatten 2D input MFCCs to 1D
        self.l1 = nn.Linear(in_dim, 256)
        self.d1 = nn.Dropout(dropout_rate)
        # self.l2 = nn.Linear(256, 256)
        # self.d2 = nn.Dropout(dropout_rate)
        # self.l3 = nn.Linear(256, 100)
        # self.d3 = nn.Dropout(dropout_rate)
        self.l4 = nn.Linear(256, 1) # one output for binary classification
    
    def forward(self, x):
        # x = self.flatten(x)
        x = self.d1(F.relu(self.l1(x)))
        # x = self.d2(F.relu(self.l2(x)))
        # x = self.d3(F.relu(self.l3(x)))
        x = torch.sigmoid(self.l4(x))
        return x


def __main__(): 
    model = WakeWord(const.NUM_MFCC, dropout_rate=dropout)
    model.to(device) # move model to device
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.1, 0.9], device=device)) # more negative samples than positive samples
    criterion = nn.BCELoss()

    train_dataset = MFCCDataset('./Data/train/train_mfccs.csv')
    train_dataloader = DataLoader(train_dataset, 
                                batch_size=batch_size, 
                                shuffle=True)
    test_dataset = MFCCDataset('./Data/test/test_mfccs.csv')
    test_dataloader = DataLoader(test_dataset,
                                batch_size=batch_size, 
                                shuffle=False)

    # Train Model 
    print("Starting training...")
    for epoch in range(epochs):
        if epoch % eval_interval == 0:
            train_loss, train_acc = evaluate(model, train_dataloader, criterion)
            test_loss, test_acc = evaluate(model, test_dataloader, criterion)
            print("Epoch: ", epoch, 
                "Training Loss: ", train_loss, 
                "Training Accuracy: ", train_acc,
                "Test Loss: ", test_loss,
                "Test Accuracy: ", test_acc)

        for X, y in train_dataloader:
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(X) # forward pass
            loss = criterion(logits, y.unsqueeze(1).float()) 
            loss.backward() # backward pass / backpropogation
            optimizer.step() # update weights

    # Number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    print("Number of parameters: ", num_params)

    # Save model
    if sys.argv[1] == "save":
        if len(sys.argv) < 3: 
            print("Please provide a model name to save.")
            return
        torch.save(model.state_dict(), "./models/" + sys.argv[2] + ".pth")

if __name__ == "__main__":
    __main__()