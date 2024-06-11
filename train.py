"""
File that trains a wake word detection model that detects the word 
"Arise" using MFCCs as input. 
Sources: 
- https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files
- 
"""

import torch 
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import constants as const
import pandas as pd 
import matplotlib.pyplot as plt
from typing import Tuple

device = 'cuda' if torch.cuda.is_available() else 'cpu'

"""
Hyperparameters
"""
epochs = 1000
eval_interval = 100
batch_size = 16
lr = 0.001
dropout = 0.5

def evaluate(model, dataloader) -> Tuple[float, float]: 
    """
    Evaluates model on given dataloader. Returns the loss and accuracy. 
    """
    model.eval()
    with torch.no_grad():
        correct, total = 0, 0 
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            
            outputs, loss = model(X, y)
            prediction = torch.max(outputs)

            total += y.size(0)
            correct += (prediction == y).sum().item()

        accuracy = correct / total
        # print(f'Accuracy: {accuracy * 100:.2f}%')
    model.train()
    return round(loss.item(), 5), round(accuracy * 100, 5)

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
    def __init__(self, in_dim, dropout_rate=0) -> None:
        super().__init__()
        # self.flatten = nn.Flatten() # Flatten 2D input MFCCs to 1D
        self.l1 = nn.Linear(in_dim, 256)
        self.r1 = nn.ReLU()
        self.d1 = nn.Dropout(dropout_rate)
        self.l2 = nn.Linear(256, 256)
        self.r2 = nn.ReLU()
        self.d2 = nn.Dropout(dropout_rate)
        self.l3 = nn.Linear(256, 2)
        # self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, x, y):
        # x = self.flatten(x)
        x = self.d1(self.r1(self.l1(x)))
        x = self.d2(self.r2(self.l2(x)))
        x = self.l3(x)
        loss = F.cross_entropy(x, y)
        x = F.softmax(x, dim=-1)
        return x, loss


model = WakeWord(const.NUM_MFCC, dropout_rate=dropout)
model.to(device) # move model to device
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

train_dataset = MFCCDataset('./Data/train/train_mfccs.csv')
train_dataloader = DataLoader(train_dataset, 
                              batch_size=batch_size, 
                              shuffle=True)

# Train Model 
print("Starting training...")
for epoch in range(epochs):
    if epoch % eval_interval == 0:
        loss, acc = evaluate(model, train_dataloader)
        print("Epoch: ", epoch, "Training Loss: ", loss, "Training Accuracy: ", acc)

    for X, y in train_dataloader:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        logits, loss = model(X, y)
        loss.backward()
        optimizer.step()

    # print(f"Epoch {epoch} loss: {loss.item()}")

# Test model 
print("Testing model...")
test_dataset = MFCCDataset('./Data/test/test_mfccs.csv')
test_dataloader = DataLoader(test_dataset,
                             batch_size=batch_size, 
                             shuffle=False)
test_loss, test_acc = evaluate(model, test_dataloader)
print("Test Loss: ", test_loss, "Test Accuracy: ", test_acc)


# torch.save(model.state_dict(), './model.pth')