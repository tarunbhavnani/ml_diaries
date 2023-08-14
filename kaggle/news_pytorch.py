# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 16:51:58 2023

@author: tarun
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchtext import data
from torchtext.datasets import AG_NEWS

# Set random seeds for reproducibility
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True

# Define the CNN model
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_filters, filter_sizes, num_classes):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([nn.Conv2d(1, num_filters, (fs, embedding_dim)) for fs in filter_sizes])
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)  # Add a channel dimension
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in x]
        x = torch.cat(x, 1)
        x = self.fc(x)
        return x

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define hyperparameters
vocab_size = 25000  # Vocabulary size (adjust according to your dataset)
embedding_dim = 100  # Dimensionality of word embeddings
num_filters = 100  # Number of filters (output channels) in each convolutional layer
filter_sizes = [3, 4, 5]  # Sizes of filters (widths of the convolutional kernels)
num_classes = 4  # Number of classes (adjust according to your dataset)
batch_size = 64  # Batch size
learning_rate = 0.001  # Learning rate
num_epochs = 10  # Number of training epochs

# Load and preprocess the AG News dataset
TEXT = data.Field(lower=True, batch_first=True, fix_length=200)
LABEL = data.LabelField(dtype=torch.float)
train_data, test_data = AG_NEWS.splits(TEXT, LABEL)
TEXT.build_vocab(train_data, max_size=vocab_size)
LABEL.build_vocab(train_data)

# Create data iterators
train_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, test_data),
    batch_size=batch_size,
    device=device
)

# Create the CNN model
model = TextCNN(vocab_size, embedding_dim, num_filters, filter_sizes, num_classes).to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
model.train()
for epoch in range(num_epochs):
    for batch in train_iterator:
        text = batch.text
        label = batch.label - 1  # Adjust labels to start from 0
        optimizer.zero_grad()
        output = model(text).squeeze(1)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

# Evaluate the model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch in test_iterator:
        text = batch.text
        label = batch.label - 1  # Adjust labels to start from 0
        output = model(text).squeeze(1)
        predicted = torch.argmax(output, dim=1)
        correct += (predicted == label).sum().item()
        total += label.size(0)

accuracy = correct / total
print(f"Test Accuracy: {accuracy * 100:.2f}%")


