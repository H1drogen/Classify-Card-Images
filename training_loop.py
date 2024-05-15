import torch
from torch import nn

from create_model import model, transform, dataset, dataloader
from models.playing_card_dataset import PlayingCardDataset

# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


