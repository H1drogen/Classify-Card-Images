import torch
from torch import nn

from create_model import model, train_dataloader, validation_dataloader

# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


num_epochs = 10
train_losses = []
validation_losses = []

for epoch in range(num_epochs):
    # Training
    model.train()
    running_loss = 0.0
    for images, labels in train_dataloader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    train_loss = running_loss / len(train_dataloader.dataset)
    train_losses.append(train_loss)

    # Validation
    model.eval()
    running_loss = 0.0
    # make sure we don't update the model weights
    with torch.no_grad():
        for images, labels in validation_dataloader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
    validation_loss = running_loss / len(validation_dataloader.dataset)
    validation_losses.append(validation_loss)

    print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {train_loss}, Validation Loss: {validation_loss}')