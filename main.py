import torch

from models.playing_card_dataset import PlayingCardDataset

tensor1 = torch.tensor([1.0, 2.0, 3.0])
tensor2 = torch.tensor([4.0, 5.0, 6.0])

print(tensor1 + tensor2)

print(torch.cuda.is_available())

dataset = PlayingCardDataset('./dataset', transform=None)
print(len(dataset))
