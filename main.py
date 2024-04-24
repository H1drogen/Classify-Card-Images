import torch
from torchvision.transforms import transforms

from models.playing_card_dataset import PlayingCardDataset

tensor1 = torch.tensor([1.0, 2.0, 3.0])
tensor2 = torch.tensor([4.0, 5.0, 6.0])

print(tensor1 + tensor2)

print(torch.cuda.is_available())

# model will expect a consistent 224x224 image
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

dataset = PlayingCardDataset('./dataset', transform)
print(len(dataset))
