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

# create a dataset
dataset = PlayingCardDataset('./dataset', transform)
print(len(dataset))

# get the first item
image, label = dataset[0]

# create a dataloader, which will handle batching and shuffling. batching is the process of combining multiple samples and shuffling is the process of randomizing the order of the samples. shuffling is done when training a model to prevent the model from learning the order of the samples.
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

