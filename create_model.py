import torch
from torchvision.transforms import transforms

from models.card_classifier import SimpleCardClassifier
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


train_folder = './dataset/archive/train'
validation_folder = './dataset/archive/valid'
test_folder = './dataset/archive/test'

# create datasets
dataset = PlayingCardDataset('./dataset', transform)
train_dataset = PlayingCardDataset(train_folder, transform)
validation_dataset = PlayingCardDataset(validation_folder, transform)
test_dataset = PlayingCardDataset(test_folder, transform)

# check the first item
image, label = dataset[0]

# create dataloaders
# Dataloader will handle batching and shuffling. batching is the process of combining multiple samples and shuffling is the process of randomizing the order of the samples. shuffling is done when training a model to prevent the model from learning the order of the samples.
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=32, shuffle=False)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

model = SimpleCardClassifier()

# shows the structure of the model accepts the input data, which is a 3x224x224 image. Returns [batch_size, num_classes] tensor with .shape
print(model(image).shape)




