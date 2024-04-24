import os

from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder


class PlayingCardDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.data = ImageFolder(root_dir, transform)
        # self.root_dir = root_dir
        # self.transform = transform
        # self.file_list = os.listdir(root_dir)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # img_name = os.path.join(self.root_dir, self.file_list[idx])
        # image = Image.open(img_name)
        # label = self.file_list[idx].split('_')[0]
        # if self.transform:
        #     image = self.transform(image)
        # return image, label
        return self.data[idx]

    @property
    def classes(self):
        return self.data.classes

