from torch.utils.data import Dataset
from torchvision import transforms


class HousesDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, house_data, house_target, transform=None):
        """
        Args:
            houses_data (numpy arry): the dataset
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dataset = house_data
        self.targets = house_target
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        target = self.targets[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample, target