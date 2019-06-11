import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
from torch.utils.data.dataloader import default_collate

class ImageFolderWithPaths(Dataset):
    """
    Custom dataset that includes image file paths.
    """

    def __init__(self, root, paths, loader=default_loader, transform=None):
        if len(paths) == 0:
            raise (RuntimeError("Found 0 files in " + root))
        self.root = root
        self.loader = loader
        self.samples = paths
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, path)
        """
        path = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, path

    def __len__(self):
        return len(self.samples)


def collate(batch):
    if isinstance(batch[0], torch.Tensor):
        return default_collate(batch)
    elif isinstance(batch[0], tuple):
        transposed = zip(*batch)
        return [collate(samples) for samples in transposed]
    return batch
