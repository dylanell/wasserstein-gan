"""
PyTorch dataset classes.
Reference:
    https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
"""

import torch
from PIL import Image

class ImageDataset(torch.utils.data.Dataset):
    """
    Make a PyTorch dataset from a dataframe of image files and labels.
    """

    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # read image and get label
        # NOTE: image must be PIL image for standard PyTorch transforms
        image = Image.open(self.df['Filename'].iloc[idx])
        label = self.df['Label'].iloc[idx]

        # apply any image transform
        if self.transform:
            image = self.transform(image)

        # construct packaged sample
        data = {'image': image, 'label': label}

        return data

class RawDataset(torch.utils.data.Dataset):
    """
    Make a Pytorch dataset from provided samples and labels.
    """

    def __init__(self, samples, labels, transform=None):
        self.samples = samples
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # get sample and label
        sample = self.samples[idx]
        label = self.labels[idx]

        # apply sample transforms
        if self.transform:
            sample = self.transform(sample)

        # get sample and label by idx
        data = {'sample': sample, 'label': label}

        return data
