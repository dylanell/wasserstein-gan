"""
Pytorch Dataloader utils.
"""

import torch
import torchvision
from torchvision.transforms import transforms

from util.pytorch_datasets import ImageDataset

# Given a dataframe of the form [img_paths, labels], construct a TensorFlow
# dataset object and perform all of the standard image dataset processing
# functions (resizing, standardization, etc.).
def build_image_dataset(
        dataframe, image_size=(32, 32), batch_size=64, num_workers=1):
    # define the transform chain to process each sample
    # as it is passed to a batch
    #   1. resize the sample (image) to 32x32 (h, w)
    #   2. convert resized sample to Pytorch tensor
    #   3. normalize sample values (pixel values) using
    #      mean 0.5 and stdev 0,5; [0, 255] -> [0, 1]
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # create dataset
    dataset = ImageDataset(dataframe, transform=transform)

    # create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    return dataset, dataloader
