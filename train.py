"""
Train a Wasserstein GAN on the MNIST dataset.
"""

import yaml

from util.pytorch_utils import build_image_dataset
from util.data_utils import generate_df_from_image_dataset
from model.wasserstein_gan import WassersteinGAN

def main():
    # parse configuration file
    with open('config.yaml', 'r') as fp:
        config = yaml.load(fp, Loader=yaml.FullLoader)

    # generate filenames/labels df from image data directory
    data_dict = generate_df_from_image_dataset(config['dataset_directory'])

    # build training dataloader
    train_set, train_loader = build_image_dataset(
        data_dict['train'],
        image_size=config['input_dimensions'][:-1],
        batch_size=config['batch_size'],
        num_workers=config['number_workers']
    )

    # initialize gan model
    gan = WassersteinGAN(config)

    # train gan on training set
    gan.train(train_loader, config['number_epochs'])

if __name__ == '__main__':
    main()
