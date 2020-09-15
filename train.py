"""
Train a Wasserstein GAN on the MNIST dataset.
"""

import argparse
import yaml

from util.dataloader_utils import make_mnist_dataloaders
from model.wasserstein_gan import WassersteinGAN

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config_file', default='config/config.yaml',
        help='path to configuration yaml file')
    args = parser.parse_args()

    # parse configuration file
    with open(args.config_file, 'r') as fp:
        config = yaml.load(fp, Loader=yaml.FullLoader)

    # initialize gan model
    gan = WassersteinGAN(config)

    # intialize MNIST dataloaders
    train_loader, test_loader = make_mnist_dataloaders(
        batch_size=config['batch_size'],
        num_workers=config['number_workers'],
        data_dir=config['dataset_directory']
    )

    # train gan on training set
    gan.train(train_loader, config['number_epochs'])

if __name__ == '__main__':
    main()
