"""
Train a Wasserstein GAN on the MNIST dataset.
"""

import argparse
import yaml

from util.pytorch_utils import build_image_dataset
from util.data_utils import generate_df_from_image_dataset
from model.suppressed_wasserstein_gan import SuppressedWassersteinGAN

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config_file', default='config/supp_config.yaml',
        help='path to configuration yaml file')
    args = parser.parse_args()
    args = parser.parse_args()

    # parse configuration file
    with open(args.config_file, 'r') as fp:
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

    # build testing dataloader
    test_set, test_loader = build_image_dataset(
        data_dict['test'],
        image_size=config['input_dimensions'][:-1],
        batch_size=config['batch_size'],
        num_workers=config['number_workers']
    )

    # initialize gan model
    gan = SuppressedWassersteinGAN(config)

    # train gan on training set
    gan.train(train_loader, config['number_epochs'])

if __name__ == '__main__':
    main()
