"""
Train a Wasserstein GAN on the MNIST dataset.
"""

import argparse

# relative imports
from cnn import CNN
from transpose_cnn import TransposeCNN
from dataloader_utils import make_mnist_dataloaders
from wasserstein_gan import WassersteinGAN

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--bs', type=int, default=64, help='batch size')
    parser.add_argument('--nw', type=int, default=1, help='number of dataloader workers')
    parser.add_argument('--ne', type=int, default=20, help='number of training epochs')
    parser.add_argument('--zdim', type=int, default=128, help='z sample dimension')
    parser.add_argument('--ntrain', type=int, default=60000, help='number of training samples')
    parser.add_argument('--ntest', type=int, default=10000, help='number of testing samples')
    parser.add_argument('--nchan', type=int, default=1, help='sample channel dimension')
    parser.add_argument('--name', type=str, default='wgan', help='model name')
    parser.add_argument('--v', type=bool, default=False, help='verbose flag')
    parser.add_argument('--ld', type=str, default='/tmp/', help='log and other output directory')
    parser.add_argument('--dd', type=str, default='/tmp/mnist_data/', help='mnist data directory')
    args = parser.parse_args()

    # initialize gan model
    gan = WassersteinGAN(args)

    # intialize MNIST dataloaders
    train_loader, test_loader = make_mnist_dataloaders(
        batch_size=args.bs,
        num_workers=args.nw,
        data_dir=args.dd
    )

    # train gan on training set
    gan.train(train_loader, args.ne)

if __name__ == '__main__':
    main()
