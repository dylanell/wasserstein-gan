"""
Convolutional neural network implemented as Pytorch nn.Module.
"""

import torch

class CNN(torch.nn.Module):
    # initialize the base class and define network layers
    def __init__(self, in_chan, out_dim, out_act=None):
        # run base initializer
        super(CNN, self).__init__()

        # set activation functions
        self.out_act = out_act

        # define convolutional layers
        self.conv_1 = torch.nn.Conv2d(in_chan, 32, 3, stride=1, padding=1)
        self.conv_2 = torch.nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv_3 = torch.nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv_4 = torch.nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.conv_5 = torch.nn.Conv2d(256, 512, 3, stride=2, padding=1)

        # define layer norm layers
        self.norm_1 = torch.nn.LayerNorm([32, 32, 32])
        self.norm_2 = torch.nn.LayerNorm([64, 16, 16])
        self.norm_3 = torch.nn.LayerNorm([128, 8, 8])
        self.norm_4 = torch.nn.LayerNorm([256, 4, 4])
        self.norm_5 = torch.nn.LayerNorm([512, 2, 2])

        # define fully connected output layer
        self.fc_1 = torch.nn.Linear(512*2*2, out_dim)

    # define network layer connections and forward propagate input x through
    # the network and return output
    def forward(self, x):
        z_1 = torch.relu(self.conv_1(x))
        z_2 = torch.relu(self.conv_2(z_1))
        z_3 = torch.relu(self.conv_3(z_2))
        z_4 = torch.relu(self.conv_4(z_3))
        z_5 = torch.relu(self.conv_5(z_4))
        z_5_flat = torch.flatten(z_5, start_dim=1)
        z_6 = self.fc_1(z_5_flat)

        # add final activation
        if self.out_act is not None:
            z_out = self.out_act(z_6)
        else:
            z_out = z_6

        return z_out
