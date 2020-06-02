"""
Convolutional neural network implemented as Pytorch nn.Module.
"""

import torch

class CNN(torch.nn.Module):
    # initialize the base class and define network layers
    def __init__(self, in_chan, out_dim, hid_act=torch.nn.ReLU(), \
                 out_act=torch.nn.Identity(), layer_norm=False):
        # run base initializer
        super(CNN, self).__init__()

        # set activation functions
        self.hid_act = hid_act
        self.out_act = out_act

        # define convolutional layers
        self.conv_1 = torch.nn.Conv2d(in_chan, 32, 3, stride=1, padding=1)
        self.conv_2 = torch.nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv_3 = torch.nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv_4 = torch.nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.conv_5 = torch.nn.Conv2d(256, 512, 3, stride=2, padding=1)

        # define layer norm layers otherwise just use identity (no norm)
        if layer_norm:
            self.norm_1 = torch.nn.LayerNorm([32, 32, 32])
            self.norm_2 = torch.nn.LayerNorm([64, 16, 16])
            self.norm_3 = torch.nn.LayerNorm([128, 8, 8])
            self.norm_4 = torch.nn.LayerNorm([256, 4, 4])
            self.norm_5 = torch.nn.LayerNorm([512, 2, 2])
        else:
            self.norm_1 = torch.nn.Identity()
            self.norm_2 = torch.nn.Identity()
            self.norm_3 = torch.nn.Identity()
            self.norm_4 = torch.nn.Identity()
            self.norm_5 = torch.nn.Identity()

        # define fully connected output layer
        self.fc_1 = torch.nn.Linear(512*2*2, out_dim)

    # define network layer connections and forward propagate input x through
    # the network and return output
    def forward(self, x):
        z = self.norm_1(self.hid_act(self.conv_1(x)))
        z = self.norm_2(self.hid_act(self.conv_2(x)))
        z = self.norm_3(self.hid_act(self.conv_3(x)))
        z = self.norm_4(self.hid_act(self.conv_4(x)))
        z = self.norm_5(self.hid_act(self.conv_5(x)))
        z = self.out_act(self.fc_1(torch.flatten(z, start_dim=1)))

        return z
