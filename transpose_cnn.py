"""
Transpose convolutional neural network implemented as Pytorch nn.Module.
"""

import torch

class TransposeCNN(torch.nn.Module):
    # initialize the base class and define network layers
    def __init__(self, in_dim, out_chan, out_act=None):
        # run base initializer
        super(TransposeCNN, self).__init__()

        # set activation functions
        self.out_act = out_act

        # define fully connected input layer
        self.fc_1 = torch.nn.Linear(in_dim, 512*2*2)

        # define convolutional layers
        self.conv_1 = torch.nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1)
        self.conv_2 = torch.nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1)
        self.conv_3 = torch.nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1)
        self.conv_4 = torch.nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1)
        self.conv_5 = torch.nn.ConvTranspose2d(
            32, out_chan, 3, stride=1, padding=1
        )

        # define layer norm layers
        self.norm_1 = torch.nn.LayerNorm([512, 2, 2])
        self.norm_2 = torch.nn.LayerNorm([256, 4, 4])
        self.norm_3 = torch.nn.LayerNorm([128, 8, 8])
        self.norm_4 = torch.nn.LayerNorm([64, 16, 16])
        self.norm_5 = torch.nn.LayerNorm([32, 32, 32])

    # define network layer connections and forward propagate input x through
    # the network and return output
    def forward(self, x):
        # get number of samples in input batch
        bs = x.shape[0]

        # propagate x through network
        z_1 = torch.relu(self.fc_1(x))
        z_1_unflat = torch.reshape(z_1, [bs, 512, 2, 2])
        z_1_unflat_norm = self.norm_1(z_1_unflat)
        z_2 = torch.relu(self.conv_1(z_1_unflat_norm, output_size=(bs, 256, 4, 4)))
        z_2_norm = self.norm_2(z_2)
        z_3 = torch.relu(self.conv_2(z_2_norm, output_size=(bs, 128, 8, 8)))
        z_3_norm = self.norm_3(z_3)
        z_4 = torch.relu(self.conv_3(z_3_norm, output_size=(bs, 64, 16, 16)))
        z_4_norm = self.norm_4(z_4)
        z_5 = torch.relu(self.conv_4(z_4_norm, output_size=(bs, 32, 32, 32)))
        z_5_norm = self.norm_5(z_5)
        z_6 = self.conv_5(z_5_norm, output_size=(bs, 1, 32, 32))

        # add final activation
        if self.out_act is not None:
            z_out = self.out_act(z_6)
        else:
            z_out = z_6

        return z_out
