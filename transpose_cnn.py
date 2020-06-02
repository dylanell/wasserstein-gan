"""
Transpose convolutional neural network implemented as Pytorch nn.Module.
"""

import torch

class TransposeCNN(torch.nn.Module):
    # initialize the base class and define network layers
    def __init__(self, in_dim, out_chan, hid_act=torch.nn.ReLU(), \
                 out_act=torch.nn.Tanh(), layer_norm=False):
        # run base initializer
        super(TransposeCNN, self).__init__()

        # set activation functions
        self.hid_act = hid_act
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

        # define layer norm layers otherwise just use identity (no norm)
        if layer_norm:
            self.norm_1 = torch.nn.LayerNorm([512, 2, 2])
            self.norm_2 = torch.nn.LayerNorm([256, 4, 4])
            self.norm_3 = torch.nn.LayerNorm([128, 8, 8])
            self.norm_4 = torch.nn.LayerNorm([64, 16, 16])
            self.norm_5 = torch.nn.LayerNorm([32, 32, 32])
        else:
            self.norm_1 = torch.nn.Identity()
            self.norm_2 = torch.nn.Identity()
            self.norm_3 = torch.nn.Identity()
            self.norm_4 = torch.nn.Identity()
            self.norm_5 = torch.nn.Identity()

    # define network layer connections and forward propagate input x through
    # the network and return output
    def forward(self, x):
        # get batch size
        bs = x.shape[0]

        # propagate x through network
        z = self.hid_act(self.fc_1(x))
        z = z.view(bs, 512, 2, 2)
        z = self.norm_1(z)
        z = self.norm_2(self.hid_act(self.conv_1(z, output_size=(bs, 256, 4, 4))))
        z = self.norm_3(self.hid_act(self.conv_2(z, output_size=(bs, 128, 8, 8))))
        z = self.norm_4(self.hid_act(self.conv_3(z, output_size=(bs, 64, 16, 16))))
        z = self.norm_5(self.hid_act(self.conv_4(z, output_size=(bs, 32, 32, 32))))
        z = self.out_act(self.conv_5(z, output_size=(bs, 1, 32, 32)))

        return z
