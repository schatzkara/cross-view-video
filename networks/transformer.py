# phase 3

import torch
import torch.nn as nn
from torchsummary import summary


class Transformer(nn.Module):
    """
    Class representing the Transformer network to be used.
    """

    def __init__(self, in_channels, out_channels, trans_name='Transformer Network'):
        """
        Initializes the Transformer Network.
        :param in_channels: (int) The number of channels in the input tensor.
        :param out_channels: (int) The number of channels desired in the output tensor.
        :param trans_name: (str, optional) The name of the Transformer (default 'Transormer Network').
        """
        super(Transformer, self).__init__()
        self.trans_name = trans_name

        # definition of all network layers
        self.conv3d_1a = nn.Conv3d(in_channels=in_channels, out_channels=in_channels, kernel_size=(3, 3, 3),
                                   stride=(1, 1, 1), padding=(1, 1, 1))
        self.relu_1a = nn.ReLU(inplace=True)
        self.conv3d_1b = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3, 3),
                                   stride=(1, 1, 1), padding=(1, 1, 1))
        self.relu_1b = nn.ReLU(inplace=True)
        self.conv3d_1c = nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3, 3),
                                   stride=(1, 1, 1), padding=(1, 1, 1))
        self.relu_1c = nn.ReLU(inplace=True)

        # print('%s Model Successfully Built \n' % self.trans_name)

    def forward(self, x):
        """
        Function to compute a single forward pass through the network, according to the architecture.
        :param x: (tensor) The input tensor to transform.
                   Must be a tensor of shape: (bsz, in_channels, frames, height, width) for this application.
        :return: A tensor representing the transformed input.
                 Shape of output is: (bsz, out_channels, frames, height, width) for this application.
        """
        x = self.conv3d_1a(x)
        x = self.relu_1a(x)
        x = self.conv3d_1b(x)
        x = self.relu_1b(x)
        x = self.conv3d_1c(x)
        x = self.relu_1c(x)

        return x


if __name__ == "__main__":
    print_summary = True

    ex = Transformer(in_channels=257, out_channels=256)

    if print_summary:
        summary(ex, input_size=(257, 4, 14, 14))
