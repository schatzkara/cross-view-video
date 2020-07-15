import torch
import torch.nn as nn
import torch.nn.functional as f
from torchsummary import summary

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class VPPredictor(nn.Module):
    """
    Class representing the Viewpoint Predictor to be used.
    """

    def __init__(self, in_channels, vpp_name='Viewpoint Predictor'):
        """
        Initializes the Generator network.
        :param in_channels: (int) The number of channels in the input tensor.
        :param vpp_name: (str, optional) The name of the network (default 'Viewpoint Predictor').
        """
        super(VPPredictor, self).__init__()
        self.vpp_name = vpp_name

        # definition of all layer channels
        layer_out_channels = {1: 128,
                              2: 32,
                              3: 1}
        layer_in_channels = {1: in_channels,
                             2: layer_out_channels[1],
                             3: layer_out_channels[2] * 3 * 3}

        # definition of all network layers
        layer = 1
        self.conv2d_1 = nn.Conv2d(in_channels=layer_in_channels[layer], out_channels=layer_out_channels[layer],
                                  kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu1 = nn.ReLU(inplace=True)
        self.avg_pool_1 = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        layer = 2
        self.conv2d_2 = nn.Conv2d(in_channels=layer_in_channels[layer], out_channels=layer_out_channels[layer],
                                  kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu2 = nn.ReLU(inplace=True)
        self.avg_pool_2 = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        layer = 3
        self.fc_1 = nn.Linear(in_features=layer_in_channels[layer], out_features=layer_out_channels[layer])
        # print('%s Model Successfully Built \n' % self.kpp_name)

    def forward(self, x):
        """
        Function to compute a single forward pass through the network, according to the architecture.
        :param x: (tensor) The input tensors from which to estimate the viewpoint change.
                   Must be a tensor of shape: (bsz, in_channels, 14, 14) for this application.
        :return: A scalar value representing the estimated viewpoint change between views 1 and 2.
        """
        x = self.conv2d_1(x)
        x = self.relu1(x)
        x = self.avg_pool_1(x)

        x = self.conv2d_2(x)
        x = self.relu2(x)
        x = self.avg_pool_2(x)

        bsz, channels, height, width = x.size()
        x = x.reshape(bsz, -1)

        x = self.fc_1(x).squeeze()

        return x  # bsz


if __name__ == "__main__":
    print_summary = True

    vpp = VPPredictor(in_channels=512)

    if print_summary:
        summary(vpp, input_size=(512, 14, 14))
