# modified from https://github.com/piergiaj/pytorch-i3d

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchsummary import summary


class MaxPool3dSamePadding(nn.MaxPool3d):
    """
    Class respresenting a 3D Max Pooling layer that computes padding necessary for legal computation with kernel.
    """

    def compute_pad(self, dim, s):
        if s % self.stride[dim] == 0:
            return max(self.kernel_size[dim] - self.stride[dim], 0)
        else:
            return max(self.kernel_size[dim] - (s % self.stride[dim]), 0)

    def forward(self, x):
        # compute 'same' padding
        (batch, channel, t, h, w) = x.size()
        # print t,h,w
        out_t = np.ceil(float(t) / float(self.stride[0]))
        out_h = np.ceil(float(h) / float(self.stride[1]))
        out_w = np.ceil(float(w) / float(self.stride[2]))
        # print out_t, out_h, out_w
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)
        # print pad_t, pad_h, pad_w

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        # print x.size()
        # print pad
        x = F.pad(x, pad)
        return super(MaxPool3dSamePadding, self).forward(x)


class Unit3D(nn.Module):
    """
    Class Representing a 3D Convolutional Unit that computes padding necessary for legal convolution with kernel.
    """

    def __init__(self, in_channels,
                 output_channels,
                 kernel_shape=(1, 1, 1),
                 stride=(1, 1, 1),
                 padding=0,
                 activation_fn=F.relu,
                 use_batch_norm=True,
                 use_bias=False,
                 name='unit_3d'):

        """Initializes Unit3D module."""
        super(Unit3D, self).__init__()

        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn
        self._use_bias = use_bias
        self.name = name
        self.padding = padding

        self.conv3d = nn.Conv3d(in_channels=in_channels,
                                out_channels=self._output_channels,
                                kernel_size=self._kernel_shape,
                                stride=self._stride,
                                padding=0,
                                # we always want padding to be 0 here.
                                # We will dynamically pad based on input size in forward function
                                bias=self._use_bias)

        if self._use_batch_norm:
            self.bn = nn.BatchNorm3d(self._output_channels, eps=0.001, momentum=0.01)

    def compute_pad(self, dim, s):
        if s % self._stride[dim] == 0:
            return max(self._kernel_shape[dim] - self._stride[dim], 0)
        else:
            return max(self._kernel_shape[dim] - (s % self._stride[dim]), 0)

    def forward(self, x):
        # compute 'same' padding
        (batch, channel, t, h, w) = x.size()
        # print t,h,w
        out_t = np.ceil(float(t) / float(self._stride[0]))
        out_h = np.ceil(float(h) / float(self._stride[1]))
        out_w = np.ceil(float(w) / float(self._stride[2]))
        # print out_t, out_h, out_w
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)
        # print pad_t, pad_h, pad_w

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        # print x.size()
        # print pad
        x = F.pad(x, pad)
        # print x.size()

        x = self.conv3d(x)
        if self._use_batch_norm:
            x = self.bn(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        return x


class InceptionModule(nn.Module):
    """
    Class representing a single Inception Module that is part of the I3D Network design.
    """

    def __init__(self, in_channels, out_channels, name):
        super(InceptionModule, self).__init__()

        self.b0 = Unit3D(in_channels=in_channels, output_channels=out_channels[0], kernel_shape=[1, 1, 1], padding=0,
                         name=name + '/Branch_0/Conv3d_0a_1x1')
        self.b1a = Unit3D(in_channels=in_channels, output_channels=out_channels[1], kernel_shape=[1, 1, 1], padding=0,
                          name=name + '/Branch_1/Conv3d_0a_1x1')
        self.b1b = Unit3D(in_channels=out_channels[1], output_channels=out_channels[2], kernel_shape=[3, 3, 3],
                          name=name + '/Branch_1/Conv3d_0b_3x3')
        self.b2a = Unit3D(in_channels=in_channels, output_channels=out_channels[3], kernel_shape=[1, 1, 1], padding=0,
                          name=name + '/Branch_2/Conv3d_0a_1x1')
        self.b2b = Unit3D(in_channels=out_channels[3], output_channels=out_channels[4], kernel_shape=[3, 3, 3],
                          name=name + '/Branch_2/Conv3d_0b_3x3')
        self.b3a = MaxPool3dSamePadding(kernel_size=[3, 3, 3],
                                        stride=(1, 1, 1), padding=0)
        self.b3b = Unit3D(in_channels=in_channels, output_channels=out_channels[5], kernel_shape=[1, 1, 1], padding=0,
                          name=name + '/Branch_3/Conv3d_0b_1x1')
        self.name = name

    def forward(self, x):
        b0 = self.b0(x)
        b1 = self.b1b(self.b1a(x))
        b2 = self.b2b(self.b2a(x))
        b3 = self.b3b(self.b3a(x))
        return torch.cat([b0, b1, b2, b3], dim=1)


class InceptionI3d(nn.Module):
    """Inception-v1 I3D architecture.
    The generator is introduced in:
        Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset
        Joao Carreira, Andrew Zisserman
        https://arxiv.org/pdf/1705.07750v1.pdf.
    See also the Inception architecture, introduced in:
        Going deeper with convolutions
        Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
        Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich.
        http://arxiv.org/pdf/1409.4842v1.pdf.
    """

    # Endpoints of the generator in order. During construction, all the endpoints up
    # to a designated `final_endpoint` are returned in a dictionary as the
    # second return value.
    VALID_ENDPOINTS = (
        'Conv3d_1a_7x7',
        'MaxPool3d_2a_3x3',
        'Conv3d_2b_1x1',
        'Conv3d_2c_3x3',
        'MaxPool3d_3a_3x3',
        'Mixed_3b',
        'Mixed_3c',
        'MaxPool3d_4a_3x3',
        'Mixed_4b',
        'Mixed_4c',
        'Mixed_4d',
        'Mixed_4e',
        'Mixed_4f',
        # 'MaxPool3d_5a_2x2',
        # 'MaxPool3d_5a_1x1',
        'Mixed_5b',
        'Mixed_5c',
        # 'Mixed_5c_small',
        # 'Logits',
        # 'Predictions',
    )
    VALID_IN_FRAMES = (8, 16)

    def __init__(self, name='modified_inception_i3d', final_endpoint='Mixed_5c', in_frames=8, in_channels=3,
                 pretrained=False, weights_path=''):
        """
        Initializes I3D generator instance.
        :param name: (str, optional) The name of the network (default 'modified_inception_i3d').
        :param final_endpoint: (str, optional) The name of the final layer used in the network (default 'Mixed_5c').
        :param in_frames: (int, optional) The number of frames in the input video (default 8).
                           Legal values: 8, 16
        :param in_channels: (int, optional) The number of channels in the input tensor (default 3).
        :param pretrained: (bool, optional) True if the generator should load pretrained weights; False otherwise.
        :param weights_path: (str, optional) The path at which the pretrained weights are located.
        Raises:
            ValueError: if 'final_ endpoint' is not recognized.
            ValueError: if 'in_frames' is not a legal value.
        """
        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % final_endpoint)
        if in_frames not in self.VALID_IN_FRAMES:
            raise ValueError('Invalid number of frames in the input: %d' % in_frames)

        super(InceptionI3d, self).__init__()
        self.i3d_name = name
        self._final_endpoint = final_endpoint
        self.in_frames = in_frames
        self.pretrained = pretrained
        self.weights_path = weights_path

        if self._final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % self._final_endpoint)

        # definition of all network layers
        self.end_points = {}
        end_point = 'Conv3d_1a_7x7'
        self.end_points[end_point] = Unit3D(in_channels=in_channels, output_channels=64, kernel_shape=[7, 7, 7],
                                            stride=(2, 2, 2), padding=(3, 3, 3), name=name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_2a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2),
                                                          padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Conv3d_2b_1x1'
        self.end_points[end_point] = Unit3D(in_channels=64, output_channels=64, kernel_shape=[1, 1, 1], padding=0,
                                            name=name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Conv3d_2c_3x3'
        self.end_points[end_point] = Unit3D(in_channels=64, output_channels=192, kernel_shape=[3, 3, 3], padding=1,
                                            name=name + end_point)
        if self._final_endpoint == end_point: return

        # Modification: change the stride and kernel of this layer depending on the number of input frames as the depth
        # must be shrunk to 1 by the end of the network.
        end_point = 'MaxPool3d_3a_3x3'
        if in_frames == 8:
            self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2),
                                                              padding=0)
        elif in_frames == 16:
            self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[2, 3, 3], stride=(2, 2, 2),
                                                              padding=0)

        if self._final_endpoint == end_point: return

        end_point = 'Mixed_3b'
        self.end_points[end_point] = InceptionModule(192, [64, 96, 128, 16, 32, 32], name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_3c'
        self.end_points[end_point] = InceptionModule(256, [128, 128, 192, 32, 96, 64], name + end_point)
        if self._final_endpoint == end_point: return

        # Modification: changed the stride of this max pooling layer so that the temporal dimension does not change.
        # now changed so that none of the dimensions change
        end_point = 'MaxPool3d_4a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[3, 3, 3], stride=(1, 1, 1),
                                                          padding=0)
        # if self._final_endpoint == end_point: return
        #
        # end_point = 'Mixed_4b'
        # self.end_points[end_point] = InceptionModule(128 + 192 + 96 + 64, [192, 96, 208, 16, 48, 64], name + end_point)
        # if self._final_endpoint == end_point: return
        #
        # end_point = 'Mixed_4c'
        # self.end_points[end_point] = InceptionModule(192 + 208 + 48 + 64, [160, 112, 224, 24, 64, 64], name + end_point)
        # if self._final_endpoint == end_point: return
        #
        # end_point = 'Mixed_4d'
        # self.end_points[end_point] = InceptionModule(160 + 224 + 64 + 64, [128, 128, 256, 24, 64, 64], name + end_point)
        # if self._final_endpoint == end_point: return
        #
        # end_point = 'Mixed_4e'
        # self.end_points[end_point] = InceptionModule(128 + 256 + 64 + 64, [112, 144, 288, 32, 64, 64], name + end_point)
        # if self._final_endpoint == end_point: return
        #
        # end_point = 'Mixed_4f'
        # self.end_points[end_point] = InceptionModule(112 + 288 + 64 + 64, [256, 160, 320, 32, 128, 128],
        #                                              name + end_point)
        # if self._final_endpoint == end_point: return
        #
        # # Modification: changed the stride of this max pooling layer so that none of the dimensions get smaller.
        # # end_point = 'MaxPool3d_5a_2x2'
        # # self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[2, 2, 2], stride=(2, 2, 2),
        # #                                                   padding=0)
        # # if self._final_endpoint == end_point: return
        #
        # # end_point = 'MaxPool3d_5a_1x1'
        # # self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[2, 2, 2], stride=(1, 1, 1),
        # #                                                   padding=0)
        # # if self._final_endpoint == end_point: return
        #
        # end_point = 'Mixed_5b'
        # self.end_points[end_point] = InceptionModule(256 + 320 + 128 + 128, [256, 160, 320, 32, 128, 128],
        #                                              name + end_point)
        # if self._final_endpoint == end_point: return
        #
        # end_point = 'Mixed_5c'
        # self.end_points[end_point] = InceptionModule(256 + 320 + 128 + 128, [384, 192, 384, 48, 128, 128],
        #                                              name + end_point)

        self.build()

        self.feature_layer = Unit3D(in_channels=384 + 384 + 128 + 128, output_channels=256, kernel_shape=[3, 3, 3],
                                    padding=1, name='features')  # 256 + 320 + 128 + 128

    def build(self):
        for k in self.end_points.keys():
            self.add_module(k, self.end_points[k])
        if self.pretrained:
            state_dict = load_layer_weights(self.weights_path)
            self.load_state_dict(state_dict)
        # print('%s Model Successfully Built \n' % self.i3d_name)

    def forward(self, x):
        """
        Function to compute a single forward pass through the network, according to the architecture.
        :param x: (tensor) The batch of videos input to the network.
                   Must be a tensor of size: (bsz, 3, 8, 112, 112) for this application.
        :return: A tensor representing the feature map representation of the 'action' in the video.
                 Shape of output is: (bsz, 256, 4, 7, 7) for this application.
        """
        return_features = []

        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x)  # use _modules to work with dataparallel
                # if end_point in ['Conv3d_1a_7x7', 'Conv3d_2c_3x3']:
                #     return_features.append(x)

        # x = self.feature_layer(x)
        # return_features.append(x)

        # return return_features  # returns 64x8x56x56, 192x8x28x28, 256x4x14x14
        return x


def load_layer_weights(weights_path):
    state_dict = torch.load(weights_path)
    remove_layers = ['Mixed_4b',
                     'Mixed_4c',
                     'Mixed_4d',
                     'Mixed_4e',
                     'Mixed_4f',
                     # 'MaxPool3d_5a_2x2',
                     # 'MaxPool3d_5a_1x1',
                     'Mixed_5b',
                     'Mixed_5c', 'Logits', 'Predictions']
    remove_layers.extend([l.lower() for l in remove_layers])
    new_state_dict = {}
    for item, state in state_dict.items():
        layer = item[:item.index('.')]
        # if the state is not one of the layers to remove, then use it
        if layer not in remove_layers:
            new_state_dict[item] = state

    return new_state_dict


# if __name__ == "__main__":
#     print_summary = True
#
#     i3d = InceptionI3d(final_endpoint='Mixed_5c', in_frames=16)
#
#     if print_summary:
#         summary(i3d, input_size=(3, 16, 112, 112))

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Discriminator(nn.Module):
    """
    Class representing the Discriminator network to be used.
    """

    def __init__(self, in_channels=3, pretrained=False, weights_path='', disc_name='Video Discriminator'):
        """
        Initializes the Discriminator network.
        :param in_channels: (int) The number of channels in the input tensor.
        :param disc_name: (str, optional) The name of the network (default 'Video Generator').
        Raises:
            ValueError: if 'out_frames' is not a legal value.
        """

        super(Discriminator, self).__init__()
        self.disc_name = disc_name

        self.i3d = InceptionI3d(name='discriminator', final_endpoint='MaxPool3d_4a_3x3', in_frames=16, in_channels=3,
                                  pretrained=pretrained, weights_path=weights_path)
        self.fc_1 = nn.Linear(in_features=480*4*14*14, out_features=1)
        self.sigmoid = nn.Sigmoid()

        # print('%s Model Successfully Built \n' % self.gen_name)

    def forward(self, vid):
        """
        Function to compute a single forward pass through the network, according to the architecture.
        :param vid: (tensor) .
        :return: .
        """
        x = self.i3d(vid)

        bsz, channels, frames, height, width = x.size()
        # print(x.size())
        x = x.reshape(bsz, -1)

        # print(x.size())

        x = self.fc_1(x).squeeze()
        x = self.sigmoid(x)

        return x


if __name__ == "__main__":
    print_summary = True

    dis = Discriminator()

    if print_summary:
        summary(dis, input_size=(3, 16, 112, 112))
