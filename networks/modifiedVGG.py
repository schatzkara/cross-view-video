# phase 3

# modified from https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py

import torch
import torch.nn as nn
from torchsummary import summary

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


class VGG(nn.Module):
    """
    Class representing the modified VGG network to be used.
    """

    def __init__(self, features, num_classes=1000, init_weights=True,
                 pretrained=False, weights_path=''):
        """
        Initializes the modified VGG network.
        :param features: All the network layers.
        :param num_classes: (int) The number of classes used for classification.
        :param init_weights: (bool) True if the network weights should be initialized; False otherwise.
        :param pretrained: (bool) True if the network should be pretrained; False otherwise.
        :param weights_path: (str) The path at which to pretrained weights are located.
        """
        super(VGG, self).__init__()
        self.features = features
        # self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        # self.classifier = nn.Sequential(
        #     nn.Linear(512 * 7 * 7, 4096),
        #     nn.ReLU(True),
        #     nn.Dropout(),
        #     nn.Linear(4096, 4096),
        #     nn.ReLU(True),
        #     nn.Dropout(),
        #     nn.Linear(4096, num_classes),
        # )
        if init_weights:
            self._initialize_weights()

        if pretrained:
            self.load_weights(weights_path=weights_path)

    def load_weights(self, weights_path):
        state_dict = torch.load(weights_path)
        bad_weights = ["features.17", "features.19", "features.21",
                       "features.24",
                       "features.26", "features.28", "classifier.0", "classifier.3",
                       "classifier.6"]
        new_state_dict = {}
        for key, weight in state_dict.items():
            first_per = key.index('.')
            second_per = key[first_per + 1:].index('.')
            id_ = key[:first_per + second_per + 1]
            if id_ not in bad_weights:
                new_state_dict[key] = weight
        self.load_state_dict(new_state_dict)

    def forward(self, x):
        return_features = []

        for i in range(len(self.features)):
            x = self.features[i](x)
            if i in [8, 15, 16]:  # 22]:
                # feature_size = x.size()[-1]
                return_features.append(x)
        # x = self.features(x)
        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.classifier(x)

        return return_features  # this has 128x56x56, 256x28x28, 256x14x14

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, num_layers=None, batch_norm=False):
    if num_layers is not None:
        cfg = cfg[:num_layers]
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    # print('Modified VGG Model Successfully Built \n')

    # print(nn.Sequential(*layers))
    return nn.Sequential(*layers)


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],  # , 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],  # , 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],  # , 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],  # , 'M'],
}

num_layers_to_use = {
    'A': 7,
    'B': 9,
    'D': 10,  # 13,
    'E': 11,
}


def _vgg(arch, cfg, batch_norm, pretrained, progress, weights_path='', **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfgs[cfg], num_layers=num_layers_to_use[cfg], batch_norm=batch_norm),
                pretrained=pretrained, weights_path=weights_path, **kwargs)
    # if pretrained:
    #     state_dict = load_state_dict_from_url(model_urls[arch],
    #                                           progress=progress)
    #     model.load_state_dict(state_dict)
    return model


def vgg11(pretrained=False, progress=True, **kwargs):
    """VGG 11-layer model (configuration "A")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg11', 'A', False, pretrained, progress, **kwargs)


def vgg11_bn(pretrained=False, progress=True, **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg11_bn', 'A', True, pretrained, progress, **kwargs)


def vgg13(pretrained=False, progress=True, **kwargs):
    """VGG 13-layer model (configuration "B")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg13', 'B', False, pretrained, progress, **kwargs)


def vgg13_bn(pretrained=False, progress=True, **kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg13_bn', 'B', True, pretrained, progress, **kwargs)


def vgg16(pretrained=False, progress=True, weights_path='', **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16', 'D', False, pretrained=pretrained, progress=progress, weights_path=weights_path, **kwargs)


def vgg16_bn(pretrained=False, progress=True, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16_bn', 'D', True, pretrained, progress, **kwargs)


def vgg19(pretrained=False, progress=True, **kwargs):
    """VGG 19-layer model (configuration "E")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg19', 'E', False, pretrained, progress, **kwargs)


def vgg19_bn(pretrained=False, progress=True, **kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg19_bn', 'E', True, pretrained, progress, **kwargs)


if __name__ == "__main__":
    print_summary = True

    vgg = vgg16()

    if print_summary:
        summary(vgg, input_size=(3, 112, 112))
