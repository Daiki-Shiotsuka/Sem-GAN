import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models
from torchvision.models.vgg import VGG

def conv(in_channels, out_channels, kernel_size=4, stride=2, padding=1, instance_norm=True, relu=True, relu_slope=None, init_zero_weights=False):
    layers = []
    conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
    if init_zero_weights:
        conv_layer.weight.data = torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.001
    else:
        nn.init.normal_(conv_layer.weight.data, 0.0, 0.02)
    layers.append(conv_layer)

    if instance_norm:
        layers.append(nn.InstanceNorm2d(out_channels))

    if relu:
        if relu_slope:
            relu_layer = nn.LeakyReLU(relu_slope, True)
        else:
            relu_layer = nn.ReLU(inplace=True)
        layers.append(relu_layer)
    return layers

def deconv(in_channels, out_channels, kernel_size=4, stride=2, padding=1, output_padding=1, instance_norm=True, relu=True, relu_slope=None, init_zero_weights=False):

    layers = []
    deconv_layer = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=True)
    if init_zero_weights:
        deconv_layer.weight.data = torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.001
    else:
        nn.init.normal_(deconv_layer.weight.data, 0.0, 0.02)
    layers.append(deconv_layer)

    if instance_norm:
        layers.append(nn.InstanceNorm2d(out_channels))

    if relu:
        if relu_slope:
            relu_layer = nn.LeakyReLU(relu_slope, True)
        else:
            relu_layer = nn.ReLU(inplace=True)
        layers.append(relu_layer)
    return layers

class ResidualBlock(nn.Module):
    def __init__(self, input_features):
        super(ResidualBlock, self).__init__()

        conv_layers = [
                nn.ReflectionPad2d(1),
                *conv(input_features, input_features, kernel_size=3, stride=1, padding=0),
                nn.ReflectionPad2d(1),
                *conv(input_features, input_features, kernel_size=3, stride=1, padding=0, relu=False)
            ]
        self.model = nn.Sequential(*conv_layers)

    def forward(self, input_data):
        return input_data + self.model(input_data)

class CycleGenerator(nn.Module):

    def __init__(self, in_channels=3, out_channels=3, res_blocks=9):
        super(CycleGenerator, self).__init__()

        # First 7 x 7 convolutional layer
        layers = [
            nn.ReflectionPad2d(3),
            *conv(in_channels, 64, 7, stride=1, padding=0)
        ]

        # Two 3 x 3 convolutional layers
        input_features = 64
        output_features = input_features * 2
        for _ in range(2):
            layers += conv(input_features, output_features, 3)
            input_features, output_features = output_features, output_features * 2

        # Residual blocks
        for _ in range(res_blocks):
            layers += [ResidualBlock(input_features)]

        # Two 3 x 3 deconvolutional layers
        output_features = input_features // 2
        for _ in range(2):
            layers += deconv(input_features, output_features, 3)
            input_features, output_features = output_features, output_features // 2

        # Output layer
        layers += [
                nn.ReflectionPad2d(3),
                nn.Conv2d(input_features, out_channels, 7),
                nn.Tanh()
            ]
        self.model = nn.Sequential(*layers)

    def forward(self, real_image):
        return self.model(real_image)

class Discriminator(nn.Module):

    def __init__(self, in_channels=3, conv_dim=64):
        super(Discriminator, self).__init__()

        C64 = conv(in_channels, conv_dim, instance_norm=False, relu_slope=0.2)
        C128 = conv(conv_dim, conv_dim * 2, relu_slope=0.2)
        C256 = conv(conv_dim * 2, conv_dim * 4, relu_slope=0.2)
        C512 = conv(conv_dim * 4, conv_dim * 8, stride = 1, relu_slope=0.2)
        C1 = conv(conv_dim * 8, 1, stride=1, instance_norm=False, relu=False)

        self.model = nn.Sequential(
                *C64,
                *C128,
                *C256,
                *C512,
                *C1
            )

    def forward(self, image):
        return self.model(image)



class FCNs(nn.Module):

    def __init__(self, pretrained_net, n_class):
        super().__init__()
        self.n_class = n_class
        self.pretrained_net = pretrained_net
        self.relu    = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5     = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

    def forward(self, x):
        output = self.pretrained_net(x)
        x5 = output['x5']  # size=(N, 512, x.H/32, x.W/32)
        x4 = output['x4']  # size=(N, 512, x.H/16, x.W/16)
        x3 = output['x3']  # size=(N, 256, x.H/8,  x.W/8)
        x2 = output['x2']  # size=(N, 128, x.H/4,  x.W/4)
        x1 = output['x1']  # size=(N, 64, x.H/2,  x.W/2)

        score = self.bn1(self.relu(self.deconv1(x5)))     # size=(N, 512, x.H/16, x.W/16)
        score = score + x4                                # element-wise add, size=(N, 512, x.H/16, x.W/16)
        score = self.bn2(self.relu(self.deconv2(score)))  # size=(N, 256, x.H/8, x.W/8)
        score = score + x3                                # element-wise add, size=(N, 256, x.H/8, x.W/8)
        score = self.bn3(self.relu(self.deconv3(score)))  # size=(N, 128, x.H/4, x.W/4)
        score = score + x2                                # element-wise add, size=(N, 128, x.H/4, x.W/4)
        score = self.bn4(self.relu(self.deconv4(score)))  # size=(N, 64, x.H/2, x.W/2)
        score = score + x1                                # element-wise add, size=(N, 64, x.H/2, x.W/2)
        score = self.bn5(self.relu(self.deconv5(score)))  # size=(N, 32, x.H, x.W)
        score = self.classifier(score)                    # size=(N, n_class, x.H/1, x.W/1)

        return score  # size=(N, n_class, x.H/1, x.W/1)


class VGGNet(VGG):
    def __init__(self, pretrained=True, model='vgg16', requires_grad=True, remove_fc=True, show_params=False):
        super().__init__(make_layers(cfg[model]))
        self.ranges = ranges[model]

        if pretrained:
            exec("self.load_state_dict(models.%s(pretrained=True).state_dict())" % model)

        if not requires_grad:
            for param in super().parameters():
                param.requires_grad = False

        if remove_fc:  # delete redundant fully-connected layer params, can save memory
            del self.classifier

        if show_params:
            for name, param in self.named_parameters():
                print(name, param.size())

    def forward(self, x):
        output = {}

        # get the output of each maxpooling layer (5 maxpool in VGG net)
        for idx in range(len(self.ranges)):
            for layer in range(self.ranges[idx][0], self.ranges[idx][1]):
                x = self.features[layer](x)
            output["x%d"%(idx+1)] = x

        return output

ranges = {
    'vgg11': ((0, 3), (3, 6),  (6, 11),  (11, 16), (16, 21)),
    'vgg13': ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25)),
    'vgg16': ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31)),
    'vgg19': ((0, 5), (5, 10), (10, 19), (19, 28), (28, 37))
}

# cropped version from https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def make_layers(cfg, batch_norm=False):
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
    return nn.Sequential(*layers)
