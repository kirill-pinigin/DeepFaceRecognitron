import torch
import torch.nn as nn
from NeuralModels import SILU, Perceptron
from DeepFaceRecognitron import IMAGE_SIZE, DIMENSION, CHANNELS

import math

def conv_bn(inp, oup, stride , activation = nn.ReLU()):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        activation,
    )

def conv_1x1_bn(inp, oup, activation = nn.ReLU()):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        activation,
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, activation = nn.ReLU()):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and inp == oup

        self.conv = nn.Sequential(
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            activation,
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3, stride, 1, groups=inp * expand_ratio, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            activation,
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, dimension=1000, channels = 3, input_size=224, width_mult=1.):
        super(MobileNetV2, self).__init__()
        # setting of inverted residual blocks
        self.interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0
        input_channel = int(32 * width_mult)
        self.last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280
        self.features = [conv_bn(channels, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in self.interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(InvertedResidual(input_channel, output_channel, s, t))
                else:
                    self.features.append(InvertedResidual(input_channel, output_channel, 1, t))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        self.features.append(nn.AvgPool2d(input_size // 32))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self.last_channel, dimension),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.last_channel)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

MOBILE_NET_V2_UTR = 'https://s3-us-west-1.amazonaws.com/models-nima/mobilenetv2.pth.tar'

import requests
import os

def download_file(url, local_filename, chunk_size=1024):
    if os.path.exists(local_filename):
        return local_filename
    r = requests.get(url, stream=True)
    with open(local_filename, 'wb') as f:
        for chunk in r.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
    return local_filename


def mobile_net_v2(dimension=1000, channels = 3, input_size=224, width_mult=1., pretrained = True):
    model = MobileNetV2(input_size=input_size)
    if pretrained:
        path_to_model = './mobilenetv2.pth.tar'
        if not os.path.exists(path_to_model):
            path_to_model = download_file(MOBILE_NET_V2_UTR, path_to_model)
        state_dict = torch.load(path_to_model, map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict)
    return model


class MobilePredictor(nn.Module):
    def __init__(self, activation=SILU(), pretrained=True):
        super(MobilePredictor, self).__init__()
        self.activation = activation
        base_model = mobile_net_v2(dimension=DIMENSION, channels = CHANNELS, input_size=IMAGE_SIZE, width_mult=1., pretrained = pretrained)
        base_model = nn.Sequential(*list(base_model.children())[:-1])
        conv = nn.Conv2d(CHANNELS, 32, kernel_size=3, stride=2, padding=3, bias=False)
        weight = torch.FloatTensor(32, CHANNELS, 3, 3)
        parameters = list(base_model.parameters())
        for i in range(32):
            if CHANNELS == 1:
                weight[i, :, :, :] = parameters[0].data[i].mean(0)
            else:
                weight[i, :, :, :] = parameters[0].data[i]
        conv.weight.data.copy_(weight)

        for m in base_model.modules():
            if isinstance(m, InvertedResidual):
                m.conv[2] = activation
                m.conv[5] = activation

        self.features =base_model
        self.features[0][0]= conv

        self.predictor = nn.Sequential(
            Perceptron(1280, 1280),
            nn.Dropout(p=0),
            activation,
            Perceptron( 1280, DIMENSION),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.predictor(x)
        return x
