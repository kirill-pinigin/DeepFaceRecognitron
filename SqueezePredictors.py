import torch
import torch.nn as nn
from torchvision import  models
import torch.nn.init as init

from NeuralModels import SILU, Perceptron

from DeepFaceRecognitron import IMAGE_SIZE, DIMENSION, CHANNELS
LATENT_DIM = int(512)
LATENT_DIM_2 = int(LATENT_DIM // 2) if LATENT_DIM > 2 else 1


class FireConvNorm(nn.Module):
    def __init__(self, inplanes=128, squeeze_planes=11,
                 expand1x1_planes=11, expand3x3_planes=11, activation = nn.ReLU()):
        super(FireConvNorm, self).__init__()
        self.outplanes = int(expand1x1_planes + expand3x3_planes)
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.activation = activation

        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)

        self.norm_sq = nn.BatchNorm2d(squeeze_planes)
        self.norm1x1 = nn.BatchNorm2d(expand1x1_planes)
        self.norm3x3 = nn.BatchNorm2d(expand3x3_planes)

    def ConfigureNorm(self):
        self.norm_sq = nn.BatchNorm2d(self.squeeze.out_channels)
        self.norm1x1 = nn.BatchNorm2d(self.expand3x3.out_channels)
        self.norm3x3 = nn.BatchNorm2d(self.expand3x3.out_channels)

    def forward(self, x):
        x = self.activation(self.norm_sq(self.squeeze(x)))
        return torch.cat([
            self.activation(self.norm1x1(self.expand1x1(x))),
            self.activation(self.norm3x3(self.expand3x3(x)))], 1)


class SqueezeSimplePredictor(nn.Module):
    def __init__(self, activation = nn.ReLU(), pretrained = True):
        super(SqueezeSimplePredictor, self).__init__()
        self.activation = activation
        first_norm_layer = nn.BatchNorm2d(96)
        final_norm_layer = nn.BatchNorm2d(DIMENSION)
        self.conv1 = nn.Conv2d(CHANNELS, 96, kernel_size=7, stride=2)
        self.norm1 = first_norm_layer
        self.downsample1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.fire1 = FireConvNorm(96, 16, 64, 64, activation=activation)
        self.fire2 = FireConvNorm(128, 16, 64, 64, activation=activation)
        self.fire3 = FireConvNorm(128, 32, 128, 128, activation=activation)
        self.downsample2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.fire4 = FireConvNorm(256, 32, 128, 128, activation=activation)
        self.fire5 = FireConvNorm(256, 48, 192, 192, activation=activation)
        self.fire6 = FireConvNorm(384, 48, 192, 192, activation=activation)
        self.fire7 = FireConvNorm(384, 64, 256, 256, activation=activation)
        self.downsample3 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.fire8 = FireConvNorm(512, 64, 256, 256, activation=activation)
        if pretrained:
            model = models.squeezenet1_0(pretrained=True).features
            if CHANNELS == 3:
                self.conv1 = model[0]

            self.fire1.squeeze = model[3].squeeze
            self.fire1.expand1x1 = model[3].expand1x1
            self.fire1.expand3x3 = model[3].expand3x3
            self.fire1.ConfigureNorm()

            self.fire2.squeeze = model[4].squeeze
            self.fire2.expand1x1 = model[4].expand1x1
            self.fire2.expand3x3 = model[4].expand3x3
            self.fire2.ConfigureNorm()

            self.fire3.squeeze = model[5].squeeze
            self.fire3.expand1x1 = model[5].expand1x1
            self.fire3.expand3x3 = model[5].expand3x3
            self.fire3.ConfigureNorm()

            self.fire4.squeeze = model[7].squeeze
            self.fire4.expand1x1 = model[7].expand1x1
            self.fire4.expand3x3 = model[7].expand3x3
            self.fire4.ConfigureNorm()

            self.fire5.squeeze = model[8].squeeze
            self.fire5.expand1x1 = model[8].expand1x1
            self.fire5.expand3x3 = model[8].expand3x3
            self.fire5.ConfigureNorm()

            self.fire6.squeeze = model[9].squeeze
            self.fire6.expand1x1 = model[9].expand1x1
            self.fire6.expand3x3 = model[9].expand3x3
            self.fire6.ConfigureNorm()

            self.fire7.squeeze = model[10].squeeze
            self.fire7.expand1x1 = model[10].expand1x1
            self.fire7.expand3x3 = model[10].expand3x3
            self.fire7.ConfigureNorm()

            self.fire8.squeeze = model[12].squeeze
            self.fire8.expand1x1 = model[12].expand1x1
            self.fire8.expand3x3 = model[12].expand3x3
            self.fire8.ConfigureNorm()

        else:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    init.kaiming_uniform(m.weight)
                    if m.bias is not None:
                        init.constant(m.bias, 0)

        self.predictor = nn.Sequential(
            nn.Dropout(p=0),
            nn.Conv2d(LATENT_DIM, DIMENSION, kernel_size=1),
            final_norm_layer,
            activation,
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.downsample1(x)
        x = self.fire1(x)
        x = self.fire2(x)
        x = self.fire3(x)
        x = self.downsample2(x)
        x = self.fire4(x)
        x = self.fire5(x)
        x = self.fire6(x)
        x = self.fire7(x)
        x = self.downsample3(x)
        x = self.fire8(x)
        x = self.predictor(x)
        x = x.view(x.size(0), -1)
        return torch.sigmoid(x)


class SqueezeResidualPredictor(SqueezeSimplePredictor):
    def __init__(self,  activation=nn.ReLU(), pretrained=True):
        super(SqueezeResidualPredictor, self).__init__(activation, pretrained)
        final_norm_layer = nn.BatchNorm2d(LATENT_DIM)

        self.features = nn.Sequential(
            nn.Conv2d(LATENT_DIM, LATENT_DIM, kernel_size=1),
            final_norm_layer,
            activation,
            nn.AdaptiveAvgPool2d(1),
        )

        reduce_number = int((LATENT_DIM + CHANNELS) / 2.0)
        sub_dimension = reduce_number if reduce_number > CHANNELS else (reduce_number + CHANNELS)

        self.predictor = nn.Sequential(
            Perceptron(LATENT_DIM, sub_dimension),
            nn.Dropout(p=0),
            activation,
            Perceptron(sub_dimension, CHANNELS),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.downsample1(x)
        f1 = self.fire1(x)

        x = self.fire2(f1)
        x = torch.add(x,f1)
        x = self.fire3(x)
        d2 = self.downsample2(x)
        x = self.fire4(d2)
        x = torch.add(x, d2)
        f5 = self.fire5(x)
        x = self.fire6(f5)
        x = torch.add(x, f5)
        x = self.fire7(x)
        d3 = self.downsample3(x)
        x = self.fire8(d3)
        x = torch.add(x, d3)
        x = self.features(x)
        x = self.predictor(x)
        return torch.sigmoid(x)


class SqueezeShuntPredictor(SqueezeResidualPredictor):
    def __init__(self, activation = nn.ReLU(),  type_norm = 'batch', pretrained = False):
        super(SqueezeShuntPredictor, self).__init__(activation=activation, type_norm=type_norm, pretrained=pretrained)
        self.shunt1 = nn.Sequential(nn.ReLU(), nn.Conv2d(96,128, kernel_size=1), nn.Sigmoid())
        self.shunt2 = nn.Sequential(nn.ReLU(), nn.Conv2d(128, 256, kernel_size=1), nn.Sigmoid())
        self.shunt3 = nn.Sequential(nn.ReLU(), nn.Conv2d(256, 384, kernel_size=1), nn.Sigmoid())
        self.shunt4 = nn.Sequential(nn.ReLU(), nn.Conv2d(384, 512, kernel_size=1), nn.Sigmoid())

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)
        d1 = self.downsample1(x)
        f1 = self.fire1(d1)
        s1 = self.shunt1(d1)
        x = torch.mul(f1, s1)
        x = self.fire2(x)
        s2 = self.shunt2(x)
        x = self.fire3(x)
        x = torch.mul(x, s2)
        d2 = self.downsample2(x)
        x = self.fire4(d2)
        s3 = self.shunt3(x)
        f5 = self.fire5(x)
        x = torch.mul(f5, s3)
        x = self.fire6(x)
        s4 = self.shunt4(x)
        x = self.fire7(x)
        x = torch.mul(x, s4)
        d3 = self.downsample3(x)
        x = self.fire8(d3)
        x = self.features(x)
        x = self.predictor(x)
        return torch.sigmoid(x)
