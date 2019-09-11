import torch
import torch.nn as nn
from torchvision import models
from NeuralModels import SILU, Perceptron
from DeepFaceRecognitron import DIMENSION, CHANNELS

LATENT = 512
REDUCTION = 7
FEATURES = 64


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, activation = nn.ReLU()):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.activation = activation
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.activation(out)

        return out

    def configure(self, other):
        self.conv1.weight.data.copy_(other.conv1.weight)
        self.conv2.weight.data.copy_(other.conv2.weight)

        self.bn1.weight.data.copy_(other.bn1.weight)
        self.bn1.bias.data.copy_(other.bn1.bias)
        self.bn1.affine = (other.bn1.affine)
        self.bn2.weight.data.copy_(other.bn2.weight)
        self.bn2.bias.data.copy_(other.bn2.bias)
        self.bn2.affine = (other.bn2.affine)

        if self.downsample is not None and other.downsample is not None:
            self.downsample = other.downsample


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, activation = nn.ReLU()):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.activation = activation
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.activation(out)

        return out


class ResidualPredictor(nn.Module):
    def __init__(self, activation = SILU(), pretrained = True):
        super(ResidualPredictor, self).__init__()
        self.activation = activation
        self.inplanes = FEATURES
        layers = [2,2,2,2]
        block = BasicBlock
        base_model = models.resnet18(pretrained=pretrained)
        conv = nn.Conv2d(CHANNELS, FEATURES, kernel_size=7, stride=2, padding=3, bias=False)
        weight = torch.FloatTensor(FEATURES, CHANNELS, 7, 7)
        parameters = list(base_model.parameters())
        for i in range(FEATURES):
            if CHANNELS == 1:
                weight[i, :, :, :] = parameters[0].data[i].mean(0)
            else:
                weight[i, :, :, :] = parameters[0].data[i]
        conv.weight.data.copy_(weight)

        self.conv1 = conv
        self.bn1 = nn.BatchNorm2d(FEATURES)
        self.activation = activation
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64,  layers[0], stride=1, activation = activation)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, activation = activation)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, activation = activation)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, activation = activation)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        if pretrained:
            self.bn1.weight.data.copy_(base_model.bn1.weight)
            self.bn1.bias.data.copy_(base_model.bn1.bias)
            self.bn1.affine = (base_model.bn1.affine)
            for i in range(layers[0]):
                self.layer1[i].configure(base_model.layer1[i])
            for i in range(layers[0]):
                self.layer2[i].configure(base_model.layer2[i])
            for i in range(layers[0]):
                self.layer3[i].configure(base_model.layer3[i])
            for i in range(layers[0]):
                self.layer4[i].configure(base_model.layer4[i])

        self.predictor = nn.Sequential(
            Perceptron(LATENT, LATENT),
            nn.Dropout(p=0),
            activation,
            Perceptron(LATENT, DIMENSION),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)

        x = self.predictor(x)
        return torch.sigmoid(x)

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.fc.parameters():
            param.requires_grad = True

    def unfreeze(self):
        for param in self.model.parameters():
            param.requires_grad = True

    def get_dropout(self):
        return self.predictor[1].p

    def set_dropout(self, proba = 0):
        if proba < 0:
            proba = 0
        if proba > 0.99:
            proba = 0.99
        self.predictor[1].p = proba

    def _make_layer(self, block, planes, blocks, stride=1, activation = nn.ReLU):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        llayers = []
        llayers.append(block(self.inplanes, planes, stride, downsample, activation=activation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            llayers.append(block(inplanes=self.inplanes, planes=planes, activation=activation))

        return nn.Sequential(*llayers)

    def _copy_layer(self, dest, source ):
        source_parameters = []
        for child in source.children():
            source_parameters.append(child.parameters())





'''
class ResidualPredictor(nn.Module):
    def __init__(self, channels = 1, dimension=11, activation = SILU(), pretrained = True):
        super(ResidualPredictor, self).__init__()
        self.activation = activation

        self.model = models.resnet18(pretrained=pretrained)
        conv = nn.Conv2d(channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        weight = torch.FloatTensor(64, channels, 7, 7)
        parameters = list(self.model.parameters())
        for i in range(64):
            if channels == 1:
                weight[i, :, :, :] = parameters[0].data[i].mean(0)
            else:
                weight[i, :, :, :] = parameters[0].data[i]
        conv.weight.data.copy_(weight)
        self.model.conv1 = conv

        self.model.avgpool = nn.AvgPool2d(8)
        num_ftrs = self.model.fc.in_features

        self.model.fc = nn.Sequential(
            Perceptron(num_ftrs, num_ftrs),
            activation,
            Perceptron(num_ftrs, dimension),
        )

    def forward(self, x):
        return self.model(x)

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.fc.parameters():
            param.requires_grad = True

    def unfreeze(self):
        for param in self.model.parameters():
            param.requires_grad = True

'''