import pandas as pd
import numpy as np
import os
import torchvision
import torch

from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import PIL
from PIL import ImageFilter, ImageEnhance, Image
from DeepFaceRecognitron import IMAGE_SIZE, DIMENSION, CHANNELS

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def load_image(filepath):
    if CHANNELS == 3:
        image = Image.open(filepath).convert('RGB')
    else :
        image = Image.open(filepath).convert('L')

    return image


class FaceDataset(Dataset):
    def __init__(self, image_dir, augmentation=True, should_invert=True):
        self.imageFolderDataset =  torchvision.datasets.ImageFolder(image_dir)

        transforms_list = [
            torchvision.transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            torchvision.transforms.ToTensor(),
        ]

        if augmentation:
            transforms_list = [
                                  torchvision.transforms.RandomHorizontalFlip(),
                                  torchvision.transforms.ColorJitter(0.2, 0.2),
                                  RandomNoise(),
                                  RandomSharp(),
                              ] + transforms_list

        self.transform = torchvision.transforms.Compose(transforms_list)
        self.should_invert = should_invert

    def __getitem__(self, index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)
        # we need to make sure approx 50% of images are in the same class
        should_get_same_class = random.randint(0, 1)
        if should_get_same_class:
            while True:
                # keep looping till the same class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] == img1_tuple[1]:
                    break
        else:
            while True:
                # keep looping till a different class image is found

                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] != img1_tuple[1]:
                    break

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])

        if CHANNELS == 1:
            img0 = img0.convert("L")
            img1 = img1.convert("L")

        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1, torch.from_numpy(np.array([int(img1_tuple[1] != img0_tuple[1])], dtype=np.float32))

    def __len__(self):
        return len(self.imageFolderDataset.imgs)


def make_dataloaders (dataset, batch_size, splitratio = 0.2):
    print(' split ratio ', splitratio)
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(splitratio * dataset_size))
    np.random.seed(42)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    # print(train_indices, val_indices)
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    print(train_sampler, valid_sampler)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=4,
                                               sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=4,
                                                    sampler=valid_sampler)
    print(train_loader, validation_loader)
    dataloaders = {'train': train_loader, 'val': validation_loader}
    return dataloaders

import random

class RandomSharp(object):
    def __init__(self, factor = 0.5):
        self.range = [0.0, factor]

    def __call__(self, input):
        enhancer = ImageEnhance.Sharpness(input)
        img = enhancer.enhance(random.uniform( self.range[0],  self.range[1]))
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.factors)

class RandomNoise(object):
    def __init__(self):
        self.noises = [GaussianNoise, UniformNoise]
        self.factors = [0, 0.02]

    def __call__(self, input):
        factor = np.random.choice(self.factors)
        index = int(random.uniform(0, len(self.noises)))
        noise = self.noises[index](factor)
        return noise(input)

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.noises)


class GaussianNoise(object):
    def __init__(self, factor: float = 0.1):
        self.factor = factor

    def __call__(self, input):
        img = np.array(input)
        img = img.astype(dtype=np.float32)
        noisy_img = img + np.random.normal(0.0, 255.0 * self.factor, img.shape)
        noisy_img = np.clip(noisy_img, 0.0, 255.0)
        noisy_img = noisy_img.astype(dtype=np.uint8)
        return Image.fromarray(noisy_img)

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.factor)


class UniformNoise(object):
    def __init__(self, factor: float = 0.1):
        self.factor = factor

    def __call__(self, input):
        img = np.array(input)
        img = img.astype(dtype=np.float32)
        noisy_img = img + np.random.uniform(0.0, 255.0 * self.factor, img.shape)
        noisy_img = np.clip(noisy_img, 0.0, 255.0)
        noisy_img = noisy_img.astype(dtype=np.uint8)
        return Image.fromarray(noisy_img)

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.factor)
