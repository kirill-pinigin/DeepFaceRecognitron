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
    def __init__(self, imageFolderDataset, transform=None, should_invert=True):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform
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


class ImagePredictionCSVDataSet(Dataset):
    def __init__(self, dir, csv_path,  augmentation: bool = False):
        self.root_dir = dir
        transforms_list = [
            torchvision.transforms.CenterCrop(IMAGE_SIZE),
            torchvision.transforms.ToTensor(),
        ]

        if augmentation:
            transforms_list = [
                                  torchvision.transforms.RandomCrop(IMAGE_SIZE),
                                  torchvision.transforms.RandomHorizontalFlip(),
                                  RandomNoise(),
                                  #RandomBlur(),
                                  torchvision.transforms.RandomRotation(10),

                              ] + transforms_list

        self.transforms = torchvision.transforms.Compose(transforms_list)

        self.data_info = pd.read_csv(csv_path, header=0)
        self.image_arr = np.asarray(self.data_info.iloc[0:, 0])
        self.label_arr = np.asarray(self.data_info.iloc[0:, 1:DIMENSION + 1])
        self.data_len = len(self.data_info.index)
        self.statistics()

    def __getitem__(self, index):
        single_image_name = os.path.join(self.root_dir, self.image_arr[index])
        img_as_img = load_image(single_image_name)
        image = self.transforms(img_as_img)
        target = torch.from_numpy(self.label_arr[index]).float()
        #target = (target - self.min)/ (self.max - self.min)
        return image, target

    def __len__(self):
        return self.data_len

    def statistics(self):
        print('maximum value = ', np.max(self.label_arr))
        self.max = float(np.max(self.label_arr))
        print('minimum value = ', np.min(self.label_arr))
        self.min = float(np.min(self.label_arr))
        print('average value = ', np.mean(self.label_arr))
        self.mean = float(np.mean(self.label_arr))
        print('dispersion value = ', np.std(self.label_arr))
        self.std = float(np.std(self.label_arr))
        print(self.label_arr.shape)


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

class RandomBlur(object):
    def __init__(self):
        self.blurring_filters = [ImageFilter.GaussianBlur, ImageFilter.BoxBlur]
        self.radius = [0, 1]

    def __call__(self, input):
        index = int(random.uniform(0,  len(self.blurring_filters)))
        radius = np.random.choice(self.radius)
        blurring_filter = self.blurring_filters[index](radius)
        return input.filter(blurring_filter)

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.blurring_filters)

class RandomNoise(object):
    def __init__(self):
        self.noises = [GaussianNoise, UniformNoise]
        self.factors = [0, 0.01, 0.02]

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
