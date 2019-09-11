import argparse
import os
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision

from DeepFaceRecognitron import DeepFaceRecognitron , IMAGE_SIZE
from  FaceDataset import  FaceDataset
from FaceRecognitionLoss import ContrastiveLoss
from MobilePredictor import MobilePredictor

from ResidualPredictor import ResidualPredictor
from SqueezePredictors import  SqueezeSimplePredictor, SqueezeResidualPredictor, SqueezeShuntPredictor
from NeuralModels import SILU

parser = argparse.ArgumentParser()
parser.add_argument('--image_dir',      type = str,   default='./SmallFaceDataset256/', help='path to dataset')
parser.add_argument('--result_dir',     type = str,   default='./RESULTS/', help='path to result')
parser.add_argument('--predictor',      type = str,   default='SqueezeResidualPredictor', help='type of image generator')
parser.add_argument('--activation',     type = str,   default='LeakyReLU', help='type of activation')
parser.add_argument('--criterion',      type = str,   default='BCE', help='type of criterion')
parser.add_argument('--optimizer',      type = str,   default='Adam', help='type of optimizer')
parser.add_argument('--lr',             type = float, default=1e-3)
parser.add_argument('--l2',             type = float, default=0)
parser.add_argument('--batch_size',     type = int,   default=64)
parser.add_argument('--epochs',         type = int,   default=64)
parser.add_argument('--resume_train',   type = bool,  default=True, help='type of training')

args = parser.parse_args()

predictor_types = { 'ResidualPredictor'        : ResidualPredictor,
                    'MobilePredictor'          : MobilePredictor,
                    'SqueezeSimplePredictor'   : SqueezeSimplePredictor,
                    'SqueezeResidualPredictor' : SqueezeResidualPredictor,
                    'SqueezeShuntPredictor'    : SqueezeShuntPredictor
                    }

activation_types = {'ReLU'      : nn.ReLU(),
                    'LeakyReLU' : nn.LeakyReLU(),
                    'PReLU'     : nn.PReLU(),
                    'ELU'       : nn.ELU(),
                    'SELU'      : nn.SELU(),
                    'SILU'      : SILU()
              }

optimizer_types = {
                    'Adam'     : optim.Adam,
                    'RMSprop'  : optim.RMSprop,
                    'SGD'      : optim.SGD
                    }

model = (predictor_types[args.predictor] if args.predictor in predictor_types else predictor_types['ResidualPredictor'])
function = (activation_types[args.activation] if args.activation in activation_types else activation_types['ReLU'])
predictor = model(activation=function)
optimizer =(optimizer_types[args.optimizer] if args.optimizer in optimizer_types else optimizer_types['Adam'])(predictor.parameters(), lr = args.lr, weight_decay=args.l2)

criterion = ContrastiveLoss()

augmentations = {'train' : True, 'val' : False}
shufles = {'train' : True, 'val' : False}
train_transforms_list = [
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ColorJitter(0.2, 0.2),
        torchvision.transforms.Resize((int(IMAGE_SIZE * 1), int(IMAGE_SIZE * 1)), interpolation=3),
        torchvision.transforms.ToTensor(),
        ]

val_transforms_list = [
        torchvision.transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=3),
        torchvision.transforms.ToTensor(),
        ]

data_transforms = {
    'train':    torchvision.transforms.Compose(train_transforms_list ),
    'val':      torchvision.transforms.Compose(val_transforms_list),
}

folder_datasets = {x: torchvision.datasets.ImageFolder(os.path.join(args.image_dir, x))for x in ['train', 'val']}
image_datasets = {x: FaceDataset(folder_datasets[x], transform=data_transforms[x])for x in ['train', 'val']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size,
                                             shuffle=shufles[x], num_workers=4)
                for x in ['train', 'val']}

testloader =  torch.utils.data.DataLoader(dataloaders['val'], batch_size=1, shuffle=False, num_workers=4)

framework = DeepFaceRecognitron(predictor = predictor, criterion = criterion, optimizer = optimizer,  directory = args.result_dir)
framework.approximate(dataloaders = dataloaders, num_epochs=args.epochs, resume_train=args.resume_train)
framework.estimate(testloader)
