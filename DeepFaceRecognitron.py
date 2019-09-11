import time
import sys
import os
import torch
from torch.autograd import Variable
import shutil
import numpy as np

IMAGE_SIZE = 256
CHANNELS = 3
DIMENSION = 128

LR_THRESHOLD = 1e-7
TRYING_LR = 3
DEGRADATION_TOLERANCY = 7
ACCURACY_TRESHOLD = float(0.0625)


class DeepFaceRecognitron(object):
    def __init__(self, predictor,  criterion, optimizer, directory):
        self.predictor = predictor
        self.criterion = criterion
        self.accuracy = torch.nn.L1Loss()
        self.optimizer = optimizer
        self.use_gpu = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.cudas = list(range(torch.cuda.device_count()))

        config = str(predictor.__class__.__name__) + '_' + str(predictor.activation.__class__.__name__)
        config += '_' + str(criterion.__class__.__name__)
        config += "_" + str(optimizer.__class__.__name__)

        print(self.device)
        print(torch.cuda.device_count())

        reportPath = os.path.join(directory, config + "/report/")
        flag = os.path.exists(reportPath)
        if flag != True:
            os.makedirs(reportPath)
            print('os.makedirs("reportPath")')

        self.modelPath = os.path.join(directory, config + "/model/")
        flag = os.path.exists(self.modelPath)
        if flag != True:
            os.makedirs(self.modelPath)
            print('os.makedirs("/modelPath/")')

        self.images = os.path.join(directory, config + "/images/")
        flag = os.path.exists(self.images)
        if flag != True:
            os.makedirs(self.images+'/bad/')
            os.makedirs(self.images + '/good/')
            print('os.makedirs("/images/")')
        else:
            shutil.rmtree(self.images)

        self.report = open(reportPath  + '/' + config + "_Report.txt", "w")
        _stdout = sys.stdout
        sys.stdout = self.report
        print(config)
        print(predictor)
        print(criterion)
        self.report.flush()
        sys.stdout = _stdout
        self.predictor = self.predictor.to(self.device)

    def __del__(self):
        self.report.close()

    def approximate(self, dataloaders, num_epochs = 20, resume_train = False):
        path = self.modelPath +"/"+ str(self.predictor.__class__.__name__) +  str(self.predictor.activation.__class__.__name__)
        if resume_train and os.path.isfile(path + '_Best.pth'):
            print( "RESUME training load Bestpredictor")
            self.predictor.load_state_dict(torch.load(path + '_Best.pth'))

        since = time.time()
        best_loss = 10000.0
        best_acc = 0.0
        counter = 0
        i = int(0)
        degradation = 0

        for epoch in range(num_epochs):
            _stdout = sys.stdout
            sys.stdout = self.report
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)
            self.report.flush()
            sys.stdout = _stdout
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            for phase in ['train', 'val']:

                if phase == 'train':
                    self.predictor.train(True)
                else:
                    self.predictor.train(False)

                running_loss = 0.0
                running_corrects = 0

                for data in dataloaders[phase]:
                    inputs1, inputs2, targets = data

                    inputs1 = Variable(inputs1.to(self.device))
                    inputs2 = Variable(inputs2.to(self.device))
                    targets = Variable(targets.to(self.device))

                    self.optimizer.zero_grad()
                    outputs1 = torch.nn.parallel.data_parallel(module=self.predictor, inputs=inputs1, device_ids = self.cudas)
                    outputs2 = torch.nn.parallel.data_parallel(module=self.predictor, inputs=inputs2,device_ids=self.cudas)
                    #diff = self.accuracy(outputs, targets)
                    #diff = float(1.0) - diff
                    loss = self.criterion(outputs1,outputs2, targets)

                    if phase == 'train':
                        loss.backward()
                        self.optimizer.step()

                    running_loss += loss.item() * targets.size(0)
                    #running_corrects += diff.item() * inputs.size(0)

                epoch_loss = float(running_loss) / float(len(dataloaders[phase].dataset))
                epoch_acc = float(running_corrects) / float(len(dataloaders[phase].dataset))

                _stdout = sys.stdout
                sys.stdout = self.report
                print('{} Loss: {:.4f} Accuracy  {:.4f} '.format(
                    phase, epoch_loss, epoch_acc))
                self.report.flush()

                sys.stdout = _stdout
                print('{} Loss: {:.4f} Accuracy  {:.4f} '.format(
                    phase, epoch_loss, epoch_acc))
                self.report.flush()

                if phase == 'val' and epoch_loss < best_loss:
                    counter = 0
                    degradation = 0
                    best_acc = epoch_acc
                    best_loss = epoch_loss
                    print('curent best_loss ', best_loss)
                    self.save('Best')
                else:
                    counter += 1
                    self.save('Regular')

            if counter > TRYING_LR * 2:
                for param_group in self.optimizer.param_groups:
                    lr = param_group['lr']
                    if lr >= LR_THRESHOLD:
                        param_group['lr'] = lr * 0.2
                        print('! Decrease LearningRate !', lr)

                counter = 0
                degradation += 1
            if degradation > DEGRADATION_TOLERANCY:
                print('This is the end! Best val best_loss: {:4f}'.format(best_loss))
                return best_acc

        time_elapsed = time.time() - since

        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val best_loss: {:4f}'.format(best_loss))
        return best_acc


    def estimate(self, test_loader, isSaveImages = True, modelPath=None):
        counter = 0
        if modelPath is not None:
            self.predictor.load_state_dict(torch.load(modelPath))
            print('load Predictor model')
        else:
            self.predictor.load_state_dict(torch.load(self.modelPath +"/"+ str(self.predictor.__class__.__name__) +  str(self.predictor.activation.__class__.__name__) + '_BestPredictor.pth'))
            print('load BestPredictor ')
        print(len(test_loader.dataset))
        i = 0
        since = time.time()
        self.predictor.train(False)
        self.predictor.eval()
        self.predictor = self.predictor.to(self.device)
        running_loss = 0.0
        running_corrects = 0
        for data in test_loader:
            inputs, targets = data
            inputs = Variable(inputs.to(self.device))
            targets = Variable(targets.to(self.device))
            outputs = self.predictor(inputs)
            diff = self.accuracy(outputs, targets)
            diff = float(1.0) - diff
            loss = self.criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += diff.item() * inputs.size(0)

            if isSaveImages and test_loader.batch_size == 1:
                result = torch.round(outputs).data.cpu().numpy()
                result = np.squeeze(result)
                indexes = np.nonzero(result)
                image = inputs.clone()
                image = image.data.cpu().float()
                counter = counter + 1
                if float(diff.item()) > 0.5:
                    filename = self.images + '/bad/'
                else:
                    filename = self.images + '/good/'

                filename+= str(counter) + "__" + str(float(diff.item()))+'__.png'
                #torchvision.utils.save_image(image, filename)

        epoch_loss = float(running_loss) / float(len(test_loader.dataset))
        epoch_acc = float(running_corrects) / float(len(test_loader.dataset))

        time_elapsed = time.time() - since

        print('Evaluating complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Loss: {:.4f} Accuracy {:.4f} '.format( epoch_loss, epoch_acc))
        #self.report.flush()

    def save(self, model):
        self.predictor = self.predictor.cpu()
        self.predictor.eval()
        x = Variable(torch.zeros(1, CHANNELS, IMAGE_SIZE, IMAGE_SIZE))
        path = self.modelPath +"/"+ str(self.predictor.__class__.__name__ ) +  str(self.predictor.activation.__class__.__name__)
        torch.save(self.predictor.state_dict(), path + "_" + model + ".pth")
        source = "Color" if CHANNELS == 3 else "Gray"
        torch_out = torch.onnx._export(self.predictor, x, path + source + str(IMAGE_SIZE) + "_" + model + ".onnx", export_params=True)
        self.predictor = self.predictor.to(self.device)

