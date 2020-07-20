from __future__ import print_function

import time
import os

import torch
import torch.nn as nn

import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import argparse
from torchsummary import summary

from mobilenet_v1_pytorch import *

parser = argparse.ArgumentParser(description='Training a pytorch model')
parser.add_argument('-ep', '--epochs', default=150, type=int)
parser.add_argument('-b', '--batch_size', default=32, type=int)
# parser.add_argument("-g", '--use_gpu', default=True, action='store_false', help='Bool type gpu')
# parser.add_argument("-p", '--use_parallel', default=True, action='store_false', help='Bool type to use_parallel')
parser.add_argument("-lr", '--learning_rate', default=0.001, type=float, help='Learning rate for model')
parser.add_argument('-w', '--workers', default=15, type=int, help='Number of additional worker processes for dataloadin')
parser.add_argument('-d', '--data_dir', default="./ILSVRC2012_Pytorch/dataset_100/", type=str, help='path to dataset')
parser.add_argument('-m', '--model', default="ResNet50", type=str, help='model name')

args = parser.parse_args()

# Batch Size for training and testing
batch_size = args.batch_size

# Number of epochs to train for
num_epochs = args.epochs

# Starting Learning Rate
learning_rate = args.learning_rate

# Number of additional worker processes for dataloading
workers = args.workers

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Initialize Dataloaders...")
# Define the transform for the data. Notice, we must resize to 224x224 with this dataset and model.
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                         std = [ 0.229, 0.224, 0.225 ]),
])

# Training dataset and Validate dataset
# Initialize Datasets. MNIST will automatically download if not present
traindir = os.path.join(args.data_dir, 'train')
valdir = os.path.join(args.data_dir, 'val')
trainset = datasets.ImageFolder(traindir, transform)
# valset = datasets.ImageFolder(valdir, transform)

# Create the Dataloaders to feed data to the training and validation steps
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True, drop_last=True)
# val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=workers)

# Construct the model
if args.model == "ResNet50":
    model = models.resnet50(pretrained=False).to(device)
elif args.model == "MobileNetV2":
    model = models.mobilenet_v2(pretrained=False).to(device)
elif args.model == "MobileNetV1":
    model = mobilenet_v1().to(device)
else:
    print("wrong model name, please try model = ResNet50 or MobileNetV2")
    exit()
summary(model, (3, 224, 224))

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training the model
def train(train_loader, model, criterion, optimizer, epoch):
    # switch to train mode
    model.train()

    torch.backends.cudnn.benchmark = True

    batch_start = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        batch_reader_end = time.time()

        input, target = input.to(device), target.to(device)
        optimizer.zero_grad()

        # forward
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))

        # backward
        loss.backward()

        # update
        optimizer.step()

        # measure elapsed time
        train_batch_cost = time.time() - batch_start
        if i % 10 == 0:
            print('Epoch: [%d][%d/%d] avg_loss %.5f, acc_top1 %.5f, acc_top5 %.5f, batch_cost: %.5f s, reader_cost: %.5f' % (epoch, i, len(train_loader), loss, prec1, prec5, train_batch_cost, batch_reader_end - batch_start))
        batch_start = time.time()

best_prec1 = 0

for epoch in range(num_epochs):
    # Train for one epoch
    print("\nBegin Training Epoch {}".format(epoch+1))
    epoch_start_time = time.time()
    train(train_loader, model, criterion, optimizer, epoch)
    epoch_end_time = time.time()
    print("\nAfter Training Epoch {} time is: {:.4f}".format(epoch+1, epoch_end_time - epoch_start_time))

    # Evaluate on validation set
    # print("Begin Validation @ Epoch {}".format(epoch+1))
    # prec1 = validate(val_loader, model, criterion)
    # best_prec1 = max(prec1, best_prec1)
    # print("Epoch Summary: ")
    # print("\tEpoch Accuracy: {}".format(prec1))
    # print("\tBest Accuracy: {}".format(best_prec1))

