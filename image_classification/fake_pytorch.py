from __future__ import print_function

import time
import os
import numpy as np

import torch
import torch.nn as nn

import torchvision.models as models

import argparse
from torchsummary import summary

from mobilenet_v1_pytorch import *

parser = argparse.ArgumentParser(description='Training a pytorch model')
parser.add_argument('-ep', '--epochs', default=150, type=int)
parser.add_argument('-b', '--batch_size', default=32, type=int)
# parser.add_argument("-g", '--use_gpu', default=True, action='store_false', help='Bool type gpu')
# parser.add_argument("-p", '--use_parallel', default=True, action='store_false', help='Bool type to use_parallel')
parser.add_argument("-lr", '--learning_rate', default=0.001, type=float, help='Learning rate for model')
parser.add_argument('-m', '--model', default="ResNet50", type=str, help='model name')
parser.add_argument('-d', '--data_dir', default="./ILSVRC2012_Pytorch/dataset_100/", type=str, help='path to dataset')

args = parser.parse_args()

# Batch Size for training and testing
batch_size = args.batch_size

# Number of epochs to train for
num_epochs = args.epochs

# Starting Learning Rate
learning_rate = args.learning_rate

train_images = 128660

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Construct the model
if args.model == "ResNet50":
    model = models.resnet50(pretrained=False).to(device)
elif args.model == "MobileNetV2":
    model = models.mobilenet_v2(pretrained=False).to(device)
elif args.model == "MobileNetV1":
    model = mobilenet_v1().to(device)
elif args.model == "ResNet101":
    model = models.resnet101(pretrained=False).to(device)
else:
    print("wrong model name, please try model = ResNet50 or MobileNetV2")
    exit()
summary(model, (3, 224, 224))

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
def Train(model, criterion, optimizer, epoch, input_fake, target_fake):
    # switch to train mode
    model.train()

    global train_images
    global batch_size
    batch_number = train_images / batch_size

    for i in range(int(batch_number)):
        compute_start_time = time.time()
        input, target = input_fake.to(device), target_fake.to(device)
        optimizer.zero_grad()

        # forward
        output = model(input)
        loss = criterion(output, target)

        # backward
        loss.backward()

        # update
        optimizer.step()

        # measure elapsed time
        compute_end_time = time.time()

        print('Epoch: [%d][%d] forward_backward %2.4f' % (epoch, i, compute_end_time - compute_start_time))

for epoch in range(num_epochs):
    # Train for one epoch
    input  = np.ones((batch_size, 3, 224, 224)).astype(np.float32)
    target = np.ones((batch_size,)).astype(np.int)
    input, target = torch.from_numpy(input), torch.from_numpy(target)
    print("\nBegin Train Epoch {}".format(epoch+1))
    epoch_start_time = time.time()
    Train(model, criterion, optimizer, epoch, input, target)
    epoch_end_time = time.time()
    print("\nAfter Train Epoch {} time is: {:.4f}".format(epoch+1, epoch_end_time - epoch_start_time))

