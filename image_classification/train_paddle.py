# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#order: standard library, third party, local library 
import time
import sys
import math
import argparse
import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.base import to_variable
import reader2 as reader
from utils import *
from mobilenet_v1 import *
from mobilenet_v2 import *
from resnet import *

args = parse_args()
print_arguments(args)


def train_model():
    place = fluid.CUDAPlace(0)
    
    with fluid.dygraph.guard(place):
        # 1. init net and optimizer
        if args.model == "MobileNetV1":
            net = MobileNetV1(class_dim=args.class_dim, scale=1.0)
        elif args.model == "MobileNetV2":
            net = MobileNetV2(class_dim=args.class_dim, scale=1.0)
        elif args.model == "ResNet50":
            net = ResNet()
        else:
            print(
                "wrong model name, please try model = ResNet50 or MobileNetV1 or MobileNetV2"
            )
            exit()

        optimizer = fluid.optimizer.AdamOptimizer(parameter_list=net.parameters())
        # for param in net.parameters():
        #     print(param.name, param.shape)


        # 2. reader
        train_data_loader = fluid.io.DataLoader.from_generator(capacity=32, use_double_buffer=True, iterable=True, return_list=True, use_multiprocess=True)
        #test_data_loader = fluid.io.DataLoader.from_generator(capacity=64, use_double_buffer=True, iterable=True, return_list=True, use_multiprocess=True)
        imagenet_reader = reader.ImageNetReader(0)
        train_reader = imagenet_reader.train(settings=args)
        #test_reader = imagenet_reader.val(settings=args)
        train_data_loader.set_sample_list_generator(train_reader, place)
        #test_data_loader.set_sample_list_generator(test_reader, place)

        # 3. train loop
        for eop in range(args.num_epochs):
            net.train()

            print("\nBegin Training Epoch {}".format(eop+1))
            epoch_start_time = time.time()

            batch_id = 0
            t_last = 0
            for img, label in train_data_loader():
                t1 = time.time()
                
                out = net(img)
                softmax_out = fluid.layers.softmax(out, use_cudnn=False)
                loss = fluid.layers.cross_entropy(input=softmax_out, label=label)
                avg_loss = fluid.layers.mean(x=loss)
                acc_top1 = fluid.layers.accuracy(input=out, label=label, k=1)
                acc_top5 = fluid.layers.accuracy(input=out, label=label, k=5)
                avg_loss.backward()
                optimizer.minimize(avg_loss)
                net.clear_gradients()
                t2 = time.time() 
                train_batch_elapse = t2 - t1
                
                if batch_id % args.print_step == 0:
                    print( "epoch id: %d, batch step: %d,  avg_loss %0.5f acc_top1 %0.5f acc_top5 %0.5f forward_backward %2.4f read_t:%2.4f" % (eop, batch_id, avg_loss.numpy(), acc_top1.numpy(), acc_top5.numpy(), train_batch_elapse, t1 - t_last))
                # sys.stdout.flush()
                batch_id += 1
                t_last = time.time()

            epoch_end_time = time.time()
            print("\nAfter Training Epoch {} time is: {:.4f}".format(eop+1, epoch_end_time - epoch_start_time))


if __name__ == '__main__':
    train_model()
