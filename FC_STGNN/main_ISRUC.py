from data_loader_ISRUC import data_generator

import torch as tr
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
import math
import os

import Model

import argparse
import matplotlib.pyplot as plt
import random
class Train():
    def __init__(self, args):


        self.train, self.valid, self.test = data_generator('./ISRUC/', args)

        self.args = args
        self.net = Model.FC_STGNN_SSC(args.patch_size,args.conv_out, args.lstmhidden_dim, args.lstmout_dim,args.conv_kernel,args.hidden_dim,args.time_denpen_len, args.num_sensor, args.num_windows,args.moving_window,args.stride, args.decay, args.pool_choice, args.n_class, args.dropout)

        self.net = self.net.cuda() if tr.cuda.is_available() else self.net
        self.loss_function = nn.CrossEntropyLoss()
        self.optim = optim.Adam(self.net.parameters())

    def data_preparation(self, data, label):
        for i in range(len(data)):

            print(data[i].shape)


    def Train_batch(self):
        self.net.train()
        loss_ = 0
        for data, label in self.train:
            data = data.cuda() if tr.cuda.is_available() else data
            label = label.cuda() if tr.cuda.is_available() else label
            self.optim.zero_grad()
            prediction = self.net(data)
            loss = self.loss_function(prediction, label)
            loss.backward()
            self.optim.step()
            loss_ = loss_ + loss.item()
        return loss_

    def Train_model(self):
        epoch = self.args.epoch
        cross_accu = 0
        test_accu_ = []
        prediction_ = []
        real_ = []

        for i in range(epoch):
            time0 = time.time()
            loss = self.Train_batch()
            # print('one iteration time is {}'.format(time.time() - time0))
            if i%self.args.show_interval == 0:
                accu_val = self.Cross_validation()

                if accu_val > cross_accu:
                    cross_accu = accu_val
                    test_accu, prediction, real = self.Prediction()

                    print('In the {}th epoch, TESTING accuracy is {}%'.format(i, np.round(test_accu, 3)))
                    test_accu_.append(test_accu)
                    prediction_.append(prediction)
                    real_.append(real)

        np.save('./experiment/{}.npy'.format(self.args.save_name),[test_accu_, prediction_, real_])

    def cuda_(self, x):
        x = tr.Tensor(np.array(x))
        # x = tr.Tensor(x)

        if tr.cuda.is_available():
            return x.cuda()
        else:
            return x


    def Cross_validation(self):
        self.net.eval()
        prediction_ = []
        real_ = []
        for data, label in self.valid:
            data = data.cuda() if tr.cuda.is_available() else data
            real_.append(label)
            prediction = self.net(data)
            prediction_.append(prediction.detach().cpu())
        prediction_ = tr.cat(prediction_,0)
        real_ = tr.cat(real_,0)

        prediction_ = tr.argmax(prediction_,-1)

        accu = self.accu_(prediction_, real_)
        # print(accu)
        return accu

    def Prediction(self):
        '''
        This is to predict the results for testing dataset
        :return:
        '''
        self.net.eval()
        prediction_ = []
        real_ = []
        for data, label in self.test:
            data = data.cuda() if tr.cuda.is_available() else data
            real_.append(label)
            prediction = self.net(data)
            prediction_.append(prediction.detach().cpu())
        prediction_ = tr.cat(prediction_, 0)
        real_ = tr.cat(real_, 0)

        prediction_ = tr.argmax(prediction_, -1)
        accu = self.accu_(prediction_, real_)
        return accu, prediction_.numpy(), real_.numpy()


    def accu_(self, predicted, real):
        num = predicted.size(0)
        real_num = 0
        for i in range(num):
            if predicted[i] == real[i]:
                real_num+=1
        return 100*real_num/num


if __name__ == '__main__':
    from args import args

    args = args()


    def args_config_ISRUC(args):
        args.epoch = 41
        args.k = 1

        args.decay = 0.8
        args.pool_choice = 'mean'
        args.moving_window = [2, 2]
        args.stride = [1, 2]
        args.lr = 1e-3
        args.batch_size = 100

        args.window_sample = 300
        args.conv_time_CNN = 6

        args.lstmout_dim = 24
        args.lstmhidden_dim = 24
        args.hidden_dim = 16

        args.num_sensor = 10
        args.n_class = 5

        args.conv_kernel = 5
        args.dropout = 0.4

        args.patch_size = 75
        args.time_denpen_len = int(args.window_sample / args.patch_size)
        args.conv_out = 12
        args.num_windows = 5

        return args

    args = args_config_ISRUC(args)

    train = Train(args)
    train.Train_model()
