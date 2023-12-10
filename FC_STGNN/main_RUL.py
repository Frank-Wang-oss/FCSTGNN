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

from data_loader_RUL import CMPDataIter_graph
import matplotlib.pyplot as plt
import random

class Train():
    def __init__(self, args):


        data_iter = CMPDataIter_graph('./',
                                data_set='FD00{}'.format(str(args.data_sub)),
                                max_rul=args.max_rul,
                                seq_len=args.patch_size,
                                time_denpen_len=args.time_denpen_len,
                                window_sample=args.window_sample,
                                net_name=1)


        self.args = args
        self.train_data = self.cuda_(data_iter.out_x)
        self.train_ops = self.cuda_(data_iter.out_ops)
        self.train_label = self.cuda_(data_iter.out_y)

        self.val_data = self.cuda_(data_iter.cross_val_x)
        self.val_ops = self.cuda_(data_iter.cross_val_ops)
        self.val_label = self.cuda_(data_iter.cross_val_y)

        self.test_data = self.cuda_(data_iter.test_x)
        self.test_ops = self.cuda_(data_iter.test_ops)
        self.test_label = self.cuda_(data_iter.test_y)

        self.train_data, self.train_ops = self.data_preprocess_transpose(self.train_data, self.train_ops)
        self.test_data, self.test_ops = self.data_preprocess_transpose(self.test_data, self.test_ops)
        self.val_data, self.val_ops = self.data_preprocess_transpose(self.val_data, self.val_ops)

        self.net = Model.FC_STGNN_RUL(args.patch_size,args.conv_out, args.lstmhidden_dim, args.lstmout_dim,args.conv_kernel, args.hidden_dim,args.conv_time_CNN, args.num_sensor, args.num_windows,args.moving_window,args.stride, args.decay, args.pool_choice, 1)


        self.net = self.net.cuda() if tr.cuda.is_available() else self.net
        self.loss_function = nn.MSELoss()
        self.optim = optim.Adam(self.net.parameters(), lr=args.lr)

    def Train_batch(self):
        self.net.train()
        loss_ = 0
        batch_size = self.args.batch_size
        iter = int(self.train_data.size(0) / batch_size)
        remain = self.train_data.size(0) - iter * batch_size
        for i in range(iter):

            data = self.train_data[i * batch_size:(i + 1) * batch_size]
            label = self.train_label[i * batch_size:(i + 1) * batch_size]
            self.optim.zero_grad()
            prediction = self.net(data)
            loss = self.loss_function(prediction, label)

            loss.backward()
            self.optim.step()
            loss_ = loss_ + loss.item()

        if remain != 0:

            data = self.train_data[-remain:]
            label = self.train_label[-remain:]
            self.optim.zero_grad()
            prediction = self.net(data)

            loss = self.loss_function(prediction, label)

            loss.backward()
            self.optim.step()
            loss_ = loss_ + loss.item()
        return loss_

    def Train_model(self):
        epoch = self.args.epoch
        cross_loss = np.inf
        test_RMSE = []
        test_score = []
        RUL_predicted = []
        RUL_real = []

        train_RMSE = []
        train_score = []
        train_RUL_predicted = []
        train_RUL_real = []

        for i in range(epoch):
            loss = self.Train_batch()
            if i%self.args.show_interval == 0:
                loss_val = self.Cross_validation()

                if loss_val < cross_loss:
                    cross_loss = loss_val
                    test_RMSE_, test_score_, test_result_predicted, test_result_real = self.Prediction()
                    train_RMSE_, train_score_, train_result_predicted, train_result_real = self.Prediction_training()

                    print('In the {}th epoch, TESTING RMSE is {}, TESTING Score is {}'.format(i, test_RMSE_, test_score_))
                    test_RMSE.append(test_RMSE_)
                    test_score.append(test_score_[0])
                    RUL_predicted.append(test_result_predicted)
                    RUL_real.append(test_result_real)

                    train_RMSE.append(train_RMSE_)
                    train_score.append(train_score_[0])
                    train_RUL_predicted.append(train_result_predicted)
                    train_RUL_real.append(train_result_real)

        index_best_test = np.argmin(test_RMSE)
        predicted_RUL = RUL_predicted[index_best_test]
        real_RUL = RUL_real[index_best_test]

        index_best_train = np.argmin(train_RMSE)
        train_predicted_RUL = train_RUL_predicted[index_best_train]
        train_real_RUL = train_RUL_real[index_best_train]


        print(test_RMSE[index_best_test])
        print(test_score[index_best_test])
        test_RMSE = np.stack(test_RMSE,0)
        test_score = np.stack(test_score,0)

        test_results = np.stack([test_RMSE, test_score],0)
        np.save('./experiment/{}.npy'.format(self.args.save_name),test_results)

    def cuda_(self, x):
        x = tr.Tensor(x)

        if tr.cuda.is_available():
            return x.cuda()
        else:
            return x

    def data_preprocess_transpose(self, data, ops):
        '''

        :param data: size is [bs, time_length, dimension, Num_nodes]
        :return: size is [bs, time_length, Num_nodes, dimension]
        '''

        data = tr.transpose(data,2,3)
        ops = tr.transpose(ops,2,3)

        return data, ops

    def Cross_validation(self):
        self.net.eval()
        loss_ = 0
        batch_size = self.args.batch_size
        iter = int(self.val_data.size(0) / batch_size)
        remain = self.val_data.size(0) - iter * batch_size
        for i in range(iter):
            data = self.val_data[i * batch_size:(i + 1) * batch_size]
            label = self.val_label[i * batch_size:(i + 1) * batch_size]
            prediction = self.net(data)
            loss = self.loss_function(prediction, label)
            loss_ = loss_ + loss.item()

        if remain != 0:
            data = self.val_data[-remain:]
            label = self.val_label[-remain:]
            prediction = self.net(data)
            loss = self.loss_function(prediction, label)
            loss_ = loss_ + loss.item()
        # print(loss_)
        return loss_

    def Prediction(self):
        '''
        This is to predict the results for testing dataset
        :return:
        '''
        self.net.eval()
        prediction = self.net(self.test_data)
        predicted_RUL = prediction
        real_RUL = self.test_label
        MSE = self.loss_function(predicted_RUL, real_RUL)

        RMSE = tr.sqrt(MSE)*self.args.max_rul
        score = self.scoring_function(predicted_RUL, real_RUL)
        return RMSE.detach().cpu().numpy(),\
               score.detach().cpu().numpy(), \
               predicted_RUL.detach().cpu().numpy(), \
               real_RUL.detach().cpu().numpy()

    def Prediction_training(self):
        self.net.eval()
        sample_idx = random.sample(range(len(self.train_data)), 100)
        train_data_sample = self.train_data[sample_idx]
        train_label_sample = self.train_label[sample_idx]
        prediction = self.net(train_data_sample)
        MSE = self.loss_function(prediction, train_label_sample)
        RMSE = tr.sqrt(MSE) * self.args.max_rul
        score = self.scoring_function(prediction, train_label_sample)
        return RMSE.detach().cpu().numpy(), \
               score.detach().cpu().numpy(), \
               prediction.detach().cpu().numpy(), \
               train_label_sample.detach().cpu().numpy()




    def visualization(self, prediction, real):
        fig = plt.figure()
        sub = fig.add_subplot(1, 1, 1)

        sub.plot(prediction, color = 'red', label = 'Predicted Labels')
        sub.plot(real, 'black', label = 'Real Labels')
        sub.legend()
        plt.show()

    def scoring_function(self, predicted, real):
        score = 0
        num = predicted.size(0)
        for i in range(num):

            if real[i] > predicted[i]:
                score = score+ (tr.exp((real[i]*self.args.max_rul-predicted[i]*self.args.max_rul)/13)-1)

            elif real[i]<= predicted[i]:
                score = score + (tr.exp((predicted[i]*self.args.max_rul-real[i]*self.args.max_rul)/10)-1)

        return score


if __name__ == '__main__':
    from args import args

    args = args()


    def args_config(data_sub, args):

        args.epoch = 41
        args.k = 1
        args.batch_size = 100
        args.conv_kernel = 2
        args.lr = 1e-3
        args.moving_window = [2, 2]
        args.stride = [1, 2]
        args.pool_choice = 'mean'
        args.decay = 0.7

        if data_sub == 1:
            args.data_sub = 1
            args.patch_size = 5
            args.time_denpen_len = 6
            args.conv_out = 7
            args.num_windows = 8
            args.lstmout_dim = 32
            args.hidden_dim = 8
            args.window_sample = 30
            args.conv_time_CNN = 6
            args.lstmhidden_dim = 8

        if data_sub == 2:
            args.data_sub = 2
            args.patch_size = 5
            args.time_denpen_len = 10
            args.conv_out = 7
            args.num_windows = 14
            args.lstmout_dim = 12
            args.hidden_dim = 8
            args.window_sample = 50
            args.conv_time_CNN = 10
            args.lstmhidden_dim = 8

        if data_sub == 3:
            args.data_sub = 3
            args.patch_size = 2
            args.time_denpen_len = 25
            args.conv_out = 4
            args.num_windows = 36
            args.lstmout_dim = 6
            args.hidden_dim = 24
            args.window_sample = 50
            args.conv_time_CNN = 25
            args.lstmhidden_dim = 8


        if data_sub == 4:
            args.data_sub = 4
            args.patch_size = 5
            args.time_denpen_len = 10
            args.conv_out = 7
            args.num_windows = 14
            args.lstmout_dim = 6
            args.hidden_dim = 8
            args.window_sample = 50
            args.conv_time_CNN = 10
            args.lstmhidden_dim = 8

        return args

    args = args_config(1, args)
    train = Train(args)
    train.Train_model()