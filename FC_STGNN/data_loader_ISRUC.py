import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import numpy as np


class Load_Dataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, X_train,y_train, args):
        super(Load_Dataset, self).__init__()

        X_train = X_train[:,:,::10]
        y_train = np.argmax(y_train,-1)
        if len(X_train.shape) < 3:
            X_train = X_train.unsqueeze(2)

        if X_train.shape.index(min(X_train.shape)) != 1:  # make sure the Channels in second dim
            X_train = X_train.permute(0, 2, 1)

        if isinstance(X_train, np.ndarray):
            self.x_data = torch.from_numpy(X_train).float()
            self.y_data = torch.from_numpy(y_train).long()
        else:
            self.x_data = X_train.float()
            self.y_data = y_train.long()


        self.len = X_train.shape[0]
        shape = self.x_data.size()
        self.x_data = self.x_data.reshape(shape[0],shape[1],args.time_denpen_len, args.window_size)
        self.x_data = torch.transpose(self.x_data, 1,2)


    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

def data_preparation(Fold_data, Fold_Label):
    train_data = []
    train_label = []
    val_data = []
    val_label = []
    test_data = []
    test_label = []
    for i in range(len(Fold_data)):
        data_idx = Fold_data[i]
        label_idx = Fold_Label[i]
        len_idx = len(data_idx)
        num_train = int(len_idx*0.6)
        num_val = int(len_idx*0.2)
        idx = np.arange(len_idx)
        np.random.shuffle(idx)

        data_idx = data_idx[idx]
        label_idx = label_idx[idx]

        train_data.append(data_idx[:num_train])
        train_label.append(label_idx[:num_train])
        val_data.append(data_idx[num_train:(num_train+num_val)])
        val_label.append(label_idx[num_train:(num_train+num_val)])
        test_data.append(data_idx[(num_train+num_val):])
        test_label.append(label_idx[(num_train+num_val):])

    train_data = np.concatenate(train_data,0)
    train_label = np.concatenate(train_label,0)
    val_data = np.concatenate(val_data,0)
    val_label = np.concatenate(val_label,0)
    test_data = np.concatenate(test_data,0)
    test_label = np.concatenate(test_label,0)

    len_train = train_data.shape[0]
    idx = np.arange(len_train)
    np.random.shuffle(idx)
    train_data = train_data[idx]
    train_label = train_label[idx]

    return train_data, train_label, val_data, val_label, test_data, test_label

def data_generator(path, args):
    path = path+'/ISRUC_S3.npz'
    ReadList = np.load(path, allow_pickle=True)
    Fold_Num = ReadList['Fold_len']  # Num of samples of each fold
    Fold_Data = ReadList['Fold_data']  # Data of each fold
    Fold_Label = ReadList['Fold_label']  # Labels of each fold

    train_data, train_label, val_data, val_label, test_data, test_label = data_preparation(Fold_Data,Fold_Label)

    #
    train_dataset = Load_Dataset(train_data, train_label, args)
    valid_dataset = Load_Dataset(val_data, val_label, args)
    test_dataset = Load_Dataset(test_data, test_label, args)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                                               shuffle=True, drop_last=args.drop_last,
                                               num_workers=0)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=args.batch_size,
                                               shuffle=False, drop_last=args.drop_last,
                                               num_workers=0)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size,
                                              shuffle=False, drop_last=False,
                                              num_workers=0)

    return train_loader, valid_loader, test_loader



