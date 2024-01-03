import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import numpy as np


class Load_Dataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, dataset, args):
        super(Load_Dataset, self).__init__()

        X_train = dataset["samples"]
        y_train = dataset["labels"]

        if len(X_train.shape) < 3:
            X_train = X_train.unsqueeze(2)

        if X_train.shape.index(min(X_train.shape)) != 1:  # make sure the Channels in second dim
            X_train = X_train.permute(0, 2, 1)

        if isinstance(X_train, np.ndarray):
            self.x_data = torch.from_numpy(X_train)
            self.y_data = torch.from_numpy(y_train).long()
        else:
            self.x_data = X_train.float()
            self.y_data = y_train.long()


        self.len = X_train.shape[0]
        shape = self.x_data.size()
        self.x_data = self.x_data.reshape(shape[0],shape[1],args.time_denpen_len, args.patch_size)
        self.x_data = torch.transpose(self.x_data, 1,2)


    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


def data_generator(data_path, args):

    train_dataset = torch.load(os.path.join(data_path, "train.pt"))
    valid_dataset = torch.load(os.path.join(data_path, "val.pt"))
    test_dataset = torch.load(os.path.join(data_path, "test.pt"))

    train_dataset = Load_Dataset(train_dataset, args)
    valid_dataset = Load_Dataset(valid_dataset, args)
    test_dataset = Load_Dataset(test_dataset, args)


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
