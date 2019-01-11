import os
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset

def get_data(trainset, data):
    return trainset.train_data.numpy()[(trainset.train_labels == data).numpy() == 1]

trainset = datasets.MNIST(root="./", train=True,download=True, transform=transforms.Compose([transforms.ToTensor()]))

testset = datasets.MNIST(root="./",train=True,download=True,transform=transforms.Compose([transforms.ToTensor()]))

class CustomSet(Dataset):
    def __init__(self, data, transform=transforms.Compose([transforms.ToTensor()])):
        self._data = data
        self._transform = transform

    def __len__(self):
        return len(self._data)

    def __getitem__(self, i):
        if self._transform:
            return self._transform(self._data[i])
        return self._data[i]

# normal data
four_data = get_data(trainset, 4)[:220].astype('float32')

# anomaly data
zero_data = get_data(trainset, 0)[:5].astype('float32')
seven_data = get_data(trainset, 7)[:3].astype('float32')
nine_data = get_data(trainset, 9)[:3].astype('float32')

normal = four_data
anomaly = np.concatenate((zero_data, seven_data, nine_data), axis=0)

normal = np.expand_dims(normal, axis=3)
anomaly = np.expand_dims(anomaly, axis=3)

trainset = CustomSet(normal)
testset = CustomSet(anomaly)

# trainset.train_data = torch.from_numpy(normal)
# trainset.train_labels = torch.LongTensor(trainset.train_labels.numpy()[(trainset.train_labels == 4).numpy() == 1][:220])
# trainset_numpy = trainset.train_data.numpy()
trainloader = DataLoader(trainset, batch_size=len(trainset), num_workers=1)
testloader = DataLoader(testset, batch_size=len(testset))

# testset.train_data = torch.ByteTensor(anomaly)
# testset.train_labels = torch.LongTensor(np.zeros(len(anomaly)))
# testloader = DataLoader(testset, batch_size=len(anomaly), num_workers=4)


class CAE_Paper_MNIST(nn.Module):
    def __init__(self):
        super(CAE_Paper_MNIST, self).__init__()
        self.conv2d_1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.elu1 = nn.ELU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 16 x 14 x 14

        self.conv2d_2 = nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1)
        self.elu2 = nn.ELU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 8 x 7 x 7

        self.conv2d_3 = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1)
        self.elu3 = nn.ELU()

        # 8 x 7 x 7

        self.dense_1 = nn.Linear(8 * 7 * 7, 32)

        # 8 x 7 x 7 -> 32

    def forward(self, x):
        x = self.conv2d_1(x)
        x = self.elu1(x)
        x = self.pool1(x)
        x = self.conv2d_2(x)
        x = self.elu2(x)
        x = self.pool2(x)
        x = self.conv2d_3(x)
        x = self.elu3(x)
        x = x.permute(0, 2, 3, 1).contiguous().view(x.size(0), -1)
        x = self.dense_1(x)
        return x


params = np.load("paper_encoder_weights.npz")

params = OrderedDict({ k : params[k].item() for k in params })

model = CAE_Paper_MNIST()

model_params_dict = model.state_dict()

for k in model_params_dict.keys():
    key_tokens = k.split('.')
    if 'dense' in key_tokens[0]:
        if not key_tokens[1].endswith('s'):
            key_tokens[1] += 's'
            model_params_dict[k] = nn.Parameter(torch.FloatTensor(params[key_tokens[0]][key_tokens[1]].T))
        else:
            model_params_dict[k] = nn.Parameter(torch.FloatTensor(params[key_tokens[0]][key_tokens[1]]))
    else:
        if not key_tokens[1].endswith('s'):
            key_tokens[1] += 's'
            model_params_dict[k] = nn.Parameter(torch.FloatTensor(np.transpose(np.asarray(params[key_tokens[0]][key_tokens[1]]), (3, 2, 0, 1))))
        else:
            model_params_dict[k] = nn.Parameter(torch.FloatTensor(params[key_tokens[0]][key_tokens[1]]))

model.load_state_dict(model_params_dict)

mbs = 1
cae_type = 'CAE_Basic'


for img in trainloader:
    img = Variable(img)
    l_img = img.data.numpy()
    features_normal = model(img).data.numpy()

np.savez('our_train_features.npz', features_normal)

for img in testloader:
    img = Variable(img)
    features_anomaly = model(img).data.numpy()

np.savez('our_test_features.npz', features_anomaly)
