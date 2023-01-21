import torch
from torch import nn

class conv_block(nn.Module):
    def __init__(self, nIn, nOut, ks):
        super(conv_block, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels=nIn, out_channels=nOut, kernel_size=ks)
        self.activation = torch.nn.ReLU()
        self.bn = nn.BatchNorm2d(nOut)

    def forward(self, X):
        X = self.conv(X)
        X = self.activation(X)
        X = self.bn(X)
        return X

class linear_block(nn.Module):
    def __init__(self, nIn, nOut, is_bn=True):
        super(linear_block, self).__init__()
        self.linear = nn.Linear(in_features=nIn, out_features=nOut)
        self.activation = nn.ReLU()
        self.is_bn = is_bn
        if(is_bn):
            self.bn = nn.BatchNorm1d(nOut)

    def forward(self, X):
        X = self.linear(X)
        X = self.activation(X)
        if(self.is_bn):
            X = self.bn(X)
        return X

class Recognizer(nn.Module):
    def __init__(self):
        super(Recognizer,self).__init__()
        self.input_size = 30
        self.conv1 = conv_block(1, 32, (3,3))
        self.conv2 = conv_block(32, 32, (3,3))
        self.pool1 = nn.MaxPool2d((2,2))
        self.conv3 = conv_block(32, 64, (3,3))
        self.conv4 = conv_block(64, 128, (3,3))
        self.flatten = nn.Flatten()
        self.lin1 = linear_block(8*8*128,8*128)
        self.lin2 = linear_block(8*128,128)
        self.lin3 = linear_block(128,10)

    def forward(self,X):
        if(torch.cuda.is_available()):
            X = X.to('cuda')
        X = self.conv1(X)
        X = self.conv2(X)
        X = self.pool1(X)
        X = self.conv3(X)
        X = self.conv4(X)
        X = self.flatten(X)
        X = self.lin1(X)
        X = self.lin2(X)
        X = self.lin3(X)
        return X



