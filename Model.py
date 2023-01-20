import torch
from torch import nn


class Recognizer(nn.Module):
    def __init__(self):
        super(Recognizer,self).__init__()
        self.input_size = 30
        self.model = self.create_model()

    def conv_block(self, model,name, nIn, nOut, ks):
        model.add_module(f"conv_{name}",torch.nn.Conv2d(in_channels=nIn,out_channels=nOut,kernel_size=ks))
        model.add_module(f"ReLU_{name}",nn.ReLU())
        model.add_module(f"BN+{name}",nn.BatchNorm2d(nOut))
        return model

    def linear_block(self, model,name, nIn, nOut, bn=True):
        model.add_module(f"Linear_{name}", nn.Linear(in_features=nIn, out_features=nOut))
        model.add_module(f"RelU_{name}",nn.ReLU())
        if(bn):
            model.add_module(f"BN",nn.BatchNorm1d(nOut))
        return model

    def create_model(self):
        model = nn.Sequential()
        model = self.conv_block(model,"01",nIn=1, nOut=32,ks=(3,3)) #(b,1,28,28) -> (b,32,26,26)
        model = self.conv_block(model,"02",nIn=32, nOut=32,ks=(3,3)) # (b,32,26,26) -> (b,32,24,24)
        model.add_module("pool_1", nn.MaxPool2d(kernel_size=(2,2))) # (b,32,24,24) -> (b,32,12,12)
        model = self.conv_block(model,"03",nIn=32, nOut=64,ks=(3,3)) # (b,64,12,12) -> (b,64,10,10)
        model = self.conv_block(model,"04",nIn=64, nOut=128,ks=(3,3)) # (b,64,10,10) -> (b,128,8,8)
        model.add_module("flatten",nn.Flatten())
        model = self.linear_block(model,"01",nIn=8*8*128, nOut=8*128, bn=True)
        model = self.linear_block(model, "02", nIn=8*128, nOut=128, bn=True)
        model = self.linear_block(model, "03", nIn=128, nOut=10, bn=False)
        return model

    def conv_block_X(self, X, nIn, nOut, ks):
        X = torch.nn.Conv2d(in_channels=nIn,out_channels=nOut,kernel_size=ks)(X)
        X = nn.ReLU()(X)
        X = nn.BatchNorm2d(nOut)(X)
        return X

    def linear_block_X(self, X, nIn, nOut, bn=True):
        X = nn.Linear(in_features=nIn, out_features=nOut)(X)
        X = nn.ReLU()(X)
        if(bn):
            X = nn.BatchNorm1d(nOut)(X)
        return X

    def pool_block_X(selfself,X,ks):
        X = nn.MaxPool2d(ks)(X)
        return X

    def flatten_block_X(self,X):
        X = nn.Flatten()(X)
        return X

    def forward(self,X):
        if(torch.cuda.is_available()):
            X = X.to('cuda')
        X = self.conv_block_X(X,1,32,(3,3))
        X = self.conv_block_X(X,32,32,(3,3))
        X = self.pool_block_X(X, (2,2))
        X = self.conv_block_X(X,32,64,(3,3))
        X = self.conv_block_X(X,64,128,(3,3))
        X = self.flatten_block_X(X)
        X = self.linear_block_X(X,128*8*8,128*8)
        X = self.linear_block_X(X,128*8,128)
        X = self.linear_block_X(X,128,10)
        return X



