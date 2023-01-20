import torch.utils.data
from torchvision import datasets, transforms


class Dataset():
    def __init__(self, batch_size=32):
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,)),
                                        ])
        trainset = datasets.MNIST("./", download=True, train=True, transform=transform)
        valset = datasets.MNIST("./", download=True, train=False, transform=transform)

        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        self.valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=True)



