import torch

import Model
import Dataset
from torch import nn, optim
from tqdm import tqdm
from collections import OrderedDict

if(__name__=="__main__"):
    data_loader = Dataset.Dataset(batch_size=32)
    train_loader = data_loader.trainloader
    val_loader = data_loader.valloader
    model = Model.Recognizer()
    if(torch.cuda.is_available()):
        model = model.cuda()

    criteration = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)

    epochs=50
    print("start training")
    for epoch in range(epochs):
        pbar = tqdm(train_loader, desc='Epoch: [%d]/[%d] Training' % (epoch, epochs), leave=True)
        model.train()
        running_loss = 0
        for batch_nb, batch in enumerate(pbar):
            image = batch[0]
            labels = batch[1]
            pbar.set_postfix(OrderedDict({"loss":running_loss}))
            optimizer.zero_grad()
            output = model(image)
            loss = criteration(output,labels)
            loss.backward()
            optimizer.step()
            running_loss = loss.item()

        print("Epoch {} - Training loss: {}".format(epoch, running_loss / len(train_loader)))



