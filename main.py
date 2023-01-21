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

    criteration = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)

    epochs=50
    print("start training")
    for epoch in range(epochs):
        pbar = tqdm(train_loader, desc='Epoch: [%d]/[%d] Training' % (epoch, epochs), leave=True)

        running_loss = 0
        feed_size = 0
        wr = 0
        for batch_nb, batch in enumerate(pbar):
            image = batch[0]
            labels = nn.functional.one_hot(batch[1],num_classes=10).to(dtype=torch.float32)
            if(torch.cuda.is_available()):
                image = image.cuda()
                labels = labels.cuda()
            feed_size += len(image)
            optimizer.zero_grad()
            output = model(image)
            wr += float(torch.sum(torch.abs(output - labels)))/2
            loss = criteration(output,labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pbar.set_postfix(OrderedDict({"train_loss":running_loss/feed_size, "precision":1-wr/feed_size}))


        print("Epoch {} - Training loss: {}".format(epoch, running_loss / feed_size))

        running_loss_eval = 0

        pbar_eval = tqdm(val_loader, desc='Epoch: [%d]/[%d] validation' % (epoch, epochs), leave=True)
        wr = 0
        feed_size=0
        for batch_nb, batch in enumerate(pbar_eval):
            image = batch[0]
            labels =  nn.functional.one_hot(batch[1],num_classes=10).to(dtype=torch.float32)
            if(torch.cuda.is_available()):
                image = image.cuda()
                labels = labels.cuda()
            feed_size += len(image)
            optimizer.zero_grad()
            output = model(image)
            loss = criteration(output,labels)
            wr += float(torch.sum(torch.abs(output - labels)))/2
            loss.backward()
            optimizer.step()
            running_loss_eval += loss.item()
            pbar_eval.set_postfix(OrderedDict({"val_loss":running_loss/feed_size, "precision":1-wr/feed_size}))

        print("Epoch {} - Eval loss: {}".format(epoch, running_loss_eval / len(val_loader)))
        print(f"precision={1-wr/feed_size}")





