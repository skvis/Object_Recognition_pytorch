import config
import data_preprocess
import model
import os
import torch
import torch.nn as nn
import torch.optim as optim


def train_cnn():

    trainloader, _ = data_preprocess.data_augumentation()

    net = model.Net()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(),
                          lr=config.LEARNING_RATE,
                          momentum=config.MOMENTUM)

    # train the network

    for epoch in range(0, config.NUM_EPOCHS):
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            # print(outputs.shape)
            # print(labels.shape)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:
                print('[%d, %5d] Loss: %.3f' %
                      (epoch + 1, i + 1, running_loss/2000))
                running_loss = 0.0

    print('Finished Training')

    # Save the model
    torch.save(net.state_dict(),
               os.path.join(config.MODEL_PATH, config.MODEL_NAME))


if __name__ == '__main__':
    train_cnn()
