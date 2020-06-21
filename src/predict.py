import config
import data_preprocess
import model
import os
import torch


def predict():

    _, testloader = data_preprocess.data_augumentation()

    net = model.Net()
    net.load_state_dict(torch.load(os.path.join(config.MODEL_PATH,
                                                config.MODEL_NAME)))

    # predict on whole data
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: %5d %%' %
          (100 * correct / total))

    # predict class wise
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        print('Accuracy of %5s : %2d %%' %
              (config.CLASSES[i], 100 * class_correct[i] / class_total[i]))


if __name__ == '__main__':
    predict()
