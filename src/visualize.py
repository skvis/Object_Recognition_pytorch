import config
import data_preprocess
import numpy as np
import matplotlib.pyplot as plt
import torchvision


def image_show(image):
    image = image / 2 + 0.5
    image = np.array(image)
    image = np.transpose(image, (1, 2, 0))
    plt.imshow(image)
    plt.show()


def load_data(trainloader, testloader):
    sample_data = iter(trainloader)
    images, _ = sample_data.next()
    image_show(torchvision.utils.make_grid(images))
    print(" ".join('%5s' % config.CLASSES[j] for j in range(4)))


if __name__ == '__main__':
    trainloader, testloader = data_preprocess.data_augumentation()
    load_data(trainloader, testloader)
