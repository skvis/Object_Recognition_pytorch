import config
import torch
import torchvision
import torchvision.transforms as transforms


def data_augumentation():

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5),
                                                         (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root=config.DATA_PATH,
                                            train=True,
                                            transform=transform,
                                            download=True)

    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=config.BATCH_SIZE,
                                              shuffle=config.SHUFFLE,
                                              num_workers=config.NUM_WORKER)

    testset = torchvision.datasets.CIFAR10(root=config.DATA_PATH,
                                           train=False,
                                           transform=transform,
                                           download=True)

    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=config.BATCH_SIZE,
                                             num_workers=config.NUM_WORKER)

    return trainloader, testloader
