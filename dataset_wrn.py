from torchvision.datasets import CIFAR10, CIFAR100, MNIST

def __init__(self, arg):
    if arg == 'CIFAR10':
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                            shuffle=True, num_workers=2, drop_last=True)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)

        testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                            shuffle=False, num_workers=2, drop_last=True)

        return trainloader, testloader

    elif arg == 'CIFAR100':
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                            download=True, transform=transform)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                            shuffle=True, num_workers=2, drop_last=True)

        testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                        download=True, transform=transform)

        testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                            shuffle=False, num_workers=2, drop_last=True)

        return trainloader, testloader

    elif arg = 'MNIST':
        trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                            download=True, transform=transform)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                            shuffle=True, num_workers=2, drop_last=True)

        testset = torchvision.datasets.MNIST(root='./data', train=False,
                                        download=True, transform=transform)

        testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                            shuffle=False, num_workers=2, drop_last=True)

        return trainloader, testloader

    else:
        print("| *Dataset NOT available!!* |")
        return 0