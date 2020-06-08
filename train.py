import torch
import torch.nn as nn
import torchvision
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from model import WideResNet
from model import conv_init
import torchvision.transforms as transforms
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import argparse

parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='Initial learning_rate')
parser.add_argument('--gamma', default=0.2, type=int, help='learning rate decay factor')
parser.add_argument('--depth', default=28, type=int, help='depth of model')
parser.add_argument('--width_factor', default=8, type=int, help='width factor (k) to use for the model')
parser.add_argument('--dropout_rate', default=0.3, type=float, help='dropout_rate')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset = [cifar10/cifar100]')
parser.add_argument('--testOnly', '-t', action='store_true', help='Test mode with the saved model')
args = parser.parse_args()


# N is the number of times each block is repeated
N = int((args.depth - 4)/6) # There are 3 conv blocks, each having 2 conv layers -> 6 total conv layers

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Assuming that we are on a CUDA machine, this should print a CUDA device:
print(device)

# Load the network and initalise the parameters
net = WideResNet(args.dropout_rate, num_classes = 10, width_factor = 10, N = N)
net.apply(conv_init)

# Mean & std pre-processsing gives better results as mentioned in the paper (below settings are for cifar 10)
train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
     transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
])


test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
])

print("============[Phase 1: Data Preperation]=============")
# Batch size of 128 is used in the paper
train_set = CIFAR10(root='./data', train = True, download = True, transform=train_transform)
train_loader = DataLoader(train_set, batch_size=128, shuffle=True, drop_last=True, num_workers=2)
print("| Performing Mean|STD pre-processing on data")
test_set = CIFAR10(root='./data', train = False, download = True, transform=test_transform)
test_loader = DataLoader(test_set, batch_size=128, shuffle=True, drop_last=True, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(train_loader)
images, labels = dataiter.next()

# show images
#imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

# Define the loss function and Optimizer
num_epochs = 200
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9,  weight_decay=5e-4, nesterov=True)
scheduler = MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)


print("=============[Phase 3 : Model Training]==================")
# Push the model to the GPU
net.to(device)

for epoch in range(num_epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_loader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 200 == 0:    # print every 200 mini-batches
            print('[%d, %5d] loss: %.5f' %
                  (epoch + 1, i + 1, running_loss / 200))
            running_loss = 0.0
    scheduler.step()

print('Finished Training')

PATH = 'drive/My Drive/wide_resnet_cifar10.pth'
torch.save(net.state_dict(), PATH)

net.eval()
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))
