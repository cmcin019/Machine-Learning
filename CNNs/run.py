# Author	: Cristopher McIntyre Garcia 
# Email	 : cmcin019@uottawa.ca
# S-N	   : 300025114

# Imports
from CNNs import get_CNN_Net_3x3_Same, get_CNN_Net_3x3_Valid, get_CNN_Net_5x5_Same
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
torch.cuda.is_available()

parser = argparse.ArgumentParser()
parser.add_argument('-q', help="Enter number of the question who's code you want to run (-1 for all)", type=int, default=-1)

args = parser.parse_args()

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

import torch.nn as nn
import torch.nn.functional as F

# Original model   
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 120)
        self.fc3 = nn.Linear(120, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# CNN model: This model works for every cnn in this assignment
# Specific cnns can be found in the CNNs.py file
# For the purpose of this assignments, the CNNs from the file CNNs.py will be used
class CNN_Net(nn.Module):
    def __init__(self, filter=5, padding='valid', activation=F.relu):
        super(CNN_Net, self).__init__()
        if filter == 3:
          lin_in = 64
        else :
          lin_in = 16
        self.conv1 = nn.Conv2d(3, 6, kernel_size=filter, padding=padding)
        self.pool = nn.MaxPool2d(2, 2) # Features div by 2 
        self.conv2 = nn.Conv2d(6, 16, kernel_size=filter, padding=padding)
        self.fc1 = nn.Linear(lin_in * filter * filter, 120)
        self.fc1_same = nn.Linear(1024, 120)
        self.fc2 = nn.Linear(120, 120)
        self.fc3 = nn.Linear(120, 10)
        self.activation = activation
        self.filter = filter
        self.lin_in = lin_in
        self.padding = padding
    def forward(self, x):
        x = self.pool(self.activation(self.conv1(x)))
        x = self.pool(self.activation(self.conv2(x)))
        if self.padding == 'same':
          x = x.view(-1, 1024)
          x = self.activation(self.fc1_same(x))
        else:
          x = x.view(-1, self.lin_in * self.filter * self.filter)
          x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x

# FC model  
class FC_Net(nn.Module):
    def __init__(self, hidden_layers=1):
        super(FC_Net, self).__init__()
        self.ffn = nn.Linear(32 * 32 * 3, 10)
        self.fc_in = nn.Linear(32 * 32 * 3, 120)
        self.fc = nn.Linear(120, 120)
        self.fc_out = nn.Linear(120, 10)
        self.hidden_layers = hidden_layers

    def forward(self, x):
        x = x.view(-1, 32 * 32 * 3)
        if self.hidden_layers < 1:
          return self.ffn(x)
        x = F.relu(self.fc_in(x))
        for _ in range(self.hidden_layers-1):
          x = F.relu(self.fc(x))
        x = self.fc_out(x)
        return x

import torch.optim as optim
# Original
cnn_net = Net()
print("Original CNN (Valid - 5x5 kernel)")
print(cnn_net)
print()

if args.q == 1 or args.q == -1:
  # Question 01 nets
  fcnet_list=[FC_Net(hidden_layers=x) for x in range(5)]

  print("FC Net ")
  print(fcnet_list[0])
  print()

if args.q == 2 or args.q == -1:
  # Question 02 nets
  cnn_sig_net = CNN_Net(activation=torch.sigmoid)
  print("CNN with Sigmoid")
  print(cnn_sig_net)
  print()

# if args.q == 3 or args.q == -1:
#   # Question 03 nets
#   cnn_same_net = CNN_Net(padding='same')
#   print("CNN - same 5x5")
#   print(cnn_same_net)
#   print()

#   cnn_f3_net = CNN_Net(filter=3)
#   print("CNN - valid 3x3")
#   print(cnn_f3_net)
#   print()

#   cnn_same_f3_net = CNN_Net(filter=3, padding='same')
#   print("CNN - same 3x3")
#   print(cnn_same_f3_net)
#   print()
  
if args.q == 3 or args.q == -1:
  # Question 03 nets
  cnn_same_net = get_CNN_Net_5x5_Same()
  print("CNN - same 5x5")
  print(cnn_same_net)
  print()

  cnn_f3_net = get_CNN_Net_3x3_Valid()
  print("CNN - valid 3x3")
  print(cnn_f3_net)
  print()

  cnn_same_f3_net = get_CNN_Net_3x3_Same()
  print("CNN - same 3x3")
  print(cnn_same_f3_net)
  print()

# Let's first define our device as the first visible cuda device if we have
# CUDA available: 
CUDA=torch.cuda.is_available()
device = torch.device("cuda:0" if CUDA else "cpu")


# Assuming that we are on a CUDA machine, this should print a CUDA device:
print(device)

# Train the network
def train_net(net):
  net.to(device=device)
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
  accuracy_values=[]
  epoch_number=[]
  for epoch in range(10):  # loop over the dataset multiple times. Here 10 means 10 epochs
      running_loss = 0.0
      for i, (inputs,labels) in enumerate(trainloader, 0):
          # get the inputs; data is a list of [inputs, labels]
          inputs = inputs.to(device=device)
          labels = labels.to(device=device)
          
          # zero the parameter gradients
          optimizer.zero_grad()

          # forward + backward + optimize
          outputs = net(inputs)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()

          # print statistics
          running_loss += loss.item()
          if i % 2000 == 1999:    # print every 2000 mini-batches
              # print('[epoch%d, itr%5d] loss: %.3f' %
              #       (epoch + 1, i + 1, running_loss / 2000))
              running_loss = 0.0
          if CUDA:
            inputs = inputs.to(device='cpu')
            labels = labels.to(device='cpu') 

      correct = 0
      total = 0
      with torch.no_grad():
          for images, labels in testloader:

              images = images.to(device=device)
              labels = labels.to(device=device)

              outputs = net(images)
              _, predicted = torch.max(outputs.data, 1)
              total += labels.size(0)
              if CUDA:
                correct += (predicted.to(device='cpu')==labels.to(device='cpu')).sum().item()
              else:
                correct += (predicted==labels).sum().item()
              
          TestAccuracy = 100 * correct / total
          epoch_number += [epoch+1]
          accuracy_values += [TestAccuracy] 
          print('Epoch=%d \tTest Accuracy=%.3f' %
                    (epoch + 1, TestAccuracy))
      
  print('Finished Training\n')
  if CUDA:
    net.to(device='cpu')
  return epoch_number, accuracy_values

import matplotlib.pyplot as plt
import numpy as np

print("Original CNN")
fig, ax = plt.subplots()
o_epoch_number, o_accuracy_values = train_net(net=cnn_net)
ax.plot(o_epoch_number, o_accuracy_values, label='ConvNet')
ax.set(xlabel='Epoch', ylabel='Accuracy')
print()
# Add a legend
plt.legend()

# TODO
# Code for question 01
if args.q == 1 or args.q == -1:
  # Fully connected networks
  print("Question 01")
  fig, ax = plt.subplots()
  for x in range(len(fcnet_list)):
    # Plot the data
    print('FC model (' + str(x) + ' hidden layer)')
    epoch_number, accuracy_values = train_net(fcnet_list[x])
    ax.plot(epoch_number, accuracy_values, label='FC model (' + str(x) + ' hidden layer)')

  # CNN - original
  print('CNN - original')
  ax.plot(o_epoch_number, o_accuracy_values, label='ConvNet')
  ax.set(xlabel='Epoch', ylabel='Accuracy')
  print()
  # Add a legend
  plt.legend()

# TODO
# Code for question 02
if args.q == 2 or args.q == -1:
  # CNN - Sigmoid
  print("Question 02")
  fig, ax = plt.subplots()
  print('CNN - Sigmoid')
  epoch_number, accuracy_values = train_net(net=cnn_sig_net)
  ax.plot(epoch_number, accuracy_values, label='Sigmoid')

  # CNN - original
  # print('CNN - Relu')
  # epoch_number, accuracy_values = train_net(net=cnn_net)
  ax.plot(o_epoch_number, o_accuracy_values, label='Relu')
  ax.set(xlabel='Epoch', ylabel='Accuracy')
  print()
  # Add a legend
  plt.legend()

# TODO
# Code for question 03
if args.q == 3 or args.q == -1:
  # CNN - Same & Filter 5x5
  print("Question 03")
  fig, ax = plt.subplots()
  print("Same & Filter 5x5")
  epoch_number, accuracy_values = train_net(net=cnn_same_net)
  ax.plot(epoch_number, accuracy_values, label='Same & Filter 5x5')

  # CNN - Valid & Filter 3x3
  print("Valid & Filter 3x3")
  epoch_number, accuracy_values = train_net(net=cnn_f3_net)
  ax.plot(epoch_number, accuracy_values, label='Valid & Filter 3x3')

  # CNN - Same & Filter 3x3
  print("Same & Filter 3x3")
  epoch_number, accuracy_values = train_net(net=cnn_same_f3_net)
  ax.plot(epoch_number, accuracy_values, label='Same & Filter 3x3')

  # CNN - Valid & Filter 5x5
  # print("Valid & Filter 5x5")
  # epoch_number, accuracy_values = train_net(net=cnn_net)
  ax.plot(o_epoch_number, o_accuracy_values, label='Original')
  ax.set(xlabel='Epoch', ylabel='Accuracy')
  # Add a legend
  plt.legend()

# Show the plot
plt.show()

## Anything better than 10% accuracy (randomly picking a class out of 10 classes)
# means the network has learned something

