import torch.nn as nn
import torch.nn.functional as F

# CNN model   
class CNN_Net_5x5_Same(nn.Module):
    def __init__(self, filter=5, padding='same', activation=F.relu):
        super(CNN_Net_5x5_Same, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=filter, padding=padding)
        self.pool = nn.MaxPool2d(2, 2) # Features div by 2 
        self.conv2 = nn.Conv2d(6, 16, kernel_size=filter, padding=padding)
        self.fc1 = nn.Linear(1024, 120)
        self.fc2 = nn.Linear(120, 120)
        self.fc3 = nn.Linear(120, 10)
        self.activation = activation
    def forward(self, x):
        x = self.pool(self.activation(self.conv1(x)))
        x = self.pool(self.activation(self.conv2(x)))
        x = x.view(-1, 1024)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x

# CNN model   
class CNN_Net_3x3_Same(nn.Module):
    def __init__(self, filter=3, padding='same', activation=F.relu):
        super(CNN_Net_3x3_Same, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=filter, padding=padding)
        self.pool = nn.MaxPool2d(2, 2) # Features div by 2 
        self.conv2 = nn.Conv2d(6, 16, kernel_size=filter, padding=padding)
        self.fc1 = nn.Linear(1024, 120)
        self.fc2 = nn.Linear(120, 120)
        self.fc3 = nn.Linear(120, 10)
        self.activation = activation
    def forward(self, x):
        x = self.pool(self.activation(self.conv1(x)))
        x = self.pool(self.activation(self.conv2(x)))
        x = x.view(-1, 1024)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x


# CNN model   
class CNN_Net_3x3_Valid(nn.Module):
    def __init__(self, filter=3, padding='valid', activation=F.relu):
        super(CNN_Net_3x3_Valid, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=filter, padding=padding)
        self.pool = nn.MaxPool2d(2, 2) # Features div by 2 
        self.conv2 = nn.Conv2d(6, 16, kernel_size=filter, padding=padding)
        self.fc1 = nn.Linear(64 * filter * filter, 120)
        self.fc2 = nn.Linear(120, 120)
        self.fc3 = nn.Linear(120, 10)
        self.activation = activation
    def forward(self, x):
        x = self.pool(self.activation(self.conv1(x)))
        x = self.pool(self.activation(self.conv2(x)))
        x = x.view(-1, 64 * 3 * 3)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x

def get_CNN_Net_5x5_Same():
    return CNN_Net_5x5_Same()

def get_CNN_Net_3x3_Same():
    return CNN_Net_3x3_Same()

def get_CNN_Net_3x3_Valid():
    return CNN_Net_3x3_Valid()