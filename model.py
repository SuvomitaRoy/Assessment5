import torch
import torch.nn as nn

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

        self.act_function = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        layers = [1, 6, 16, 120, 84, 10]

        self.conv1 = torch.nn.Conv2d(layers[0], layers[1], 5, padding = 2)
        self.conv2 = torch.nn.Conv2d(layers[1], layers[2], 5)
        self.fc1 = torch.nn.Linear(5 * 5 * layers[2], layers[3])
        self.fc2 = torch.nn.Linear(layers[3], layers[4])
        self.fc3 = torch.nn.Linear(layers[4], layers[5])


    def forward(self, x):
        x = self.conv1(x)
        x = self.act_function(x)
        x = torch.nn.functional.max_pool2d(x, 2)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.act_function(x)
        x = torch.nn.functional.max_pool2d(x, 2)
        x = self.dropout(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.act_function(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.act_function(x)
        x = self.dropout(x)

        x = self.fc3(x)
        x = torch.nn.functional.log_softmax(x, dim = 1)

        return x