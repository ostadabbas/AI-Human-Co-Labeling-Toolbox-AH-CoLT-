import torch
import torch.nn as nn
import torch.nn.functional as F

'''
class Classifier1(torch.nn.Module):
    def __init__(self):
        super(Classifier1, self).__init__()
        self.conv1 = torch.nn.Conv2d(3,32,5,stride = 1)
        self.conv2 = torch.nn.Conv2d(32,64,3)
        self.conv3 = torch.nn.Conv2d(64,128,3)

        self.fc1 = torch.nn.Linear(128*2*2, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 4)

        self.pool = torch.nn.MaxPool2d(2,2)

        self.conv1_bn = torch.nn.BatchNorm2d(32)
        self.conv2_bn = torch.nn.BatchNorm2d(64)
        self.conv3_bn = torch.nn.BatchNorm2d(128)
        self.fc1_bn = torch.nn.BatchNorm2d(128)
        self.fc2_bn = torch.nn.BatchNorm2d(64)

        self.dropout_lin = torch.nn.Dropout(0.1)
        self.dropout_conv = torch.nn.Dropout(0.1)

    def forward(self,x):
        x = self.dropout_conv(self.pool(F.relu(self.conv1_bn(self.conv1(x)))))
        x = self.dropout_conv(self.pool(F.relu(self.conv2_bn(self.conv2(x)))))
        x = self.dropout_conv(self.pool(F.relu(self.conv3_bn(self.conv3(x)))))

        x = x.view(-1, 128*2*2)
        x = self.dropout_lin(F.relu(self.fc1_bn(self.fc1(x))))
        x = self.dropout_lin(F.relu(self.fc2_bn(self.fc2(x))))
        x = self.fc3(x)

        return x
'''

class Classifier1(nn.Module):
    def __init__(self):
        super(Classifier1, self).__init__()

        self.conv_layer = nn.Sequential(
            # Conv Layer block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Dropout(0.1),

            # Conv Layer block 1
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Dropout(0.1),

            # Conv Layer block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Dropout(0.1)

        )

        self.fc_layer = nn.Sequential(
            nn.Linear(128*2*2, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(64, 4)

        )

    def forward(self, x):
        """Perform forward"""

        # conv layers
        x = self.conv_layer(x)

        # flatten
        x = x.view(x.size(0), -1)
        #print(x.size())
        # fc layer
        x = self.fc_layer(x)

        return x
