import torch
import torch.nn as nn
import numpy as np

class Classifier_rgb(nn.Module):
    def __init__(self):
        super(Classifier_rgb, self).__init__()
        self.fc_layer = nn.Sequential(
            nn.Linear(150528 ,1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Dropout(p=0.2),

            nn.Linear(1024, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),

            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
            nn.Dropout(p=0.2),
            nn.Linear(64,16),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(16),
            nn.Linear(16,4),
            nn.Softmax()
        )

    def forward(self, x):
        """Perform forward"""
        # flatten
        x = x.view(x.size(0), -1)
        # print(x.size())
        # fc layer
        x = self.fc_layer(x)

        return x


# class Classifier_rgb(nn.Module):
#     def __init__(self):
#         super(Classifier_rgb, self).__init__()
#
#         self.conv_layer = nn.Sequential(
#             # Conv Layer block 1
#             nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
#             nn.BatchNorm2d(16),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Dropout2d(p=0.2),
#
#             nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Dropout2d(p=0.2),
#
#             nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Dropout2d(p=0.3)
#
#         )
#
#         self.fc_layer = nn.Sequential(
#
#             nn.Linear(100352, 1024),
#             nn.ReLU(inplace=True),
#             nn.Dropout(p=0.2),
#             nn.Linear(1024, 512),
#             nn.ReLU(inplace=True),
#             nn.Dropout(p=0.2),
#             nn.Linear(512, 4),
#             nn.Softmax()
#
#         )
#
#     def forward(self, x):
#         """Perform forward"""
#
#         # conv layers
#         x = self.conv_layer(x)
#         print(np.shape(x))
#         # flatten
#         x = x.view(x.size(0), -1)
#         #print(x.size())
#         # fc layer
#         x = self.fc_layer(x)
#
#         return x


'''
            # Conv Layer block 1
            nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(3, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

             # Conv Layer block 1
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),


            # Conv Layer block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),

            # Conv Layer block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
'''
