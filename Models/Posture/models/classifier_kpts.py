import torch
import torch.nn as nn


class Classifier_kpts(nn.Module):
    def __init__(self):
        super(Classifier_kpts, self).__init__()
        self.fc_layer = nn.Sequential(
            nn.Linear(24 ,128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(p=0.2),
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
        # x = x.view(x.size(0), -1)
        #print(x.size())
        # fc layer
        x = self.fc_layer(x)

        return x

