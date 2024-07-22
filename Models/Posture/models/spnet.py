import torch
import torch.nn as nn


class SPNet(nn.Module):

    def __init__(self):
        super(SPNet, self).__init__()

        V = torch.empty((32, 4), requires_grad=True) # initialize kernel matrix 3x2
        nn.init.orthogonal_(V)
        self.V = nn.Parameter(V)
        self.register_parameter("sp_v", self.V)

        self.class_classifier = nn.Sequential()
        #self.class_classifier.add_module('d_fc1', nn.Linear(1 * 32 * 32, 10))   # without sp layer
        self.class_classifier.add_module('d_fc1', nn.Linear(1 * 4 * 32, 10))    # with sp layer
        self.class_classifier.add_module('d_bn1', nn.BatchNorm1d(10))
        self.class_classifier.add_module('d_relu1', nn.ReLU(True))

        self.class_classifier.add_module('d_fc2', nn.Linear(10, 2))
        self.class_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

    def forward(self, input_data):
        #input_data = input_data.expand(input_data.data.shape[0], 1, 3, 32)    # 10x1x3x32

        # sp layer
        feature = torch.matmul(torch.transpose(self.V, 0, 1), input_data)      # 2x3  *  10x1x3x32
        feature = feature.view(-1, feature.size()[1] * feature.size()[2] * feature.size()[3])   # 10x1x2x32  -> 10x64
        '''
        # without sp layer
        feature = input_data.view(-1, input_data.size()[1] * input_data.size()[2] * input_data.size()[3])  # 10x1x3x32 -> 10x96
        '''

        class_output = self.class_classifier(feature)

        return feature, class_output

'''
# test toy data
if __name__ == "__main__":
    net = SPNet()
    f, y = net(torch.randn(2,1,32,32))
    print(y)
'''

'''

self.class_classifier.add_module('d_fc2', nn.Linear(512, 256))   # without sp layer
#self.class_classifier.add_module('d_fc1', nn.Linear(1 * 3 * 32, 10))    # with sp layer
self.class_classifier.add_module('d_bn2', nn.BatchNorm1d(256))
self.class_classifier.add_module('d_relu2', nn.ReLU(True))

self.class_classifier.add_module('d_fc3', nn.Linear(256, 128))   # without sp layer
#self.class_classifier.add_module('d_fc1', nn.Linear(1 * 3 * 32, 10))    # with sp layer
self.class_classifier.add_module('d_bn3', nn.BatchNorm1d(128))
self.class_classifier.add_module('d_relu3', nn.ReLU(True))

self.class_classifier.add_module('d_fc4', nn.Linear(128, 10))   # without sp layer
#self.class_classifier.add_module('d_fc1', nn.Linear(1 * 3 * 32, 10))    # with sp layer
self.class_classifier.add_module('d_bn4', nn.BatchNorm1d(10))
self.class_classifier.add_module('d_relu4', nn.ReLU(True))
'''