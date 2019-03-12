import torch
import torch.nn as nn
import torch.nn.functional as func
from functools import reduce    

import torch
import torch.nn as nn
import torch.nn.functional as func
from functools import reduce


class Network(nn.Module):
    def __init__(self, height=32, width=32, conv_channels=3,
                conv_config=(
                    {'kernel_size': 7, 'stride': 1, 'padding': 3, 'dilation': 1},
                    {'kernel_size': 7, 'stride': 1, 'padding': 3, 'dilation': 1},
                    {'kernel_size': 7, 'stride': 1, 'padding': 3, 'dilation': 1},
                )):
        super(Network, self).__init__()
        
        # field initialization
        self.conv_channels = conv_channels
        self.relu_slope = 0.0
        
        # layers           
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=conv_channels, out_channels=32, bias=True, **conv_config[0]),
            nn.Conv2d(in_channels=32, out_channels=16, bias=True, **conv_config[1]),
            nn.Conv2d(in_channels=16, out_channels=8, bias=True, **conv_config[2]),
        ])
        
        self.linears = nn.ModuleList([
            nn.Linear(in_features=self.calc_flattened_size(conv_channels, height, width), out_features=32, bias=True),
            nn.Linear(in_features=32, out_features=16, bias=True)
        ])
        
        self.output_layer = nn.Linear(in_features=16, out_features=10, bias=False) 
    
    
    def calc_flattened_size(self, conv_channels, height, width):
        '''calculate number of items in each image'''
        with torch.no_grad():
            x = self.run_convs(torch.ones((1, conv_channels, height, width)))
            return self.flatten(x).size()[1]
    
    
    @staticmethod
    def flatten(x):
        '''flatten tensor of size [a, b, c, ...] to [a, b*c*d...]'''
        return x.view(-1, reduce(lambda a, b: a*b, list(x.size()[1:])))

    
    def run_convs(self, x):
        '''transform x with convolutional module list'''
        for i in range(len(self.convs)):
            #print(f'input size: {x.shape}')
            x = self.convs[i](x)
            # ReLU activation
            x = func.leaky_relu(x, negative_slope=self.relu_slope)
            # normalize
            x = func.normalize(x, p=2, dim=1) #batch_norm(input, running_mean, running_var, weight=None, bias=None, training=self.training, momentum=0.1, eps=1e-05) 
            # max pool
            x = func.max_pool2d(x, kernel_size=(2,2))
        
        return x
    
    
    def run_linears(self, x):
        '''transform x with linear module list'''
        for i in range(len(self.linears)):
            x = self.linears[i](x)
            # ReLU activation
            x = func.leaky_relu(x, negative_slope=self.relu_slope)
            # normalize
            x = func.normalize(x, p=2, dim=1) #batch_norm(input, running_mean, running_var, weight=None, bias=None, training=self.training, momentum=0.1, eps=1e-05)    
        
        return x
    
    
    def forward(self, x):        
        # convolutional layers
        x = self.run_convs(x)
        #print('conv done, x size', x.size())
        
        # flatten to 2 dimensions
        x = self.flatten(x)
        #print('flattened x size', x.size())
        
        # linear layers
        x = self.run_linears(x)
        #print('linears done, x size', x.size())
        
        # output layer
        x = self.output_layer(x)
        #x = torch.nn.functional.softmax(x, dim=1)
        return x