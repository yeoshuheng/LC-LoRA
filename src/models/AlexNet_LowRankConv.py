import torch.nn as nn
from src.compression.LowRankLinear import LowRankLinear
from src.compression.LowRankConv import LowRankConv


def getBase(model):
    """
    @param model : The original AlexNet.
    
    @return The weights and bias needed to act as 
        the base for the low-rank version of the custom linear layers.
    """
    wd = model.state_dict()
    w = [wd['classifier.1.weight'], wd['classifier.4.weight']]
    b = [wd['classifier.1.bias'], wd['classifier.4.bias']]
    w_conv = [wd['feature.2.weight'], wd['feature.5.weight'], 
              wd['feature.7.weight'], wd['feature.9.weight']]
    b_conv = [wd['feature.2.bias'], wd['feature.5.bias'], 
              wd['feature.7.bias'], wd['feature.9.bias']]
    return w, b, w_conv, b_conv

def load_sd_decomp(org_sd, model, decomposed_layers, decomposed_conv_layers):
    """
    @param org_sd : The state_dict when the model is ongoing.
    @param model : The decomp model with decomposed layers.
    @param decomposed_layers : The decomposed linear layers in decomp model.
    @param decomposed_conv_layers : The decomposed conv layers in decomp model.

    @return The new model with the old state dictionary loaded in.
    """
    new_sd = model.state_dict()
    for k, v in org_sd.items():
        if k not in decomposed_layers and k not in decomposed_conv_layers:
            new_sd[k] = v
    model.load_state_dict(new_sd)

class AlexNet_LowRank(nn.Module):   
    def __init__(self, weights : list, bias : list, conv_weights : list, 
                 conv_bias : list, num=10, rank = -1):
        """
        @param weights : List of initial bases for the loRA linear layers, kept as a parameter.
        @param bias : List of initial biases for the loRA linear layers, kept as a parameter.
        @param conv_weights : List of initial bases for the loRA conv layers, kept as a parameter.
        @param conv_bias : List of initial biases for the loRA conv layers, kept as a parameter.
        @param rank : The rank of the original model to be kept.
        """
        super(AlexNet_LowRank, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=1),
            nn.ReLU(inplace=True),
            LowRankConv(32, 64, conv_weights[0], conv_bias[0], rank = rank),
            nn.ReLU(inplace=True),   
            nn.MaxPool2d( kernel_size=2, stride=2),
            LowRankConv(64, 96, conv_weights[1], conv_bias[1], rank = rank),
            nn.ReLU(inplace=True),                         
            LowRankConv(96, 64, conv_weights[2], conv_bias[2], rank = rank),
            nn.ReLU(inplace=True),                         
            LowRankConv(64, 32, conv_weights[3], conv_bias[3], rank = rank),
            nn.ReLU(inplace=True),
            nn.MaxPool2d( kernel_size=2, stride=1),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            LowRankLinear(32*12*12, 2048, weights[0], bias[0], rank = rank),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            LowRankLinear(2048, 1024, weights[1], bias[1], rank = rank),
            nn.ReLU(inplace=True),
            nn.Linear(1024,num),
        )
    
    def forward(self, x):
        x = self.feature(x)
        x = x.view(-1,32*12*12)
        x = self.classifier(x)
        return x