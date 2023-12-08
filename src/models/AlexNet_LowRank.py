import torch.nn as nn
from src.compression.LowRankLinear import LowRankLinear


def getBase(model):
    """
    @param model : The original AlexNet.
    
    @return The weights and bias needed to act as the base for the low-rank version of the custom linear layers.
    """
    wd = model.state_dict()
    w = [wd['classifier.1.weight'], wd['classifier.4.weight']]
    b = [wd['classifier.1.bias'], wd['classifier.4.bias']]
    return w, b

def load_sd_decomp(org_sd, model, decomposed_layers):
    """
    @param org_sd : The state_dict when the model is ongoing.
    @param model : The decomp model with decomposed layers.
    @param decomposed_layers : The decomposed layers in decomp model.

    @return The new model with the old state dictionary loaded in.
    """
    new_sd = model.state_dict()
    for k, v in org_sd.items():
        if k not in decomposed_layers:
            new_sd[k] = v
    model.load_state_dict(new_sd)

class AlexNet_LowRank(nn.Module):   
    def __init__(self, base : list, bias : list, num=10, rank = 100):
        """
        @param base : List of initial bases for the linear layers, kept as a parameter.
        @param rank : The rank of the original model to be kept.
        """
        super(AlexNet_LowRank, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=1),
            nn.ReLU(inplace=True), 
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),   
            nn.MaxPool2d( kernel_size=2, stride=2),
            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),                         
            nn.Conv2d(96, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),                         
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d( kernel_size=2, stride=1),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            LowRankLinear(32*12*12, 2048, rank, base[0], bias[0]),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            LowRankLinear(2048, 1024, rank, base[1], bias[1]),
            nn.ReLU(inplace=True),
            nn.Linear(1024,num),
        )
    
    def forward(self, x):
        x = self.feature(x)
        x = x.view(-1,32*12*12)
        x = self.classifier(x)
        return x