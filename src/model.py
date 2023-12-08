import torchvision.models as models
import torch.nn as nn

class Resnet(nn.Module):
    def __init__(self, num_classes=1000):
        super(Resnet, self).__init__()
        
        self.resnet = models.resnet50()
        self.linear = nn.Linear(1000, num_classes)
    
    def forward(self, x):
        resnet_out = self.resnet(x)
        
        return self.linear(resnet_out)