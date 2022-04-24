import torch
from torchvision import models

class VGGLoss(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(VGGLoss, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23,30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False


    def forward(self, x1, x2, layer):
        
        loss = 0

        h1 = self.slice1(x1)
        h2 = self.slice1(x2)

        if 1 in layer:
            loss += torch.mean(torch.pow(h1-h2,2))
        if layer[-1] == 1:
            return loss

        h1 = self.slice2(h1)
        h2 = self.slice2(h2)
        if 2 in layer:
            loss += torch.mean(torch.pow(h1-h2,2))
        if layer[-1] == 2:
            return loss 
        
        h1 = self.slice3(h1)
        h2 = self.slice3(h2)
        if 3 in layer:
            loss += torch.mean(torch.pow(h1-h2,2))
        if layer[-1] == 3:
            return loss 
        
        h1 = self.slice4(h1)
        h2 = self.slice4(h2)
        if 4 in layer:
            loss += torch.mean(torch.pow(h1-h2,2))
        if layer[-1] == 4:
            return loss 

        h1 = self.slice5(h1)
        h2 = self.slice5(h2)
        if 5 in layer:
            loss += torch.mean(torch.pow(h1-h2,2))
        if layer[-1] == 5:
            return loss         


            