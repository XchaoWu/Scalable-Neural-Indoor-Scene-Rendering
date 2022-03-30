import torch 
import torch.nn as nn 


class Unet(nn.Module):
    def __init__(self, in_channel, mode='bilinear',align_corners=True):
        super(Unet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=64, 
                               kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32,
                               kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64,
                               kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64,
                               kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128,
                               kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128,
                               kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(in_channels=192, out_channels=64,
                               kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(in_channels=64, out_channels=64,
                               kernel_size=3, stride=1, padding=1)
        self.conv9 = nn.Conv2d(in_channels=96, out_channels=32,
                               kernel_size=3, stride=1, padding=1)
        self.conv10 = nn.Conv2d(in_channels=32, out_channels=3,
                                kernel_size=3, stride=1, padding=1)

        self.relu = nn.ReLU()

        self.down = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode=mode,align_corners=align_corners)
    
    def forward(self,x):

        O1 = self.relu(self.conv1(x)) # [N, 64, H, W]

        O2 = self.relu(self.conv2(O1)) # [N, 32, H, W]

        O3 = self.down(O2) # [N, 32, H//2, W//2]

        O4 = self.relu(self.conv3(O3)) # [N, 64, H//2, W//2]

        O5 = self.relu(self.conv4(O4)) # [N, 64, H//2, W//2]

        O6 = self.down(O5) # [N, 64, H//4, W//4]

        O7 = self.relu(self.conv5(O6)) # [N, 128, H//4, W//4]

        E7 = self.relu(self.conv6(O7)) # [N, 128, H//4, W//4]

        E6 = self.up(E7) # [N, 128, H//2, W//2]

        E5 = self.relu(self.conv7(torch.cat([E6,O5],dim=1))) # [N, 64, H//2, W//2]

        E4 = self.relu(self.conv8(E5)) # [N, 64, H//2, W//2]

        E3 = self.up(E4) # [N, 64, H, W]

        E2 = self.relu(self.conv9(torch.cat([E3,O2],dim=1))) # [N, 32, H, W]

        E1 = self.relu(self.conv10(E2)) # [N, 3, H, W]

        return E1

class CNN(nn.Module):
    def  __init__(self):
        super(CNN, self).__init__()
        self.cur_inchannel = 5 
        self.RCONS = Unet(self.cur_inchannel)    
    def forward(self, x):
        out = self.RCONS(x)
        return out