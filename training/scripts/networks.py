import torch 
import torch.nn as nn 

class Positional_Encoding(nn.Module):
    def __init__(self, L):
        super(Positional_Encoding, self).__init__()
        self.L = L 
    def embed(self, x, L):
        rets = [x]
        for i in range(L):
            for fn in [torch.sin, torch.cos]:
                rets.append(fn(2.**i*x))
        return torch.cat(rets, -1)   
    def forward(self, x):
        return self.embed(x, self.L)

class GeneralMLP(nn.Module):
    def __init__(self, num_in, num_out, activation,
                 hiden_depth=4, hiden_width=64):
        super(GeneralMLP, self).__init__()
        assert(hiden_depth >= 2)
        layers = [nn.Linear(num_in, hiden_width)]
        layers.append(activation)
        for i in range(hiden_depth-2):
            layers.append(nn.Linear(hiden_width, hiden_width))
            layers.append(activation)
        layers.append(nn.Linear(hiden_width, num_out))
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.mlp(x)


class MLP(nn.Module):
    def __init__(self, hiden_depth, hiden_width, in_channel, L, num_out, activation):
        super(MLP, self).__init__()
        self.mlp = GeneralMLP(num_in=L*2*in_channel+in_channel,num_out=num_out,
                              activation=activation, hiden_depth=hiden_depth, hiden_width=hiden_width)
        self.PE = Positional_Encoding(L)
    def forward(self,x):
        return self.mlp(self.PE(x))



def init_model(model, mode = 'default'):
    assert mode in ['xavier', 'kaiming', 'zeros', 'default', "small"]
    def kaiming_init(layer):
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            layer.bias.data.fill_(0.)
    def xavier_init(layer):
        if isinstance(layer, nn.Linear):
            nn.init.xavier_normal_(layer.weight)
            layer.bias.data.fill_(0.)
    def zeros_init(layer):
        if isinstance(layer, nn.Linear):
            nn.init.zeros_(layer.weight)
            layer.bias.data.fill_(0.)
    def small_init(layer):
        if isinstance(layer, nn.Linear):
            nn.init.ones_(layer.weight) * 0.0001
            layer.bias.data.fill_(0.0001)
    if mode == 'default':
        return model 
    elif mode == 'kaiming':
        model.apply(kaiming_init)
        print('\n====== Kaiming Init ======\n')
    elif mode == 'xavier':
        model.apply(xavier_init)
        print('\n====== Xavier Init ======\n')
    elif mode == 'zeros':
        model.apply(zeros_init)
        print('\n====== zeros Init ======\n')
    elif mode == "small":
        model.apply(small_init)
        print('\n====== small Init ======\n')
    return model 