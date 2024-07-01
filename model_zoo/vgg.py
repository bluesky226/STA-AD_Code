import torch.nn as nn
import torchvision


class VGGEncoder(nn.Module):
    
    def __init__(self, layers=[4, 9, 16, 23]):
    
        super(VGGEncoder, self).__init__()
        vgg = torchvision.models.vgg16(pretrained=True).features
        self.encoder = nn.ModuleList()
        temp_seq = nn.Sequential()
        for i in range(max(layers) + 1):
            temp_seq.add_module(str(i), vgg[i])
            if i in layers:
                
                self.encoder.append(temp_seq)
                temp_seq = nn.Sequential()

        
        
        

    def forward(self, x):
        features = []
        for layer in self.encoder:
            x = layer(x)
            features.append(x)
        return features
