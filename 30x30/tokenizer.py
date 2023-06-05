import torch
import torch.nn as nn

# We envision the input to be Nx3x32x32, where each row is the time-series data from each charge/ discharge (c/d) cycle
# Hence, we want convolutions to tokenize along rows (hence, H=1, W=16), instead of mixing data across rows and adding information
# from different parts of the c/d curve together through H=W!=1. With W=16, H=1, tokens are generated from 
# consecutive curve patches, which is more reasonable
# Hence, final output is Nx64x32x2, which we avg pool by a (1x2) filter -> Nx64x32x1 tensor
# After flatten and transpose, it will be Nx32x64
# It's better to avpool instead of maxpool, since we would lose info about curves if we use max. Instead, 
# using av pooling allows to use data from adjacent curve patches together, while reducing amount of tokens

class ConvTokenizer(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, kernel_size=(1,16), stride=(1,16), padding=(1,1),
                 avgpool_kernel=(1,2), avgpool_stride=(1,2), avgpool_padding=0):
        super().__init__()

        self.embed_layers = nn.Sequential(nn.Conv2d(in_channels=in_channels,out_channels=out_channels,
                                                    kernel_size=kernel_size,
                                                    stride=stride,padding=padding), nn.ReLU(),
                                          nn.AvgPool2d(avgpool_kernel,avgpool_stride, avgpool_padding))
        self.flatten = nn.Flatten(2,3)

    def seq_len(self, in_channels, height, width):
        return self.forward(torch.zeros(1, in_channels, height, width)).size()[1]

    def forward(self, x):
        out = self.embed_layers(x)
        return self.flatten(out).transpose(-2,-1)