import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizer import * 
from conv_transformer import *

# https://discuss.pytorch.org/t/how-to-pad-one-side-in-pytorch/21212
# https://pytorch.org/docs/stable/generated/torch.nn.ReflectionPad2d.html#torch.nn.ReflectionPad2d
# Purposefully avoided using pooling and strided conv layers here:
# 1. No need for pooling since image sizes are just 30x30 at best. Pooling here, causes rapid decrease
# in resolution of images and hence, a smaller network
# 2. Strided conv layers cause loss of discriminative features; each pixel may get sampled different number of times
# when the stride !=1, and hence subsequent layers will lose information from previous layers
# Total number of params: 432+4608+9216+4608+2304+2304+1296000+100100 = 1419572
# Total datapoints in C1: 30*30*3*4071 = 10991700

class Seq_Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3),
                                    nn.ReLU())
        
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc = nn.Sequential(nn.Flatten(),nn.Linear(9*9*16,1000),
                                nn.ReLU(),
                                nn.Linear(1000,100),
                                nn.ReLU(),
                                nn.Linear(100,1))

        
    def forward(self, x):
        cap = None
        x1 = self.layer1(x)
        x2 = self.pool1(x1)
        return self.fc(x2)
    

# Fire module as described in paper: 1x1 'squeeze' layers, which feed into 1x1 and 3x3 'expand' layers
# Squeeze layer filters num can be as low as 1/8th of expand layer filters num.
# Hence, to compensate the severe dimensionality reduction, bypass connections can be added 
# between Fire modueles, as in ResNet
class Fire(nn.Module):
    """
    Parameters:
    squeeze_in (int): Number of input channels to squeeze layer
    squeeze_out (int): Number of output channels from squeeze layer, to both expand layer filters 
    expand1_out (int): Number of output channels from 1x1 expand layer
    expand3_out (int): Number of output channels from 3x3 expand layer
    """
    def __init__(self, squeeze_in, squeeze_out, expand1_out, expand3_out):
        super().__init__()

        self.squeeze = nn.Conv2d(in_channels=squeeze_in, out_channels=squeeze_out, kernel_size=1)
        self.expand_1 = nn.Conv2d(in_channels=squeeze_out, out_channels=expand1_out, kernel_size=1)
        self.expand_3 = nn.Conv2d(in_channels=squeeze_out, out_channels=expand3_out, kernel_size=3, padding=1)

    def forward(self, x):

        squeeze1 = F.conv2d(x, self.squeeze.weight)
        relu_squeeze1 = F.relu(squeeze1)
        expand1 = F.conv2d(relu_squeeze1, self.expand_1.weight)
        relu_expand1 = F.relu(expand1)
        expand3 = F.conv2d(relu_squeeze1, self.expand_3.weight, padding=1)
        relu_expand3 = F.relu(expand3)
        # Concatenate along channel dim, NxCxHxW
        return torch.concat([relu_expand1, relu_expand3], dim=1)


class SqueezeNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1), nn.ReLU())
        # Can add simple residual links between Fire modules here
        self.squeeze2 = nn.Sequential(Fire(16,16,64,64), nn.ReLU()) 
        self.squeeze3 = nn.Sequential(Fire(128,16,64,64), nn.ReLU())
        self.squeeze4 = nn.Sequential(Fire(128,64,128,128), nn.ReLU())
        self.maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Can add simple residual links between Fire modules here
        self.squeeze6 = nn.Sequential(Fire(256,48,192,192), nn.ReLU()) 
        self.squeeze7 = nn.Sequential(Fire(384,48,192,192), nn.ReLU()) 
        self.squeeze8 = nn.Sequential(Fire(384,64,256,256), nn.ReLU())
        self.maxpool9 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        # Adding complex residual link here, 7x7x512
        self.res_link1 = nn.Conv2d(512,256,1)
        self.squeeze10 = nn.Sequential(Fire(512,64,128,128), nn.ReLU())

        # Custom additions from here
        self.conv11 = nn.Conv2d(256,64,1)
        self.conv12 = nn.Conv2d(64,16,1)
        self.fc13 = nn.Sequential(nn.Flatten(), nn.Linear(7*7*16,512),
                                nn.ReLU(),
                                nn.Linear(512,128),
                                nn.ReLU(),
                                nn.Linear(128,1))


    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.squeeze2(x1)

        # Simple residual link
        x3 = self.squeeze3(x2) + x2
        
        x4 = self.squeeze4(x3)
        x5 = self.maxpool5(x4)
        x6 = self.squeeze6(x5)
        
        # Simple residual link
        x7 = self.squeeze7(x6) + x6
        
        x8 = self.squeeze8(x7)
        x9 = self.maxpool9(x8)
        
        # Complex residual link
        x10 = self.squeeze10(x9) + F.conv2d(x9, self.res_link1.weight)
        
        x11 = self.conv11(x10)
        relu_x11 = F.relu(x11)
        x12 = self.conv12(relu_x11)
        relu_x12 = F.relu(x12)
        return self.fc13(relu_x12)
    

class ConvTransformer(nn.Module):
    def __init__(self,
                 img_size=30,
                 embedding_dim=64,
                 in_channels=3,
                 kernel_size=(1,16),
                 stride=(1,16),
                 padding=1,
                 avgpool_kernel=(1,2),
                 avgpool_stride=(1,2),
                 avgpool_padding=0,
                 dropout=0.,
                 attn_dropout=0.1,
                 num_encoder_layers=3,
                 nheads=2,
                 mlp_ratio=4.0,
                 num_classes=1,
                 positional_embedding=True, 
                 seq_pool=True) -> None:
        super().__init__()
        self.tokenizer = ConvTokenizer(in_channels, embedding_dim, kernel_size, stride, padding,
                                       avgpool_kernel, avgpool_stride, avgpool_padding)
        
        seq_len = self.tokenizer.seq_len(in_channels, img_size, img_size)
        dim_feedforward = int(embedding_dim*mlp_ratio)
        # self.regressor = RegressionTransformer()
        self.regressor = RegressionTransformer(embedding_dim, nheads, seq_len,
                                         num_encoder_layers, dim_feedforward, dropout, attn_dropout, mlp_ratio,
                                         num_classes, seq_pool, positional_embedding)

    def forward(self, x):
        tokens = self.tokenizer(x)
        return self.regressor(tokens) 
