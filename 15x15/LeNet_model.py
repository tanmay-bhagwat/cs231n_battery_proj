import torch.nn as nn
import torch.nn.functional as F
from utils import zero_initialize

# https://discuss.pytorch.org/t/how-to-pad-one-side-in-pytorch/21212
# https://pytorch.org/docs/stable/generated/torch.nn.ReflectionPad2d.html#torch.nn.ReflectionPad2d

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=2)
        nn.init.kaiming_normal_(self.conv1.weight)
        self.convb1 = zero_initialize((16,))

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=2)
        nn.init.kaiming_normal_(self.conv2.weight)
        self.convb2 = zero_initialize((32,))

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=2)
        nn.init.kaiming_normal_(self.conv3.weight)
        self.convb3 = zero_initialize((16,))

        self.conv4 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=2)
        nn.init.kaiming_normal_(self.conv4.weight)
        self.convb4 = zero_initialize((16,))

        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(3*3*16,50)
        self.fcb1 = zero_initialize((50,))
        
        self.fc2 = nn.Linear(50,1)
        self.fcb2 = zero_initialize((1,))

        
    def forward(self, x):
        cap = None

        x1 = F.conv2d(x, self.conv1.weight, bias=self.convb1)
        relu_x1 = F.relu(x1)
        pool_x1 = F.max_pool2d(relu_x1, 2, stride=2)

        x2 = F.conv2d(pool_x1, self.conv2.weight, bias=self.convb2)
        relu_x2 = F.relu(x2)
        pool_x2 = F.max_pool2d(relu_x2, 2, stride=2)

        # See comments on top for docs
        p1 = nn.ConstantPad2d((1,0,1,0),0)
        x3 = p1(pool_x2)

        x3 = F.conv2d(x3, self.conv3.weight, bias=self.convb3)
        relu_x3 = F.relu(x3)

        # See comments on top for docs
        x4 = p1(relu_x3)

        x4 = F.conv2d(x4, self.conv4.weight, bias=self.convb4)
        relu_x4 = F.relu(x4)

        flat_in = self.flatten(relu_x4)

        x5 = F.linear(flat_in, self.fc1.weight, bias=self.fcb1)
        relu_x5 = F.relu(x5)

        cap = F.linear(relu_x5, self.fc2.weight, bias=self.fcb2)
        
        return cap