import torch.nn as nn
import torch.nn.functional as F
from utils import zero_initialize

# https://discuss.pytorch.org/t/how-to-pad-one-side-in-pytorch/21212
# https://pytorch.org/docs/stable/generated/torch.nn.ReflectionPad2d.html#torch.nn.ReflectionPad2d

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)
        nn.init.kaiming_normal_(self.conv1.weight)
        self.convb1 = zero_initialize((16,))
        
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        nn.init.kaiming_normal_(self.conv2.weight)
        self.convb2 = zero_initialize((32,))

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        nn.init.kaiming_normal_(self.conv3.weight)
        self.convb3 = zero_initialize((32,))

        self.conv4 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3)
        nn.init.kaiming_normal_(self.conv4.weight)
        self.convb4 = zero_initialize((16,))

        self.conv5 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3)
        nn.init.kaiming_normal_(self.conv5.weight)
        self.convb5 = zero_initialize((16,))

        self.conv6 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3)
        nn.init.kaiming_normal_(self.conv6.weight)
        self.convb6 = zero_initialize((16,))
        
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(9*9*16,1000)
        self.fcb1 = zero_initialize((1000,))
        
        self.fc2 = nn.Linear(1000,100)
        self.fcb2 = zero_initialize((100,))

        self.fc3 = nn.Linear(100,1)
        self.fcb3 = zero_initialize((1,))

        
    def forward(self, x):
        cap = None

        x1 = F.conv2d(x, self.conv1.weight, bias=self.convb1)
        relu_x1 = F.relu(x1)

        x2 = F.conv2d(relu_x1, self.conv2.weight, bias=self.convb2)
        relu_x2 = F.relu(x2)

        x3 = F.conv2d(relu_x2, self.conv3.weight, bias=self.convb3)
        relu_x3 = F.relu(x3)

        x4 = F.conv2d(relu_x3, self.conv4.weight, bias=self.convb4)
        relu_x4 = F.relu(x4)

        x5 = F.conv2d(relu_x4, self.conv5.weight, bias=self.convb5)
        relu_x5 = F.relu(x5)

        x6 = F.conv2d(relu_x5, self.conv6.weight, bias=self.convb6)
        relu_x6 = F.relu(x6)

        x7 = F.max_pool2d(relu_x6, 2, 2)
        relu_x7 = F.relu(x7)

        flat_in = self.flatten(relu_x7)

        x8 = F.linear(flat_in, self.fc1.weight, bias=self.fcb1)
        relu_x8 = F.relu(x8)

        x9 = F.linear(relu_x8, self.fc2.weight, bias=self.fcb2)
        relu_x9 = F.relu(x9)

        cap = F.linear(relu_x9, self.fc3.weight, bias=self.fcb3)
        
        return cap