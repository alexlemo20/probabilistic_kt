import torch.nn as nn
import torch.nn.functional as F


class YT_Small(nn.Module):

    def __init__(self, num_classes=10):
        super(YT_Small, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3)
        self.conv1_bn = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.conv2_bn = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3_bn = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3)
        self.conv4_bn = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3)
        self.conv5_bn = nn.BatchNorm2d(64)

        self.fc1 = nn.Linear(in_features=7744, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=num_classes)

    def forward(self, x):
        # Convolutional block 1
        out = F.relu(self.conv1_bn(self.conv1(x)))
        out = F.relu(self.conv2_bn(self.conv2(out)))
        out = F.max_pool2d(out, 2)

        # Convolutional block 2
        out = F.relu(self.conv3_bn(self.conv3(out)))
        out = F.relu(self.conv4_bn(self.conv4(out)))
        out = F.max_pool2d(out, 2)

        # Convolutional Block 3
        out = F.relu(self.conv5_bn(self.conv5(out)))
        out = out.view(out.size(0), -1)

        # FC layer
        out = F.relu(self.fc1(out))
        out = self.fc2(out)

        return out

    def get_features(self, x, level=0):
        # Convolutional block 1
        out = F.relu(self.conv1_bn(self.conv1(x)))
        out = F.relu(self.conv2_bn(self.conv2(out)))
        out = F.max_pool2d(out, 2)

        # Convolutional block 2
        out = F.relu(self.conv3_bn(self.conv3(out)))
        out = F.relu(self.conv4_bn(self.conv4(out)))
        out = F.max_pool2d(out, 2)

        # Convolutional Block 3
        out = F.relu(self.conv5_bn(self.conv5(out)))
        out = out.view(out.size(0), -1)

        # FC layer
        out = F.relu(self.fc1(out))

        return out
        

class FaceNet(nn.Module):
    def __init__(self, num_classes=10):
        super(FaceNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv1_bn = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 64, kernel_size=3, stride=1, padding=1)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3_bn = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv4_bn = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv5_bn = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv6_bn = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv7_bn = nn.BatchNorm2d(512)

        self.fc1 = nn.Linear(in_features=512 * 8 * 8, out_features=4096)
        self.fc2 = nn.Linear(in_features=4096, out_features=num_classes)

    def forward(self, x):
        # Convolutional block 1
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)

        # Convolutional block 2
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = F.max_pool2d(out, 2)

        # Convolutional block 3
        out = F.relu(self.conv5(out))
        out = F.relu(self.conv6(out))
        out = F.relu(self.conv7(out))
        out = F.max_pool2d(out, 2)

        out = out.view(out.size(0), -1)

        # FC layers
        out = F.relu(self.fc1(out))
        out = self.fc2(out)

        return out
        
    def get_features(self, x, level=0):
        # Convolutional block 1
        out = F.relu(self.conv1_bn(self.conv1(x)))
        out = F.relu(self.conv2_bn(self.conv2(out)))
        out = F.max_pool2d(out, 2)

        # Convolutional block 2
        out = F.relu(self.conv3_bn(self.conv3(out)))
        out = F.relu(self.conv4_bn(self.conv4(out)))
        out = F.max_pool2d(out, 2)

        # Convolutional Block 3
        out = F.relu(self.conv5_bn(self.conv5(out)))
        out = F.relu(self.conv6_bn(self.conv6(out)))        
        out = F.relu(self.conv7_bn(self.conv7(out)))
        out = F.max_pool2d(out, 2)
                
        out = out.view(out.size(0), -1)

        # FC layer
        out = F.relu(self.fc1(out))

        return out
	
        

