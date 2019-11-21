import torch
import torch.nn as nn
from .ResNet import ResNet, BasicBlock, Bottleneck

__all__ = ['scan_resnet18']


class SCAN(nn.Module):
    def __init__(self, channels, b_stride=2, final_channels=512, num_classes=100, selection=False):
        super(SCAN, self).__init__()
        self.attention = nn.Sequential(
                    nn.Conv2d(channels, channels, kernel_size=3, padding=1, stride=2),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(),
                    nn.ConvTranspose2d(channels, channels, kernel_size=4, padding=1, stride=2),
                    nn.BatchNorm2d(channels),
                    nn.Sigmoid()
                )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(channels, 128, kernel_size=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=b_stride, stride=b_stride),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, final_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(final_channels),
            nn.ReLU(),
            nn.AvgPool2d(4,4)
        )
        self.shallow_classifier = nn.Linear(final_channels, num_classes)
        
    def forward(self, x):
        x = self.attention(x)
        features = self.bottleneck(x)
        features = features.view(x.size(0), -1)
        x = self.shallow_classifier(features)
        
        return x, features

class SCAN_ResNet(ResNet):
    def __init__(self, block, layers, num_classes=100):
        super(SCAN_ResNet, self).__init__(block, layers, num_classes)

        self.scan1 = SCAN(
            channels=64,
            final_channels=512,
            b_stride=8
        )
        self.scan2 = SCAN(
            channels=128,
            final_channels=512,
            b_stride=4
        )
        self.scan3 = SCAN(
            channels=256,
            final_channels=512,
            b_stride=2
        )
        self.avgpool = nn.AvgPool2d(4, 4)

    def forward(self, x):
        feature_list = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)

        feature1 = self.scan1(x)

        x = self.layer2(x)

        fea2 = self.scan2(x)
        fea2 = fea2 * x
        feature_list.append(fea2)

        x = self.layer3(x)

        fea3 = self.scan3(x)
        fea3 = fea3 * x
        feature_list.append(fea3)


        x = self.layer4(x)
        x = self.avgpool(x)
        feature_list.append(x)

        feature1 = self.scala1(feature_list[0]).view(x.size(0), -1)
        feature2 = self.scala2(feature_list[1]).view(x.size(0), -1)
        feature3 = self.scala3(feature_list[2]).view(x.size(0), -1)
        feature4 = self.scala4(feature_list[3]).view(x.size(0), -1)

        exit1 = self.fc1(feature1)
        exit2 = self.fc2(feature2)
        exit3 = self.fc3(feature3)
        exit4 = self.fc(feature4)

        return [exit1, exit2, exit3, exit4], [feature1, feature2, feature3, feature4]

    
def scan_resnet18(pretrained=False, **kwargs):
    model = SCAN_ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    
    return model