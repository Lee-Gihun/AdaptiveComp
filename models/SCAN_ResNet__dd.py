import torch.nn as nn

__all__ = ['scan_resnet18']

def ScalaNet(channel_in, channel_out, size):
    return nn.Sequential(
        nn.Conv2d(channel_in, 128, kernel_size=1, stride=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(128, 128, kernel_size=size, stride=size),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(128, channel_out, kernel_size=1, stride=1),
        nn.BatchNorm2d(channel_out),
        nn.ReLU(),
        nn.AvgPool2d(4, 4)
        )

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


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
        selection = None
        x = self.attention(x)
        features = self.bottleneck(x)
        features = features.view(x.size(0), -1)
        x = self.shallow_classifier(features)
        
        return x, features, selection

    
class SCAN(nn.Module):
    def __init__(self, channels, b_stride=2, final_channels=512, num_classes=100, selection=False):
        super(SCAN, self).__init__()
        
        if selection:
            self._selection = SelectionLayer(final_channels)
        
        self.selection = selection
            
        # activation func
        self._relu = nn.ReLU(inplace=True)
        
        # attention module
        self._attconv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1, bias=True)
        self._attbn0 = nn.BatchNorm2d(channels)
        self._attdeconv = nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1, bias=True)
        self._attbn1 = nn.BatchNorm2d(channels)
        
        # bottleneck module
        self._bot1x1_0 = nn.Conv2d(channels, 128, kernel_size=1, stride=1, bias=True)
        self._botbn0 = nn.BatchNorm2d(128)
        self._bot3x3 = nn.Conv2d(128, 128, kernel_size=b_stride, stride=b_stride, bias=True)
        self._botbn1 = nn.BatchNorm2d(128)
        self._bot1x1_1 = nn.Conv2d(128, final_channels, kernel_size=1, stride=1, bias=True)
        self._botbn2 = nn.BatchNorm2d(final_channels)
        
        # classifier module
        self._globalavgpool = nn.AvgPool2d(4, 4)
        self._shallow_classifier = nn.Conv2d(final_channels, num_classes, kernel_size=1, stride=1, bias=True)

        
    
    def forward(self, x):
        # attention
        att = self._relu(self._attbn0(self._attconv(x)))
        att = self._relu(self._attbn1(self._attdeconv(att)))
        x = x * torch.sigmoid(att)
        
        # bottleneck
        x = self._relu(self._botbn0(self._bot1x1_0(x)))
        x = self._relu(self._botbn1(self._bot3x3(x)))
        
        features = self._relu(self._botbn2(self._bot1x1_1(x)))
        features = self._globalavgpool(features)
        
        selection = None
        
        if self.selection:
            selection = self._selection(features).squeeze()
        
        x = self._shallow_classifier(features).squeeze()
        
        return x, features, selection


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=100, zero_init_residual=False, align="CONV"):
        super(ResNet, self).__init__()
        print("num_class: ", num_classes)
        self.inplanes = 64
        self.align = align
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        #   self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        print("CONV for aligning")
        self.scala1 = ScalaNet(
            channel_in=64*block.expansion,
            channel_out=512*block.expansion,
            size=8
        )
        self.scala2 = ScalaNet(
            channel_in=128 * block.expansion,
            channel_out=512 * block.expansion,
            size=4
        )
        self.scala3 = ScalaNet(
            channel_in=256 * block.expansion,
            channel_out=512 * block.expansion,
            size=2
        )
        self.scala4 = nn.AvgPool2d(4, 4)

        self.attention1 = nn.Sequential(
            nn.Conv2d(kernel_size=3, padding=1, stride=2, in_channels=64* block.expansion, out_channels=64* block.expansion),
            nn.BatchNorm2d(64* block.expansion),
            nn.ReLU(),
            nn.ConvTranspose2d(kernel_size=4, padding=1, stride=2, in_channels=64* block.expansion, out_channels=64* block.expansion),
            nn.BatchNorm2d(64* block.expansion),
            nn.Sigmoid()
        )

        self.attention2 = nn.Sequential(
            nn.Conv2d(kernel_size=3, padding=1, stride=2, in_channels=128* block.expansion, out_channels=128* block.expansion),
            nn.BatchNorm2d(128* block.expansion),
            nn.ReLU(),
            nn.ConvTranspose2d(kernel_size=4, padding=1, stride=2, in_channels=128* block.expansion, out_channels=128* block.expansion),
            nn.BatchNorm2d(128* block.expansion),
            nn.Sigmoid()
        )

        self.attention3 = nn.Sequential(
            nn.Conv2d(kernel_size=3, padding=1, stride=2, in_channels=256* block.expansion, out_channels=256* block.expansion),
            nn.BatchNorm2d(256* block.expansion),
            nn.ReLU(),
            nn.ConvTranspose2d(kernel_size=4, padding=1, stride=2, in_channels=256* block.expansion, out_channels=256* block.expansion),
            nn.BatchNorm2d(256* block.expansion),
            nn.Sigmoid()
        )

        self.fc1 = nn.Linear(512 * block.expansion, num_classes)
        self.fc2 = nn.Linear(512 * block.expansion, num_classes)
        self.fc3 = nn.Linear(512 * block.expansion, num_classes)
        self.fc4 = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        feature_list = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)

        fea1 = self.attention1(x)
        fea1 = fea1 * x
        feature_list.append(fea1)

        x = self.layer2(x)

        fea2 = self.attention2(x)
        fea2 = fea2 * x
        feature_list.append(fea2)

        x = self.layer3(x)

        fea3 = self.attention3(x)
        fea3 = fea3 * x
        feature_list.append(fea3)


        x = self.layer4(x)
        feature_list.append(x)

        feature1 = self.scala1(feature_list[0]).view(x.size(0), -1)
        feature2 = self.scala2(feature_list[1]).view(x.size(0), -1)
        feature3 = self.scala3(feature_list[2]).view(x.size(0), -1)
        feature4 = self.scala4(feature_list[3]).view(x.size(0), -1)

        exit1 = self.fc1(feature1)
        exit2 = self.fc2(feature2)
        exit3 = self.fc3(feature3)
        exit4 = self.fc4(feature4)

        return [exit1, exit2, exit3, exit4], [feature1, feature2, feature3, feature4]
        # None is prepared for Hint Learning


def scan_resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    
    return model