import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as functional


def get_resnet(net):  # sourcery skip: raise-specific-error
    if net == 'resnet18':
        return torchvision.models.resnet18(pretrained=True)
    elif net == 'resnet34':
        return torchvision.models.resnet34(pretrained=True)
    elif net == 'resnet50':
        return torchvision.models.resnet50(pretrained=True)
    elif net == 'resnet101':
        return torchvision.models.resnet101(pretrained=True)
    elif net == 'resnet152':
        return torchvision.models.resnet152(pretrained=True)
    else:
        raise Exception('invalid resnet')


class ResnetUNet(nn.Module):
    def __init__(self, net='resnet34', freeze_resnet=True):
        super().__init__()

        self.resnet = get_resnet(net)
        self.bottleneck = nn.Sequential(*[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
                                          nn.ReLU(inplace=False)])
        self.up3 = nn.Sequential(*[nn.Conv2d(512 + 256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.ReLU(inplace=False)])
        self.up2 = nn.Sequential(*[nn.Conv2d(256 + 128, 128, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.ReLU(inplace=False)])
        self.up1 = nn.Sequential(*[nn.Conv2d(128 + 64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.ReLU(inplace=False)])
        self.up0 = nn.Sequential(*[nn.Conv2d(64 + 3, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.ReLU(inplace=False)])
        self.normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        if freeze_resnet:
            for e in self.resnet.modules():
                for param in e.parameters():
                    param.requires_grad = False

                if isinstance(e, nn.BatchNorm2d):
                    e.momentum = 0
                    e.eval()

    def forward(self, x):
        x = self.normalize(x)

        x0 = self.resnet.conv1(x)
        x0 = self.resnet.bn1(x0)
        x0 = self.resnet.relu(x0)

        x1 = self.resnet.maxpool(x0)
        x1 = self.resnet.layer1(x0)
        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)

        x4 = self.bottleneck(x4)
        u4 = functional.interpolate(x4, size=(x3.shape[2], x3.shape[3]), mode='nearest')

        u3 = self.up3(torch.cat((x3, u4), dim=1))
        u3 = functional.interpolate(u3, size=(x2.shape[2], x2.shape[3]), mode='nearest')

        u2 = self.up2(torch.cat((x2, u3), dim=1))
        u2 = functional.interpolate(u2, size=(x1.shape[2], x1.shape[3]), mode='nearest')

        u1 = self.up1(torch.cat((x1, u2), dim=1))
        u1 = functional.interpolate(u1, size=(x.shape[2], x.shape[3]), mode='nearest')

        return self.up0(torch.cat((x, u1), dim=1))
