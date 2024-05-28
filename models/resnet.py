from torch import nn
import torch


class ConvBlock(nn.Module):
    def __init__(self, in_channels, filters, kernel_size, stride):
        super(ConvBlock, self).__init__()
        f1, f2, f3 = filters

        self.main_path = nn.Sequential(
            nn.Conv2d(
                in_channels, f1, kernel_size=1, stride=stride, padding=0, bias=False
            ),
            nn.BatchNorm2d(f1),
            nn.ReLU(inplace=True),
            nn.Conv2d(f1, f2, kernel_size=kernel_size, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(f2),
            nn.ReLU(inplace=True),
            nn.Conv2d(f2, f3, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(f3),
        )

        self.shortcut = nn.Sequential(
            nn.Conv2d(
                in_channels, f3, kernel_size=1, stride=stride, padding=0, bias=False
            ),
            nn.BatchNorm2d(f3),
        )

        self.out_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_copy = x.clone()
        x = self.main_path(x)
        x += self.shortcut(x_copy)
        x = self.out_relu(x)
        return x


class IdBlock(nn.Module):
    def __init__(self, in_channels, filters, kernel_size):
        super(IdBlock, self).__init__()
        f1, f2, f3 = filters

        self.main_path = nn.Sequential(
            nn.Conv2d(in_channels, f1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(f1),
            nn.ReLU(inplace=True),
            nn.Conv2d(f1, f2, kernel_size=kernel_size, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(f2),
            nn.ReLU(inplace=True),
            nn.Conv2d(f2, f3, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(f3),
        )

        self.out_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_copy = x.clone()
        x = self.main_path(x)
        x += x_copy
        x = self.out_relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, layers_blocks, num_classes, input_channels=3):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.stage_0 = nn.Sequential(
            nn.Conv2d(
                input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.stage_1 = self._make_stage(64, layers_blocks[0], stride=1)
        self.stage_2 = self._make_stage(128, layers_blocks[1], stride=2)
        self.stage_3 = self._make_stage(256, layers_blocks[2], stride=2)
        self.stage_4 = self._make_stage(512, layers_blocks[3], stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.stage_0(x)
        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.stage_3(x)
        x = self.stage_4(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def _make_stage(self, out_channels, num_blocks, stride):
        layers = []
        layers.append(
            ConvBlock(
                self.in_channels,
                filters=(out_channels, out_channels, out_channels * 4),
                kernel_size=3,
                stride=stride,
            )
        )
        self.in_channels = out_channels * 4
        for _ in range(1, num_blocks):
            layers.append(
                IdBlock(
                    self.in_channels,
                    filters=(out_channels, out_channels, out_channels * 4),
                    kernel_size=3,
                )
            )
        return nn.Sequential(*layers)


def ResNet100(num_classes, num_channels=3):
    return ResNet([2, 3, 23, 2], num_classes, num_channels)


def ResNet50(num_classes, num_channels=3):
    return ResNet([2, 3, 5, 2], num_classes, num_channels)


def ResNet18(num_classes, num_channels=3):
    return ResNet([1, 1, 1, 1], num_classes, num_channels)


