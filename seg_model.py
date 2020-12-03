import torch
import torch.nn as nn
from torch.nn.modules import Conv2d, MaxPool2d, BatchNorm2d, ReLU, UpsamplingBilinear2d, Sigmoid


class DownSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSampleBlock, self).__init__()
        self.conv = Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn = BatchNorm2d(out_channels)
        self.relu = ReLU(inplace=True)
        self.pool = MaxPool2d(2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class UpSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, size):
        super(UpSampleBlock, self).__init__()
        self.up_sample = UpsamplingBilinear2d(size)
        self.conv = Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn = BatchNorm2d(out_channels)
        self.relu = ReLU(inplace=True)

    def forward(self, x):
        x = self.up_sample(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class NaivePSPNet(nn.Module):
    def __init__(self):
        super(NaivePSPNet, self).__init__()
        self.down_sample1 = DownSampleBlock(3, 64)
        self.down_sample2 = DownSampleBlock(64, 128)
        self.down_sample3 = DownSampleBlock(128, 256)
        self.down_sample4 = DownSampleBlock(256, 512)
        self.up_sample1 = UpSampleBlock(512, 256, size=24)
        self.up_sample2 = UpSampleBlock(256, 128, size=48)
        self.up_sample3 = UpSampleBlock(128, 64, size=96)
        self.up_sample4 = UpSampleBlock(64, 1, size=192)

    def forward(self, x):
        x = self.down_sample1(x)
        x = self.down_sample2(x)
        x = self.down_sample3(x)
        x = self.down_sample4(x)
        x = self.up_sample1(x)
        x = self.up_sample2(x)
        x = self.up_sample3(x)
        x = self.up_sample4(x)
        return x


if __name__ == '__main__':
    input_data = torch.rand([1, 3, 192, 192])
    model = NaivePSPNet(6)
    output_data = model(input_data)
    print(input_data.shape)
    print(output_data.shape)
