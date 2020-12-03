import torch
import torch.nn as nn
import torchvision

from torch.nn.modules import Linear, Conv2d, ReLU, BatchNorm2d, Flatten


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ConvBlock, self).__init__()
        self.conv = Conv2d(in_channel, out_channel, 3, padding=True)
        self.bn = BatchNorm2d(out_channel)
        self.relu = ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class RegTransNet(nn.Module):
    def __init__(self):
        super(RegTransNet, self).__init__()
        self.entire_resnet = torchvision.models.resnet34(pretrained=False)
        self.output = Linear(1000, 2)

    def forward(self, x):
        x = self.entire_resnet(x)
        x = self.output(x)
        return x


if __name__ == '__main__':
    model = RegTransNet()
    # stat(model, (3, 192, 192))

    model = model.cuda()
    input_data = torch.rand(1, 3, 192, 192).cuda()
    print(input_data.shape)
    output_data = model(input_data)
    print(output_data.shape)
