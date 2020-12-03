import torch
import torch.nn as nn
import torchvision

from torchstat import stat
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
        self.backbone_resnet = nn.Sequential(*(list(self.entire_resnet.children())[:-2]))
        self.conv_block_i = ConvBlock(512, 256)
        for i in range(5):
            self.conv_block_o = ConvBlock(256, 256)

        self.flatten = Flatten()
        self.fc1 = Linear(9216, 4096)
        self.fc2 = Linear(4096, 4096)
        self.output = Linear(4096, 2)

    def forward(self, x):
        x = self.backbone_resnet(x)
        x = self.conv_block_i(x)
        for i in range(5):
            x = self.conv_block_o(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.output(x)
        return x


if __name__ == '__main__':
    model = RegTransNet()
    # stat(model, (3, 192, 192))

    model = model.cuda()
    input_data = torch.rand(32, 3, 192, 192).cuda()
    print(input_data.shape)
    output_data = model(input_data)
    print(output_data.shape)
