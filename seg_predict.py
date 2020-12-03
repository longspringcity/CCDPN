import torch
import argparse
import matplotlib.pyplot as plt
import cv2

from torchvision.transforms import transforms
from net.Unet import Unet
from lib.utils import letterbox_image

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='./SegModel_dict49.pth', help='the path of model')
opt = parser.parse_args()


def to_tensor(image):
    trans = transforms.ToTensor()
    img_tensor = trans(image)
    return img_tensor

def detect(image_path):
    model_path = opt.model_path
    model = Unet()
    model.load_state_dict(torch.load(model_path))
    model.cuda()
    image = cv2.imread(image_path)
    cv2.imshow('image', image)
    cv2.waitKey(0)
    image = letterbox_image(image, (192, 192))
    image = to_tensor(image).cuda()
    image = torch.unsqueeze(image, dim=0)
    model.eval()
    mask = model(image).detach().cpu().numpy()[0, 0, :, :]
    plt.imshow(mask)
    plt.show()


if __name__ == '__main__':
    image_path = './data_crop/Real/test/laptop/000001_rgb.png'
    detect(image_path)