import os
import cv2
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import glob

from lib.utils import letterbox_image
from PIL import Image


class SegDataset(data.Dataset):
    def __init__(self, source, mode, data_dir, img_size, cat, visualize=False):
        """
        Args:
            source: 'CAMERA', 'Real' or 'CAMERA+Real'
            mode: 'train' or 'test'
            data_dir:
            img_size: square image window
        """
        self.source = source
        self.mode = mode
        self.data_dir = data_dir
        self.img_size = img_size
        self.cat = cat
        self.visualize = visualize

        assert source in ['CAMERA', 'Real', 'CAMERA+Real']
        assert mode in ['train', 'test', 'train_test', 'test_test']

        self.cat_names = ['bottle', 'bowl', 'camera', 'can', 'laptop', 'mug']

        crop_data_path = os.path.join(self.data_dir, self.source, self.mode, self.cat_names[self.cat - 1])
        data_info_path = os.path.join(crop_data_path, 'center_points.txt')
        self.data_info = np.loadtxt(data_info_path)

        self.img_list = sorted(glob.glob(os.path.join(crop_data_path, '*rgb.png')))
        self.mask_list = sorted(glob.glob(os.path.join(crop_data_path, '*mask.png')))

        self.length = len(self.img_list)
        self.colorjitter = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.3)
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])
        print('{} images found.'.format(self.length))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        img_path = self.img_list[index]
        raw_rgb = cv2.imread(img_path)[:, :, :3]

        instance_id = self.data_info[index][2]

        mask_path = self.mask_list[index]
        mask = cv2.imread(mask_path)[:, :, 2]
        mask[mask != instance_id] = 0
        mask[mask == instance_id] = 1

        raw_rgb = letterbox_image(raw_rgb, (192, 192))
        mask = letterbox_image(mask, (192, 192), mask=True)
        mask = torch.from_numpy(mask).float()
        mask = torch.unsqueeze(mask, 0)

        if self.visualize:
            rgb = raw_rgb.copy()
        else:
            rgb = raw_rgb
        cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        if self.mode == 'train':
            # 色域扭曲
            rgb = self.colorjitter(Image.fromarray(rgb))
            rgb = np.array(rgb)
        rgb = self.transform(rgb)

        if self.visualize:
            return raw_rgb, rgb, mask
        return rgb, mask
