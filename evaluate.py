import argparse
import torch
import cv2

from reg_dataset import PoseDataset

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Real', help='CAMERA or CAMERA+Real')
parser.add_argument('--img_size', type=int, default=192, help='cropped image size')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--num_workers', type=int, default=10, help='number of data loading workers')
opt = parser.parse_args()

# data_dir = '../0020_Object-deformnet_ECCV2020/object-deformnet-master/data'

# dataset
test_dataset = PoseDataset(opt.dataset, 'test', './data_crop_test', opt.img_size, 5)
train_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1)

for epoch in range(1):
    for batch in train_dataloader:
        image = batch[0][0].numpy()
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        center_point = batch[1][0].numpy().astype(int)
        cv2.circle(image, (center_point[1], center_point[2]), 3, color=(255, 255, 0), thickness=cv2.FILLED)
        cv2.imshow('image', image)
        cv2.waitKey(0)
