import _pickle as cPickle
import os
import numpy as np
import cv2
import argparse

from tqdm import tqdm
from lib.utils import translation_to_uv

parser = argparse.ArgumentParser()
parser.add_argument('--class_id', type=int, default=5, help='bottle: 1, bowl: 2, camera: 3, can: 4, laptop: 5, mug: 6')
parser.add_argument('--mode', default='test', help='train/test')
parser.add_argument('--visualize', type=bool, default=False, help='Is circle center point')
opt = parser.parse_args()

data_dir = '/media/zhangtong/DATA/data'
result_dir = './data_crop/Real/{:s}'.format(opt.mode)
file_path = 'Real/{:s}_list.txt'.format(opt.mode)
intrinsics = np.array([[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]])
class_names = {1: 'bottle', 2: 'bowl', 3: 'camera', 4: 'can', 5: 'laptop', 6: 'mug'}
instance_count = np.zeros(len(class_names) + 1, dtype=int)

img_list = [os.path.join(file_path.split('/')[0], line.rstrip('\n'))
            for line in open(os.path.join(data_dir, file_path))]
center_point = list()

for path in tqdm(img_list, smoothing=0.9):
    img_path = os.path.join(data_dir, path)
    raw_rgb = cv2.imread(img_path + '_color.png')  # (480, 640, 3)
    raw_mask = cv2.imread(img_path + '_mask.png')[:, :, 2]
    coord = cv2.imread(img_path + '_coord.png')[:, :, :3]

    with open(img_path + '_label.pkl', 'rb') as f:
        gts = cPickle.load(f)

    num_classes = len(gts['class_ids'])

    for i in range(num_classes):
        rmin, cmin, rmax, cmax = gts['bboxes'][i]
        crop_rgb = raw_rgb[rmin:rmax, cmin:cmax]
        crop_mask = raw_mask[rmin:rmax, cmin:cmax]

        if opt.mode == 'train':
            translation = gts['translations'][i]
        elif opt.mode == 'test':
            translation = gts['poses'][i, :3, 3]
        u_2d, v_2d = translation_to_uv(translation, intrinsics)
        shift_u, shift_v = u_2d - cmin, v_2d - rmin
        if opt.visualize:
            cv2.circle(crop_rgb, (shift_u, shift_v), 3, (255, 255, 0), cv2.FILLED)

        class_id = gts['class_ids'][i]
        instance_id = gts['instance_ids'][i]
        class_name = class_names[class_id]

        # 写入文件
        result_path = os.path.join(result_dir, class_name)
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        center_points_file = os.path.join(result_path, 'center_points.txt')
        with open(center_points_file, 'a+') as f:
            f.write('{:d} {:d} {:d}\n'.format(shift_u, shift_v, instance_id))

        # 保存文件
        rgb_img_name = '{:06d}_rgb.png'.format(instance_count[class_id])
        rgb_mask_name = '{:06d}_mask.png'.format(instance_count[class_id])
        cv2.imwrite(os.path.join(result_path, rgb_img_name), crop_rgb)
        cv2.imwrite(os.path.join(result_path, rgb_mask_name), crop_mask)
        instance_count[class_id] += 1
print('Success!')
for i in range(1, len(class_names) + 1):
    print(class_names[i], ': ', instance_count[i], end=' ')
