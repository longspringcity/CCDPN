import os
import torch
# import random
import argparse
import numpy as np
# import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from net.Unet import Unet
from lib.seg_loss import SegLoss
from lib.utils import setup_logger
from seg_dataset import SegDataset
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=1e-5, help='initial learning rate')
parser.add_argument('--batch_size', type=int, default=6, help='batch size')
parser.add_argument('--num_workers', type=int, default=4, help='number of data loading workers')
parser.add_argument('--result_dir', type=str, default='results/real', help='directory to save train results')
parser.add_argument('--train_prop', type=int, default=0.9, help='Proportion of training data to total data')
parser.add_argument('--evaluate_train', type=bool, default=True, help='Is evaluate on trainset to check overfitting')
parser.add_argument('--max_epoch', type=int, default=100, help='Max number of epochs to train')
parser.add_argument('--num_val', type=int, default=3000, help='The number of validation in the test set')
opt = parser.parse_args()


def train():
    if not os.path.exists(opt.result_dir):
        os.makedirs(opt.result_dir)
    # tb_writer = tf.summary.FileWriter(opt.result_dir)
    logger = setup_logger('train_log', os.path.join(opt.result_dir, 'log.txt'))
    for key, value in vars(opt).items():
        logger.info(key + ': ' + str(value))
    # 定义model&loss
    model = Unet()
    model.cuda()
    criterion = SegLoss().cuda()
    # 准备数据
    train_dataset = SegDataset('Real', 'train', '../data_crop', 192, 5)
    # test_dataset = PoseDataset('Real', 'test', '../data_crop', 192, 5)
    train_dataset_size = train_dataset.length
    # test_dataset_size = test_dataset.length
    # test_idx = list(range(test_dataset_size))
    # 训练数据和验证数据的大小
    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True,
                                  num_workers=opt.num_workers, pin_memory=True)
    num_batch = np.ceil(train_dataset_size / opt.batch_size).astype(int)
    global_step = 0
    # 定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    for epoch in range(opt.max_epoch):
        # 训练
        model.train()
        running_bce_loss = 0
        running_iou = 0
        running_loss = 0
        for i, data in tqdm(enumerate(train_dataloader), total=num_batch,
                            desc='Training... Epoch: {:02d}, progress'.format(epoch + 1)):
            rgb, real_mask = data
            rgb = rgb.cuda()
            real_mask = real_mask.cuda()
            pred_mask = model(rgb)
            bce_loss, iou, loss = criterion(pred_mask, real_mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1
            running_loss += loss.item()
            running_bce_loss += bce_loss.item()
            running_iou += iou.item()

            # write results to tensorboard
            # summary = tf.Summary(value=[tf.Summary.Value(tag='train_loss', simple_value=loss)])
            # tb_writer.add_summary(summary, global_step)

            # if i == 0:
            #     array_real_mask = real_mask.detach().cpu().numpy()[0]
            #     array_real_mask = np.transpose(array_real_mask, (1, 2, 0))
            #     plt.title('real_mask')
            #     plt.imshow(array_real_mask)
            #     plt.show()
            #     array_pred_mask = pred_mask.detach().cpu().numpy()[0]
            #     array_pred_mask = np.transpose(array_pred_mask, (1, 2, 0))
            #     plt.title('pred_mask')
            #     plt.imshow(array_pred_mask)
            #     plt.show()
        running_loss /= num_batch
        logger.info(
            'Epoch {:02d} train finished! running loss: {:.5f} bce_loss: {:.5f}, iou: {:.5f}'.format(epoch + 1,
                                                                                                     running_loss,
                                                                                                     running_bce_loss,
                                                                                                     running_iou))
        torch.save(model.state_dict(), 'SegModel_dict{:02d}.pth'.format(epoch))


if __name__ == '__main__':
    train()
