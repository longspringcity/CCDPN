import os
import torch
import random
import argparse
import numpy as np
# import tensorflow.compat.v1 as tf

from torch.utils.data import DataLoader, sampler
from reg_model import RegTransNet
from lib.reg_loss import RegLoss
from lib.utils import setup_logger
from reg_dataset import PoseDataset
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--num_workers', type=int, default=4, help='number of data loading workers')
parser.add_argument('--result_dir', type=str, default='results/real', help='directory to save train results')
parser.add_argument('--train_prop', type=int, default=0.9, help='Proportion of training data to total data')
parser.add_argument('--evaluate_train', type=bool, default=True, help='Is evaluate on trainset to check overfitting')
parser.add_argument('--max_epoch', type=int, default=50, help='Max number of epochs to train')
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
    model = RegTransNet()
    model.cuda()
    criterion = RegLoss().cuda()
    # 准备数据
    train_dataset = PoseDataset('Real', 'train', '../data_crop', 192, 5)
    test_dataset = PoseDataset('Real', 'test', '../data_crop', 192, 5)
    train_dataset_size = train_dataset.length
    test_dataset_size = test_dataset.length
    train_idx = list(range(train_dataset_size))
    test_idx = list(range(test_dataset_size))
    # 训练数据和验证数据的大小
    train_sampler = sampler.RandomSampler(train_idx)
    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, sampler=train_sampler,
                                  num_workers=opt.num_workers, pin_memory=True)
    num_batch = np.ceil(train_dataset_size / opt.batch_size).astype(int)
    global_step = 0
    # 定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    for epoch in range(opt.max_epoch):
        # 训练
        model.train()
        running_loss = 0
        for i, data in tqdm(enumerate(train_dataloader), total=num_batch,
                            desc='Training... Epoch: {:02d}, progress'.format(epoch + 1)):
            rgb, real_pos = data
            rgb = rgb.cuda()
            real_pos = real_pos.cuda()
            pred_pos = model(rgb)
            loss = criterion(pred_pos, real_pos)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1
            running_loss += loss.item()

            # write results to tensorboard
            # summary = tf.Summary(value=[tf.Summary.Value(tag='train_loss', simple_value=loss)])
            # tb_writer.add_summary(summary, global_step)
        running_loss /= num_batch
        logger.info('>>>>----Epoch {:02d} train finished! running loss: {:.2f}---<<<<'.format(epoch + 1, running_loss))

        if opt.evaluate_train:
            # 测试训练集
            model.eval()
            train_success_5, train_success_10, train_success_20 = 0, 0, 0
            random.shuffle(train_idx)
            train_val_sampler = sampler.RandomSampler(train_idx[:opt.num_val])  # 用于测试训练集的采样器
            train_val_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, sampler=train_val_sampler,
                                              num_workers=opt.num_workers, pin_memory=True)
            for i, data in enumerate(train_val_dataloader):
                rgb, real_pos = data
                rgb = rgb.cuda()
                real_pos = real_pos.cuda()
                pred_pos = model(rgb)
                error_distances = torch.norm(pred_pos - real_pos, 2, dim=1)
                train_success_5 += torch.sum((error_distances <= 5)).item()
                train_success_10 += torch.sum((error_distances <= 10)).item()
                train_success_20 += torch.sum((error_distances <= 20)).item()
            train_acc_5 = 100 * (train_success_5 / opt.num_val)
            train_acc_10 = 100 * (train_success_10 / opt.num_val)
            train_acc_20 = 100 * (train_success_20 / opt.num_val)
            # write results to tensorboard
            # summary = tf.Summary(value=[tf.Summary.Value(tag='train_acc_5', simple_value=train_acc_5),
            #                             tf.Summary.Value(tag='train_acc_10', simple_value=train_acc_10),
            #                             tf.Summary.Value(tag='train_acc_20', simple_value=train_acc_20)])
            # tb_writer.add_summary(summary, global_step)
            logger.info(
                'train accuracies: 5px -- {:.2f}, 10px -- {:.2f}, 20px -- {:.2f}'.format(train_acc_5, train_acc_10,
                                                                                         train_acc_20))
        # 测试验证集
        model.eval()
        test_success_5, test_success_10, test_success_20 = 0, 0, 0
        random.shuffle(test_idx)
        test_sampler = sampler.RandomSampler(test_idx[:opt.num_val])
        val_dataloader = DataLoader(test_dataset, batch_size=opt.batch_size, sampler=test_sampler,
                                    num_workers=opt.num_workers, pin_memory=True)
        for i, data in enumerate(val_dataloader):
            rgb, real_pos = data
            rgb = rgb.cuda()
            real_pos = real_pos.cuda()
            pred_pos = model(rgb)
            error_distances = torch.norm(pred_pos - real_pos, 2, dim=1)
            test_success_5 += torch.sum((error_distances <= 5)).item()
            test_success_10 += torch.sum((error_distances <= 10)).item()
            test_success_20 += torch.sum((error_distances <= 20)).item()
        val_acc_5 = 100 * (test_success_5 / opt.num_val)
        val_acc_10 = 100 * (test_success_10 / opt.num_val)
        val_acc_20 = 100 * (test_success_20 / opt.num_val)
        # summary = tf.Summary(value=[tf.Summary.Value(tag='test_acc_5', simple_value=val_acc_5),
        #                             tf.Summary.Value(tag='test_acc_10', simple_value=val_acc_10),
        #                             tf.Summary.Value(tag='test_acc_20', simple_value=val_acc_20)])
        # tb_writer.add_summary(summary, global_step)
        logger.info('val accuracies: 5px -- {:.2f}, 10px -- {:.2f}, 20px -- {:.2f}'.format(val_acc_5, val_acc_10,
                                                                                           val_acc_20))


if __name__ == '__main__':
    train()
