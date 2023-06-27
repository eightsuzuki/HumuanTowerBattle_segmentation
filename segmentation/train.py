import sys
import os
import glob
import time
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import OrderedDict

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, models, transforms

from option import draw_process
from metrics import dice_coef, iou_score
import losses

from unet import UNet
from utils import split_train_val, get_imgs_and_masks

arch_names = list(UNet.__dict__.keys())
loss_names = list(losses.__dict__.keys())
loss_names.append('BCEWithLogitsLoss')

# 学習のオプションを設定
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default=None, help='model name: (default: arch+timestamp)')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='UNet', choices=arch_names, help='model architecture: ' + ' | '.join(arch_names) + ' (default: UNet)')
    parser.add_argument('--dataset', default=None, help='dataset name')
    parser.add_argument('--epochs', dest='epochs', default=500, type=int, help='number of epochs')
    parser.add_argument('--batch-size', default=16, type=int, help='batch size')
    parser.add_argument('--loss', default='BCEDiceLoss', choices=loss_names, help='loss: '+' | '.join(loss_names) +' (default: BCEDiceLoss)')
    parser.add_argument('--optimizer', default='Adam',choices=['Adam', 'SGD'],help='loss: ' +' | '.join(['Adam', 'SGD']) +' (default: Adam)')
    parser.add_argument('--lr', '--learning-rate', default=5e-5, type=float, help='initial learning rate')
    parser.add_argument('--val-percent', default=0.1, type=float, help='rate of validation')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight-decay', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--draw-process', default=True, help='draw figre of loss and accuracy')
    parser.add_argument('--early-stop', default=20, type=int, metavar='N', help='early stopping (default: 20)')
    parser.add_argument('--gpu', action='store_true', dest='gpu', default=True, help='use cuda')
    parser.add_argument('--load', action='store_true', dest='load', default=False, help='load file model')
    parser.add_argument('--height', dest='height', type=int, default=480, help='height of the images')
    parser.add_argument('--width', dest='width', type=int, default=640, help='width of the images')

    args = parser.parse_args()
    return args

class AverageMeter(object):
    # 平均値を算出
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train(args, train_loader, net, criterion, optimizer, epoch):
    losses = AverageMeter()
    ious = AverageMeter()

    net.train()

    for batch_num, (img, mask) in tqdm(enumerate(train_loader), total=len(train_loader)):
        img = img.cuda()
        mask = mask.cuda()

        pred = net(img)
        loss = criterion(pred, mask)
        iou = iou_score(pred, mask)

        losses.update(loss.item(), img.size(0))
        ious.update(iou, img.size(0))

        # 勾配の初期化
        optimizer.zero_grad()
        # 勾配の計算
        loss.backward()
        # 勾配を元にパラメータの更新
        optimizer.step()

        print('--- Loss:{:.4f} IoU:{:.4f} ---'.format(loss, iou))

    log = OrderedDict([('loss', losses.avg), ('iou', ious.avg)])
    return log

def val(args, val_loader, net, criterion):
    losses = AverageMeter()
    ious = AverageMeter()

    net.eval()

    with torch.no_grad():
        for batch_num, (img, mask) in tqdm(enumerate(val_loader), total=len(val_loader)):
            img = img.cuda()
            mask = mask.cuda()

            pred = net(img)
            loss = criterion(pred, mask)
            iou = iou_score(pred, mask)

            losses.update(loss.item(), img.size(0))
            ious.update(iou, img.size(0))

    log = OrderedDict([('loss', losses.avg), ('iou', ious.avg)])
    return log

def main():
    # 学習のオプションを読み込み
    args = parse_args()

    if args.name is None:
        args.name = '{}_{}'.format(args.dataset, args.arch)

    if not os.path.exists('models/{}'.format(args.name)):
        os.makedirs('models/{}'.format(args.name))

    # 訓練画像ファイル
    dir_img = './Aug_train/img/'
    dir_mask = './Aug_train/mask/'

    # ネットワークの読み込み
    net = UNet(n_channels=3, n_classes=1)

    net = net.cuda()

    # モデルの読み込み
    if args.load:
        net.load_state_dict(torch.load(args.load))
        print('Model loaded from {}'.format(args.load))

    # # ロスの設定
    # if args.loss == 'BCEWithLogitsLoss':
    #     criterion = nn.BCEWithLogitsLoss().cuda()
    # else:
    #     criterion = losses.__dict__[args.loss]().cuda()
    criterion = nn.BCELoss()

    # 勾配法の設定
    if args.optimizer == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr=args.lr)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(
            net.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay)

    cudnn.benchmark = True

    img_path = os.listdir(dir_img)
    mask_path = os.listdir(dir_mask)

    id_dataset = split_train_val(img_path, args.val_percent)

    N_train = len(id_dataset['train'])
    N_val = len(id_dataset['val'])

    train_img, train_mask = get_imgs_and_masks(id_dataset['train'], dir_img, dir_mask, args.height, args.width)
    val_img, val_mask = get_imgs_and_masks(id_dataset['val'], dir_img, dir_mask, args.height, args.width)

    train_data = torch.utils.data.TensorDataset(torch.tensor(train_img), torch.tensor(train_mask))
    val_data = torch.utils.data.TensorDataset(torch.tensor(val_img), torch.tensor(val_mask))

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=False)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, drop_last=False)

    learning_log = pd.DataFrame(index=[], columns=['epoch', 'lr', 'loss', 'iou', 'val_loss', 'val_iou'])

    best_iou = 0
    trigger = 0

    print('訓練開始:エポック:{} バッチサイズ:{} 学習率:{} 学習データ数:{} 検証データ数:{}'.format(args.epochs, args.batch_size, args.lr, N_train, N_val))
    print('CUDA:{}'.format(str(args.gpu)))

    # エポック
    for epoch in range(args.epochs):
        since = time.time()
        print('='*50)
        print('Strating epoch {}/{}.'.format(epoch+1, args.epochs))
        print('='*50)

        # 1エポック内での学習
        train_log = train(args, train_loader, net, criterion, optimizer, epoch)
        # 1エポック内での評価
        val_log = val(args, val_loader, net, criterion)

        tmp = pd.Series([
            epoch,
            args.lr,
            train_log['loss'],
            train_log['iou'],
            val_log['loss'],
            val_log['iou']
            ], index=['epoch', 'lr', 'loss', 'iou', 'val_loss', 'val_iou'])

        # ログの保存
        learning_log = learning_log.append(tmp, ignore_index=True)
        learning_log.to_csv('models/{}/log.csv'.format(args.name, index=False))

        if args.draw_process:
            draw_process(learning_log,'models/{}/'.format(args.name))

        trigger +=1

        print('='*50)
        print('Finished epoch {}/{}.'.format(epoch+1, args.epochs))
        print('--- loss:{:.4f}  iou:{:4f} ---'.format(train_log['loss'], train_log['iou']))
        print('--- val_loss:{:4f}  val_iou:{:4f} ---'.format(val_log['loss'], val_log['iou']))

        # 最良モデルの保存
        if val_log['iou'] > best_iou:
            torch.save(net.state_dict(), 'models/{}/model.pth'.format(args.name))
            best_iou = val_log['iou']
            print('BestModel saved!')
            trigger = 0

        time_elapsed = time.time()-since
        print('エポック所要時間:　{:.0f}m {:.0f}s'.format(time_elapsed//60, time_elapsed%60))
        print('='*50)

        torch.cuda.empty_cache()

        # early stopping
        if not args.early_stop is None:
            if trigger >= args.early_stop:
                print('Early stopping!')
                break

    print('Training finished!')

if __name__ == '__main__':
    main()
