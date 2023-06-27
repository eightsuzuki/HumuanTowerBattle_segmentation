import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image

from unet import UNet
from utils import resize_imgs, normalize, denormalize, hwc_to_chw

# 各種変数の設定
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='MODEL.pth', metavar='FILE', help='Specify the file in which is stored the model' "(default: './MODEL.pth')")
    parser.add_argument('--gpu', action='store_true', dest='gpu', default=True, help='use cuda or not')
    parser.add_argument('--height', dest='in_height', type=int, default=256, help='height of the images')
    parser.add_argument('--width', dest='in_width', type=int, default=256, help='width of the images')
    return parser.parse_args()

#　検出を行う
def predict_img(net, target, dir, save, gpu, height, width, num):
    net.eval()

    # 画像の読み込み
    org_img = Image.open(dir + target)

    # 画像の処理
    load_img = org_img.resize((width, height))
    load_img = np.array(load_img, dtype=np.float32)
    img = normalize(load_img)
    img = img[np.newaxis, :, :, :]
    img = list(map(hwc_to_chw, img))
    img = np.array(img, dtype=np.float32)
    img = torch.from_numpy(img)

    # if gpu:
    #     img = img.cuda()

    # 検出操作
    with torch.no_grad():
        predict = net(img).numpy()
        predict = denormalize(predict)
        predict = predict.reshape(height, width)
        predict = (predict > 128) * 255
        predict = Image.fromarray(np.uint8(predict))
        predict = predict.resize((org_img.size[0], org_img.size[1]))
        predict.save( save + target[:-4] + '.jpg', quality=100)

if __name__ == '__main__':
    args = get_args()

    # 検出したい画像のフォルダ
    target_files = './test/img/'

    # 検出した画像の保存先フォルダ
    output_files = './test/result/'

    # アーキテクチャの読み込み
    net = UNet(n_channels=3, n_classes=1)

    # 学習したモデルの読み込み
    print('Loading model {}'.format(args.model))
    net.load_state_dict(torch.load(args.model))
    print("Model loaded!\n")

    # CUDAがあるなら使用
    # if args.gpu:
    #     print('Using CUDA!')
    #     net.cuda()

    for i, target in enumerate(os.listdir(target_files)):
        print('Predicting image {}'.format(target))

        predict = predict_img(net=net, target=target, dir=target_files, save=output_files, gpu=args.gpu, width=args.in_width, height=args.in_height, num=i)
