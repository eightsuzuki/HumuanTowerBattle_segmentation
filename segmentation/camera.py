# coding:utf-8
import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image
import cv2
import pyautogui

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
def predict_img(net, gpu, height, width):
    net.eval()
    i = 0

    # カメラのキャプチャを開始 --- (*1)
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

    while True:
        # 画像を取得 --- (*2)
        _, img = cam.read()

        # ウィンドウに画像を表示 --- (*3)
        img = cv2.rectangle(img,(310,-10),(970,970),(255,0,0),5)
        cv2.imshow('PUSH SPACE KEY', img)

        # SPACEキーが押されたら画像を保存する
        if cv2.waitKey(1) == 32:
            img_path = './image.jpg'
            img = img[0:960 , 320:960]
            i += 1
            cv2.imwrite(img_path, img)
            print('Image captured!')

            # 画像の読み込み
            org_img = Image.open(img_path)

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
                predict = cv2.bitwise_not(predict)
                ret, markers = cv2.connectedComponents(np.uint8(predict), connectivity = 4)
                r_channel, g_channel, b_channel = cv2.split(load_img)
                alpha_channel = np.zeros(b_channel.shape, dtype=b_channel.dtype)
                alpha_channel[markers != 1] = [255]
                result = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
                result = cv2.resize(result.astype(np.float32) , (org_img.size[0]//4, org_img.size[1]//4))
                cv2.imwrite('../../tower_buttle/Assets/Resources/result{0:03d}.png'.format(i), result)
                print('result{0:03d}.png saved!'.format(i))
                time.sleep(2)
                pyautogui.hotkey('command', 'tab')
                time.sleep(4)
                pyautogui.hotkey('command', 'tab')

        # Enterキーが押されたら終了する
        if cv2.waitKey(1) == 13:
            break

    # 後始末
    cam.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    args = get_args()

    # アーキテクチャの読み込み
    net = UNet(n_channels=3, n_classes=1)

    # 学習したモデルの読み込み
    print('Loading model {}'.format(args.model))
    net.load_state_dict(torch.load(args.model, map_location=lambda storage, loc: storage))
    print("Model loaded!\n")

    # CUDAがあるなら使用
    # if args.gpu:
    #     print('Using CUDA!')
    #     net.cuda()

    predict = predict_img(net=net, gpu=args.gpu, width=args.in_width, height=args.in_height)
