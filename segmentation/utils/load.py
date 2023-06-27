import os
import numpy as np
from PIL import Image

from .utils import resize_img, resize_img_gray, normalize, hwc_to_chw

def resize_imgs(ids, dir, height, width):
    '''リストのタプルを参考に、適切なリサイズ画像を返す'''
    for id in ids:
        if not ('._' in id):
            load_img = Image.open(dir + id)
            im = resize_img(load_img, height=height, width=width)
            yield im

def resize_imgs_gray(ids, dir, height, width):
    '''リストのタプルを参考に、適切なリサイズ画像を返す（グレイスケール）'''
    for id in ids:
        if not ('._' in id):
            load_img = Image.open(dir + id)
            load_img = load_img.convert('L')
            im = resize_img_gray(load_img, height=height, width=width)
            yield im

def get_imgs_and_masks(ids, dir_img, dir_mask, height, width):
    '''画像とそのマスクを返す'''
    imgs = resize_imgs(ids, dir_img, height, width)

    # 学習用のため（H, W, C）-> (C, H, W)に変換
    imgs_switched = map(hwc_to_chw, imgs)
    imgs_normalized = map(normalize, imgs_switched)

    masks = resize_imgs_gray(ids, dir_mask, height, width)
    masks_switched = map(hwc_to_chw, masks)
    masks_normalized = map(normalize, masks_switched)

    return np.array(list(imgs_normalized)), np.array(list(masks_normalized))
