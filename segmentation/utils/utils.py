import random
import numpy as np

def split_train_val(dataset, val_percent=0.05):
    dataset = list(dataset)
    length = len(dataset)
    n = int(length * val_percent)
    return {'train': dataset[:-n], 'val': dataset[-n:]}

def hwc_to_chw(img):
    return np.transpose(img, [2, 0, 1])

def resize_img(pilimg, height, width):
    img = pilimg.resize((height, width))
    img = np.array(img, dtype=np.float32)
    return img

def resize_img_gray(pilimg, height, width):
    img = pilimg.resize((height, width))
    img = np.array(img, dtype=np.float32)
    img = img[:, :, np.newaxis]
    return img

def normalize(x):
    return x / 255

def denormalize(x):
    return x * 255
