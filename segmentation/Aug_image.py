import os
import shutil
from PIL import Image, ImageOps

def augment(file, path, aug_path, save, aug_save):
    for i in range(2):
        im = Image.open(os.path.join(path, file))
        aug_im = Image.open(os.path.join(aug_path, file[:-4] + '.png'))
        if i == 1:
            im = ImageOps.mirror(im)
            aug_im = ImageOps.mirror(aug_im)
        save_p = os.path.join(save, file)
        aug_save_p = os.path.join(aug_save, file)
        im.save(save_p[:-4] + '_' + str(i) + '.png', quality=100)
        aug_im.save(aug_save_p[:-4] + '_' + str(i) + '.png', quality=100)

def main():
    if not os.path.exists('./Aug_train'):
        os.mkdir('./Aug_train')
    if os.path.exists('./Aug_train/img'):
        shutil.rmtree('./Aug_train/img')
    os.mkdir('./Aug_train/img')
    if os.path.exists('./Aug_train/mask'):
        shutil.rmtree('./Aug_train/mask')
    os.mkdir('./Aug_train/mask')

    path = './train/img'
    aug_path = './train/mask'
    save = './Aug_train/img'
    aug_save = './Aug_train/mask'
    target = os.listdir(path)
    for file in target:
        augment(file, path, aug_path, save, aug_save)

if __name__ == "__main__":
    main()
