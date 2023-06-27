import sys
import time

import cv2
import numpy as np
from skimage import transform

import ailia

sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from detector_utils import load_image  # noqa: E402C
import webcamera_utils  # noqa: E402

# logger
from logging import getLogger  # noqa: E402

logger = getLogger(__name__)

# Import the Listener class from pynput.keyboard
from pynput.keyboard import Listener, Key

# ======================
# Parameters
# ======================

WEIGHT_PATH = 'u2net-human-seg.onnx'
MODEL_PATH = 'u2net-human-seg.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/u2net-human-seg/'

IMAGE_PATH = 'input.jpg'
SAVE_IMAGE_PATH = 'output.png'
IMAGE_SIZE = 320

# ======================
# Argument Parser Config
# ======================

parser = get_base_parser(
    'U^2-Net - human segmentation',
    IMAGE_PATH,
    SAVE_IMAGE_PATH,
)
parser.add_argument(
    '-c', '--composite',
    action='store_true',
    help='Composite input image and predicted alpha value'
)
parser.add_argument(
    '--capture',
    action='store_true',
    help='Capture an image from webcam and recognize'
)
args = update_parser(parser)


# ======================
# Utils
# ======================

def preprocess(img):
    img = transform.resize(img, (IMAGE_SIZE, IMAGE_SIZE), mode='constant')

    img = img / np.max(img)
    img[:, :, 0] = (img[:, :, 0] - 0.485) / 0.229
    img[:, :, 1] = (img[:, :, 1] - 0.456) / 0.224
    img[:, :, 2] = (img[:, :, 2] - 0.406) / 0.225
    img = img.astype(np.float32)

    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, axis=0)

    return img


# ======================
# Main functions
# ======================

def human_seg(net, img):
    h, w = img.shape[:2]

    # initial preprocesses
    img = preprocess(img)

    # feedforward
    output = net.predict([img])
    d1, d2, d3, d4, d5, d6, d7 = output
    pred = d1[:, 0, :, :]

    # post processes
    ma = np.max(pred)
    mi = np.min(pred)
    pred = (pred - mi) / (ma - mi)

    pred = pred.transpose(1, 2, 0)  # CHW -> HWC

    pred = cv2.resize(pred, (w, h), cv2.INTER_LINEAR)

    return pred

def recognize_from_image(net):
    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.info(image_path)

        img = cv2.imread(image_path)

        # inference
        logger.info('Start inference...')

        pred = human_seg(net, img)

        # Create mask for foreground
        mask = (pred > 0.5).astype(np.uint8)

        # Create transparent background image
        background = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)
        background[:, :, 3] = 255  # Set alpha channel to fully opaque

        # Create foreground image with transparency
        img_with_alpha = np.dstack((img, mask * 255))  # Set alpha channel based on the mask

        # Combine foreground and transparent background
        img_transparent = cv2.bitwise_and(img_with_alpha, background)

        # Save results with transparency
        savepath = get_savepath(args.savepath, image_path, ext='.png')
        logger.info(f'saved at: {savepath}')
        cv2.imwrite(savepath, img_transparent)

        # Save cropped input image
        input_savepath = get_savepath(args.savepath, image_path, prefix='input_', ext='.png')
        cv2.imwrite(input_savepath, img_with_alpha)

    logger.info('Script finished successfully.')


def capture_and_recognize(net):
    capture = cv2.VideoCapture(0)

    image_counter = 0
    enter_pressed = False

    def on_press(key):
        nonlocal enter_pressed
        nonlocal image_counter

        if key == Key.space:
            # Capture photo
            image_path = f'input.jpg'
            cv2.imwrite(image_path, frame)
            recognize_from_image(net)
        elif key == Key.enter:
            enter_pressed = True

    # Register keyboard event handler
    with Listener(on_press=on_press) as listener:
        while not enter_pressed:
            ret, frame = capture.read()
            if not ret:
                break

            cv2.imshow('Webcam', frame)

            if cv2.waitKey(1) == ord('q'):
                break

    capture.release()
    cv2.destroyAllWindows()


def main():
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    # initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id)


    # image mode
    if args.capture:
        # Capture and recognize from webcam
        capture_and_recognize(net)
    else:
        # Recognize from input image(s)
        recognize_from_image(net)


if __name__ == '__main__':
    main()
