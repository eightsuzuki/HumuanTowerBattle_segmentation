import time

import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation

# For static images:
# IMAGE_FILES = []
# BG_COLOR = (192, 192, 192)  # gray
# MASK_COLOR = (255, 255, 255)  # white
# with mp_selfie_segmentation.SelfieSegmentation(model_selection=0) as selfie_segmentation:
#     for idx, file in enumerate(IMAGE_FILES):
#         image = cv2.imread(file)
#         image_height, image_width, _ = image.shape
#         # Convert the BGR image to RGB before processing.
#         resu lts = selfie_segmentation.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

#         # Draw selfie segmentation on the background image.
#         # To improve segmentation around boundaries, consider applying a joint
#         # bilateral filter to "results.segmentation_mask" with "image".
#         condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
#         # Generate solid color images for showing the output selfie segmentation mask.
#         fg_image = np.zeros(image.shape, dtype=np.uint8)
#         fg_image[:] = MASK_COLOR
#         bg_image = np.zeros(image.shape, dtype=np.uint8)
#         bg_image[:] = BG_COLOR
#         output_image = np.where(condition, fg_image, bg_image)
#         cv2.imwrite("../Documents/大学関係/unilab/unilab-tower-battle/Assets/Resources" + str(idx) + ".png", output_image)

# For webcam input:
BG_COLOR = (255, 255, 255)  # white
cap = cv2.VideoCapture(0)
with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_segmentation:
    bg_image = None
    idx = 1
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, dsize=(600, 336))
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = selfie_segmentation.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw selfie segmentation on the background image.
        # To improve segmentation around boundaries, consider applying a joint
        # bilateral filter to "results.segmentation_mask" with "image".
        condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
        # The background can be customized.
        #   a) Load an image (with the same width and height of the input image) to
        #      be the background, e.g., bg_image = cv2.imread('/path/to/image/file')
        #   b) Blur the input image by applying image filtering, e.g.,
        #      bg_image = cv2.GaussianBlur(image,(55,55),0)
        if bg_image is None:
            bg_image = np.zeros(image.shape, dtype=np.uint8)
            bg_image[:] = BG_COLOR
        output_image = np.where(condition, image, bg_image)

        output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2BGRA)
        output_image[:, :, 3] = np.where(np.all(output_image == 255, axis=-1), 0, 255)

        cv2.imshow("Human Segmentation", output_image)
        
        img_path = "/Users/suzuki8/murata/ユニラボ/tower_battle/unity/Assets/Resources/"
        
        # Spaceキーが押されたら撮影
        if cv2.waitKey(1) == 32:
            cv2.imwrite(
                img_path + str(idx) + ".png", output_image
            )
            print("took a picture")
            idx += 1
            time.sleep(1)

        # Enterキーが押されたら終了
        if cv2.waitKey(1) == 13:
            break

cap.release()
