import cv2
import numpy as np
import random
import os
import matplotlib.pyplot as plt


class Main:
    def __init__(self):
        return

    def zoom(self, folder, image):
        # inter linear = zoom
        # inter area = shrinking
        # inter cubic = slow
        interpolation = [cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LINEAR]

        img = cv2.imread(f'{folder}\\{image}')
        rows, cols, channel = img.shape
        value = random.uniform(0.3, 1)
        h_taken = int(value * rows)
        w_taken = int(value * cols)
        h_start = random.randint(0, rows - h_taken)
        w_start = random.randint(0, cols - w_taken)
        img = img[h_start:h_start + h_taken, w_start:w_start + w_taken, :]
        res = cv2.resize(img, (rows, cols), random.choice(interpolation))
        # plt.imshow(res[..., ::-1])
        # open cv reversed the numpy array order, undo the reverse order to return correct color image
        return res[..., ::-1]

    def brightness(self, folder, image):
        img = cv2.imread(f'{folder}\\{image}')
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        # hsv stands for (hue, saturation, value, also known as HSB or hue, saturation, brightness)
        # to adjust brightness, manipulate v value
        # -ve to decrease brightness, +ve to increase brightness
        brightness_scale = random.randrange(-70, 70)
        v = cv2.add(v, brightness_scale)
        v[v > 255] = 255  # maximum rgb color is 255
        v[v < 0] = 0  # minimum rgb color is 0
        final_hsv = cv2.merge((h, s, v))
        res = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        # plt.imshow(res[..., ::-1])
        # open cv reversed the numpy array order, undo the reverse order to return correct color image
        return res[..., ::-1]

    def rotate_clockwise(self, folder, image):
        img = cv2.imread(f'{folder}\\{image}')
        rows, cols, channel = img.shape
        r = cv2.getRotationMatrix2D((cols/2, rows/2), 90, 1)
        res = cv2.warpAffine(img, r, (cols, rows))
        # plt.imshow(res[..., ::-1])
        # open cv reversed the numpy array order, undo the reverse order to return correct color image
        return res[..., ::-1]

    def translation(self, folder, image):
        img = cv2.imread(f'{folder}\\{image}')
        rows, cols, channel = img.shape
        r = np.float32([[1, 0, random.randrange(-200, 200)],  # randomly shift x coordinates
                        [0, 1, random.randrange(-200, 200)]]) # randomly shift y coordinates
        res = cv2.warpAffine(img, r, (cols, rows))
        # plt.imshow(res[..., ::-1])
        # open cv reversed the numpy array order, undo the reverse order to return correct color image
        return res[..., ::-1]

    def img_save(self, filename, img):
        cv2.imwrite(filename, img)

    def transform_list(self):
        return { 'zoom': self.zoom,
                'brightness': self.brightness,
                'rc': self.rotate_clockwise,
                'translate': self.translation}

# unit test case
# prod_sku_root_folder = 'Product SKU'
# prod_sku_folders = ['HEAD & SHOULDERS SHAMPOO (ASSORTED) COOL MENTHOL', 'HUGGIES ULTRA SUPER JUMBO M',
#                     'HUP SENG CREAM CRACKERS 428G', 'KLEENEX ULTRA SOFT BATH ISSUE MEGA',
#                     'NATURAL PURE OLIVE OIL 750 ML',
#                     'SUNSLIK SHAMPOO (ASSORTED) LIVELY CLEAN & FRESH']
# augmented_prod_sku_folder = 'Augmented Images'
# cwd = os.getcwd()
# image_path = f'{cwd}\\{prod_sku_root_folder}'
#
# x = Main()
# y = x.zoom(f'{image_path}\\{prod_sku_folders[1]}\\train', 'T45.jpg')
# plt.figure()
# plt.imshow(y)
# plt.show()
# y = x.translation(f'{image_path}\\{prod_sku_folders[0]}\\train', 'T3.jpg')
