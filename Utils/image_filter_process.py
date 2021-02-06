import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image


class Main:
    def __init__(self):
        self.cwd = os.getcwd()
        self.prod_sku_root_folder = 'Product SKU'
        self.prod_sku_folders = ['HEAD & SHOULDERS SHAMPOO (ASSORTED) COOL MENTHOL', 'HUGGIES ULTRA SUPER JUMBO M',
                                'HUP SENG CREAM CRACKERS 428G', 'KLEENEX ULTRA SOFT BATH ISSUE MEGA',
                                'NATURAL PURE OLIVE OIL 750 ML', 'SUNSLIK SHAMPOO (ASSORTED) LIVELY CLEAN & FRESH']
        self.finalized_prod_sku_folder = 'Finalized Images'
        self.image_path = f'{self.cwd}\\{self.prod_sku_root_folder}'

    def initiate(self, filter_mode, batch, test_input):
        switcher = {
            0: self.mean_filter,
            1: self.median_filter,
            2: self.bilateral_filter
        }

        func = switcher.get(filter_mode, lambda: 'Invalid filter mode selected')
        func(batch, test_input)

    def mean_filter(self, batch, test_input):
        mean_filter = '__import__("cv2").blur(img, (3,3))'
        filter_folder = 'mean filtered'
        if batch:
            self.filter_batch_images(mean_filter, filter_folder)
        else:
            self.filter_test_images(mean_filter, test_input)

    def median_filter(self, batch, test_input):
        median_filter = '__import__("cv2").medianBlur(img, 3)'
        filter_folder = 'median filtered'
        if batch:
            self.filter_batch_images(median_filter, filter_folder)
        else:
            self.filter_test_images(median_filter, test_input)

    def bilateral_filter(self, batch, test_input):
        bilat_filter = '__import__("cv2").bilateralFilter(img,9,75,75)'
        filter_folder = 'bilateral filtered'
        if batch:
            self.filter_batch_images(bilat_filter, filter_folder)
        else:
            self.filter_test_images(bilat_filter, test_input)

    def filter_batch_images(self, filter, filter_folder):
        train_val = ['train', 'val']
        finalized_prod_sku_folder = f'{self.image_path}\\{self.finalized_prod_sku_folder}'

        for folder in self.prod_sku_folders:
            for tv in train_val:
                for file in os.listdir(f'{finalized_prod_sku_folder}\\{folder}\\{filter_folder}\\{tv}'):
                    img = Image.open(f'{finalized_prod_sku_folder}\\{folder}\\{filter_folder}\\{tv}\\{file}')
                    img_arr = np.asarray(img)
                    # plt.figure()
                    # plt.imshow(img_arr)
                    # plt.show()
                    filtered_img_arr = eval(filter, {'img': img_arr})
                    # plt.figure()
                    # plt.imshow(filtered_img_arr)
                    # plt.show()

                    # delete original image
                    os.remove(f'{finalized_prod_sku_folder}\\{folder}\\{filter_folder}\\{tv}\\{file}')
                    # save new image
                    img = Image.fromarray(filtered_img_arr)
                    img.save(f'{finalized_prod_sku_folder}\\{folder}\\{filter_folder}\\{tv}\\{file}', optimze=True)

    def filter_test_images(self, filter, test_input):
        finalized_prod_sku_folder = f'{self.image_path}\\{self.finalized_prod_sku_folder}'
        img = cv2.imread(f'{finalized_prod_sku_folder}\\{test_input}')
        plt.figure()
        plt.imshow(img[..., ::-1])
        plt.show()
        img = eval(filter, {'img': img})
        plt.figure()
        plt.imshow(img[..., ::-1])
        plt.show()

# unit test cases
# m = Main()
# m.initiate(0, False, 'SUNSLIK SHAMPOO (ASSORTED) LIVELY CLEAN & FRESH\\unfiltered\\train\\A779.jpg')
# m.initiate(1, False, 'SUNSLIK SHAMPOO (ASSORTED) LIVELY CLEAN & FRESH\\unfiltered\\train\\A779.jpg')
# m.initiate(2, False, 'SUNSLIK SHAMPOO (ASSORTED) LIVELY CLEAN & FRESH\\unfiltered\\train\\A779.jpg')
# end