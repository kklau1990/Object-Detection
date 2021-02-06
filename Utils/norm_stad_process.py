import matplotlib.pyplot as plt
import cv2
import numpy as np
import os


class Main:
    def __init__(self):
        self.cwd = os.getcwd()
        self.prod_sku_root_folder = 'Product SKU'
        self.prod_sku_folders = ['HEAD & SHOULDERS SHAMPOO (ASSORTED) COOL MENTHOL', 'HUGGIES ULTRA SUPER JUMBO M',
                                 'HUP SENG CREAM CRACKERS 428G', 'KLEENEX ULTRA SOFT BATH ISSUE MEGA',
                                 'NATURAL PURE OLIVE OIL 750 ML', 'SUNSLIK SHAMPOO (ASSORTED) LIVELY CLEAN & FRESH']
        self.finalized_prod_sku_folder = 'Finalized Images'
        self.image_path = f'{self.cwd}\\{self.prod_sku_root_folder}'
        self.train_val = ['train', 'val']
        self.resize_width = 800
        self.resized_height = 800

    # transform data to have zero mean and unit variance to standardize pixels separately in each channel
    def standardization(self):
        finalized_prod_sku_folder = f'{self.image_path}\\{self.finalized_prod_sku_folder}'
        for folder in self.prod_sku_folders:
            for tv in self.train_val:
                for file in os.listdir(f'{finalized_prod_sku_folder}\\{folder}\\unfiltered\\{tv}'):
                    img = plt.imread(f'{finalized_prod_sku_folder}\\{folder}\\unfiltered\\{tv}\\{file}')
                    img = np.asarray(img)
                    img = img.astype('float32')
                    img = self.std_formula(img)
                    # delete original image
                    os.remove(f'{finalized_prod_sku_folder}\\{folder}\\unfiltered\\{tv}\\{file}')
                    # save resize image
                    cv2.imwrite(f'{finalized_prod_sku_folder}\\{folder}\\unfiltered\\{tv}\\{file}', img)

    def standardization_plot(self, folder, filename):
        img = plt.imread(f'{self.image_path}\\{self.finalized_prod_sku_folder}\\{folder}\\{filename}')
        self.plot(img)
        img = np.asarray(img)
        img = img.astype('float32')

        # calculate per-channel means divide by standard deviation
        img = self.std_formula(img)
        self.plot(img)
    # end

    # normalize value between 0 and 1
    def normalization(self):
        finalized_prod_sku_folder = f'{self.image_path}\\{self.finalized_prod_sku_folder}'
        for folder in self.prod_sku_folders:
            for tv in self.train_val:
                for file in os.listdir(f'{finalized_prod_sku_folder}\\{folder}\\unfiltered\\{tv}'):
                    img = plt.imread(f'{finalized_prod_sku_folder}\\{folder}\\unfiltered\\{tv}\\{file}')
                    img = np.asarray(img)
                    img = img.astype('float32')

                    norm_img = np.zeros((self.resized_height, self.resized_height))
                    final_img = cv2.normalize(img, norm_img, 0, 255, cv2.NORM_MINMAX)
                    final_img2 = np.asarray(final_img)
                    final_img2 = final_img2.astype('float32')
                    final_img2 = self.feature_scaling(final_img2)

                    # delete original image
                    os.remove(f'{finalized_prod_sku_folder}\\{folder}\\unfiltered\\{tv}\\{file}')
                    # save resize image
                    cv2.imwrite(f'{finalized_prod_sku_folder}\\{folder}\\unfiltered\\{tv}\\{file}', final_img2)

    def normalization_plot(self, folder, filename):
        img = plt.imread(f'{self.image_path}\\{self.finalized_prod_sku_folder}\\{folder}\\{filename}')
        self.plot(img)

        # cv2 normalize output with higher contrast
        norm_img = np.zeros((self.resized_height, self.resized_height))
        final_img = cv2.normalize(img, norm_img, 0, 255, cv2.NORM_MINMAX)
        self.plot(final_img)

        # rescale data from 0 to 1 using original image
        final_img2 = np.asarray(img)
        final_img2 = final_img2.astype('float32')
        final_img2 = self.feature_scaling(final_img2)
        self.plot(final_img2)

        # rescale data from 0 to 1 using cv2 normalized image
        final_img3 = np.asarray(final_img)
        final_img3 = final_img3.astype('float32')
        final_img3 = self.feature_scaling(final_img3)
        self.plot(final_img3)
    # end

    # feature scaling function to normalize image data range from 0 to 1
    def feature_scaling(self, image):
        return image / 255.0
    # end

    def std_formula(self, img):
        mean = img.mean(axis=(0, 1), dtype='float64')
        std = img.std(axis=(0, 1), dtype='float64')
        img = (img - mean) / std
        return img

    def plot(self, img):
        plt.figure()
        plt.imshow(img)
        plt.show()


# unit test cases
# nsp_obj = Main()
# to visualize differences between original, standardized and normalized image
# nsp_obj.standardization_plot('HUGGIES ULTRA SUPER JUMBO M\\unfiltered\\train', 'A332.jpg')
# nsp_obj.normalization_plot('HUGGIES ULTRA SUPER JUMBO M\\unfiltered\\train', 'A332.jpg')
# end