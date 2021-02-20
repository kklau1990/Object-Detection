import cv2
import numpy as np
import pandas as pd
from Utils import AffineTransform as af, NormStadProcess as nsp, RenameFiles as nf
from Utils import ImageFilterProcess as ifp, CreateTrainVal as ctv
import os
import matplotlib.pyplot as plt
import shutil
import ProductSKU as psku


df = pd.DataFrame(columns=['Filepath', 'Folder', 'Filename', 'Dimension'])


class Main:
    def __init__(self):
        psku_obj = psku.Main()
        self.cwd = psku_obj.cwd
        self.prod_sku_root_folder = psku_obj.prod_sku_root_folder
        self.prod_sku_folders = psku_obj.prod_sku_folders
        self.augmented_prod_sku_folder = psku_obj.augmented_prod_sku_folder
        self.finalized_prod_sku_folder = psku_obj.finalized_prod_sku_folder
        self.image_path = psku_obj.image_path
        self.resize_width = psku_obj.resize_width
        self.resized_height = psku_obj.resized_height

    # initiate data grouping into dataframe
    def data_aggregation(self, folder, excludetest=False):
        global df
        df = df[0:0]  # reset dataframe
        train_val_test = ['train', 'val', 'test']

        if folder == self.prod_sku_root_folder:
            for folder in self.prod_sku_folders:
                for tv in train_val_test:
                    for file in os.listdir(f'{self.image_path}\\{folder}\\{tv}'):
                        im = cv2.imread(f'{self.image_path}\\{folder}\\{tv}\\{file}')
                        df = df.append({'Filepath': f'{self.image_path}\\{folder}\\{tv}\\{file}', 'Folder': f'{folder}',
                                        'Filename': f'{file}', 'Dimension': str(im.shape)}, ignore_index=True)
        elif folder == self.augmented_prod_sku_folder:
            augment_image_path = f'{self.image_path}\\{self.augmented_prod_sku_folder}'
            for folder in self.prod_sku_folders:
                for file in os.listdir(f'{augment_image_path}\\{folder}'):
                    im = cv2.imread(f'{augment_image_path}\\{folder}\\{file}')
                    df = df.append({'Filepath': f'{augment_image_path}\\{folder}\\{file}', 'Folder': f'{folder}',
                                    'Filename': f'{file}', 'Dimension': str(im.shape)}, ignore_index=True)
        elif folder == self.finalized_prod_sku_folder:
            if excludetest:
                train_val_test.pop(2)  # exclude test folder
            finalized_image_path = f'{self.image_path}\\{self.finalized_prod_sku_folder}'
            for folder in self.prod_sku_folders:
                for tv in train_val_test:
                    for file in os.listdir(f'{finalized_image_path}\\{folder}\\unfiltered\\{tv}'):
                        im = cv2.imread(f'{finalized_image_path}\\{folder}\\unfiltered\\{tv}\\{file}')
                        df = df.append({'Filepath': f'{finalized_image_path}\\{folder}\\unfiltered\\{tv}\\{file}',
                                        'Folder': f'{folder}', 'Filename': f'{file}', 'Dimension': str(im.shape)},
                                       ignore_index=True)
    # end

    # data distribution visualization
    def data_dist_vis(self, df, xlabel, ylabel, title, charthv, rotdeg):
        color = [plt.cm.Paired(i) for i in range(0, len(df.drop_duplicates()))]
        ax = df.value_counts().plot(kind=charthv, color=color, rot=rotdeg)

        for index, p in enumerate(ax.patches):
            left, bottom, width, height = p.get_bbox().bounds
            val = df.value_counts()[index]
            if charthv == 'bar':
                ax.annotate(val, (p.get_x() + 0.2, height + 0.05))
            else:
                ax.annotate(val, xy=(width + 0.02, p.get_y() + 0.2))

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()
    # end

    # image visualization
    def img_vis(self):
        for folder in self.prod_sku_folders:
            df_filtered = df[df['Folder'] == folder]

            # randomly sample 15 records
            # max 15 only because huggies only have 15 records
            random_subset = df_filtered.sample(15)
            columns = 5
            rows = 3

            for j in range(1, 3):
                fig = plt.figure(figsize=(8, 8))
                fig.canvas.set_window_title(folder)

                for i in range(1, columns * rows + 1):
                    img = cv2.imread(random_subset.iloc[i-1]['Filepath'])
                    fig.add_subplot(rows, columns, i)
                    if j == 1:
                        # to visualize image selected
                        plt.imshow(img[..., ::-1])  # RGB-> BGR
                    else:
                        # to visualize pixel intensity
                        plt.hist(img.ravel(), 256, [0, 256])
                    plt.title(random_subset.iloc[i - 1]['Filename'])
                plt.show()
    # end

    # data augmentation
    def data_augmentation(self):
        af_obj = af.Main()
        transform_list = af_obj.transform_list()
        ifile = 187  # original resource has 186 images with maximum name T186. Start name from 187 index
        train_val_test = ['train', 'val', 'test']

        for folder in self.prod_sku_folders:
            for tv in train_val_test:
                for file in os.listdir(f'{self.image_path}\\{folder}\\{tv}'):
                    if file.lower().endswith('.jpg'):
                        for key in transform_list:  # loop through all affine transform functions
                            # apply all affine transform methods per image
                            res = transform_list[key](f'{self.image_path}\\{folder}\\{tv}', file)
                            af_obj.img_save(f'{self.image_path}\\{self.augmented_prod_sku_folder}\\{folder}\\'
                                            f'A{ifile}.jpg', res[..., ::-1])  # save as RGB color
                            ifile += 1
    # end

    # resize for train and validation folders so that ML model can learn faster, testing images size does not matter
    def img_resize(self):
        train_val = ['train', 'val']
        finalized_prod_sku_folder = f'{self.image_path}\\{self.finalized_prod_sku_folder}'
        for folder in self.prod_sku_folders:
            for tv in train_val:
                for file in os.listdir(f'{finalized_prod_sku_folder}\\{folder}\\unfiltered\\{tv}'):
                    img = cv2.imread(f'{finalized_prod_sku_folder}\\{folder}\\unfiltered\\{tv}\\{file}')
                    resized_img = cv2.resize(img, (self.resized_height, self.resize_width))
                    # delete original image
                    os.remove(f'{finalized_prod_sku_folder}\\{folder}\\unfiltered\\{tv}\\{file}')
                    # save resize image
                    cv2.imwrite(f'{finalized_prod_sku_folder}\\{folder}\\unfiltered\\{tv}\\{file}', resized_img)
    # end

    # copy a set of unfiltered images to filter folders
    def copy(self):
        finalized_prod_sku_folder = f'{self.image_path}\\{self.finalized_prod_sku_folder}'
        for folder in self.prod_sku_folders:
            # source folder
            unfiltered_path = f'{finalized_prod_sku_folder}\\{folder}\\unfiltered'
            ttv_folders = next(os.walk(f'{unfiltered_path}'))[1]
            for ttv_folder in ttv_folders:
                # copy a set of unfiltered test, train val images to filter folders
                filter_folders = next(os.walk(f'{finalized_prod_sku_folder}\\{folder}'))[1]
                filter_folders.remove('unfiltered')
                for filter_folder in filter_folders:
                    shutil.copytree(f'{unfiltered_path}\\{ttv_folder}',
                                    f'{finalized_prod_sku_folder}\\{folder}\\{filter_folder}\\{ttv_folder}')
    # end

    # smoothing images
    def img_filter(self, **kwargs):
        ifp_obj = ifp.Main()

        filter_mode = kwargs.pop('filter_mode', [0])  # if no filter is selected, default as 1 (mean filter)
        batch = kwargs.pop('batch', False)
        test_input = kwargs.pop('test_input', '')

        if batch and test_input:
            return 'Invalid operation. System only accepts filter files by batch or single input.'

        if not batch and not test_input:
            return 'Please insert a test input image path.'

        for i in filter_mode:
            ifp_obj.initiate(i, batch, test_input)
    # end

# rename original files, enable when necessary
nf_obj = nf
# nf.main()  # executed once only
# end

# split files into train test val for base images, enable when necessary
ctv_obj = ctv.Main()
# ctv_obj.Build('Base')  # executes only once/when required
# end

dp = Main()  # initialization
# to visualize sampled data distribution
dp.data_aggregation('Product SKU')
dp.data_dist_vis(df['Folder'], 'Total Count', 'Product SKU', 'Product SKU Data Distribution', 'barh', '1')
dp.data_dist_vis(df['Dimension'], 'Image Dimension', 'Total Count', 'Image Dimension Data Distribution', 'bar', 0)
dp.img_vis()

# synthesize data, enable when necessary
# dp.data_augmentation()

# to visualize augmented data distribution
dp.data_aggregation('Augmented Images')
dp.data_dist_vis(df['Folder'], 'Total Count', 'Product SKU', 'Product SKU Augmented Data Distribution', 'barh', '1')
dp.data_dist_vis(df['Dimension'], 'Image Dimension', 'Total Count', 'Augmented Image Dimension Data Distribution',
                 'bar', 0)

# to visualize finalized data distribution
dp.data_aggregation('Finalized Images')
dp.data_dist_vis(df['Folder'], 'Total Count', 'Product SKU', 'Product SKU Finalized Data Distribution', 'barh', '1')
dp.data_dist_vis(df['Dimension'], 'Image Dimension', 'Total Count', 'Finalized Image Dimension Data Distribution',
                 'bar', 0)

# resize finalized images, enable when necessary
# dp.img_resize()

# visualize total finalized training and validation data
dp.data_aggregation('Finalized Images', True)
dp.data_dist_vis(df['Dimension'], 'Image Dimension', 'Total Count', 'Finalized Image Dimension Data Distribution '
                                                                     'Exclude Test Data', 'bar', 0)

# to visualize differences between original, standardized and normalized image
nsp_obj = nsp.Main()
nsp_obj.standardization_plot('HUGGIES ULTRA SUPER JUMBO M\\unfiltered\\train', 'A364.jpg')
nsp_obj.normalization_plot('HUGGIES ULTRA SUPER JUMBO M\\unfiltered\\train', 'A364.jpg')
# begin normalization process, enable when necessary
# nsp_obj.normalization(observation_test=False)

# copy a set of finalized unfiltered data to bilateral, mean, and median filtered folder, enable when necessary
# dp.copy()

# initiate filter processes, enable when necessary
# dp.img_filter(filter_mode=[0, 1, 2], batch=True)

# split files into train test val for finalized images, enable when necessary
# ctv_obj.Build('Finalized')  # executes only once/when required
# end
