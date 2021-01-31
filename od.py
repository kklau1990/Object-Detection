import cv2
import numpy as np
import pandas as pd
import rename_files as nf
import create_train_val as ctv
import affine_transform as af
import image_filter_process as ifp
import norm_stad_process as nsp
import os
import matplotlib.pyplot as plt

df = pd.DataFrame(columns=['Filepath', 'Folder', 'Filename', 'Dimension'])
# constant width, height for resized image

class Main:
    def __init__(self):
        self.cwd = os.getcwd()
        self.prod_sku_root_folder = 'Product SKU'
        self.prod_sku_folders = ['HEAD & SHOULDERS SHAMPOO (ASSORTED) COOL MENTHOL', 'HUGGIES ULTRA SUPER JUMBO M',
                                'HUP SENG CREAM CRACKERS 428G', 'KLEENEX ULTRA SOFT BATH ISSUE MEGA',
                                'NATURAL PURE OLIVE OIL 750 ML', 'SUNSLIK SHAMPOO (ASSORTED) LIVELY CLEAN & FRESH']
        self.augmented_prod_sku_folder = 'Augmented Images'
        self.finalized_prod_sku_folder = 'Finalized Images'
        self.image_path = f'{self.cwd}\\{self.prod_sku_root_folder}'
        self.resize_width = 800
        self.resized_height = 800

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
                    for file in os.listdir(f'{finalized_image_path}\\{folder}\\{tv}'):
                        im = cv2.imread(f'{finalized_image_path}\\{folder}\\{tv}\\{file}')
                        df = df.append({'Filepath': f'{finalized_image_path}\\{folder}\\{tv}\\{file}', 'Folder':
                                        f'{folder}', 'Filename': f'{file}', 'Dimension': str(im.shape)},
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

            # randomly sample 12 records
            random_subset = df_filtered.sample(12)
            # end
            columns = 3
            rows = 4
            fig = plt.figure(figsize=(8, 8))
            fig.canvas.set_window_title(folder)

            for i in range(1, columns * rows + 1):
                img = cv2.imread(random_subset.iloc[i-1]['Filepath'])
                fig.add_subplot(rows, columns, i)
                plt.imshow(img[..., ::-1])  # RGB-> BGR
                plt.title(random_subset.iloc[i - 1]['Filename'])
            plt.show()

            # to visualize pixel intensity
            fig = plt.figure(figsize=(8, 8))
            fig.canvas.set_window_title(folder)
            for i in range(1, columns * rows + 1):
                img = cv2.imread(random_subset.iloc[i-1]['Filepath'])
                fig.add_subplot(rows, columns, i)
                plt.hist(img.ravel(), 256, [0, 256])
                plt.title(random_subset.iloc[i-1]['Filename'])
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
                for file in os.listdir(f'{finalized_prod_sku_folder}\\{folder}\\{tv}'):
                    img = cv2.imread(f'{finalized_prod_sku_folder}\\{folder}\\{tv}\\{file}')
                    resized_img = cv2.resize(img, (self.resized_height, self.resize_width))
                    # delete original image
                    os.remove(f'{finalized_prod_sku_folder}\\{folder}\\{tv}\\{file}')
                    # save resize image
                    cv2.imwrite(f'{finalized_prod_sku_folder}\\{folder}\\{tv}\\{file}', resized_img)
    # end

    # smoothing images
    def img_filter(self, **kwargs):
        ifp_obj = ifp.Main()

        filter_mode = kwargs.pop('filter_mode', [1])  # if no filter is selected, default as 1 (mean filter)
        batch = kwargs.pop('batch', False)
        test_input = kwargs.pop('test_input', '')

        if batch and test_input:
            return 'Invalid operation. System only accepts filter files by batch or single input.'

        if not batch and not test_input:
            return 'Please insert a test input image path.'

        for i in filter_mode:
            ifp_obj.initiate(i, batch, test_input)
    # end

    # data normalization process
    def normalization(self):
        nsp_obj = nsp.Main()
        nsp_obj.normalization()
    # end

    # test yolo model
    def yolo_predict_output(self, test_input):
        print("Initializing YoloV3......Please wait.......")
        net = cv2.dnn.readNet(f'{self.cwd}\\base weights\\yolo-obj_best.weights',
                              'F:\\APU\\Modules\\CP\\CP2\\darknet\\build\\darknet\\x64\\cfg\\yolo-obj.cfg')

        # save all the names in file o the list classes
        classes = []
        with open("F:\\APU\\Modules\\CP\\CP2\\darknet\\build\\darknet\\x64\\data\\obj.names", "r") as f:
            classes = [line.strip() for line in f.readlines()]

        # get layers of the network
        layer_names = net.getLayerNames()

        # Determine the output layer names from the YOLO model

        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        print("Yolov3 Loaded successfully.")
        print("System is predicting objects.......")
        # Capture frame-by-frame
        img = cv2.imread(f'{test_input}')
        img = cv2.resize(img, (self.resized_height, self.resize_width))
        height, width, channels = img.shape

        # USing blob function of opencv to preprocess image
        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416),
                                     swapRB=True, crop=False)
        # Detecting objects
        net.setInput(blob)
        outs = net.forward(output_layers)

        # Showing informations on the screen
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # We use NMS function in opencv to perform Non-maximum Suppression
        # we give it score threshold and nms threshold as arguments.
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                conf = '{:.2f}'.format(confidences[i])
                label = str(classes[class_ids[i]])
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(img, f'{label}: {str(conf)}', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            # font size, color, thickness
                            0.5, (255, 0, 0), 2)

        plt.figure()
        plt.imshow(img[..., ::-1])  # RGB-> BGR
        plt.show()


# rename original files
nf_obj = nf
# nf.main()  # executed once only
# end

# split files into train test val
ctv_obj = ctv
# ctv_obj.main() # executes only once/when required
# end

od = Main()  # initialization
# to visualize sampled data distribution
# od.data_aggregation('Product SKU')
# od.data_dist_vis(df['Folder'], 'Total Count', 'Product SKU', 'Product SKU Data Distribution', 'barh', '1')
# od.data_dist_vis(df['Dimension'], 'Image Dimension', 'Total Count', 'Image Dimension Data Distribution', 'bar', 0)
# od.img_vis()

# od.data_augmentation()

# to visualize augmented data distribution
# od.data_aggregation('Augmented Images')
# od.data_dist_vis(df['Folder'], 'Total Count', 'Product SKU', 'Product SKU Augmented Data Distribution', 'barh', '1')
# od.data_dist_vis(df['Dimension'], 'Image Dimension', 'Total Count', 'Augmented Image Dimension Data Distribution',
#                  'bar', 0)

# to visualize finalized data distribution
# od.data_aggregation('Finalized Images')
# od.data_dist_vis(df['Folder'], 'Total Count', 'Product SKU', 'Product SKU Finalized Data Distribution', 'barh', '1')
# od.data_dist_vis(df['Dimension'], 'Image Dimension', 'Total Count', 'Image Dimension Finalized Data Distribution',
#                  'bar', 0)

# resize finalized images
# od.img_resize()

# to visualize differences between original, standardized and normalized image
# od.standardization('HUGGIES ULTRA SUPER JUMBO M\\train', 'A332.jpg')
# od.normalization('HUGGIES ULTRA SUPER JUMBO M\\train', 'A332.jpg')

# od.data_aggregation('Finalized Images', True)
# od.data_dist_vis(df['Dimension'], 'Image Dimension', 'Total Count', 'Image Dimension Data Distribution Exclude Test '
#                  'Data', 'bar', 0)

# od.img_filter(filter_mode=[0, 1, 2], batch=True)

# od.yolo_predict_output('F:\\APU\Modules\\CP\\CP2\\Object Detection\\Product SKU\\HUP SENG CREAM CRACKERS 428G'
#                    '\\test\\TT53.jpg')

