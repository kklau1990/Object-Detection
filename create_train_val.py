import os


def main():
    prod_sku_root_folder = 'Product SKU'

    cwd = os.getcwd()

    image_path = f'{cwd}\\{prod_sku_root_folder}'
    data_path = 'F:\\APU\\Modules\\CP\\CP2\\darknet\\build\\darknet\\x64\\data'

    train_val = ['train', 'val']
    prod_sku_folders = ['HEAD & SHOULDERS SHAMPOO (ASSORTED) COOL MENTHOL', 'HUGGIES ULTRA SUPER JUMBO M',
               'HUP SENG CREAM CRACKERS 428G', 'KLEENEX ULTRA SOFT BATH ISSUE MEGA', 'NATURAL PURE OLIVE OIL 750 ML',
               'SUNSLIK SHAMPOO (ASSORTED) LIVELY CLEAN & FRESH']

    for folder in prod_sku_folders:
        for tv in train_val:
            for file in os.listdir(f'{image_path}\\{folder}\\{tv}'):
                if tv == 'train':
                    f1 = open(f'{data_path}\\train.txt', 'a+')
                    f1.write(f'{data_path}\\obj\\{file}\n')
                    f1.close()
                else:
                    f1 = open(f'{data_path}\\test.txt', 'a+')
                    f1.write(f'{data_path}\\obj\\{file}\n')
                    f1.close()

