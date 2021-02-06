import os


def main():
    prod_sku_root_folder = 'Product SKU'

    cwd = os.getcwd()

    image_path = f'{cwd}\\{prod_sku_root_folder}'

    iFile = 1
    for subdirs, dirs, sub_files in os.walk(image_path):
        for dir in dirs:
            sub_image_path = f'{cwd}\\{prod_sku_root_folder}\\{dir}'
            for sub_subdirs, ttvs, images in os.walk(sub_image_path):
                iCount = 1
                prefix = 'TT'
                for ttv in ttvs:

                    if iCount == 2:
                        prefix = 'T'
                    elif iCount == 3:
                        prefix = 'V'

                    data_folder = f'{cwd}\\{prod_sku_root_folder}\\{dir}\\{ttv}'
                    for sub_sub_subdirs, sub_ttvs, files in os.walk(data_folder):
                        for file in files:
                            os.rename(f'{data_folder}\\{file}', f'{data_folder}\\{prefix}{iFile}.jpg')
                            iFile += 1
                    iCount += 1
        break
