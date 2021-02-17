import os
import ProductSKU as psku


class Main:
    def __init__(self):
        return

    def Build(self, origin):
        psku_obj = psku.Main()
        tv = ['train', 'val']

        for folder in psku_obj.prod_sku_folders:
            train_val_name = 'train.txt'
            for f in tv:
                if f == 'val':
                    train_val_name = 'valid.txt'

                # base image path
                data_path = f'{psku_obj.image_path}\\{folder}\\{f}'
                if origin == 'Finalized':
                    # finalized image path
                    data_path = f'{psku_obj.image_path}\\{psku_obj.finalized_prod_sku_folder}\\{folder}\\unfiltered\\{f}'

                for file in os.listdir(data_path):
                    t = open(f'F:\\APU\\Modules\\CP\\CP2\\darknet\\build\\darknet\\x64\\data\\{train_val_name}', 'a')
                    t.write(f'F:\\APU\\Modules\\CP\\CP2\\darknet\\build\\darknet\\x64\data\\obj\\{file}')
                    t.write('\n')
                    t.close()

# unit test case
# dd_obj = Main()
# dd_obj.Build()
# end