import os


class Main:
    def __init__(self):
        self.cwd = os.getcwd()
        self.prod_sku_root_folder = 'Product SKU'
        self.prod_sku_folders = ['HEAD & SHOULDERS SHAMPOO (ASSORTED) COOL MENTHOL', 'HUGGIES ULTRA SUPER JUMBO M',
                                'HUP SENG CREAM CRACKERS 428G', 'KLEENEX ULTRA SOFT BATH TISSUE MEGA',
                                'NATURAL PURE OLIVE OIL 750 ML', 'SUNSLIK SHAMPOO (ASSORTED) LIVELY CLEAN & FRESH']
        self.augmented_prod_sku_folder = 'Augmented Images'
        self.finalized_prod_sku_folder = 'Finalized Images'
        self.image_path = f'{self.cwd}\\{self.prod_sku_root_folder}'
        self.resize_width = 800
        self.resized_height = 800
