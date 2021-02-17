import os, glob
import ProductSKU as psku

# to verify images and annotated txt files are aligned
img = []
txt = []
for file in glob.glob(f'F:\\APU\\Modules\\CP\\CP2\\darknet\\build\\darknet\\x64\\data\\obj\\*.jpg'):
    img.append(file.split('\\')[-1].split('.')[0])

for file in glob.glob(f'F:\\APU\\Modules\\CP\\CP2\\Object Detection\\Product SKU\\Annotated Images - '
                      f'Finalized\\*.txt'):
    f = file.split('\\')[-1]
    if f != 'classes.txt':
        txt.append(f.split('.')[0])

# if end result of img array is empty means data are aligned
for i in txt:
    img.remove(i)

# to ensure all annotated images are tagged to corresponding class
psku_obj = psku.Main()
tv = ['train', 'val']
path = psku_obj.image_path + '\\' + psku_obj.finalized_prod_sku_folder
i = 0
for folder in psku_obj.prod_sku_folders:
    for f in tv:
        for file in os.listdir(f'{path}\\{folder}\\unfiltered\\{f}'):
            t = file.split('.')[0]
            txtfile = f'F:\\APU\\Modules\\CP\\CP2\\Object Detection\\Product SKU\\Annotated Images - Finalized\\{t}.txt'
            content = open(txtfile, 'r')
            for rl in content.readlines():
                if int(rl.split()[0]) != i:
                    print(txtfile)
                    break
    i += 1
