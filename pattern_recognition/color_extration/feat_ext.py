from image import Imagem
from skimage import io
import os

imgs_folder = './imgs_selected'

image_files = [f for f in os.listdir(imgs_folder) if f.endswith('.png') or f.endswith('.JPG') or f.endswith('.jpeg')]

for filename in image_files:
    image_path = os.path.join(imgs_folder, filename)
    image = io.imread(image_path)

    op = ['_normal','_shift','_change','_change_shift']
    img = Imagem(image_path, image, filename)
    
    for i in range(len(op)):
        if '_shift' in op[i]:
            img.int_shift(1.2)
        if '_change' in op[i]:
            img.int_change(30)       
        save_folder = os.path.join(imgs_folder, filename + op[i])
        img.gen_all(save_folder)
        img.save(os.path.join(save_folder, img.name))
        img.reset()