from image import Imagem
from skimage import io
import skimage
import matplotlib.pyplot as plt
import os

imgs_folder = './imgs_selected'

image_files = [f for f in os.listdir(imgs_folder) if f.endswith('.jpg') or f.endswith('.JPG') or f.endswith('.jpeg')]
for filename in image_files:
    image_path = os.path.join(imgs_folder, filename)
    image = io.imread(image_path)

    img = Imagem(image_path, image)
    
    img_shift = Imagem(image_path, image)
    img_shift.int_shift(1.2)

    img_change = Imagem(image_path, image)
    img_change.int_change(30)

    img_change_shift = Imagem(image_path, image)
    img_change_shift.int_change(30)
    img_change_shift.int_shift(1.2)

    save_folder = os.path.join(imgs_folder, filename + '_normal')
    img.gen_hist_rgb(save_folder)
    img.gen_hist_opp(save_folder)
    img.gen_tcolor_dist(save_folder)
    
    save_folder = os.path.join(imgs_folder, filename + '_shift')
    img_shift.gen_hist_rgb(save_folder)
    img_shift.gen_hist_opp(save_folder)
    img_shift.gen_tcolor_dist(save_folder)
    img_shift.save(os.path.join(save_folder, filename + '_shifted.png'))

    save_folder = os.path.join(imgs_folder, filename + '_change')
    img_change.gen_hist_rgb(save_folder)
    img_change.gen_hist_opp(save_folder)
    img_change.gen_tcolor_dist(save_folder)
    img_change.save(os.path.join(save_folder, filename + '_change.png'))

    save_folder = os.path.join(imgs_folder, filename + '_change_shift')
    img_change_shift.gen_hist_rgb(save_folder)
    img_change_shift.gen_hist_opp(save_folder)
    img_change_shift.gen_tcolor_dist(save_folder)
    img_change_shift.save(os.path.join(save_folder, filename + '_change_shift.png'))

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 4, 1)
    plt.imshow(img.image)
    plt.title('Imagem Original')
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.imshow(img_change.image)
    plt.title('img_change')
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.imshow(img_shift.image)
    plt.title('img_shift')
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.imshow(img_change_shift.image)
    plt.title('img_change_shift')
    plt.axis('off')

    plt.tight_layout()
    plt.show()
