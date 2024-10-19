from image import Imagem
import matplotlib.pyplot as plt
from skimage import io
import os

imgs_folder = './imgs_selected'

image_files = [f for f in os.listdir(imgs_folder) if f.endswith('.jpg') or f.endswith('.JPG') or f.endswith('.jpeg')]

for filename in image_files:
    
    image_path = os.path.join(imgs_folder, filename)
    image = io.imread(image_path)

    op = ['_normal', '_shift', '_change', '_change_shift']
    feat = ['_img', '_hrgb', '_htclor', '_hop']
    imgs = []
    
    plt.figure(figsize=(19, 10))
    plt.suptitle(filename)
    for i in range(len(op)):
        plt.subplot(1, len(op), i + 1)
        img_obj = Imagem(image_path, image, filename + op[i])
        if '_shift' in op[i]:
            img_obj.int_shift(-30)
        if '_change' in op[i]:
            img_obj.int_change(0.5)
        imgs.append(img_obj)
        plt.title(img_obj.get_name())
        plt.imshow(img_obj.image)
        plt.axis('off')  
    plt.tight_layout()
    plt.savefig(filename + feat[0] + '_comp.png')
    plt.close()

    # Gera os histogramas de cores
    plt.figure(figsize=(19, 10))
    plt.suptitle("Histograma de Cores.")
    for j in range(len(imgs)):
        plt.subplot(1, len(imgs), j + 1)
        imgs[j].gen_hist_rgb()
    plt.tight_layout()
    plt.savefig(filename + feat[1] + '_comp.png')
    plt.close()

    # Gera os histogramas de cores transformadas
    plt.figure(figsize=(19, 10))
    plt.suptitle("Histograma de Cores Transformadas.")
    for j in range(len(imgs)):
        plt.subplot(1, len(imgs), j + 1)
        imgs[j].gen_tcolor_dist()
    plt.tight_layout()
    plt.savefig(filename + feat[2] + '_comp.png')
    plt.close()
    
    # Gera os histogramas de oponentes
    plt.figure(figsize=(19, 10))
    plt.suptitle("Histograma de Oponentes.")
    for j in range(len(imgs)):
        plt.subplot(1, len(imgs), j + 1)
        imgs[j].gen_hist_opp()
    plt.tight_layout()
    plt.savefig(filename + feat[3] + '_comp.png')
    plt.close()
