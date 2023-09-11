from image import Imagem
import matplotlib.pyplot as plt
from skimage import io
import os

imgs_folder = './imgs_selected'

image_files = [f for f in os.listdir(imgs_folder) if f.endswith('.jpg') or f.endswith('.JPG') or f.endswith('.jpeg')]

for filename in image_files:
    
    image_path = os.path.join(imgs_folder, filename)
    image = io.imread(image_path)

    op = ['_normal','_shift','_change','_change_shift']
    feat = ['_img','_hrgb','_htclor','__hop']
    imgs = []
    

    plt.figure(figsize=(19, 10))
    plt.suptitle(filename)
    
    #aplica tranformações
    for i in range(len(op)):
        plt.subplot(1, len(op), i+1)
        imgs.append(Imagem(image_path, image, filename + op[i]))
        if '_shift' in op[i]:
            imgs[i].int_shift(1.2)
        if '_change' in op[i]:
            imgs[i].int_change(30)
        plt.title(imgs[i].get_name())
        plt.imshow(imgs[i].image)
    plt.tight_layout()
    plt.savefig(filename + feat[0] + '_comp.png')
    plt.close()

    
    plt.figure(figsize=(19, 10))
    plt.suptitle("Histograma de Cores.")
    for j in range(len(imgs)):
        plt.subplot(1, len(imgs), j+1)
        imgs[j].gen_hist_rgb()
    plt.tight_layout()
    plt.savefig(filename + feat[1] + '_comp.png')
    plt.close()

    plt.figure(figsize=(19, 10))
    plt.suptitle("Histograma de Cores Transformadas.")
    for j in range(len(imgs)):
        plt.subplot(1, len(imgs), j+1)
        imgs[j].gen_tcolor_dist()
    
    plt.tight_layout()
    plt.savefig(filename + feat[2] + '_comp.png')
    plt.close()
    
    plt.figure(figsize=(19, 10))
    plt.suptitle("Histograma de Oponentes.")
    for j in range(len(imgs)):
        plt.subplot(1, len(imgs), j+1)
        imgs[j].gen_hist_opp()

    plt.tight_layout()
    plt.savefig(filename + feat[3] + '_comp.png')
    plt.close()    


