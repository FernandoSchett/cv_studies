import os
import skimage
from skimage import io, exposure, transform, filters

def equalize_hist(image):
    equalized_image = skimage.color.rgb2gray(image)
    equalized_image = exposure.equalize_hist(equalized_image)  
    equalized_image = skimage.img_as_ubyte(equalized_image)

    return equalized_image

def ganna_adjust(image):
    gamma_image = skimage.img_as_ubyte(image)
    gamma_image = exposure.adjust_gamma(gamma_image, 1.5, -1)
    return gamma_image

def gauss_filter(image):
    gauss_image = filters.gaussian(image, 1.5, channel_axis=None)
    gauss_image = skimage.img_as_ubyte(gauss_image)
    return gauss_image

def laplace_filter(image):
    laplace_image = filters.laplace(image)
    laplace_image = skimage.img_as_ubyte(laplace_image)
    return laplace_image

dataset_folder = './dataset'

equalized_subfolder = os.path.join(dataset_folder, "eq_train")
gamma_subfolder = os.path.join(dataset_folder, "gamma_train")
gauss_subfolder = os.path.join(dataset_folder, "gaussiano_train")
laplace_subfolder = os.path.join(dataset_folder, "laplace_train")

for folder, subfolder, arq in os.walk(dataset_folder):
    if 'treino' == folder.split('/')[-1]: 
        for tr_folder, sub_folder, arq in os.walk(folder):
            image_files = [f for f in os.listdir(tr_folder) if f.endswith('.jpg') or f.endswith('.JPG')]
            for filename in image_files:
                image_path = os.path.join(tr_folder, filename)
                image = io.imread(image_path)
                
                image = transform.resize(image, [1024, 1024])
                eq_image = equalize_hist(image.copy())
                gamma_image = ganna_adjust(image.copy())
                gauss_img = gauss_filter(image.copy())
                lap_img = laplace_filter(image.copy())
                
                # Salvar Eq imgs              
                equalized_filename = os.path.splitext(filename)[0] + '_equalized.png'
                equalized_train_folder = os.path.join(equalized_subfolder, folder.split('/')[-2])
                equalized_train_folder = os.path.join(equalized_train_folder, tr_folder.split('/')[-1])
                if not os.path.exists(equalized_train_folder):
                    os.makedirs(equalized_train_folder)

                equalized_path = os.path.join(equalized_train_folder, equalized_filename)
                io.imsave(equalized_path, eq_image)  
                
                
                # Salvar Gamma imgs
                gamma_filename = os.path.splitext(filename)[0] + '_gamma.png'
                gamma_train_folder = os.path.join(gamma_subfolder, folder.split('/')[-2])
                gamma_train_folder = os.path.join(gamma_train_folder, tr_folder.split('/')[-1])
                if not os.path.exists(gamma_train_folder):
                    os.makedirs(gamma_train_folder)

                gamma_path = os.path.join(gamma_train_folder, gamma_filename)
                io.imsave(gamma_path, gamma_image)  

                # Salvar Gauss imgs              
                gauss_filename = os.path.splitext(filename)[0] + '_gauss.png'
                gauss_train_folder = os.path.join(gauss_subfolder, folder.split('/')[-2])
                gauss_train_folder = os.path.join(gauss_train_folder, tr_folder.split('/')[-1])
                if not os.path.exists(gauss_train_folder):
                    os.makedirs(gauss_train_folder)

                gauss_path = os.path.join(gauss_train_folder, gauss_filename)
                io.imsave(gauss_path, gauss_img) 

                # Salvar Laplace imgs              
                lap_filename = os.path.splitext(filename)[0] + '_lap.png'
                lap_train_folder = os.path.join(laplace_subfolder, folder.split('/')[-2])
                lap_train_folder = os.path.join(lap_train_folder, tr_folder.split('/')[-1])
                if not os.path.exists(lap_train_folder):
                    os.makedirs(lap_train_folder)            
                    
                lap_path = os.path.join(lap_train_folder, lap_filename)
                io.imsave(lap_path, lap_img)     

print("Equalização de histograma nas imagens de treinamento concluído")