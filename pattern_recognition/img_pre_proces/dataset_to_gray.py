import os
import skimage
import copy
from skimage import io, exposure, transform, filters

dataset_folder = './dataset'

to_gray_subfolder = os.path.join(dataset_folder, "_to_gray")

for folder, subfolder, arq in os.walk(dataset_folder):
    if 'treino' == folder.split('/')[-1]: 
        for tr_folder, sub_folder, arq in os.walk(folder):
            image_files = [f for f in os.listdir(tr_folder) if f.endswith('.jpg') or f.endswith('.JPG')]
            for filename in image_files:
                image_path = os.path.join(tr_folder, filename)
                image = io.imread(image_path)
                
                image = transform.resize(image, [1024, 1024])
                eq_image = equalize_hist(copy.deepcopy(image))
                gamma_image = ganna_adjust(copy.deepcopy(image))
                gauss_img = gauss_filter(copy.deepcopy(image))
                print(filename)
                lap_img = laplace_filter(copy.deepcopy(image))
                
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