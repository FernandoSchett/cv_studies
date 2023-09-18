import os
from skimage import io
from transformer import Transformer as tr 

dataset_folder = './dataset'
operation = "_to_gray"
result_folder = dataset_folder + operation
targets = ["treino", "teste"]
std_tam = [1024, 1024]

# Verifica se o dataset existe.
if not os.path.exists(dataset_folder):
    print("Dataset folder not found.")
else:    
    for folder, subfolder, arq in os.walk(dataset_folder):
        for target in targets:
            if target in os.path.basename(folder):  # find targets in foldernames   
                for trget_dir, trget_sub_dir, trget_arq in os.walk(folder):
                    imgs_name = [f for f in os.listdir(trget_dir) if f.endswith('.jpg') or f.endswith('.JPG') or f.endswith('.jpeg')]
                    
                    print('/'.join((trget_dir.split('/'))[2:]))
                    save_place = result_folder + '/' + '/'.join((trget_dir.split('/'))[2:])
                    if not os.path.exists(save_place):
                            os.makedirs(save_place)
                    
                    for img_name in imgs_name:
                        
                        imagem_path = os.path.join(trget_dir, img_name)
                        imagem = io.imread(imagem_path)
                        imagem = tr.alterar_tam(imagem, std_tam)
                        aux_imagem = tr.to_gray(imagem)
                        io.imsave(save_place + "/" + os.path.splitext(img_name)[0] + operation + '.png', aux_imagem)
