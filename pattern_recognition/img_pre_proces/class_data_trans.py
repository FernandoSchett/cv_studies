import os
from image import Imagem
from skimage import io
import matplotlib.pyplot as plt
from nclass_transformer import Transformer as tr

dataset_folder = './dataset'
targets = ["dataset"]
std_tam = [1024, 1024]
gamma_list = [4, 16, 32]

op = ["./dataset_eq_train", "./dataset_gamma_1_train", "./dataset_gaussiano_train", "./dataset_laplace_train"]
trans_keys = ["_equalized", "_gamma_4", "_gamma_16", "_gamma_32", "_gauss", "_laplace"] 

# Verifica se o dataset existe.
if not os.path.exists(dataset_folder):
    print("Dataset folder not found.")
else:
    # Cria os diretórios de saída.
    for _ in op:
        if not os.path.exists(_):
            os.makedirs(_)        
    
    for folder, subfolder, arq in os.walk(dataset_folder):
        for target in targets:
            if target in os.path.basename(folder): 
                for trget_dir, trget_sub_dir, trget_arq in os.walk(folder):

                    imgs_name = [f for f in os.listdir(trget_dir) if f.endswith('.jpg') or f.endswith('.JPG') or f.endswith('.jpeg')]
                    for img_name in imgs_name:
                        
                        imagem_path = os.path.join(trget_dir, img_name)
                        imagem = io.imread(imagem_path)
                        imagem = tr.alterar_tam(imagem, std_tam)
                        
                        
                        imgs_trans = []
                        imgs_trans.append(imagem)
                        plt.imshow(imgs_trans[0])
                        plt.axis('off')  # Para desativar as coordenadas do eixo
                        plt.show()

                        imgs_trans.append(tr.to_gray(imagem))
                        
                        imgs_trans.append(tr.equalize_hist(tr.to_gray(imagem.copy())))
                        plt.imshow(imgs_trans[1], cmap='gray')
                        plt.axis('off')  # Para desativar as coordenadas do eixo
                        plt.show()
                        plt.imshow(imgs_trans[2], cmap='gray')
                        plt.axis('off')  # Para desativar as coordenadas do eixo
                        plt.show()
                        
                        for gamma in gamma_list:
                            imgs_trans.append(tr.gamma_adjust(imagem.copy(), gamma, 1))
                        imgs_trans.append(tr.gauss_filter(imagem.copy(), 1.5))
                        imgs_trans.append(tr.laplace_filter(imagem.copy()))

                        for fodase in imgs_trans:
                            plt.imshow(fodase)
                            plt.axis('off')  # Para desativar as coordenadas do eixo
                            plt.show()
                        """
                        for i in range(len(imgs_trans)):
                            img_trans_name = os.path.splitext(img_name[i])[0] + trans_keys[i] + '.png'
                            img_trans_save_dir = os.path.join(op[i], "/".join(trget_dir.split('/')[1:]) )
                            img_trans_save_path = os.path.join(img_trans_save_dir

                        # Salvar Eq imgs              
                        equalized_train_folder = os.path.join(equalized_subfolder, folder.split('/')[-2])
                        equalized_train_folder = os.path.join(equalized_train_folder, tr_folder.split('/')[-1])
                        if not os.path.exists(equalized_train_folder):
                            os.makedirs(equalized_train_folder)

                        equalized_path = os.path.join(equalized_train_folder, equalized_filename)
                        io.imsave(equalized_path, eq_imagem)
                        """       
