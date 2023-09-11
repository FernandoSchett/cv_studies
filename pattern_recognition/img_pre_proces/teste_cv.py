from image import Imagem
import matplotlib.pyplot as plt
from skimage import io
import copy
import numpy as np
import os
import cv2

imgs_folder = './imgs_selected'

image_files = [f for f in os.listdir(imgs_folder) if f.endswith('.jpg') or f.endswith('.JPG') or f.endswith('.jpeg')]

for filename in image_files:
    image_path = os.path.join(imgs_folder, filename)
    imagem = cv2.imread(image_path)

    op = ['_normal','_shift','_change','_change_shift']
    img = Imagem(image_path, imagem, filename)

    # Carregue a imagem

    # Verifique se a imagem foi carregada com sucesso
    if imagem is None:
        print('Não foi possível carregar a imagem.')
    else:
        # Aumente em 30 o valor de todos os canais de cor
        print(imagem)
        print("===========================================================")
        imagem_aumentada = imagem.astype(np.uint16)
        imagem_aumentada = imagem_aumentada + 100

        print(imagem_aumentada)
        print("===========================================================")

        # Certifique-se de que os valores não excedam 255
        imagem_aumentada[imagem_aumentada > 255] = 255

        # Salve a imagem aumentada
        cv2.imwrite('imagem_aumentada.jpg', imagem_aumentada)

        # Exiba a imagem original e a imagem aumentada (opcional)
        cv2.imshow('Imagem Original', imagem)
        cv2.imshow('Imagem Aumentada', imagem_aumentada)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
