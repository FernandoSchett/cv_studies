"""
    Author: Fernando Schettini
    How to run: python3 feature_extractor.py
    Description: Classe que extrai todas as features especificas de todas as imagens em uma pasta e salva em uma grande matriz de features.
    Last update: 25 november 2024.
"""

import os
import cv2
import numpy as np
from glob import glob
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.filters import sobel


class FeatureExtractor:
    def __init__(self, input_dir, output_dir="features"):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.feature_matrix = []  # Matriz de features que inclui as features e a coluna binária

    def load_images(self):
        """Load all images and determine their labels based on the folder structure."""
        # Define as classes de cada imagem com base na subpasta.
        for class_label, class_dir in zip([0, 1], ["0_N", "4_ADH"]):
            folder_path = os.path.join(self.input_dir, class_dir)
            for img_path in glob(os.path.join(folder_path, "*.png")):
                yield img_path, class_label

    def extract_lbp_features(self, img):
        """Extract LBP features from an image."""
        return local_binary_pattern(img, P=24, R=3, method="uniform").flatten()

    def extract_sobel_features(self, img):
        """Extract Sobel features from an image."""
        return sobel(img).flatten()

    def extract_haralick_features(self, img):
        """Extract all Haralick features from an image."""
        glcm = graycomatrix(img, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                            levels=256, symmetric=True, normed=True)
        haralick_features = []
        for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']:
            haralick_features.extend(graycoprops(glcm, prop).flatten())
        return np.array(haralick_features)

    def extract_features(self):
        """Extract features from all images and build the feature matrix."""
        for img_path, label in self.load_images():
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Ler como escala de cinza

            # Extrair as features separadamente
            lbp_features = self.extract_lbp_features(img)
            sobel_features = self.extract_sobel_features(img)
            haralick_features = self.extract_haralick_features(img)

            # Combine todas as features em um único vetor
            feature_row = np.concatenate([lbp_features, sobel_features, haralick_features, [label]])
            self.feature_matrix.append(feature_row)

    def save_features(self):
        """Save the feature matrix as a single .npy file."""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)  # Criar o diretório apenas ao salvar

        # Nome do arquivo baseado no nome do diretório de entrada
        base_dir_name = os.path.basename(os.path.normpath(self.input_dir))
        feature_file = os.path.join(self.output_dir, f"features_{base_dir_name}.npy")

        # Salvar a matriz de features como um arquivo binário
        np.save(feature_file, np.array(self.feature_matrix))
        print(f"Features saved to {feature_file}")

    def load_features(self, feature_file):
        """Load features from a .npy file."""
        if not os.path.exists(feature_file):
            raise FileNotFoundError(f"Feature file {feature_file} not found!")
        features = np.load(feature_file)
        print(f"Features loaded from {feature_file}")
        return features

    def show_sample(self, n_samples=5):
        """Display a sample of the feature matrix."""
        if not self.feature_matrix:
            print("No features extracted yet!")
            return
        print("Feature Sample:")
        for i in range(min(n_samples, len(self.feature_matrix))):
            print(f"Features: {self.feature_matrix[i][:10]}...")  # Mostra os primeiros 10 valores
            print(f"Label: {self.feature_matrix[i][-1]}")
            print("-" * 30)


if __name__ == "__main__":
    input_directory = 'data/bracs-binary/train'
    extractor = FeatureExtractor(input_directory)
    
    # Extrair as features separadamente
    extractor.extract_features()

    # Mostrar uma amostra da matriz de features
    extractor.show_sample()

    # Salvar a matriz de features em disco
    extractor.save_features()

    # Carregar a matriz de features de volta
    output_file = os.path.join(extractor.output_dir, "features_train.npy")
    features = extractor.load_features(output_file)
    print(f"Loaded features shape: {features.shape}")
