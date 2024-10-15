import os
import cv2
from glob import glob
import matplotlib.pyplot as plt
import numpy as np

class ImageProcessor:
    def __init__(self, input_dir, output_dir=None, size=(256, 256)):
        self.input_dir = input_dir
        self.size = size
        self.images = {}  # Imagens processadas
        self.original_images = {}  # Cópia das imagens originais
        self.operations = {}  # Histórico de operações aplicadas em cada imagem

        # Configura o diretório de saída com base nas operações aplicadas se não for especificado
        if output_dir is None:
            self.output_dir = self._generate_output_dir()
        else:
            self.output_dir = output_dir

    def _generate_output_dir(self):
        """Gera automaticamente o diretório de saída com base na pasta de entrada e nas operações aplicadas."""
        base_name = os.path.basename(os.path.normpath(self.input_dir))
        operations_str = "_".join(set(sum(self.operations.values(), []))) if self.operations else "original"
        output_dir = f"{base_name}_{operations_str}" if operations_str else base_name

        # Criar o caminho final, no mesmo diretório da pasta de entrada
        parent_dir = os.path.dirname(self.input_dir)
        return os.path.join(parent_dir, output_dir)

    def load_images(self):
        """Load all images into memory and store the originals."""
        for img_path in glob(os.path.join(self.input_dir, '**', '*.png'), recursive=True):
            img = cv2.imread(img_path)
            relative_path = img_path.replace(self.input_dir, '').lstrip(os.sep)  # Corrige caminho relativo
            self.images[relative_path] = img
            self.original_images[relative_path] = img.copy()  # Armazenar uma cópia original
            self.operations[relative_path] = []  # Iniciar o histórico de operações como uma lista vazia

    def _save_image(self, img, relative_path):
        """Helper method to save a processed image to the output directory."""
        # Concatenar as operações no nome do arquivo, mas apenas se houver operações
        base_name, ext = os.path.splitext(os.path.basename(relative_path))
        operations_str = "_".join(self.operations[relative_path]) if self.operations[relative_path] else ""
        new_name = f"{base_name}_{operations_str}{ext}" if operations_str else f"{base_name}{ext}"
        
        save_path = os.path.join(self.output_dir, os.path.dirname(relative_path), new_name).lstrip(os.sep)
        
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))  # Criar diretórios, se necessário
        cv2.imwrite(save_path, img)
    
    def save_images(self):
        """Save all images in memory to the output directory."""
        # Recalcular o diretório de saída com base nas operações
        self.output_dir = self._generate_output_dir()
        for relative_path, img in self.images.items():
            img = self._convert_to_uint8(img)  # Converter para uint8 antes de salvar
            self._save_image(img, relative_path)

    def _convert_to_uint8(self, img):
        """Convert image to uint8, scaling pixel values to 0-255 if necessary."""
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255)  # Garante que os valores estejam no intervalo correto
            img = img.astype(np.uint8)
        return img
        
    ######### Operações com imagens no dataset ############

    def reset_images(self):
        """Reset all images to their original state."""
        self.images = {k: v.copy() for k, v in self.original_images.items()}
        self.operations = {k: [] for k in self.original_images.keys()}  # Resetar as operações também

    def resize_images(self):
        """Resize all images in memory."""
        for relative_path, img in self.images.items():
            self.images[relative_path] = cv2.resize(img, self.size)
            self.operations[relative_path].append(f"resize{self.size[0]}x{self.size[1]}")  # Registrar operação

    def equalize_histogram(self):
        """Apply histogram equalization to all images in memory."""
        for relative_path, img in self.images.items():
            if len(img.shape) == 2:  # Grayscale
                self.images[relative_path] = cv2.equalizeHist(img)
            else:  # Color
                img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
                img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
                self.images[relative_path] = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
            self.operations[relative_path].append("equalize_histogram")  # Registrar operação

    def gamma_correction(self, gamma):
        """Apply gamma correction to all images in memory."""
        for relative_path, img in self.images.items():
            gamma_corrected = cv2.pow(img / 255.0, gamma) * 255.0
            self.images[relative_path] = gamma_corrected
            self.operations[relative_path].append(f"gamma_{gamma}")  # Registrar operação

    def gaussian_smoothing(self, kernel_size=(5, 5)):
        """Apply Gaussian smoothing to all images in memory."""
        for relative_path, img in self.images.items():
            self.images[relative_path] = cv2.GaussianBlur(img, kernel_size, 0)
            self.operations[relative_path].append(f"gaussian_blur_{kernel_size[0]}x{kernel_size[1]}")  # Registrar operação
    
    def convert_to_grayscale(self):
        """Convert all images in the dataset to grayscale."""
        for relative_path, img in self.images.items():
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Converter a imagem para escala de cinza
            self.images[relative_path] = gray_img  # Atualizar a imagem no dataset
            self.operations[relative_path].append("grayscale")  # Registrar a operação

    def laplacian_sharpening(self):
        """Apply Laplacian sharpening to all images in memory."""
        for relative_path, img in self.images.items():
            if len(img.shape) == 2:  # Grayscale
                laplacian_img = cv2.Laplacian(img, cv2.CV_64F)
            else:  # Color
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                laplacian_img = cv2.Laplacian(gray_img, cv2.CV_64F)
            
            # Converta a imagem de volta para uint8
            laplacian_img = np.absolute(laplacian_img)  # Converte valores negativos para positivos
            laplacian_img = np.uint8(laplacian_img)  # Converte para uint8
            
            # Se a imagem original era colorida, converter para BGR para manter as 3 dimensões
            if len(img.shape) == 3:  # Color
                laplacian_img = cv2.cvtColor(laplacian_img, cv2.COLOR_GRAY2BGR)
            
            self.images[relative_path] = laplacian_img
            self.operations[relative_path].append("laplacian_sharpening")
    
    def light_intensity_change(self, a):
        """Apply light intensity change (multiplication by factor `a`) to all images."""
        for relative_path, img in self.images.items():
            img = img * a  # Multiplicar os canais R, G, B por a
            img = np.clip(img, 0, 255)  # Garantir que os valores estejam no intervalo [0, 255]
            self.images[relative_path] = img.astype(np.uint8)  # Garantir que a imagem esteja em uint8
            self.operations[relative_path].append(f"light_intensity_change_a_{a}")

    def light_intensity_shift(self, o1):
        """Apply light intensity shift (add constant `o1`) to all images."""
        for relative_path, img in self.images.items():
            img = img + o1  # Somar a constante o1 a cada canal R, G, B
            img = np.clip(img, 0, 255)  # Garantir que os valores estejam no intervalo [0, 255]
            self.images[relative_path] = img.astype(np.uint8)  # Garantir que a imagem esteja em uint8
            self.operations[relative_path].append(f"light_intensity_shift_o1_{o1}")

    def light_intensity_change_and_shift(self, a, o1):
        """Apply light intensity change and shift to all images (multiplication by `a` and shift by `o1`)."""
        for relative_path, img in self.images.items():
            img = (img * a) + o1  # Multiplicar por a e adicionar o1
            img = np.clip(img, 0, 255)  # Garantir que os valores estejam no intervalo [0, 255]
            self.images[relative_path] = img.astype(np.uint8)  # Garantir que a imagem esteja em uint8
            self.operations[relative_path].append(f"light_intensity_change_and_shift_a_{a}_o1_{o1}")

    def show_sample_images(self, num_images=5, max_title_length=10):
        """Display a sample of processed images with their filenames and applied operations."""
        sample_keys = list(self.images.keys())[:num_images]  # Get first N image keys

        plt.figure(figsize=(15, 5))  # Set the figure size

        for i, key in enumerate(sample_keys):
            img = self._convert_to_uint8(self.images[key])  # Converter para uint8 antes de exibir
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for Matplotlib

            # Obter nome da imagem e as operações aplicadas
            base_name = os.path.basename(key)
            if self.operations[key]:
                operations_str = "_".join(self.operations[key])
            else:
                operations_str = ""

            # Limitar o comprimento do nome das operações para evitar que o título fique muito longo
            if len(operations_str) > max_title_length:
                operations_str = operations_str[:max_title_length] + "..."

            # Definir o título como o número da imagem, nome e operações aplicadas
            plt.subplot(1, num_images, i + 1)  # Create subplots
            plt.imshow(img)
            plt.title(f"({i+1}) {base_name} ({operations_str})", fontsize=8)  # Nome e operações
            plt.axis('off')  # Hide the axes

        # Ajustar o espaçamento vertical entre as subplots para evitar sobreposição
        plt.subplots_adjust(hspace=0.6)  # Aumenta o espaço vertical entre as imagens

        plt.tight_layout()
        plt.show()
    
    def show_comparative_sample_images(self, num_images=5, max_title_length=10):
        """Display a comparative sample of images before and after transformation, with concatenated operation names."""
        sample_keys = list(self.images.keys())[:num_images]  # Get first N image keys

        plt.figure(figsize=(15, 10))  # Ajustar o tamanho da figura conforme o número de imagens

        for i, key in enumerate(sample_keys):
            # Obter nome do arquivo sem a extensão
            base_name = os.path.basename(key)
            operations_str = "_".join(self.operations[key]) if self.operations[key] else "original"

            # Limitar o comprimento do título, se necessário
            if len(operations_str) > max_title_length:
                operations_str = operations_str[:max_title_length] + "..."

            # Imagem original (antes da transformação)
            original_img = self._convert_to_uint8(self.original_images[key])  # Converter original para uint8
            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for Matplotlib

            # Imagem processada (depois da transformação)
            processed_img = self._convert_to_uint8(self.images[key])  # Converter processada para uint8
            processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for Matplotlib

            # Mostrar imagem original
            plt.subplot(2, num_images, i + 1)  # Primeira linha: imagens originais
            plt.imshow(original_img)
            plt.title(f"({i+1}) Original: {base_name}", fontsize=10)  # Diminuir tamanho da fonte
            plt.axis('off')

            # Mostrar imagem processada
            plt.subplot(2, num_images, num_images + i + 1)  # Segunda linha: imagens processadas
            plt.imshow(processed_img)
            plt.title(f"({num_images + i}) Processed: {base_name}_{operations_str}", fontsize=8)  # Diminuir tamanho da fonte
            plt.axis('off')

        # Ajustar espaçamento para evitar sobreposição
        
        plt.subplots_adjust(hspace=1.0)  # Aumentar o espaço vertical entre os subplots
        plt.subplots_adjust(wspace=10.0)
        plt.tight_layout()
        plt.show()
    
    def get_images(self):
        "Return dict with images and their names."
        processed_images = {}

        for key in self.images.keys():
            base_name = os.path.basename(key)
            operations_str = "_".join(self.operations[key]) if self.operations[key] else ""

            image_name_with_operations = f"{base_name}_{operations_str}" if operations_str else base_name
            processed_images[image_name_with_operations] = self.images[key]

        return processed_images
    
    ###### EXTRAIR FEATURES #########
    
    def extract_rgb_histogram(self):
        """Extract RGB histogram features from all images."""
        features = {}
        for relative_path, img in self.images.items():
            # Calcular o histograma para os canais R, G, B
            hist_r = cv2.calcHist([img], [0], None, [256], [0, 256])
            hist_g = cv2.calcHist([img], [1], None, [256], [0, 256])
            hist_b = cv2.calcHist([img], [2], None, [256], [0, 256])
            # Concatenar os histogramas em uma única feature
            hist_rgb = np.concatenate((hist_r, hist_g, hist_b)).flatten()
            features[relative_path] = hist_rgb
        return features

    def extract_o1_o2(self):
        """Extract O1, O2 features (used to separate intensity of colors)."""
        features = {}
        for relative_path, img in self.images.items():
            # Convertendo para YUV e separando o canal de intensidade (Y)
            img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            Y = img_yuv[:, :, 0]  # Intensidade
            U = img_yuv[:, :, 1]  # O1
            V = img_yuv[:, :, 2]  # O2
            features[relative_path] = (U.mean(), V.mean())  # Média de O1 e O2
        return features

    def extract_color_transform(self):
        """Apply color transform (e.g., convert to HSV) and return mean of each channel."""
        features = {}
        for relative_path, img in self.images.items():
            # Convertendo para HSV
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            # Calcular a média de cada canal HSV
            H_mean = img_hsv[:, :, 0].mean()
            S_mean = img_hsv[:, :, 1].mean()
            V_mean = img_hsv[:, :, 2].mean()
            features[relative_path] = (H_mean, S_mean, V_mean)  # Retornar a média de H, S, V
        return features


if __name__ == "__main__":
    input_directory = 'data/bracs-binary/train'

    processor = ImageProcessor(input_directory)
    processor.load_images()

    processor.resize_images()
    processor.gamma_correction(1.5)

    processor.show_sample_images(num_images=5)

    processor.save_images()