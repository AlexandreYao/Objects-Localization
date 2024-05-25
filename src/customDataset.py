import os
import cv2
import glob
import torch
import numpy as np
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, image_folders, image_extension=".jpg", image_size=416):
        """
        Initialise un CustomDataset.
        Args:
            image_folders (list): Liste des chemins vers les dossiers contenant les images et les annotations.
            image_size (int, optional): Taille des images. Par défaut, 416.
        """
        self.image_folders = image_folders
        self.image_size = image_size
        self.image_extension = image_extension
        self.data = self.prepare_data()

    def __len__(self):
        """
        Renvoie la taille du jeu de données.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Renvoie un échantillon du jeu de données.
        Args:
            idx (int): Indice de l'échantillon à récupérer.
        Returns:
            dict: Un dictionnaire contenant l'image à utiliser en pytorch, l'image original, ses annotations et le chemin vers l'image.
        """
        # Récupère les chemins de l'image et des annotations pour l'index donné
        image_path, annotations_path = self.data[idx]
        # Charge l'image à partir du chemin
        image = cv2.imread(image_path)
        # Récupère les dimensions de l'image
        height, width, _ = image.shape
        # Redimensionne l'image à la taille spécifiée
        image_resized = cv2.resize(image, (self.image_size, self.image_size))
        # Transpose les dimensions de l'image et convertit en tenseur PyTorch
        input_img = image_resized[:, :, ::-1].transpose((2, 0, 1)).copy()
        input_img = torch.from_numpy(input_img).float().div(255.0)
        # Charge les annotations de l'image
        filled_labels = self.load_annotations(
            annotation_file=annotations_path,
            orig_height=height,
            orig_width=width,
        )
        # Crée un dictionnaire contenant l'échantillon d'image et ses informations associées
        sample = {
            "input_img": input_img,
            "label": filled_labels,
        }
        return sample

    def load_annotations(self, annotation_file, orig_height, orig_width):
        """
        Charge les annotations à partir d'un fichier.
        Args:
            annotation_file (str): Chemin vers le fichier d'annotations.
            orig_height (int): Hauteur de l'image d'origine.
            orig_width (int): Largeur de l'image d'origine.
        Returns:
            torch.Tensor: Annotations remplies.
        """
        # Charge les annotations depuis le fichier et les reshape pour obtenir un tableau 2D
        labels = np.loadtxt(annotation_file).reshape(-1, 5)
        # Convertit les coordonnées YOLO en coordonnées des coins
        x_center = labels[:, 1] * orig_width
        y_center = labels[:, 2] * orig_height
        width = labels[:, 3] * orig_width
        height = labels[:, 4] * orig_height
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        # Redimensionne les coordonnées des coins pour correspondre à l'image redimensionnée
        x1 = x1 * self.image_size / orig_width
        y1 = y1 * self.image_size / orig_height
        x2 = x2 * self.image_size / orig_width
        y2 = y2 * self.image_size / orig_height
        # Convertit les coordonnées des coins en coordonnées YOLO normalisées
        new_x_center = (x1 + x2) / 2 / self.image_size
        new_y_center = (y1 + y2) / 2 / self.image_size
        new_width = (x2 - x1) / self.image_size
        new_height = (y2 - y1) / self.image_size
        labels[:, 1] = new_x_center
        labels[:, 2] = new_y_center
        labels[:, 3] = new_width
        labels[:, 4] = new_height
        # Convertit le tableau numpy en tensor PyTorch et le retourne
        filled_labels = torch.from_numpy(labels).float()
        return filled_labels

    def prepare_data(self):
        """
        Prépare les données en associant chaque image à ses annotations.
        Returns:
            list: Liste des paires (chemin de l'image, chemin de l'annotation) pour chaque image valide.
        """
        # Initialise une liste pour stocker les paires image-annotations
        data = []
        # Parcourt chaque dossier contenant des images
        for image_folder in self.image_folders:
            annotation_folder = image_folder
            # Parcourt chaque fichier d'annotation dans le dossier
            for annotations_path in glob.glob(os.path.join(annotation_folder, "*.txt")):
                # Construit le chemin de l'image correspondant au fichier d'annotation
                image_name = (
                    os.path.splitext(os.path.basename(annotations_path))[0]
                    + self.image_extension
                )
                image_path = os.path.join(image_folder, image_name)
                # Vérifie si l'image correspondante existe
                if os.path.exists(image_path):
                    # Ajoute la paire image-annotation à la liste des données
                    data.append((image_path, annotations_path))
        # Renvoie la liste complète des paires image-annotation
        return data