from PIL import Image, ImageDraw
import os
import random


def create_image_with_rectangle(image_size):
    # Créer une image noire (L mode = 8-bit pixels, black and white)
    img = Image.new("L", image_size, 0)
    draw = ImageDraw.Draw(img)
    # Générer aléatoirement la position et la taille du rectangle
    x1 = random.randint(0, image_size[0] // 2)
    y1 = random.randint(0, image_size[1] // 2)
    x2 = random.randint(image_size[0] // 2, image_size[0])
    y2 = random.randint(image_size[1] // 2, image_size[1])
    # S'assurer que x2 > x1 et y2 > y1
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    # Dessiner le rectangle
    draw.rectangle([x1, y1, x2, y2], outline=255, fill=255)
    # Convertir l'image en RGB (3 canaux)
    img_rgb = img.convert("RGB")
    # Calculer les coordonnées normalisées pour le format YOLO
    x_center = (x1 + x2) / 2 / image_size[0]
    y_center = (y1 + y2) / 2 / image_size[1]
    width = (x2 - x1) / image_size[0]
    height = (y2 - y1) / image_size[1]
    bbox = (0, x_center, y_center, width, height)  # Classe 0 (unique classe)
    return img_rgb, bbox


# Configuration
image_size = (100, 100)
output_dirs = {
    "train": (r"..\data\black-and-white-rectangle\train", 1000),
    "val": (r"..\data\black-and-white-rectangle\val", 200),
}
for key, (output_dir, num_images) in output_dirs.items():
    print(key)
    # Créer le répertoire de sortie s'il n'existe pas
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Générer les images et les annotations
    for i in range(num_images):
        img, bbox = create_image_with_rectangle(image_size)
        img_path = os.path.join(output_dir, f"image_{i}.png")
        annotation_path = os.path.join(output_dir, f"image_{i}.txt")
        img.save(img_path)
        # Écrire l'annotation au format YOLO dans un fichier séparé
        bbox_str = " ".join(map(str, bbox))
        with open(annotation_path, "w") as f:
            f.write(f"{bbox_str}\n")
        print("\t", img_path)
print("Génération d'images et annotations terminée.")