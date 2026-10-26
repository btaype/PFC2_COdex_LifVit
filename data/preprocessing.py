from PIL import Image, ImageOps


def load_rgb(path):
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def check_prepared_face(image):
    if min(image.size) < 32:
        raise ValueError("Imagen demasiado pequeña para un rostro preparado")
