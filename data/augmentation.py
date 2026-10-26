import io
import random

from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode


class JPEGCompression:
    def __init__(self, probability, quality):
        self.probability, self.quality = probability, quality

    def __call__(self, image):
        if random.random() >= self.probability:
            return image
        stream = io.BytesIO()
        image.save(stream, format="JPEG", quality=random.randint(*self.quality))
        stream.seek(0)
        with Image.open(stream) as compressed:
            return compressed.convert("RGB")


def build_transform(config, train=False):
    a = config["augmentation"]
    operations = []
    if train:
        operations.extend([
            T.RandomResizedCrop(224, scale=tuple(a["crop_scale"]), ratio=tuple(a["crop_ratio"]), interpolation=InterpolationMode.BICUBIC, antialias=True),
            T.RandomHorizontalFlip(a["horizontal_flip"]),
            T.RandomApply([T.ColorJitter(a["brightness"], a["contrast"], a["saturation"], a["hue"])], p=a["color_jitter_probability"]),
            T.RandomApply([T.GaussianBlur(3, tuple(a["blur_sigma"]))], p=a["blur_probability"]),
            JPEGCompression(a["jpeg_probability"], a["jpeg_quality"]),
        ])
    else:
        operations.append(T.Resize((224, 224), interpolation=InterpolationMode.BICUBIC, antialias=True))
    return T.Compose(operations + [T.ToTensor()])
