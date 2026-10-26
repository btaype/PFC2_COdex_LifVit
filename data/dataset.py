from pathlib import Path

import torch
from torch.utils.data import Dataset

from .augmentation import build_transform
from .preprocessing import check_prepared_face, load_rgb


class FaceDataset(Dataset):
    def __init__(self, records, root, config, split):
        self.records = [r for r in records if r["split"] == split]
        if not self.records:
            raise ValueError(f"No hay imágenes en {split}")
        self.root = Path(root)
        self.transform = build_transform(config, train=split == "train")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        image = load_rgb(self.root / row["path"])
        check_prepared_face(image)
        return {"image": self.transform(image), "label": torch.tensor([row["label"]], dtype=torch.float32),
                "generator": row["generator"], "path": row["path"]}
