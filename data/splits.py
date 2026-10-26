"""Manifiestos auditables. Nunca se infiere la identidad desde el nombre del archivo."""
import csv
import hashlib
from pathlib import Path

from PIL import Image, ImageOps

from models.weights import file_hash
from utils.config import resolve

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
FIELDS = ["path", "split", "label", "generator", "group_id", "sha256", "pixel_sha256", "source_id", "generator_checkpoint", "sampler"]


def image_hash(path):
    with Image.open(path) as image:
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        return hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()


def safe_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Ruta fuera del dataset: {relative}")
    return path


def folder_images(root):
    root = Path(root)
    return sorted(path for split in ("train", "val", "test")
                  for path in (root / split).rglob("*") if path.is_file() and path.suffix.lower() in EXTENSIONS)


def build_manifest(root, destination, metadata=None, require_groups=False):
    root, destination = Path(root), Path(destination)
    if destination.exists():
        raise FileExistsError(f"El manifiesto ya existe: {destination}; elige otro nombre")
    meta = {}
    if metadata:
        with Path(metadata).open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                key = row["path"].replace("\\", "/")
                if key in meta:
                    raise ValueError(f"Metadatos repetidos: {key}")
                meta[key] = row
    records = []
    for path in folder_images(root):
        relative = path.relative_to(root).as_posix()
        parts = path.relative_to(root).parts
        split = parts[0]
        if parts[1] == "real":
            label, generator = 0, "real"
        elif parts[1] == "fake" and len(parts) >= 4:
            label, generator = 1, parts[2].lower()
        else:
            raise ValueError(f"Usa split/real/imagen o split/fake/generador/imagen: {relative}")
        row = {"path": relative, "split": split, "label": label, "generator": generator,
               "sha256": file_hash(path), "pixel_sha256": image_hash(path)}
        for key in ("group_id", "source_id", "generator_checkpoint", "sampler"):
            row[key] = meta.get(relative, {}).get(key) or ""
        records.append(row)
    if not records:
        raise ValueError(f"No hay imágenes todavía en {root}. Agrega los datasets antes de crear el manifiesto.")
    unused = set(meta) - {r["path"] for r in records}
    if unused:
        raise ValueError(f"Metadatos sin archivo: {sorted(unused)[:10]}")
    validate_records(records, require_groups=require_groups)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)
    return records


def validate_records(records, require_groups=False):
    if not records:
        raise ValueError("Manifiesto vacío")
    paths, groups, contents, pixels = set(), {}, {}, {}
    for row in records:
        path, split, label = row["path"], row["split"], int(row["label"])
        if split not in ("train", "val", "test") or label not in (0, 1):
            raise ValueError(f"Split/etiqueta inválidos: {row}")
        if (label == 0) != (row["generator"] == "real"):
            raise ValueError(f"Etiqueta/generador incoherentes: {path}")
        if path in paths:
            raise ValueError(f"Ruta repetida: {path}")
        paths.add(path)
        group = row.get("group_id", "").strip()
        if require_groups and not group:
            raise ValueError(f"Falta group_id de identidad/origen en {path}; completa metadata.csv")
        if group:
            if group in groups and groups[group] != split:
                raise ValueError(f"Fuga de identidad/origen entre particiones: {group}")
            groups[group] = split
        for seen, key in ((contents, "sha256"), (pixels, "pixel_sha256")):
            digest = row.get(key, "")
            if not digest:
                raise ValueError(f"Falta {key} en {path}")
            if digest in seen:
                raise ValueError(f"Imagen duplicada ({key}): {path} y {seen[digest]}")
            seen[digest] = path


def read_manifest(path, root, require_groups=False, verify_files=True):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        records = list(csv.DictReader(stream))
    validate_records(records, require_groups)
    for row in records:
        row.setdefault("group_id", "")
        row["label"] = int(row["label"])
        file = safe_path(root, row["path"])
        if verify_files and (not file.is_file() or file_hash(file) != row["sha256"] or image_hash(file) != row["pixel_sha256"]):
            raise ValueError(f"Archivo ausente o modificado respecto al manifiesto: {file}")
    return records


def load_records(data_config, create_if_missing=False):
    """Indexa carpetas sin inventar grupos; conserva inmutable un manifiesto existente."""
    root, manifest = resolve(data_config["root"]), resolve(data_config["manifest"])
    strict = data_config.get("require_group_ids", False)
    if not manifest.exists() and create_if_missing and data_config.get("auto_manifest", True):
        metadata = data_config.get("metadata_path")
        build_manifest(root, manifest, resolve(metadata) if metadata else None, require_groups=strict)
    records = read_manifest(manifest, root, require_groups=strict)
    actual = {path.relative_to(root).as_posix() for path in folder_images(root)}
    indexed = {r["path"] for r in records}
    if actual != indexed:
        raise ValueError("Las carpetas cambiaron respecto al manifiesto. Crea otro manifiesto para los nuevos datos; no se modifica el de un entrenamiento existente.")
    return records


def group_audit(records):
    known = sum(bool(r.get("group_id", "").strip()) for r in records)
    return {"total_images": len(records), "images_with_group_id": known, "images_without_group_id": len(records) - known,
            "declared_groups_checked": len({r["group_id"].strip() for r in records if r.get("group_id", "").strip()}),
            "group_check_coverage": "complete" if known == len(records) else "partial" if known else "unavailable",
            "identity_independence": "not_proven_by_folder_layout_or_hashes"}


def validate_protocol(records, cfg):
    for split, key in (("train", "train_generators"), ("val", "validation_generators")):
        rows = [r for r in records if r["split"] == split]
        expected = set(cfg["data"][key])
        present = {r["generator"] for r in rows if r["label"] == 1}
        if present != expected or not any(r["label"] == 0 for r in rows):
            raise ValueError(f"{split}: se necesitan reales y exactamente {sorted(expected)}; encontrados {sorted(present)}")
