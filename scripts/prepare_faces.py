import argparse
import csv
from pathlib import Path

from PIL import Image

from data.preprocessing import load_rgb
from data.splits import safe_path
from utils.config import resolve


def main():
    parser = argparse.ArgumentParser(description='Recorta imágenes con cajas externas explícitas; no inventa un detector facial.')
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--boxes", required=True, help="CSV path,x,y,width,height; una caja auditada por imagen")
    parser.add_argument("--margin", type=float, default=0.2, help="Fracción del lado añadida por cada borde")
    args = parser.parse_args()
    if args.margin < 0:
        parser.error("El margen no puede ser negativo")
    input_root, output_root = resolve(args.input_root).resolve(), resolve(args.output_root).resolve()
    if output_root == input_root or output_root.is_relative_to(input_root) or input_root.is_relative_to(output_root):
        parser.error("Las carpetas de entrada y salida deben estar separadas")
    if output_root.exists():
        parser.error("Elige una carpeta de salida nueva para no sobrescribir rostros")
    with resolve(args.boxes).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or len({r["path"] for r in rows}) != len(rows):
        parser.error("Se requiere una caja única por imagen y al menos una imagen")
    plans, destinations = [], set()
    for row in rows:
        source = safe_path(input_root, row["path"])
        destination = safe_path(output_root, str(Path(row["path"]).with_suffix(".png")))
        if destination in destinations:
            parser.error(f"Dos imágenes producen el mismo PNG: {destination}")
        destinations.add(destination)
        image = load_rgb(source)
        x, y, w, h = [float(row[k]) for k in ("x", "y", "width", "height")]
        if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > image.width or y + h > image.height:
            parser.error(f"Caja fuera de la imagen: {source}")
        side = max(w, h) * (1 + 2 * args.margin)
        cx, cy = x + w / 2, y + h / 2
        box = (max(0, round(cx - side / 2)), max(0, round(cy - side / 2)), min(image.width, round(cx + side / 2)), min(image.height, round(cy + side / 2)))
        plans.append((source, destination, box))
    output_root.mkdir(parents=True, exist_ok=False)
    for source, destination, box in plans:
        image = load_rgb(source).crop(box).resize((224, 224), Image.Resampling.BICUBIC)
        destination.parent.mkdir(parents=True, exist_ok=True)
        image.save(destination, format="PNG")
    print(f"Rostros guardados: {len(plans)}; política uniforme; salida: {output_root}")


if __name__ == "__main__":
    main()
