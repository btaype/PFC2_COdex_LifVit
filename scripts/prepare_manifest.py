import argparse
from collections import Counter

from data.splits import build_manifest
from utils.config import resolve


def main():
    parser = argparse.ArgumentParser(description="Indexa rostros ya recortados. Rechaza fugas y duplicados.")
    parser.add_argument("--root", default="datasets/diffusion")
    parser.add_argument("--output", default="manifests/diffusion.csv")
    parser.add_argument("--metadata", help="CSV opcional con path,group_id y procedencia conocida")
    parser.add_argument("--require-group-ids", action="store_true", help="Exigir group_id en todas las imágenes")
    args = parser.parse_args()
    try:
        records = build_manifest(resolve(args.root), resolve(args.output), resolve(args.metadata) if args.metadata else None,
                                 require_groups=args.require_group_ids)
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        parser.exit(2, f"Datos no preparados: {exc}\n")
    print(dict(Counter((r["split"], r["generator"]) for r in records)))
    print(f"Manifiesto guardado: {resolve(args.output)}")


if __name__ == "__main__":
    main()
