import argparse
import json

from utils.config import resolve
from utils.reproducibility import environment, save_json


def main():
    parser = argparse.ArgumentParser(description="Detecta versiones, CUDA y memoria; no instala ni descarga nada")
    parser.add_argument("--output", default="docs/environment.json")
    args = parser.parse_args()
    info = environment()
    save_json(resolve(args.output), info)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
