"""Descarga e importa HASYv2 con sus IDs de origen y etiquetas originales."""

import argparse
import tarfile

from src.catalog import PROJECT_ROOT
from src.hasy_import import download_hasy, prepare_hasy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", default=PROJECT_ROOT / "data" / "external" / "HASYv2.tar.bz2")
    parser.add_argument("--output", default=PROJECT_ROOT / "data" / "hasy")
    parser.add_argument("--per-class", type=int, default=800)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        if args.per_class < 1:
            raise ValueError("--per-class debe ser positivo.")
        from pathlib import Path
        if Path(args.output).exists():
            raise ValueError("El destino ya existe; conserva el subconjunto o indica otro --output.")
        print("Comprobando la descarga oficial HASYv2 (34.6 MB)...", flush=True)
        archive = download_hasy(args.archive)
        print("Seleccionando las seis clases, validando IDs de origen y descartando duplicados...", flush=True)
        metadata = prepare_hasy(archive, args.output, args.per_class, args.seed)
        print(f"Preparadas {metadata['n_samples']} imágenes con {metadata['n_participants']} IDs de origen.")
        print("Los IDs de HASY no equivalen a personas; la evaluación se separa por imágenes.")
        print(f"Datos: {args.output}/processed")
        print("Origen: HASYv2 público; las capturas propias permanecen separadas.")
        return 0
    except (ValueError, OSError, KeyError, tarfile.ReadError) as error:
        print(f"No se pudo preparar HASY: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
