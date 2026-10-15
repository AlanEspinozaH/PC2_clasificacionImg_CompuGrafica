"""Inspecciona la colección preparada sin necesitar PyTorch ni entrenar."""

import argparse
from collections import Counter

from src.catalog import PROJECT_ROOT
from src.training_data import load_dataset, split_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=PROJECT_ROOT / "data" / "processed")
    args = parser.parse_args()
    try:
        dataset = load_dataset(args.data)
        print(f"Origen: {dataset.source_description} ({dataset.source_type})")
        print(f"Muestras utilizables: {len(dataset.y)}; pruebas DEMO excluidas: {dataset.excluded_demo}")
        print("Categoría   Muestras   Meta")
        for symbol in dataset.catalog["symbols"]:
            count = int((dataset.y == symbol["class_index"]).sum())
            print(f"{symbol['id']:<11} {count:>7} {dataset.catalog['target_per_class']:>6}")
        identity = "ID de origen" if dataset.source_type == "hasyv2" else "Participante"
        print(f"\n{identity:<13} alpha   xi   infty   equiv   partial   in")
        for participant in sorted(set(dataset.participants.tolist())):
            counts = Counter(dataset.y[dataset.participants == participant].tolist())
            print(f"{participant:<13} " + "  ".join(f"{counts[i]:>5}" for i in range(6)))
        try:
            split_dataset(dataset)
            if dataset.source_type == "hasyv2":
                print("\nPartición estratificada por imágenes; los user_id de HASY no son personas fiables.")
            else:
                print("\nLa colección admite una partición por persona. Revisa también cantidad y etiquetas.")
        except ValueError as error:
            print(f"\nAntes de entrenar: {error}")
        return 0
    except (ValueError, OSError, KeyError) as error:
        print(f"No se pudo inspeccionar: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
